import os
import json
import subprocess
import shutil
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request
import urllib.parse

app = FastAPI(title="Benetti Agent Core", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_ROOT = os.path.abspath(os.environ.get("WORKSPACE_ROOT", "./workspace"))
DEFAULT_PIN = os.environ.get("AGENT_PIN", "2026")
GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

os.makedirs(WORKSPACE_ROOT, exist_ok=True)

def verify_pin(x_pin: str = Header(None)):
    if not x_pin or x_pin.strip() != DEFAULT_PIN:
        raise HTTPException(status_code=401, detail="PIN non valido.")
    return x_pin

class CommandRequest(BaseModel):
    command: str
    cwd: str = ""

class FileWriteRequest(BaseModel):
    filepath: str
    content: str

class FileReadRequest(BaseModel):
    filepath: str

class AgentTaskRequest(BaseModel):
    instruction: str
    context: str = ""

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Benetti Agent Core",
        "workspace": WORKSPACE_ROOT
    }

# --- TOOL 1: ESECUZIONE COMANDI DA TERMINALE LINUX ---
@app.post("/api/tools/run-command", dependencies=[Depends(verify_pin)])
def run_command(req: CommandRequest):
    work_dir = os.path.abspath(os.path.join(WORKSPACE_ROOT, req.cwd)) if req.cwd else WORKSPACE_ROOT
    if not work_dir.startswith(WORKSPACE_ROOT):
        raise HTTPException(status_code=400, detail="Percorso di lavoro non consentito.")

    try:
        res = subprocess.run(
            req.command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        return {
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "Errore: comando interrotto per timeout (max 120s)."}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}

# --- TOOL 2: SCRITTURA FILE E CREAZIONE PROGETTI ---
@app.post("/api/tools/write-file", dependencies=[Depends(verify_pin)])
def write_file(req: FileWriteRequest):
    full_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, req.filepath.lstrip("/\\")))
    if not full_path.startswith(WORKSPACE_ROOT):
        raise HTTPException(status_code=400, detail="Percorso non consentito.")

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(req.content)

    return {"success": True, "filepath": req.filepath, "bytes": len(req.content)}

# --- TOOL 3: LETTURA FILE ---
@app.post("/api/tools/read-file", dependencies=[Depends(verify_pin)])
def read_file(req: FileReadRequest):
    full_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, req.filepath.lstrip("/\\")))
    if not full_path.startswith(WORKSPACE_ROOT) or not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File non trovato.")

    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return {"filepath": req.filepath, "content": content}

# --- TOOL 4: LISTA FILE NEL WORKSPACE ---
@app.get("/api/tools/list-files", dependencies=[Depends(verify_pin)])
def list_files(subdir: str = ""):
    target_dir = os.path.abspath(os.path.join(WORKSPACE_ROOT, subdir.lstrip("/\\")))
    if not target_dir.startswith(WORKSPACE_ROOT) or not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail="Directory non trovata.")

    items = []
    for root, dirs, files in os.walk(target_dir):
        rel_root = os.path.relpath(root, WORKSPACE_ROOT)
        for d in dirs:
            items.append({"name": d, "type": "dir", "path": os.path.join(rel_root, d).replace("\\", "/")})
        for f in files:
            items.append({"name": f, "type": "file", "path": os.path.join(rel_root, f).replace("\\", "/")})
    return {"workspace": WORKSPACE_ROOT, "items": items}

# --- TOOL 5: LOOP AUTONOMO DI RAGIONAMENTO & AZIONE (AGENTIC LOOP) ---
AGENT_TOOLS_SCHEMA = [
    {
        "name": "run_terminal_command",
        "description": "Esegue un comando nel terminale Linux (es. git, npm, python, pip, mkdir, curl, build).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "Il comando da eseguire nella shell."}
            },
            "required": ["command"]
        }
    },
    {
        "name": "write_project_file",
        "description": "Crea o modifica un file all'interno del workspace del progetto.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "filepath": {"type": "STRING", "description": "Percorso relativo del file (es: src/index.html)."},
                "content": {"type": "STRING", "description": "Contenuto completo del file."}
            },
            "required": ["filepath", "content"]
        }
    },
    {
        "name": "read_project_file",
        "description": "Legge il contenuto di un file esistente.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "filepath": {"type": "STRING", "description": "Percorso del file da leggere."}
            },
            "required": ["filepath"]
        }
    }
]

def call_gemini_raw(contents: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload: Dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{
                "text": "Sei l'Agente Autonomo per la famiglia Benetti. Hai a disposizione un sistema operativo Linux e strumenti (Tools) per creare file, eseguire comandi da terminale, compilare progetti e testare siti web. Quando ti viene chiesto di realizzare un progetto, usa i Tool per creare fisicamente i file necessari ed eseguire comandi. Spiega sempre cosa hai realizzato in italiano chiaro, strutturato e professionale."
            }]
        },
        "generationConfig": {"temperature": 0.4}
    }
    if tools:
        payload["tools"] = [{"functionDeclarations": tools}]

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))

@app.post("/api/agent/run-task", dependencies=[Depends(verify_pin)])
def run_agent_task(req: AgentTaskRequest):
    """
    Esegue un compito autonomo con ciclo ReAct: Gemini decide quali strumenti usare,
    il server li esegue e invia i risultati a Gemini fino al completamento.
    """
    history: List[Dict[str, Any]] = [
        {"role": "user", "parts": [{"text": req.instruction}]}
    ]

    action_log = []
    max_steps = 8

    for step in range(max_steps):
        res = call_gemini_raw(history, AGENT_TOOLS_SCHEMA)
        candidate = res.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        history.append({"role": "model", "parts": parts})

        function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not function_calls:
            # Nessuna ulteriore azione richiesta, risposta finale
            final_text = "".join(p.get("text", "") for p in parts)
            return {
                "success": True,
                "reply": final_text,
                "action_log": action_log
            }

        tool_responses = []
        for fc in function_calls:
            fn_name = fc["name"]
            fn_args = fc.get("args", {})

            if fn_name == "run_terminal_command":
                cmd = fn_args.get("command", "")
                action_log.append(f"Esecuzione comando: {cmd}")
                cmd_res = run_command(CommandRequest(command=cmd))
                out = cmd_res.get("stdout", "") + "\n" + cmd_res.get("stderr", "")
                tool_responses.append({
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"output": out[:2000], "exit_code": cmd_res.get("exit_code")}
                    }
                })

            elif fn_name == "write_project_file":
                fp = fn_args.get("filepath", "")
                data_cnt = fn_args.get("content", "")
                action_log.append(f"Scrittura file: {fp} ({len(data_cnt)} bytes)")
                wr_res = write_file(FileWriteRequest(filepath=fp, content=data_cnt))
                tool_responses.append({
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"success": wr_res.get("success"), "filepath": fp}
                    }
                })

            elif fn_name == "read_project_file":
                fp = fn_args.get("filepath", "")
                action_log.append(f"Lettura file: {fp}")
                try:
                    rd_res = read_file(FileReadRequest(filepath=fp))
                    tool_responses.append({
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"content": rd_res.get("content", "")[:3000]}
                        }
                    })
                except Exception as ex:
                    tool_responses.append({
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"error": str(ex)}
                        }
                    })

        history.append({"role": "user", "parts": tool_responses})

    return {
        "success": True,
        "reply": "Raggiunto il numero massimo di iterazioni per il task.",
        "action_log": action_log
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

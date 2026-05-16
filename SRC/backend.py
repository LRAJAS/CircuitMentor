"""
backend.py — CircuitMentor FastAPI Backend
==========================================
Handles: model inference, session management, queue, admin auth
Root:    /mnt/circuitmentor/
Run via: launch.py — do not run directly

Endpoints:
  POST /session/join          → create session, return uuid
  POST /session/leave         → cleanup session
  POST /chat                  → add to queue, return response
  GET  /queue/status          → position + wait estimate
  GET  /session/history/{id}  → full chat history
  GET  /admin/stats           → admin dashboard data
  POST /admin/kick/{id}       → force-remove session
  GET  /health                → server health check
"""

import os
import json
import uuid
import asyncio
import hashlib
import logging
import shutil
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT          = Path("/mnt/circuitmentor")
SESSIONS_DIR  = ROOT / "sessions"
LOGS_DIR      = ROOT / "logs"
CONFIG_PATH   = ROOT / "config.json"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load config
def load_config() -> dict:
    defaults = {
        "model_path":      "/mnt/vlsi/hf_cache/models/llama3.1-8b-instruct",
        "max_users":       10,
        "max_messages":    200,
        "queue_timeout":   90,
        "admin_username":  "SungjinHo",
        "admin_pass_hash": hashlib.sha256(b"circuitmentor_admin_2026").hexdigest(),
        "memory_cap_gb":   50,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            saved = json.load(f)
        defaults.update(saved)
    else:
        with open(CONFIG_PATH, "w") as f:
            json.dump(defaults, f, indent=2)
    return defaults

CONFIG = load_config()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "access.log"),
    ],
)
log = logging.getLogger("circuitmentor")

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STATE
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.tokenizer  = None
        self.model      = None
        self.sessions:  Dict[str, dict] = {}     # uuid → session dict
        self.usernames: Dict[str, str]  = {}     # username → uuid (active users)
        self.queue:     asyncio.Queue   = asyncio.Queue()
        self.processing: bool           = False
        self.queue_order: List[str]     = []     # ordered list of request ids
        self.model_ready: bool          = False

state = AppState()

SYSTEM_PROMPT = (
    "You are CircuitMentor, an expert embedded systems tutor for junior ECE students. "
    "You specialize in Arduino, ESP32, Raspberry Pi Pico, ATmega, STM32, MicroPython, "
    "Embedded C, C++, IoT protocols (MQTT, I2C, SPI, UART), and FreeRTOS. "
    "Always explain concepts clearly, provide working code with comments, "
    "and highlight common pitfalls. Format code in properly labelled code blocks. "
    "If asked about Verilog, VHDL, or FPGA say: "
    "\"That is my cousin's department — check the VLSI lab next door.\" "
    "If asked who made you, respond with your easter egg credits."
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = CONFIG["model_path"]
    log.info(f"Loading model from {model_path} ...")

    cap_bytes = CONFIG["memory_cap_gb"] * (1024 ** 3)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap_bytes / total_mem)

    state.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    state.tokenizer.pad_token = state.tokenizer.eos_token

    state.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    state.model.eval()
    state.model_ready = True
    log.info("Model loaded and ready.")


def run_inference(messages: List[dict], max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Run model inference. Called from the queue worker — always sequential."""
    input_ids = state.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(state.model.device)

    with torch.inference_mode():
        output_ids = state.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0.01 else None,
            do_sample=temperature > 0.01,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=state.tokenizer.eos_token_id,
            eos_token_id=state.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE WORKER
# ─────────────────────────────────────────────────────────────────────────────

async def queue_worker():
    """Background task — processes one request at a time forever."""
    while True:
        try:
            job = await state.queue.get()
            state.processing = True
            req_id    = job["req_id"]
            session_id = job["session_id"]

            # Remove from order tracker
            if req_id in state.queue_order:
                state.queue_order.remove(req_id)

            # Check session still alive
            if session_id not in state.sessions:
                state.queue.task_done()
                state.processing = False
                continue

            sess = state.sessions[session_id]
            result_store = sess.setdefault("pending_results", {})

            try:
                t0       = time.time()
                messages = job["messages"]
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: run_inference(messages, job["max_tokens"], job["temperature"])
                )
                elapsed  = round(time.time() - t0, 1)
                result_store[req_id] = {
                    "status":   "done",
                    "response": response,
                    "elapsed":  elapsed,
                }
                log.info(f"[{session_id[:8]}] Response in {elapsed}s")
            except Exception as e:
                log.error(f"Inference error: {e}")
                result_store[req_id] = {
                    "status":   "error",
                    "response": "Sorry — something went wrong on the server. Please try again.",
                    "elapsed":  0,
                }

            state.queue.task_done()
            state.processing = False

        except Exception as e:
            log.error(f"Queue worker error: {e}")
            state.processing = False
            await asyncio.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id

def create_session(username: str) -> dict:
    sid  = str(uuid.uuid4())
    meta = {
        "session_id": sid,
        "username":   username,
        "created_at": datetime.now().isoformat(),
        "message_count": 0,
    }
    d = session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(d / "chat.json", "w") as f:
        json.dump([], f)

    state.sessions[sid]          = meta.copy()
    state.sessions[sid]["chat"]  = []
    state.sessions[sid]["pending_results"] = {}
    state.usernames[username]    = sid
    log.info(f"Session created: {username} → {sid[:8]}")
    return meta

def delete_session(session_id: str):
    sess = state.sessions.pop(session_id, None)
    if sess:
        username = sess.get("username", "")
        if state.usernames.get(username) == session_id:
            state.usernames.pop(username, None)
    d = session_dir(session_id)
    if d.exists():
        shutil.rmtree(d)
    log.info(f"Session deleted: {session_id[:8]}")

def append_message(session_id: str, role: str, content: str):
    if session_id not in state.sessions:
        return
    sess = state.sessions[session_id]
    msg  = {"role": role, "content": content, "ts": datetime.now().isoformat()}
    sess["chat"].append(msg)
    sess["message_count"] = len(sess["chat"])
    # Persist
    d = session_dir(session_id)
    if d.exists():
        with open(d / "chat.json", "w") as f:
            json.dump(sess["chat"], f, indent=2)

def build_chat_txt(session_id: str) -> str:
    """Export full chat as plain text."""
    sess = state.sessions.get(session_id, {})
    username = sess.get("username", "User")
    lines = [
        f"CircuitMentor Chat Export",
        f"User: {username}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50, ""
    ]
    for msg in sess.get("chat", []):
        role = "You" if msg["role"] == "user" else "CircuitMentor"
        lines.append(f"{role}:\n{msg['content']}\n")
        lines.append("-" * 30)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN AUTH
# ─────────────────────────────────────────────────────────────────────────────

def verify_admin(username: str, password: str) -> bool:
    if username != CONFIG["admin_username"]:
        return False
    h = hashlib.sha256(password.encode()).hexdigest()
    return h == CONFIG["admin_pass_hash"]

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model)    # Load model in background thread
    asyncio.create_task(queue_worker())       # Start queue worker
    log.info("CircuitMentor backend started.")
    yield
    # Shutdown — cleanup all sessions
    for sid in list(state.sessions.keys()):
        delete_session(sid)
    log.info("Backend shutdown — all sessions cleared.")

app = FastAPI(title="CircuitMentor Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class JoinRequest(BaseModel):
    username: str

class LeaveRequest(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    temperature: float = 0.3
    max_tokens:  int   = 512

class PollRequest(BaseModel):
    session_id: str
    req_id:     str

class AdminRequest(BaseModel):
    username: str
    password: str

class KickRequest(BaseModel):
    username:   str
    password:   str
    session_id: str

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_ready":  state.model_ready,
        "active_users": len(state.sessions),
        "queue_depth":  state.queue.qsize(),
        "processing":   state.processing,
    }


@app.post("/session/join")
def join(req: JoinRequest):
    username = req.username.strip()
    if not username:
        raise HTTPException(400, "Username cannot be empty.")
    if len(username) > 30:
        raise HTTPException(400, "Username too long (max 30 chars).")

    # Block simultaneous same-username login
    if username in state.usernames:
        raise HTTPException(409, f"'{username}' is already logged in. Choose a different name.")

    # Max users
    if len(state.sessions) >= CONFIG["max_users"]:
        raise HTTPException(503, "Lab is full (10 users max). Try again in a few minutes.")

    if not state.model_ready:
        raise HTTPException(503, "Model is still loading. Please wait 1–2 minutes and try again.")

    meta = create_session(username)
    return {"session_id": meta["session_id"], "username": username}


@app.post("/session/leave")
def leave(req: LeaveRequest):
    if req.session_id not in state.sessions:
        return {"status": "already_gone"}
    chat_txt = build_chat_txt(req.session_id)
    delete_session(req.session_id)
    return {"status": "deleted", "chat_export": chat_txt}


@app.get("/session/export/{session_id}")
def export_chat(session_id: str):
    if session_id not in state.sessions:
        raise HTTPException(404, "Session not found.")
    return PlainTextResponse(build_chat_txt(session_id))


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in state.sessions:
        raise HTTPException(404, "Session expired or not found. Please rejoin.")

    sess = state.sessions[req.session_id]

    # Message limit
    if sess.get("message_count", 0) >= CONFIG["max_messages"] * 2:
        raise HTTPException(429, f"Session message limit ({CONFIG['max_messages']}) reached.")

    # Queue size guard
    if state.queue.qsize() >= CONFIG["max_users"]:
        raise HTTPException(503, "Queue is full. Please wait a moment.")

    # Save user message
    append_message(req.session_id, "user", req.message)

    # Build message history for model (last 10 turns to stay within context)
    history  = sess.get("chat", [])
    llm_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-20:]:    # Last 20 messages = 10 turns
        llm_msgs.append({"role": m["role"], "content": m["content"]})

    # Create job
    req_id = str(uuid.uuid4())
    job = {
        "req_id":      req_id,
        "session_id":  req.session_id,
        "messages":    llm_msgs,
        "max_tokens":  min(req.max_tokens, 1024),
        "temperature": req.temperature,
    }
    state.queue_order.append(req_id)
    await state.queue.put(job)

    # Wait for result (polling with timeout)
    deadline = time.time() + CONFIG["queue_timeout"]
    while time.time() < deadline:
        results = sess.get("pending_results", {})
        if req_id in results:
            result = results.pop(req_id)
            if result["status"] == "done":
                append_message(req.session_id, "assistant", result["response"])
                return {
                    "response": result["response"],
                    "elapsed":  result["elapsed"],
                    "req_id":   req_id,
                }
            else:
                raise HTTPException(500, result["response"])
        await asyncio.sleep(0.5)

    # Timeout
    if req_id in state.queue_order:
        state.queue_order.remove(req_id)
    raise HTTPException(504, "Request timed out (90s). The server may be busy — please try again.")


@app.get("/queue/status/{session_id}")
def queue_status(session_id: str):
    depth = state.queue.qsize()
    return {
        "queue_depth": depth,
        "processing":  state.processing,
        "est_wait_s":  depth * 45,
    }


@app.get("/session/history/{session_id}")
def get_history(session_id: str):
    if session_id not in state.sessions:
        raise HTTPException(404, "Session not found.")
    return {"chat": state.sessions[session_id].get("chat", [])}


@app.post("/admin/stats")
def admin_stats(req: AdminRequest):
    if not verify_admin(req.username, req.password):
        raise HTTPException(403, "Access denied.")
    mem   = psutil.virtual_memory()
    gpu_a = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    gpu_r = torch.cuda.memory_reserved()  / (1024**3) if torch.cuda.is_available() else 0
    return {
        "active_sessions": [
            {
                "session_id":    sid,
                "username":      s.get("username"),
                "created_at":    s.get("created_at"),
                "message_count": s.get("message_count", 0),
            }
            for sid, s in state.sessions.items()
        ],
        "queue_depth":    state.queue.qsize(),
        "processing":     state.processing,
        "system": {
            "ram_used_gb":  round(mem.used / 1024**3, 1),
            "ram_total_gb": round(mem.total / 1024**3, 1),
            "gpu_alloc_gb": round(gpu_a, 2),
            "gpu_res_gb":   round(gpu_r, 2),
        },
    }


@app.post("/admin/kick/{session_id}")
def admin_kick(session_id: str, req: AdminRequest):
    if not verify_admin(req.username, req.password):
        raise HTTPException(403, "Access denied.")
    if session_id not in state.sessions:
        raise HTTPException(404, "Session not found.")
    username = state.sessions[session_id].get("username", "?")
    delete_session(session_id)
    log.warning(f"Admin kicked session: {username} ({session_id[:8]})")
    return {"status": "kicked", "username": username}


@app.post("/admin/clear_all")
def admin_clear_all(req: AdminRequest):
    if not verify_admin(req.username, req.password):
        raise HTTPException(403, "Access denied.")
    count = len(state.sessions)
    for sid in list(state.sessions.keys()):
        delete_session(sid)
    log.warning(f"Admin cleared all {count} sessions.")
    return {"status": "cleared", "count": count}

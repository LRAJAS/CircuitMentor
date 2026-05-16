"""
launch.py — CircuitMentor Launcher
====================================
Single command to start the entire CircuitMentor stack:
  1. Creates /mnt/circuitmentor/ directory structure
  2. Starts FastAPI backend (port 8000)
  3. Starts Streamlit frontend (port 8501)
  4. Opens ngrok tunnel → prints public URL
  5. Watches for Ctrl+C → clean shutdown

Usage:
    pip install fastapi uvicorn streamlit pyngrok requests psutil torch transformers peft
    python launch.py
    python launch.py --ngrok-token YOUR_TOKEN   (first time only)
    python launch.py --no-ngrok                 (local only, no public URL)
    python launch.py --port-backend 8000 --port-frontend 8501
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CircuitMentor Launcher")
    p.add_argument("--ngrok-token",    default=None,  help="ngrok authtoken (first run only)")
    p.add_argument("--no-ngrok",       action="store_true", help="Skip ngrok, local only")
    p.add_argument("--port-backend",   type=int, default=8000)
    p.add_argument("--port-frontend",  type=int, default=8501)
    p.add_argument("--model-path",     default="/mnt/vlsi/hf_cache/models/llama3.1-8b-instruct")
    p.add_argument("--memory-cap-gb",  type=int, default=50)
    return p.parse_args()

args = parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT         = Path("/mnt/circuitmentor")
SESSIONS_DIR = ROOT / "sessions"
LOGS_DIR     = ROOT / "logs"
CONFIG_PATH  = ROOT / "config.json"
BACKEND_PATH = Path(__file__).parent / "backend.py"
FRONTEND_PATH= Path(__file__).parent / "frontend.py"

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

BLUE  = "\033[94m"
GREEN = "\033[92m"
AMBER = "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def log(msg, color=RESET):
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_directories():
    for d in [ROOT, SESSIONS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log(f"Directory structure ready at {ROOT}", GREEN)

def write_config():
    import hashlib
    config = {
        "model_path":      args.model_path,
        "max_users":       10,
        "max_messages":    200,
        "queue_timeout":   90,
        "admin_username":  "SungjinHo",
        "admin_pass_hash": hashlib.sha256(b"circuitmentor_admin_2026").hexdigest(),
        "memory_cap_gb":   args.memory_cap_gb,
        "port_backend":    args.port_backend,
        "port_frontend":   args.port_frontend,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Config written to {CONFIG_PATH}", GREEN)
    return config

def check_dependencies():
    required = ["fastapi", "uvicorn", "streamlit", "requests", "torch",
                "transformers", "peft", "psutil"]
    missing  = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log(f"Missing packages: {', '.join(missing)}", RED)
        log(f"Run: pip install {' '.join(missing)}", AMBER)
        sys.exit(1)
    log("All dependencies found.", GREEN)

def check_model():
    model_path = Path(args.model_path)
    if not model_path.exists():
        log(f"Model not found at {args.model_path}", RED)
        log("SCP the model to the Orin first.", AMBER)
        sys.exit(1)
    config_file = model_path / "config.json"
    if not config_file.exists():
        log(f"Model directory incomplete (no config.json).", RED)
        sys.exit(1)
    log(f"Model found at {args.model_path}", GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

processes = []

def start_backend():
    log("Starting FastAPI backend...", BLUE)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            f"{BACKEND_PATH.stem}:app",
            "--host", "0.0.0.0",
            "--port", str(args.port_backend),
            "--workers", "1",       # Single worker — one model instance
            "--loop", "asyncio",
        ],
        cwd=str(BACKEND_PATH.parent),
        stdout=open(LOGS_DIR / "backend.log", "a"),
        stderr=subprocess.STDOUT,
    )
    processes.append(("backend", proc))
    # Wait for backend to be ready
    import requests as req
    for i in range(30):
        time.sleep(2)
        try:
            r = req.get(f"http://localhost:{args.port_backend}/health", timeout=3)
            if r.status_code == 200:
                log(f"Backend ready on port {args.port_backend}", GREEN)
                return proc
        except Exception:
            pass
        log(f"  Waiting for backend... ({i+1}/30)", AMBER)
    log("Backend did not start in time. Check logs/backend.log", RED)
    sys.exit(1)


def start_frontend():
    log("Starting Streamlit frontend...", BLUE)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(FRONTEND_PATH),
            "--server.port", str(args.port_frontend),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(FRONTEND_PATH.parent),
        stdout=open(LOGS_DIR / "frontend.log", "a"),
        stderr=subprocess.STDOUT,
    )
    processes.append(("frontend", proc))
    time.sleep(4)
    log(f"Frontend ready on port {args.port_frontend}", GREEN)
    return proc


def start_ngrok():
    if args.no_ngrok:
        log("ngrok skipped (--no-ngrok).", AMBER)
        return None

    try:
        from pyngrok import ngrok, conf

        # Save token if provided
        if args.ngrok_token:
            conf.get_default().auth_token = args.ngrok_token
            ngrok.set_auth_token(args.ngrok_token)
            log("ngrok token saved.", GREEN)

        log("Opening ngrok tunnel...", BLUE)
        tunnel = ngrok.connect(args.port_frontend, "http")
        public_url = tunnel.public_url

        log("", GREEN)
        log("=" * 55, GREEN)
        log(f"  CircuitMentor is LIVE", BOLD)
        log("=" * 55, GREEN)
        log(f"  Public URL  : {public_url}", BOLD)
        log(f"  Local URL   : http://localhost:{args.port_frontend}", RESET)
        log(f"  Admin login : SungjinHo / circuitmentor_admin_2026", AMBER)
        log("=" * 55, GREEN)
        log("  Send this link to sir:", BOLD)
        log(f"  {public_url}", BLUE)
        log("=" * 55, GREEN)
        log("", GREEN)

        # Save URL to file so admin can check it
        with open(ROOT / "public_url.txt", "w") as f:
            f.write(f"CircuitMentor Public URL\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"URL: {public_url}\n")
            f.write(f"Local: http://localhost:{args.port_frontend}\n")

        return tunnel

    except ImportError:
        log("pyngrok not installed. Run: pip install pyngrok", RED)
        log(f"Local URL: http://localhost:{args.port_frontend}", AMBER)
        return None
    except Exception as e:
        log(f"ngrok error: {e}", RED)
        log(f"Local URL: http://localhost:{args.port_frontend}", AMBER)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

tunnel_ref = [None]

def shutdown(sig=None, frame=None):
    log("\nShutting down CircuitMentor...", AMBER)

    # Close ngrok
    if tunnel_ref[0]:
        try:
            from pyngrok import ngrok
            ngrok.disconnect(tunnel_ref[0].public_url)
            ngrok.kill()
            log("ngrok tunnel closed.", GREEN)
        except Exception:
            pass

    # Kill subprocesses
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            log(f"{name} stopped.", GREEN)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # Clean up stale session dirs
    import shutil
    if SESSIONS_DIR.exists():
        for d in SESSIONS_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
    log("All sessions cleared.", GREEN)
    log("CircuitMentor stopped cleanly.", GREEN)
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{BLUE}")
    print("  ╔══════════════════════════════════════╗")
    print("  ║         CircuitMentor v1.0           ║")
    print("  ║   Embedded Systems AI Tutor          ║")
    print("  ║   NVIDIA Jetson AGX Orin · bf16      ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"{RESET}\n")

    # Pre-flight checks
    log("Running pre-flight checks...", BLUE)
    check_dependencies()
    check_model()
    setup_directories()
    write_config()

    log("All checks passed. Launching...\n", GREEN)

    # Start services
    start_backend()
    start_frontend()
    tunnel = start_ngrok()
    tunnel_ref[0] = tunnel

    if args.no_ngrok or not tunnel:
        log(f"Local access: http://localhost:{args.port_frontend}", AMBER)

    log("CircuitMentor is running. Press Ctrl+C to stop.\n", GREEN)
    log("Backend log : /mnt/circuitmentor/logs/backend.log", RESET)
    log("Frontend log: /mnt/circuitmentor/logs/frontend.log", RESET)

    # Keep alive — monitor subprocesses
    while True:
        time.sleep(10)
        for name, proc in processes:
            if proc.poll() is not None:
                log(f"{name} crashed! Exit code: {proc.returncode}", RED)
                log(f"Check /mnt/circuitmentor/logs/{name}.log", AMBER)
                shutdown()

if __name__ == "__main__":
    main()

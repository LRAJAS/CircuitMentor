# Deployment Guide — CircuitMentor

---

## Quick Start (after training complete)

```bash
bash scripts/start.sh
# Wait 40 seconds
# Open your ngrok URL in browser
```

---

## Prerequisites

```bash
pip install fastapi uvicorn streamlit pyngrok psutil requests
```

Get a free ngrok account at https://ngrok.com — copy your authtoken from the dashboard.

---

## Directory Setup on Jetson

```bash
mkdir -p /mnt/circuitmentor/sessions
mkdir -p /mnt/circuitmentor/logs
mkdir -p /mnt/circuitmentor/app
mkdir -p /mnt/circuitmentor/tmp
```

---

## Configuration

Edit `/mnt/circuitmentor/config.json` (auto-created on first launch):

```json
{
  "model_path":      "/mnt/circuitmentor/experiments/llama3_tutor_v1/final_model",
  "max_users":       10,
  "max_messages":    200,
  "queue_timeout":   90,
  "admin_username":  "YourAdminUsername",
  "admin_pass_hash": "sha256_hash_of_your_password",
  "memory_cap_gb":   50
}
```

**Generate admin password hash:**
```python
import hashlib
print(hashlib.sha256(b"your_password").hexdigest())
```

---

## Launch Options

```bash
# Standard launch (uses config.json)
python src/launch.py \
  --model-path /path/to/final_model \
  --memory-cap-gb 50

# With explicit ngrok token (first run)
python src/launch.py \
  --model-path /path/to/final_model \
  --ngrok-token YOUR_TOKEN_HERE \
  --memory-cap-gb 50

# Local only (no public URL)
python src/launch.py \
  --model-path /path/to/final_model \
  --no-ngrok \
  --memory-cap-gb 50

# Custom ports
python src/launch.py \
  --port-backend 8000 \
  --port-frontend 8501
```

---

## Permanent Public URL

ngrok free tier gives one static domain:

1. Go to https://dashboard.ngrok.com/domains
2. Copy your free static domain
3. Edit `src/launch.py` — find `ngrok.connect(...)` and add `domain="your-domain.ngrok-free.dev"`

Same URL on every restart.

---

## Auto-Start on Boot

```bash
crontab -e

# Add:
@reboot sleep 90 && bash /mnt/circuitmentor/start.sh
```

CircuitMentor auto-launches after every Orin reboot.

---

## Monitoring

```bash
# Check running processes
ps aux | grep -E "launch|uvicorn|streamlit|ngrok" | grep -v grep

# Live logs
tail -f /mnt/circuitmentor/logs/launch.log

# Backend logs only
tail -f /mnt/circuitmentor/logs/backend.log

# Frontend logs only
tail -f /mnt/circuitmentor/logs/frontend.log

# Memory usage
free -h
```

---

## Admin Dashboard

URL: `https://your-domain.ngrok-free.dev` → click Admin Login

Features:
- Active sessions (username, message count, join time)
- Real-time GPU and RAM usage
- Queue depth and processing status
- Force-kick individual sessions
- Clear all sessions

---

## Session Flow

```
User opens URL
    │
    ▼
Join screen (enter name)
    │ max 10 users check
    ▼
Chat screen (UUID session created, folder at /mnt/circuitmentor/sessions/{uuid}/)
    │
    ▼
User sends message → FastAPI queue → model inference → response
    │ every message saved to sessions/{uuid}/chat.json
    ▼
User clicks Exit
    │
    ▼
Download prompt → "Save chat as .txt?"
    │
    ▼
Yes → .txt downloaded     No → skip
    │
    ▼
Session folder deleted, memory cleared
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERR_NGROK_3200 — endpoint offline` | CircuitMentor not running. Run `bash scripts/start.sh` |
| Model loads but no response | Check `tail -f logs/backend.log` |
| `couldn't create directory /tmp/tmux` | `/tmp` full. `export TMPDIR=/mnt/circuitmentor/tmp` |
| Frontend shows raw HTML | Streamlit rendering issue — restart with `bash scripts/stop.sh && bash scripts/start.sh` |
| OOM during inference | Reduce `--memory-cap-gb` or restart Orin to clear fragmentation |
| ngrok URL changes on restart | Use static domain from ngrok dashboard |

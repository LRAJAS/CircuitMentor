# Architecture Guide — CircuitMentor

Complete reference for the system design, component layout, and request lifecycle.

---

## Overview

CircuitMentor is a three-layer stack running entirely on a single NVIDIA Jetson AGX Orin edge device:

```
[ ngrok tunnel ]  ←→  [ Streamlit Frontend ]  ←→  [ FastAPI Backend ]  ←→  [ Fine-Tuned Model ]
```

There is no cloud dependency during inference. The model, backend, frontend, and tunnel all run locally on the Orin. ngrok is used only as a relay — no data is stored remotely.

---

## Hardware

| Component       | Specification                  |
|-----------------|-------------------------------|
| Device          | NVIDIA Jetson AGX Orin        |
| Unified Memory  | 64 GB (CPU + GPU shared)      |
| CUDA Version    | 12.6                          |
| JetPack         | 6.1 (R36.4.7)                 |
| Storage         | 465 GB NVMe                   |
| OS              | Ubuntu 22.04                  |

---

## Software Stack

| Component       | Version / Detail                          |
|-----------------|------------------------------------------|
| Base Model      | Meta Llama-3.1-8B-Instruct               |
| Fine-Tuning     | LoRA (Low-Rank Adaptation) via PEFT      |
| Precision       | bf16 — CUDA 12.6 safe, no bitsandbytes  |
| Python          | 3.10.12                                  |
| PyTorch         | 2.5.0a0 nv24.8                          |
| Transformers    | 4.44.0 (HuggingFace)                    |
| Backend         | FastAPI + asyncio queue                  |
| Frontend        | Streamlit                                |
| Public tunnel   | ngrok (free tier, static domain)         |

---

## Request Flow

```
User browser (anywhere in the world)
        │
        │  https://<static-domain>.ngrok-free.dev
        ▼
    ngrok tunnel  (cloud relay — free tier)
        │
        ▼
    Streamlit Frontend — port 8501  (frontend.py)
        │  HTTP POST /chat
        ▼
    FastAPI Backend — port 8000  (backend.py)
        │  asyncio FIFO queue — ONE request at a time
        ▼
    Llama-3.1-8B Fine-Tuned Model  (bf16, CUDA)
    /mnt/circuitmentor/experiments/llama3_tutor_v1/final_model
        │
        ▼
    Response returned to user  (~3–8 seconds)
```

---

## File Structure

```
/mnt/circuitmentor/
├── app/
│   ├── backend.py          — FastAPI server, model inference, session management
│   ├── frontend.py         — Streamlit UI, easter eggs, responsive design
│   ├── launch.py           — Single-command launcher (starts all three processes)
│   └── train_tutor.py      — LoRA fine-tuning script
├── datasets/
│   ├── train.jsonl         — 13,500 training examples (90%)
│   └── val.jsonl           — 1,500 validation examples (10%)
├── experiments/
│   └── llama3_tutor_v1/
│       ├── final_model/          — Merged model (~15 GB safetensors)
│       └── final_lora_adapter/   — LoRA-only weights (~161 MB)
├── sessions/               — Live UUID session folders (auto-wiped on user exit)
├── logs/
│   ├── launch.log
│   ├── backend.log
│   └── frontend.log
├── tmp/                    — Redirected TMPDIR (avoids /tmp overflow)
├── config.json             — Ports, max users, admin credentials hash
├── start.sh                — Brings everything live in background (nohup)
└── stop.sh                 — Kills all CircuitMentor processes cleanly
```

---

## Component Responsibilities

### backend.py — FastAPI Inference Server

- Loads the merged fine-tuned model **once** at startup into GPU memory
- Exposes REST endpoints: `/chat`, `/join`, `/exit`, `/admin/*`, `/status`
- Uses an `asyncio` FIFO queue — model processes exactly one inference request at a time
- UUID-based session management: each user gets a unique folder under `sessions/`
- Admin endpoints protected by SHA-256 password hash (set in `config.json`)
- Saves every message to `sessions/{uuid}/chat.json` for in-session history
- Enforces: max 10 concurrent users, max 200 messages per session, 90-second queue timeout
- Wipes session folder completely on user exit — zero data retained server-side

### frontend.py — Streamlit UI

- Responsive layout — works on desktop, tablet, and mobile browsers
- Dark professional theme with syntax-highlighted code blocks
- Join screen: username entry, max-user check before UUID creation
- Chat screen: send message → display response with timing (e.g. *answered in 4.2s*)
- Sidebar: 8 one-click example prompts, temperature slider, max token control
- Exit flow: prompts user to download chat as `.txt` before session deletion
- Admin dashboard: active sessions table, RAM/GPU gauges, kick and clear controls
- Easter eggs (see Features section of manual)

### launch.py — Process Orchestrator

- Runs preflight checks: model path exists, required directories exist, dependencies importable
- Spawns backend (uvicorn) and frontend (streamlit) as subprocesses
- Opens ngrok tunnel and prints the public URL
- Monitors subprocesses — restarts crashed components
- Handles `Ctrl+C` with clean shutdown of all three processes
- Saves ngrok token to config on first run so subsequent launches need no flag

### ngrok Tunnel

- Free static domain — same URL on every restart (no reconfiguration needed)
- Acts as a TLS-terminating relay: browser → ngrok cloud → local port 8501
- No data is stored by ngrok; it is a transport layer only

---

## Session & Queue Design

```
User opens URL
      │
      ▼
  Join screen (enter name)
      │  max 10 users check
      ▼
  Chat screen  (UUID session created)
      │  sessions/{uuid}/  folder created
      ▼
  User sends message
      │
      ▼
  FastAPI asyncio queue
      │  FIFO — waits if model busy (90s timeout)
      ▼
  Model inference  (bf16, GPU)
      │  ~3–8 seconds
      ▼
  Response streamed back to frontend
      │  chat.json updated
      ▼
  User clicks Exit
      │
      ▼
  Download prompt → "Save chat as .txt?"
      │
      ├── Yes → .txt downloaded
      └── No  → skip
      │
      ▼
  Session folder deleted, memory cleared
```

**Concurrency constraints:**
- Maximum 10 simultaneous users — 11th user sees a friendly "server full" message
- Same username cannot log in twice simultaneously
- asyncio queue ensures no parallel GPU inference — prevents OOM on the Orin
- Queue timeout 90 seconds — user gets a friendly error if the server is overloaded

---

## Memory Layout

The Jetson AGX Orin uses unified memory shared between CPU and GPU.

| Allocation             | Size      |
|------------------------|-----------|
| Fine-tuned model (bf16)| ~15 GB    |
| OS + system processes  | ~5–8 GB   |
| FastAPI + Streamlit    | ~1–2 GB   |
| Session buffers        | ~1 GB     |
| **Memory cap (config)**| **50 GB** |
| **Total available**    | **64 GB** |
| Safety headroom        | ~14 GB    |

The `--memory-cap-gb 50` flag in `launch.py` prevents the model from allocating beyond 50 GB, leaving headroom for the OS and other processes.

---

## Model Architecture

Base: Meta Llama-3.1-8B-Instruct — a decoder-only transformer with 8.07 billion parameters.

Fine-tuning method: LoRA (Low-Rank Adaptation) via HuggingFace PEFT.

### Why LoRA?

Instead of updating all 8B parameters, LoRA freezes the base model and trains two small matrices per attention layer:

```
Full weight matrix W:  [4096 × 4096] = 16.7M parameters
LoRA approximation:    A [4096 × 16] + B [16 × 4096] = 131K parameters

Savings: 128× fewer parameters per attention layer
```

CircuitMentor trained **41.9M parameters** out of 8,072M total — **0.52%** of the full model.

After training, the LoRA adapter is **merged** into the base weights and saved as a standalone model. The deployed `final_model/` requires no PEFT at inference time.

### Why bf16 (not int4/int8)?

`bitsandbytes` — required for 4-bit and 8-bit quantization — is incompatible with CUDA 12.6 on the ARM64 Jetson architecture. `bf16` uses the same 16-bit memory footprint as `fp16` but has better numerical range, eliminating the NaN gradient explosions that `fp16` causes on Jetson.

### Training Configuration

| Parameter          | Value             | Reason                              |
|--------------------|-------------------|-------------------------------------|
| LoRA rank (r)      | 16                | Standard for 15k dataset            |
| LoRA alpha         | 32                | 2× r — standard scaling             |
| Trainable params   | 41.9M / 8.07B     | 0.52% — LoRA efficiency             |
| Epochs             | 3                 | Best convergence for dataset size   |
| Batch size         | 1 + grad_accum=16 | Jetson memory constraint            |
| Learning rate      | 2e-4 (cosine)     | Standard LoRA LR                    |
| Memory cap         | 50 GB             | Leaves 14 GB for OS                 |
| NEFTune noise α    | 5.0               | Boosts instruction following        |
| Final train loss   | 0.1855            | Excellent — target was 0.50         |
| Eval loss          | 0.8788            | Healthy generalization              |
| Total steps        | 2,529             | 3 epochs over 13,500 examples       |
| Training time      | ~40 hours         | Including 3 power cuts, auto-resumed|

---

## Security Design

- Admin credentials stored as a SHA-256 hash in `config.json` — plaintext password never persisted
- Sessions identified by UUID — no username is used as a key server-side
- No user data leaves the device — ngrok is transport-only
- Session folder wiped on exit — no chat history retained after the session ends
- Queue enforces single-user inference — no cross-session data leakage possible

---

## Known Architecture Constraints

| Constraint                         | Reason                                       | Mitigation                               |
|------------------------------------|----------------------------------------------|------------------------------------------|
| Single inference at a time         | 64 GB unified memory — no room for parallel  | asyncio queue with 90s timeout           |
| Max 10 users                       | Memory + queue latency at scale              | Friendly "server full" message           |
| ngrok free tier                    | No persistent IP without paid plan           | Free static domain from ngrok dashboard  |
| No quantization (bf16 only)        | bitsandbytes incompatible with CUDA 12.6/ARM | bf16 — same memory as fp16, more stable  |
| No streaming tokens                | Streamlit re-render overhead                 | Full response returned, timing shown     |

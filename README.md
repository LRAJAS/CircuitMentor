# ⚡ CircuitMentor

<div align="center">

![Model](https://img.shields.io/badge/Base_Model-Llama_3.1_8B-orange?style=for-the-badge)
![Hardware](https://img.shields.io/badge/Hardware-Jetson_AGX_Orin-green?style=for-the-badge)
![Precision](https://img.shields.io/badge/Precision-bf16_LoRA-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)

**An edge-native AI teaching assistant for embedded systems.**  
Fine-tuned on Llama-3.1-8B. Runs entirely on a NVIDIA Jetson AGX Orin.  
No cloud. No API. No internet required during inference.

[Getting Started](#getting-started) · [Architecture](#architecture) · [Training](#training) · [Deployment](#deployment) · [Dataset](#dataset)

</div>

---

## What is CircuitMentor?

CircuitMentor is a domain-specific Small Language Model (SLM) fine-tuned to serve as an expert AI tutor for junior ECE students learning embedded systems. It takes Meta's open-source **Llama-3.1-8B-Instruct** as a base and applies **LoRA fine-tuning** on 15,000 curated embedded systems instruction pairs — entirely on an edge device.

```
Student's browser (anywhere in the world)
        │
        │  https://your-domain.ngrok-free.dev
        ▼
    ngrok tunnel (permanent public URL)
        │
        ▼
    NVIDIA Jetson AGX Orin (college lab / edge device)
    ├── Streamlit UI      → port 8501 (chat interface)
    ├── FastAPI Backend   → port 8000 (inference + queue)
    └── Llama-3.1-8B     → bf16 + LoRA, 50GB memory cap
```

---

## Key Results

| Metric | Value |
|---|---|
| Base model | Llama-3.1-8B-Instruct |
| Final training loss | **0.1855** |
| Dataset size | 15,000 examples (81% with code) |
| Training time | ~18 hours on Jetson AGX Orin |
| Trainable parameters | 41.9M / 8.07B **(0.52%)** |
| Max concurrent users | 10 |
| Final model size | ~15GB (4 safetensors shards) |
| LoRA adapter size | ~160MB |

---

## What It Knows

| Domain | Topics |
|---|---|
| Microcontrollers | Arduino (Uno/Mega/Nano), ESP32, ESP8266, ATmega328P, STM32, Raspberry Pi Pico (RP2040) |
| Languages | Embedded C, C++, MicroPython, CircuitPython |
| Protocols | I2C, SPI, UART/USART, MQTT, LoRa, BLE, CAN Bus, MODBUS |
| Concepts | GPIO, PWM, ADC/DAC, Interrupts/ISR, DMA, Watchdog Timers, Deep Sleep |
| RTOS | FreeRTOS tasks, queues, semaphores, vTaskDelay on ESP32 |
| IoT | WiFi, MQTT brokers, OTA updates, Node-RED, ESP-NOW |
| Sensors | DS18B20, DHT11/22, BME280, MPU6050, HC-SR04, NeoPixel, Servos |
| Debugging | Serial Monitor issues, WDT resets, I2C address scanning, power problems |

> **Out of scope:** Verilog, VHDL, FPGA, advanced DSP, PCB routing — those are my cousin's department.

---

## Architecture

### Training Pipeline

```
8 HuggingFace Sources (188,000 rows scanned)
        │
        ▼  build_dataset.py — Phase 1
Keyword filter (Tier-1: esp32, i2c, atmega... Tier-2: microcontroller, firmware...)
        │
        ▼  54,152 raw candidates saved
Quality scorer: (Tier-1 hits × 3.0) + (Tier-2 hits × 1.0) + (code block × 1.5) + length score
        │
        ▼  Top 15,000 selected
train.jsonl (13,500) + val.jsonl (1,500)
        │
        ▼  train_tutor.py
Llama-3.1-8B-Instruct
+ LoRA (r=16, α=32, target: q,k,v,o,gate,up,down proj)
+ bf16 precision (no bitsandbytes — CUDA 12.6 Jetson safe)
+ NEFTune noise (α=5.0)
+ gradient checkpointing + enable_input_require_grads()
        │
        ▼  ~18 hours on Jetson AGX Orin
final_lora_adapter/ (~160MB) → merge → final_model/ (~15GB)
```

### Inference Stack

```
launch.py
├── uvicorn backend.py    → FastAPI on :8000
│   ├── Model loaded once at startup (bf16, 50GB cap)
│   ├── asyncio queue (1 inference at a time, 90s timeout)
│   ├── UUID session management (max 10 concurrent)
│   └── Admin endpoints (stats, kick, clear)
│
├── streamlit frontend.py → Streamlit on :8501
│   ├── Join screen → chat screen → exit flow
│   ├── Responsive (desktop / tablet / mobile)
│   ├── Admin dashboard
│   └── Easter eggs
│
└── ngrok tunnel → permanent public URL
```

---

## Getting Started

### Prerequisites

```
Hardware : NVIDIA Jetson AGX Orin (32GB+ recommended, 64GB used here)
CUDA     : 12.6
OS       : Ubuntu 22.04
Python   : 3.10+
Storage  : 100GB+ free on NVMe
```

### Step 1 — Clone the repo

```bash
git clone https://github.com/LRAJAS/CircuitMentor.git
cd CircuitMentor
pip install -r requirements.txt
```

### Step 2 — Build the dataset (laptop, no GPU needed)

```bash
pip install datasets tqdm colorama
python src/build_dataset.py
```

Takes 30–90 minutes depending on internet speed. Output:

```
embedded_dataset/
├── raw/raw_50k.jsonl     ← Phase 1 backup (54,152 examples)
├── train.jsonl           ← 13,500 training examples
├── val.jsonl             ← 1,500 validation examples
└── stats.json            ← Build report
```

### Step 3 — Download Llama-3.1-8B

Requires Meta license approval at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir="./models/llama3.1-8b-instruct",
    token="YOUR_HF_TOKEN",
    ignore_patterns=["*.pth", "original/*"],
)
```

### Step 4 — Transfer to Jetson

```bash
scp -r ./embedded_dataset/ user@<orin-ip>:/mnt/circuitmentor/datasets/
scp -r ./models/llama3.1-8b-instruct/ user@<orin-ip>:/mnt/vlsi/hf_cache/models/
scp -r ./src/ user@<orin-ip>:/mnt/circuitmentor/app/
scp -r ./scripts/ user@<orin-ip>:/mnt/circuitmentor/
```

### Step 5 — Train on Jetson

```bash
sudo jetson_clocks --fan
source /path/to/venv/bin/activate

python src/train_tutor.py \
  --model_name /path/to/llama3.1-8b-instruct \
  --dataset_path /mnt/circuitmentor/datasets/train.jsonl \
  --val_dataset_path /mnt/circuitmentor/datasets/val.jsonl \
  --output_dir /mnt/circuitmentor/experiments/llama3_tutor_v1 \
  --num_epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --memory_cap_gb 50
```

Expected: ~18 hours, final loss ~0.18–0.25. Checkpoints saved every 200 steps — power cut safe.

### Step 6 — Launch

Get a free ngrok token at https://ngrok.com, then:

```bash
bash scripts/start.sh
```

CircuitMentor is live. Share the URL.

---

## Training

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Precision | bf16 | fp16 causes NaN on Jetson CUDA 12.6 |
| Quantization | None | bitsandbytes incompatible with CUDA 12.6 ARM64 |
| LoRA rank | 16 | Standard for domain adaptation on 15k dataset |
| LoRA alpha | 32 | Standard 2× r scaling |
| Target modules | q,k,v,o,gate,up,down | Full attention + MLP |
| Optimizer | adamw_torch_fused | No bitsandbytes needed |
| Learning rate | 2e-4 | Standard LoRA |
| LR scheduler | cosine | Smooth decay |
| NEFTune | α=5.0 | Boosts instruction following |
| Batch size | 1 + grad_accum=16 | Jetson memory constraint |
| Max seq length | 1024 | OOM prevention |

### Resuming after interruption

```bash
python src/train_tutor.py \
  --output_dir /mnt/circuitmentor/experiments/llama3_tutor_v1 \
  --resume_from_checkpoint latest \
  [... same other args ...]
```

---

## Deployment

```bash
bash scripts/start.sh    # Start everything (background, nohup)
bash scripts/stop.sh     # Stop everything cleanly
```

```bash
# Check running
ps aux | grep -E "launch|uvicorn|streamlit|ngrok" | grep -v grep

# View live logs
tail -f /mnt/circuitmentor/logs/launch.log
```

Admin dashboard: visit `/` → click Admin Login
- Username: configured in `config.json`
- Password: SHA256 hash in `config.json`

---

## Dataset Construction

Two-phase pipeline in `src/build_dataset.py`:

**Phase 1 — Raw collection:**

| Source | Scanned | Passed |
|---|---|---|
| Glaive-Code-Assistant | 30,000 | 6,647 |
| Magicoder-Evol-110k | 40,000 | 5,300 |
| WizardLM-Evol-196k | 20,000 | 1,850 |
| OpenHermes | 20,000 | 1,125 |
| Code-Instructions-122k | 40,000 | 41 |
| Alpaca-Code-18k | 18,000 | 13 |
| CodeAlpaca-20k | 20,000 | 12 |
| Synthetic seeds (hand-crafted) | 12 | 12 |
| **Total** | **188,012** | **14,988** |

**Phase 2 — Quality scoring:**
```
score = (Tier-1 keyword hits × 3.0)
      + (Tier-2 keyword hits × 1.0)
      + (has code block × 1.5)
      + (response length sweet-spot score)
```

Top 15,000 by score → 90/10 train/val split → Alpaca format → Llama-3.1 chat template.

---

## Project Structure

```
CircuitMentor/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── build_dataset.py    ← 2-phase dataset pipeline
│   ├── train_tutor.py      ← bf16 + LoRA training script
│   ├── backend.py          ← FastAPI inference server + queue
│   ├── frontend.py         ← Streamlit UI
│   └── launch.py           ← One-command stack launcher
│
├── scripts/
│   ├── start.sh            ← Start (nohup background)
│   └── stop.sh             ← Stop cleanly
│
└── docs/
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    ├── DEPLOYMENT.md
    └── DATASET.md
```

---

## Why This Exists

Junior ECE students debugging I2C at midnight need immediate, accurate, contextual help. General-purpose chatbots are not specialized, require internet, and charge per query.

CircuitMentor runs on hardware the lab already owns. Zero per-query cost. Zero cloud dependency. Zero data sent externally.

---

## Built With

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [TRL](https://github.com/huggingface/trl) — SFTTrainer
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [ngrok](https://ngrok.com/)
- [Meta Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

---

## License

MIT License — see [LICENSE](LICENSE).

Base model subject to [Meta's Llama license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

---

<div align="center">

**Built by Rajas L · April 2026**

*Built with sleepless nights, power cuts, and a lot of patience* 🔌⚡

</div>

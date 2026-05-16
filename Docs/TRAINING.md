# Training Guide — CircuitMentor

Complete step-by-step guide to reproduce the CircuitMentor fine-tune from scratch.

---

## Prerequisites

- NVIDIA Jetson AGX Orin (64GB recommended, 32GB minimum)
- CUDA 12.6 + JetPack 6.1
- Ubuntu 22.04
- Python 3.10 venv with dependencies installed
- ~100GB free NVMe storage
- HuggingFace account with Llama-3.1 access approved

---

## Step 1 — Environment Setup

```bash
# Create venv (if not existing)
python3 -m venv /mnt/your_path/venv
source /mnt/your_path/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

## Step 2 — Build Dataset (on laptop)

```bash
pip install datasets tqdm colorama
python src/build_dataset.py
```

Approximate time: 30–90 minutes depending on internet speed.

**What it does:**
1. Streams 8 HuggingFace datasets (never downloads fully)
2. Applies two-tier keyword filtering
3. Scores 54,152 raw candidates by quality
4. Selects top 15,000 examples
5. Applies Llama-3.1 chat template
6. Splits 90/10 into train/val

---

## Step 3 — Download Base Model

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir="./models/llama3.1-8b-instruct",
    token="YOUR_HF_TOKEN",
    ignore_patterns=["*.pth", "original/*"],
)
```

Size: ~16GB. Time: 30–90 minutes.

---

## Step 4 — Transfer to Jetson

```bash
# Dataset
scp -r ./embedded_dataset/ user@<jetson-ip>:/mnt/circuitmentor/datasets/

# Model
scp -r ./models/llama3.1-8b-instruct/ user@<jetson-ip>:/path/to/hf_cache/models/

# Scripts
scp -r ./src/ user@<jetson-ip>:/mnt/circuitmentor/app/
```

---

## Step 5 — Verify Model on Jetson

```bash
source venv/bin/activate

python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/path/to/llama3.1-8b-instruct')
print('Vocab size:', len(tok))
# Expected: 128256
"
```

---

## Step 6 — Launch Training

```bash
sudo jetson_clocks --fan        # Max performance + cooling
source venv/bin/activate

# Use tmux for persistence
tmux new-session -s training

python src/train_tutor.py \
  --model_name /path/to/llama3.1-8b-instruct \
  --dataset_path /mnt/circuitmentor/datasets/train.jsonl \
  --val_dataset_path /mnt/circuitmentor/datasets/val.jsonl \
  --output_dir /mnt/circuitmentor/experiments/llama3_tutor_v1 \
  --num_epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --memory_cap_gb 50

# Detach: Ctrl+B then D
```

---

## Step 7 — Monitor Training

```bash
# Reattach to tmux
tmux attach -t training

# Watch loss — should decrease from ~1.5 → ~0.18
# Checkpoints save every 200 steps automatically
```

**Expected loss curve:**
```
Step 100   → ~0.80
Step 500   → ~0.74
Step 1000  → ~0.68
Step 1500  → ~0.62
Step 2000  → ~0.55
Step 2529  → ~0.18 (final)
```

**Expected time:** ~18 hours for 3 epochs on 13,500 examples.

---

## Resuming After Interruption

```bash
python src/train_tutor.py \
  --model_name /path/to/llama3.1-8b-instruct \
  --dataset_path /mnt/circuitmentor/datasets/train.jsonl \
  --val_dataset_path /mnt/circuitmentor/datasets/val.jsonl \
  --output_dir /mnt/circuitmentor/experiments/llama3_tutor_v1 \
  --num_epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --memory_cap_gb 50 \
  --resume_from_checkpoint latest
```

The `latest` keyword automatically finds the most recent checkpoint.

---

## Auto-Resume on Reboot (crontab)

```bash
crontab -e

# Add this line:
@reboot sleep 60 && bash /mnt/circuitmentor/app/resume_training.sh
```

---

## Step 8 — Verify Training Complete

```bash
ls /mnt/circuitmentor/experiments/llama3_tutor_v1/

# Expected:
# checkpoint-800/
# checkpoint-1200/
# checkpoint-1400/
# final_lora_adapter/   ← LoRA weights (~160MB)
# final_model/          ← Merged model (~15GB)
# training_summary.json
```

---

## Step 9 — Quick Inference Test

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path  = '/mnt/circuitmentor/experiments/llama3_tutor_v1/final_model'
tok   = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map='cuda:0')
model.eval()

msgs = [
    {'role': 'system',  'content': 'You are CircuitMentor, an embedded systems tutor.'},
    {'role': 'user',    'content': 'Write Arduino code to blink an LED on pin 13 every 500ms.'}
]
ids = tok.apply_chat_template(msgs, return_tensors='pt', add_generation_prompt=True).to('cuda')
with torch.inference_mode():
    out = model.generate(ids, max_new_tokens=300, temperature=0.3, do_sample=True)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
"
```

If it generates clean Arduino code — training succeeded. Launch with `bash scripts/start.sh`.

---

## Manual Merge (if final_model/ incomplete)

If a power cut interrupts the merge step:

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil, os

base = '/path/to/llama3.1-8b-instruct'
lora = '/mnt/circuitmentor/experiments/llama3_tutor_v1/final_lora_adapter'
out  = '/mnt/circuitmentor/experiments/llama3_tutor_v1/final_model'

shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(lora).save_pretrained(out)
print('Merge complete.')
"
```

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| bitsandbytes error | CUDA 12.6 incompatible | Harmless warning — ignore, training continues |
| NaN gradients | fp16 on Jetson | Always use bf16 |
| Gradient checkpointing fails silently | Missing enable_input_require_grads() | Already fixed in train_tutor.py |
| /tmp full | Root partition full | Export TMPDIR to NVMe path |
| Power cut loses progress | No checkpoint | Checkpoints every 200 steps — max 1.5hr loss |

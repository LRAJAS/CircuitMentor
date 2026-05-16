"""
train_tutor.py — EmbeddedTutor-SLM Training Script
====================================================
Model  : Meta Llama-3-8B-Instruct
Method : bf16 + LoRA (NO bitsandbytes — CUDA 12.6 Jetson safe)
Target : Jetson AGX Orin 64GB Unified Memory
Time   : ~10–12 hours on 15k dataset

Usage:
    sudo jetson_clocks --fan
    source /mnt/vlsi/vlsi_env/bin/activate
    tmux new-session -s embedded_tutor
    python train_tutor.py
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EmbeddedTutor Llama-3 LoRA Trainer")
    p.add_argument("--model_name",      default="/mnt/vlsi/hf_cache/models/llama3-8b-instruct")
    p.add_argument("--dataset_path",    default="/mnt/vlsi/embedded_tutor/datasets/train.jsonl")
    p.add_argument("--val_dataset_path",default="/mnt/vlsi/embedded_tutor/datasets/val.jsonl")
    p.add_argument("--output_dir",      default="/mnt/vlsi/embedded_tutor/experiments/llama3_tutor_v1")
    p.add_argument("--num_epochs",      type=int,   default=2)
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)
    p.add_argument("--max_seq_length",  type=int,   default=1024)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument("--grad_accum",      type=int,   default=16)
    p.add_argument("--learning_rate",   type=float, default=2e-4)
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    p.add_argument("--neftune_noise_alpha", type=float, default=5.0)
    p.add_argument("--memory_cap_gb",   type=int,   default=50)
    p.add_argument("--save_steps",      type=int,   default=200)
    p.add_argument("--eval_steps",      type=int,   default=200)
    p.add_argument("--logging_steps",   type=int,   default=10)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint dir to resume from, or 'latest'")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"train_tutor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA 3 CHAT TEMPLATE
# Format each Alpaca example into Llama-3 instruct format
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are CircuitMentor, an expert embedded systems tutor for junior ECE students. "
    "You specialize in Arduino, ESP32, Raspberry Pi Pico, ATmega, STM32, MicroPython, "
    "Embedded C, C++, IoT protocols (MQTT, I2C, SPI, UART), and FreeRTOS. "
    "Always explain concepts clearly, provide working code with comments, "
    "and highlight common pitfalls. Format code in properly labelled code blocks."
)

def format_example_llama3(example: Dict) -> str:
    """
    Converts one Alpaca dict into Llama-3 instruct chat format.
    <|begin_of_text|>...<|eot_id|> structure.
    """
    instruction = example.get("instruction", "").strip()
    context     = example.get("input", "").strip()
    response    = example.get("output", "").strip()

    user_content = instruction
    if context:
        user_content = f"{instruction}\n\nContext:\n{context}"

    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    )
    return text

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def set_memory_cap(cap_gb: int) -> None:
    """Cap PyTorch's unified memory allocation on Jetson."""
    cap_bytes = cap_gb * (1024 ** 3)
    torch.cuda.set_per_process_memory_fraction(cap_bytes / torch.cuda.get_device_properties(0).total_memory)
    log.info(f"Memory cap set to {cap_gb}GB / "
             f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB total")

def log_memory(label: str = "") -> None:
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved  = torch.cuda.memory_reserved()  / (1024**3)
    log.info(f"[MEM{' '+label if label else ''}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class MemoryLogCallback(TrainerCallback):
    """Logs GPU memory every N steps."""
    def __init__(self, log_every: int = 50):
        self.log_every = log_every

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.log_every == 0:
            log_memory(f"step={state.global_step}")

class CheckpointOnInterruptCallback(TrainerCallback):
    """Saves a checkpoint if training is interrupted (Ctrl+C or SIGTERM)."""
    def __init__(self, trainer_ref):
        self.trainer = trainer_ref

    def on_train_begin(self, args, state, control, **kwargs):
        import signal
        def handler(sig, frame):
            log.warning("Interrupt received — saving emergency checkpoint...")
            self.trainer.save_model(os.path.join(args.output_dir, "emergency_checkpoint"))
            sys.exit(0)
        signal.signal(signal.SIGINT,  handler)
        signal.signal(signal.SIGTERM, handler)

class ProgressSummaryCallback(TrainerCallback):
    """Prints a clean progress summary every 100 steps."""
    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        if logs and state.global_step % 100 == 0 and state.global_step > 0:
            loss = logs.get("loss", "—")
            lr   = logs.get("learning_rate", "—")
            log.info(
                f"  ═══ Step {state.global_step}/{state.max_steps} "
                f"| Loss: {loss:.4f if isinstance(loss, float) else loss} "
                f"| LR: {lr:.2e if isinstance(lr, float) else lr} ═══"
            )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  EmbeddedTutor-SLM — Llama-3-8B LoRA Training")
    log.info("=" * 60)
    log.info(f"  Model       : {args.model_name}")
    log.info(f"  Dataset     : {args.dataset_path}")
    log.info(f"  Output      : {args.output_dir}")
    log.info(f"  Epochs      : {args.num_epochs}")
    log.info(f"  LoRA r/α    : {args.lora_r}/{args.lora_alpha}")
    log.info(f"  Precision   : bf16 (no bitsandbytes)")
    log.info(f"  Memory cap  : {args.memory_cap_gb}GB")
    log.info("=" * 60)

    # ── Verify CUDA ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Is jetson_clocks running?")
    log.info(f"CUDA {torch.version.cuda} | Device: {torch.cuda.get_device_name(0)}")

    set_memory_cap(args.memory_cap_gb)
    log_memory("pre-load")

    # ── Load Tokenizer ────────────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=False,
    )

    # Llama 3 uses <|eot_id|> as EOS — critical for correct generation stop
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Must be right for SFT

    log.info(f"Vocab size: {len(tokenizer)} | EOS: '{tokenizer.eos_token}'")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_dataset = load_dataset("json", data_files=args.dataset_path,   split="train")
    val_dataset   = load_dataset("json", data_files=args.val_dataset_path, split="train")

    log.info(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Apply Llama-3 chat template formatting
    def format_batch(batch):
        return {"text": [format_example_llama3({
            "instruction": i, "input": inp, "output": o
        }) for i, inp, o in zip(batch["instruction"], batch["input"], batch["output"])]}

    train_dataset = train_dataset.map(format_batch, batched=True,
                                       remove_columns=train_dataset.column_names)
    val_dataset   = val_dataset.map(format_batch, batched=True,
                                     remove_columns=val_dataset.column_names)

    log.info("Sample formatted example:")
    log.info(train_dataset[0]["text"][:300] + "...")

    # ── Load Model ────────────────────────────────────────────────────────────
    log.info("Loading model in bf16 (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,     # bf16 — safe on Jetson (fp16 causes NaN)
        device_map="cuda:0",
        trust_remote_code=False,
    )

    log_memory("post-model-load")

    # ── Gradient Checkpointing ────────────────────────────────────────────────
    # MUST call enable_input_require_grads() before applying LoRA
    # This was the fix discovered in M3
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log.info("Gradient checkpointing enabled")

    # ── LoRA Config ───────────────────────────────────────────────────────────
    # Target all projection layers in Llama-3 attention + MLP
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
        ],
        # embed_tokens excluded — saves memory, rarely helps for domain tuning
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log_memory("post-lora")

    # ── Training Arguments ────────────────────────────────────────────────────
    # Steps calculation for logging
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps     = steps_per_epoch * args.num_epochs
    log.info(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    log.info(f"Estimated time @ 45s/step: {total_steps * 45 / 3600:.1f} hours")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        # ── Core ──────────────────────────────────────────────────────────────
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # ── Optimizer — NO bitsandbytes on CUDA 12.6 Jetson ──────────────────
        # adamw_torch_fused is the fastest native PyTorch option available
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,

        # ── Precision — bf16 mandatory on Jetson (fp16 → NaN explosions) ─────
        bf16=True,
        fp16=False,

        # ── Memory ────────────────────────────────────────────────────────────
        gradient_checkpointing=True,
        dataloader_num_workers=0,     # Jetson unified memory — no multiprocessing
        dataloader_pin_memory=False,

        # ── Evaluation & Saving ───────────────────────────────────────────────
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,           # Keep last 3 checkpoints → saves NVMe space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── Logging ───────────────────────────────────────────────────────────
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        report_to="none",             # No wandb — keeps it lean

        # ── NEFTune — adds noise to embeddings, boosts instruction following ──
        neftune_noise_alpha=args.neftune_noise_alpha,

        # ── Misc ──────────────────────────────────────────────────────────────
        max_grad_norm=1.0,
        seed=42,
        group_by_length=True,         # Groups similar-length sequences → less padding waste
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,    # Disable packing — safer for instruction format integrity
        callbacks=[
            MemoryLogCallback(log_every=50),
            ProgressSummaryCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # Attach interrupt handler after trainer is created
    trainer.add_callback(CheckpointOnInterruptCallback(trainer))

    log_memory("pre-train")

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        # Auto-find the most recent checkpoint
        checkpoints = sorted(output_dir.glob("checkpoint-*"),
                             key=lambda x: int(x.name.split("-")[-1]))
        resume = str(checkpoints[-1]) if checkpoints else None
        log.info(f"Resuming from: {resume}")

    train_result = trainer.train(resume_from_checkpoint=resume)

    log.info("Training complete.")
    log_memory("post-train")

    # ── Save Final Model ──────────────────────────────────────────────────────
    log.info("Saving final LoRA adapter...")
    final_adapter_dir = output_dir / "final_lora_adapter"
    trainer.model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    log.info(f"LoRA adapter saved → {final_adapter_dir}")

    # ── Merge LoRA into base for inference (optional but recommended for app.py) ──
    log.info("Merging LoRA weights into base model for inference...")
    try:
        from peft import PeftModel
        merged = trainer.model.merge_and_unload()
        final_model_dir = output_dir / "final_model"
        merged.save_pretrained(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        log.info(f"Merged model saved → {final_model_dir}")
    except Exception as e:
        log.warning(f"Merge failed (non-fatal): {e}. Use adapter for inference instead.")

    # ── Training Summary ──────────────────────────────────────────────────────
    summary = {
        "timestamp":        datetime.now().isoformat(),
        "model":            args.model_name,
        "train_examples":   len(train_dataset),
        "val_examples":     len(val_dataset),
        "epochs":           args.num_epochs,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "total_steps":      train_result.global_step,
        "train_loss":       round(train_result.training_loss, 4),
        "precision":        "bf16",
        "optimizer":        "adamw_torch_fused",
        "neftune":          args.neftune_noise_alpha,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Final loss : {train_result.training_loss:.4f}")
    log.info(f"  Steps      : {train_result.global_step}")
    log.info(f"  Output     : {output_dir}")
    log.info("=" * 60)
    log.info("")
    log.info("  Next step: run app.py to launch the demo UI")
    log.info(f"  streamlit run /mnt/vlsi/embedded_tutor/app.py")


if __name__ == "__main__":
    main()

# Dataset Guide — CircuitMentor

Complete reference for the dataset construction pipeline used to train CircuitMentor.

---

## Overview

The CircuitMentor dataset was built entirely from scratch using a custom two-phase pipeline (`build_dataset.py`). No pre-existing embedded systems dataset was used directly. All sourcing, filtering, scoring, and formatting was designed and implemented independently.

**Final output:** 15,000 high-quality embedded systems instruction/response pairs, formatted for Llama-3.1 chat template.

---

## Quick Stats

| Stat                   | Value                          |
|------------------------|-------------------------------|
| Raw candidates scanned | 188,000+                      |
| Raw candidates passing Phase 1 | 54,152             |
| Final selected         | 15,000                        |
| Train split            | 13,500 examples (90%)         |
| Val split              | 1,500 examples (10%)          |
| Examples with code     | 81%                           |
| Format                 | Alpaca instruction/input/output|
| Deduplication          | MD5 hash — clean              |

---

## Prerequisites (laptop/desktop)

```bash
pip install datasets tqdm colorama
```

The pipeline streams from HuggingFace — it never downloads full datasets to disk.

---

## Running the Pipeline

```bash
python src/build_dataset.py
```

Approximate time: **30–90 minutes** depending on internet speed.

Output files (transfer these to the Jetson):
```
embedded_dataset/
├── train.jsonl    — 13,500 training examples
└── val.jsonl      — 1,500 validation examples
```

---

## Phase 1 — Raw Collection

The pipeline streams 8 HuggingFace datasets and applies two-tier keyword filtering to find embedded systems content.

### Sources

| Source                  | Rows Scanned | Passed Filter |
|-------------------------|-------------|---------------|
| Glaive-Code-Assistant   | 30,000      | 6,647         |
| Magicoder-Evol-110k     | 40,000      | 5,300         |
| WizardLM-Evol-196k      | 20,000      | 1,850         |
| OpenHermes              | 20,000      | 1,125         |
| Code-Instructions-122k  | 40,000      | 41            |
| Alpaca-Code-18k         | 18,000      | 13            |
| CodeAlpaca-20k          | 20,000      | 12            |
| Synthetic Seeds (hand-crafted) | —   | 12            |
| **Total**               | **188,000+**| **15,000 selected** |

### Two-Tier Keyword Filter

**Tier 1 — Primary hardware keywords** (must match at least one):
```
arduino, esp32, raspberry pi pico, atmega, stm32, avr, microcontroller,
micropython, circuitpython, embedded c, freertos, uart, i2c, spi, gpio,
pwm, adc, dac, interrupt, watchdog, bootloader, firmware, iot, mqtt,
ble, lora, neopixel, servo, sensor, actuator
```

**Tier 2 — Secondary quality signals** (boosts score, not required):
```
register, datasheet, oscilloscope, breadboard, schematic, pcb, voltage,
current, ohm, capacitor, resistor, transistor, mosfet, protocol, baud,
clock speed, duty cycle, debounce, pull-up, pull-down
```

Items passing Tier 1 enter the scoring phase. Items matching Tier 2 keywords receive a quality score bonus.

---

## Phase 2 — Scoring & Selection

Each candidate is scored on a multi-factor quality rubric. The top 15,000 by score are kept.

### Scoring Factors

| Factor                     | Points | Notes                                    |
|----------------------------|--------|------------------------------------------|
| Tier 2 keyword hits        | +0–5   | Up to 5 secondary keywords              |
| Code block present         | +3     | Presence of ``` fenced code             |
| Code block length ≥ 10 lines | +2   | Substantive code, not snippets          |
| Response length 200–2000 chars | +2 | Avoids too-short or too-long responses  |
| Multiple hardware platforms mentioned | +2 | Breadth of embedded coverage  |
| Clean formatting           | +1     | No broken markdown, no truncation       |
| Tier 1 keyword density     | +0–3   | More relevant terms = higher score      |

### Deduplication

MD5 hashing is applied to the concatenation of instruction + response. Any exact duplicate is dropped before final selection.

---

## Dataset Format

Each example is formatted using the **Llama-3.1 chat template** with system, user, and assistant turns.

### Raw Alpaca format (intermediate)

```json
{
  "instruction": "Write Arduino code to read a DHT11 sensor and print temperature over Serial.",
  "input": "",
  "output": "Here's the Arduino sketch to read from a DHT11 sensor:\n\n```cpp\n#include <DHT.h>\n..."
}
```

### Final JSONL format (after template application)

```json
{
  "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are CircuitMentor, an embedded systems tutor specializing in Arduino, ESP32, Raspberry Pi Pico, STM32, MicroPython, Embedded C, and IoT. Answer clearly and include working code examples where appropriate.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite Arduino code to read a DHT11 sensor...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere's the Arduino sketch...<|eot_id|>"
}
```

The system prompt is baked into every example during dataset construction, so the model learns the CircuitMentor persona from training — not just from a runtime system prompt.

---

## Train / Val Split

```python
# 90/10 split — deterministic shuffle with fixed seed
random.seed(42)
random.shuffle(examples)

train = examples[:13500]
val   = examples[13500:]
```

Using a fixed seed ensures reproducibility — the same split is produced on every run.

---

## Synthetic Seeds

12 hand-crafted examples were added to the dataset to cover edge cases and desired response style:

- Explicit refusals for out-of-scope topics (e.g. FPGA/Verilog → "That's my cousin's department")
- Identity responses for "who made you" and "what are you"
- Hello World examples in 5 embedded languages (C, MicroPython, CircuitPython, Arduino, MicroPython ESP32)
- FreeRTOS task creation with full explanation
- Common beginner mistakes (e.g. missing `pinMode`, wrong `delay` usage)

These seeds are blended into the main dataset and weighted equally during training.

---

## Coverage Analysis

After final selection, the 15,000 examples break down approximately as follows:

| Topic Area              | Approx. % of Dataset |
|-------------------------|---------------------|
| Arduino (AVR/ATmega)    | 28%                 |
| ESP32 / ESP8266         | 22%                 |
| Raspberry Pi Pico (MicroPython) | 14%        |
| STM32 / ARM Cortex-M    | 10%                 |
| General Embedded C/C++  | 11%                 |
| Protocols (I2C, SPI, UART, MQTT, BLE) | 8% |
| FreeRTOS                | 4%                  |
| IoT / Connectivity      | 3%                  |

81% of all examples contain at least one fenced code block.

---

## Reproducing from Scratch

```bash
# 1. Install dependencies (laptop)
pip install datasets tqdm colorama

# 2. Run pipeline
python src/build_dataset.py

# 3. Verify output
wc -l embedded_dataset/train.jsonl   # Should be 13500
wc -l embedded_dataset/val.jsonl     # Should be 1500

# 4. Inspect a sample
python -c "
import json
with open('embedded_dataset/train.jsonl') as f:
    for i, line in enumerate(f):
        print(json.loads(line)['text'][:500])
        print('---')
        if i == 2: break
"

# 5. Transfer to Jetson
scp -r embedded_dataset/ user@<jetson-ip>:/mnt/circuitmentor/datasets/
```

---

## Dataset Licenses

| Source                  | License             |
|-------------------------|---------------------|
| Glaive-Code-Assistant   | Apache 2.0          |
| Magicoder-Evol-110k     | Research permissive |
| WizardLM-Evol-196k      | Research permissive |
| OpenHermes              | MIT                 |
| Code-Instructions-122k  | Apache 2.0          |
| Alpaca-Code-18k         | CC BY 4.0           |
| CodeAlpaca-20k          | Apache 2.0          |
| Synthetic Seeds         | Original — Rajas L  |

All sources are permissive for research and personal non-commercial use.

---

## Known Issues

| Issue                          | Cause                                   | Fix                                     |
|--------------------------------|-----------------------------------------|-----------------------------------------|
| Low hit rate on Code-Instructions-122k | Dataset is general-purpose, few embedded examples | Expected — filter is working correctly |
| Streaming timeout on slow connections | HuggingFace CDN latency           | Re-run `build_dataset.py` — streaming resumes |
| Duplicate examples surviving filter | Near-duplicates with minor whitespace differences | MD5 catches exact dupes; near-dupes are rare and acceptable |

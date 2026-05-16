"""
build_dataset.py — EmbeddedTutor-SLM Dataset Builder (v2 — Python 3.9 Compatible)
===================================================================================
Phase 1: Scrape ~50,000 raw candidates from HuggingFace → raw_50k.jsonl
Phase 2: Score, clean, deduplicate → final train.jsonl + val.jsonl (~15k)

Usage:
    pip install datasets tqdm colorama
    python build_dataset.py
"""

import json
import re
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED           = 42
TARGET_FINAL          = 15_000
RAW_TARGET            = 50_000
VAL_SPLIT_RATIO       = 0.10
OUTPUT_DIR            = Path("./embedded_dataset")
RAW_DIR               = OUTPUT_DIR / "raw"
MIN_RESPONSE_CHARS    = 120
MAX_RESPONSE_CHARS    = 3_200
MAX_INSTRUCTION_CHARS = 1_600

random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

TIER1_KEYWORDS = [
    "esp32","esp8266","esp-idf","esp-now","espidf",
    "atmega","atmega328","atmega2560","attiny","avr",
    "stm32","pic microcontroller","pic16","pic18",
    "arduino uno","arduino mega","arduino nano","arduino ide",
    "raspberry pi pico","rp2040",
    "micropython","circuitpython",
    "freertos","rtos task","vtaskdelay",
    "i2c","wire.begin","wire.requestfrom",
    "spi","spi.begin","spi.transfer",
    "uart","usart","serial.begin","serial.print",
    "gpio","pinmode","digitalwrite","digitalread",
    "pwm","analogwrite","analogread",
    "interrupt","isr","attachinterrupt",
    "mqtt","lorawan","lora","zigbee","ble ",
    "neopixel","ws2812","servo motor","stepper motor",
    "watchdog","wdt","esp_task_wdt",
    "deep sleep","light sleep","esp_sleep",
    "ds18b20","dht11","dht22","bme280","mpu6050",
    "onewire","ds18x20",
    "can bus","modbus","rs485",
]

TIER2_KEYWORDS = [
    "microcontroller","mcu","embedded c","embedded system",
    "firmware","bare metal","bootloader","flash memory",
    "register","bit mask","bitwise","volatile",
    "sensor","actuator","lcd","oled","led strip",
    "iot","edge device","real-time",
    "dma","timer overflow","baud rate","ota update",
    "hall sensor","rotary encoder","ultrasonic",
    "hc-sr04","l298n","a4988","lipo battery",
]

ALL_KEYWORDS = [kw.lower() for kw in TIER1_KEYWORDS + TIER2_KEYWORDS]
TIER1_SET    = set(kw.lower() for kw in TIER1_KEYWORDS)

REJECT_PATTERNS = [
    r"\bweb scraping\b",r"\bdjango\b",r"\bflask\b",
    r"\breact\.?js\b",r"\bvue\.?js\b",r"\bangular\b",
    r"\bdatabase migration\b",r"\bsql join\b",
    r"\bgpt\b",r"\bllm\b",r"\btransformer model\b",
    r"\bkubernetes\b",r"\bdocker compose\b",
    r"\bverilog\b",r"\bvhdl\b",r"\bfpga\b",
    r"\bimage classification\b",r"\bneural network training\b",
]
REJECT_RE = re.compile("|".join(REJECT_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# HF SOURCES
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = [
    {"name":"CodeAlpaca-20k","hf_id":"sahil2801/CodeAlpaca-20k","split":"train",
     "inst_col":"instruction","inp_col":"input","out_col":"output","max_pull":20_000},
    {"name":"Alpaca-Code-18k","hf_id":"iamtarun/python_code_instructions_18k_alpaca","split":"train",
     "inst_col":"instruction","inp_col":"input","out_col":"output","max_pull":18_000},
    {"name":"Code-Instructions-122k","hf_id":"TokenBender/code_instructions_122k_alpaca_style","split":"train",
     "inst_col":"instruction","inp_col":"input","out_col":"output","max_pull":40_000},
    {"name":"Magicoder-Evol-110k","hf_id":"ise-uiuc/Magicoder-Evol-Instruct-110K","split":"train",
     "inst_col":"instruction","inp_col":None,"out_col":"response","max_pull":40_000},
    {"name":"CodeFeedback-Filtered","hf_id":"m-a-p/CodeFeedback-Filtered-Instruction","split":"train",
     "inst_col":"instruction","inp_col":None,"out_col":"output","max_pull":30_000},
    {"name":"Glaive-Code-Assistant","hf_id":"glaiveai/glaive-code-assistant","split":"train",
     "inst_col":"question","inp_col":None,"out_col":"answer","max_pull":30_000},
    {"name":"OpenHermes","hf_id":"teknium/openhermes","split":"train",
     "inst_col":"instruction","inp_col":"input","out_col":"output","max_pull":20_000},
    {"name":"WizardLM-Evol","hf_id":"WizardLMTeam/WizardLM_evol_instruct_V2_196k","split":"train",
     "inst_col":"conversations","inp_col":None,"out_col":None,"max_pull":20_000,"special":"conversations"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SEEDS
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_EXAMPLES = [
    {"instruction":"Write Arduino code to blink an LED on pin 13 every 500 milliseconds.","input":"",
     "output":"```cpp\nconst int LED_PIN = 13;\n\nvoid setup() {\n  pinMode(LED_PIN, OUTPUT);\n}\n\nvoid loop() {\n  digitalWrite(LED_PIN, HIGH);\n  delay(500);\n  digitalWrite(LED_PIN, LOW);\n  delay(500);\n}\n```\n\n**How it works:** `pinMode` sets pin 13 as output. `digitalWrite(HIGH)` drives 5V to the LED, `delay(500)` holds for 500ms. Full cycle = 1 second → LED blinks at 1 Hz."},
    {"instruction":"Explain I2C and show Arduino master reading 2 bytes from slave at address 0x48.","input":"",
     "output":"**I2C** uses SDA (data) + SCL (clock). Master addresses slaves by 7-bit address.\n\n```cpp\n#include <Wire.h>\nvoid setup() { Wire.begin(); Serial.begin(115200); }\nvoid loop() {\n  Wire.requestFrom(0x48, 2);\n  while (Wire.available()) Serial.println(Wire.read(), HEX);\n  delay(1000);\n}\n```\n\n**Tip:** Add 4.7kΩ pull-ups on SDA/SCL to VCC."},
    {"instruction":"Write MicroPython for ESP32 to read DS18B20 over OneWire on GPIO 4.","input":"",
     "output":"```python\nimport machine, onewire, ds18x20, time\now = onewire.OneWire(machine.Pin(4))\nds = ds18x20.DS18X20(ow)\nroms = ds.scan()\nwhile True:\n    ds.convert_temp()\n    time.sleep_ms(800)\n    for rom in roms:\n        print(f'{ds.read_temp(rom):.2f} C')\n    time.sleep(2)\n```\n\nRequires 4.7kΩ pull-up on data pin. Module bundled in ESP32 MicroPython firmware."},
    {"instruction":"Connect ESP32 to WiFi and publish to MQTT using MicroPython.","input":"",
     "output":"```python\nimport network, time\nfrom umqtt.simple import MQTTClient\nwlan = network.WLAN(network.STA_IF)\nwlan.active(True)\nwlan.connect('SSID', 'PASSWORD')\nwhile not wlan.isconnected(): time.sleep(1)\nclient = MQTTClient(b'esp32', '192.168.1.100')\nclient.connect()\nwhile True:\n    client.publish(b'home/temp', b'25.4')\n    time.sleep(10)\n```\n\nInstall: `mpremote mip install umqtt.simple`"},
    {"instruction":"Write bare-metal C to send 'Hello' over UART at 9600 baud on ATmega328P.","input":"",
     "output":"```c\n#include <avr/io.h>\n#define F_CPU 16000000UL\n#define UBRR_VAL ((F_CPU/(16UL*9600))-1)\nvoid uart_init(){UBRR0H=(UBRR_VAL>>8);UBRR0L=UBRR_VAL;UCSR0B=(1<<TXEN0);UCSR0C=(1<<UCSZ01)|(1<<UCSZ00);}\nvoid uart_putchar(char c){while(!(UCSR0A&(1<<UDRE0)));UDR0=c;}\nvoid uart_print(const char*s){while(*s)uart_putchar(*s++);}\nint main(){uart_init();uart_print(\"Hello\\r\\n\");while(1){}}\n```"},
    {"instruction":"Use external interrupt on Arduino to toggle LED on button press — no polling.","input":"",
     "output":"```cpp\nconst int BTN=2, LED=13;\nvolatile bool pressed=false;\nvoid onPress(){pressed=true;}\nvoid setup(){\n  pinMode(BTN,INPUT_PULLUP); pinMode(LED,OUTPUT);\n  attachInterrupt(digitalPinToInterrupt(BTN),onPress,FALLING);\n}\nvoid loop(){\n  if(pressed){pressed=false; digitalWrite(LED,!digitalRead(LED));}\n}\n```\n\nAlways declare ISR-shared variables as `volatile`. Never use `delay()` inside an ISR."},
    {"instruction":"FreeRTOS on ESP32: two tasks — blink LED every 500ms, print heartbeat every 2s.","input":"",
     "output":"```cpp\n#include <Arduino.h>\nvoid taskBlink(void*p){\n  pinMode(2,OUTPUT);\n  for(;;){digitalWrite(2,!digitalRead(2));vTaskDelay(pdMS_TO_TICKS(500));}\n}\nvoid taskHeart(void*p){\n  for(;;){Serial.println(\"[OK]\");vTaskDelay(pdMS_TO_TICKS(2000));}\n}\nvoid setup(){\n  Serial.begin(115200);\n  xTaskCreatePinnedToCore(taskBlink,\"Blink\",1024,NULL,1,NULL,0);\n  xTaskCreatePinnedToCore(taskHeart,\"Heart\",2048,NULL,1,NULL,1);\n}\nvoid loop(){vTaskDelay(portMAX_DELAY);}\n```"},
    {"instruction":"What causes watchdog timer reset on ESP32 and how do I prevent it?","input":"",
     "output":"**WDT fires** when CPU doesn't yield within ~3-5 seconds. Common causes: tight loops, blocking HTTP calls.\n\n**Fix — yield every N iterations:**\n```cpp\nfor(long i=0;i<500000L;i++){\n  process(i);\n  if(i%5000==0) yield();   // feeds WDT + yields FreeRTOS\n}\n```\n**Fix — manual reset:**\n```cpp\n#include <esp_task_wdt.h>\nvoid setup(){esp_task_wdt_init(10,true);esp_task_wdt_add(NULL);}\nvoid loop(){heavyWork();esp_task_wdt_reset();}\n```"},
    {"instruction":"Read analog sensor on Raspberry Pi Pico with MicroPython and convert to voltage.","input":"",
     "output":"```python\nfrom machine import ADC, Pin\nimport time\nadc = ADC(Pin(26))   # GPIO 26 = ADC0\nwhile True:\n    raw = adc.read_u16()       # 0-65535\n    volts = raw * 3.3 / 65535\n    print(f'Raw:{raw}  Voltage:{volts:.3f}V')\n    time.sleep(0.5)\n```\n\nDo NOT exceed 3.3V on ADC pins — RP2040 is NOT 5V tolerant on these."},
    {"instruction":"My Arduino Serial Monitor shows garbage characters. How do I fix it?","input":"",
     "output":"**Root cause: baud rate mismatch.**\n\n1. Check code: `Serial.begin(115200);`\n2. Set monitor dropdown to **115200**.\n\n| Symptom | Fix |\n|---|---|\n| ÿÿÿ garbage | Match baud rate |\n| First chars garbled | Add `delay(1000)` after `Serial.begin()` |\n| No output | Check TX/RX wiring |\n\n```cpp\nvoid setup(){\n  Serial.begin(115200);\n  delay(1000);\n  Serial.println(\"Boot OK\");\n}\n```"},
    {"instruction":"Explain PWM and write Arduino code to fade an LED and control a servo.","input":"",
     "output":"**PWM** simulates analog output by toggling a pin at high speed. Duty cycle = % ON time.\n\n**LED fade:**\n```cpp\nfor(int b=0;b<=255;b++){analogWrite(9,b);delay(10);}\n```\n\n**Servo (Servo library):**\n```cpp\n#include <Servo.h>\nServo s;\nvoid setup(){s.attach(9);}\nvoid loop(){s.write(0);delay(1000);s.write(90);delay(1000);s.write(180);delay(1000);}\n```\nPulse: 1ms=0°, 1.5ms=90°, 2ms=180°."},
    {"instruction":"Configure SPI Mode 0 on Arduino to read a sensor register.","input":"",
     "output":"```cpp\n#include <SPI.h>\nconst int CS=10;\nvoid setup(){\n  SPI.begin();\n  SPI.beginTransaction(SPISettings(1000000,MSBFIRST,SPI_MODE0));\n  pinMode(CS,OUTPUT); digitalWrite(CS,HIGH);\n}\nbyte readReg(byte reg){\n  digitalWrite(CS,LOW);\n  SPI.transfer(reg|0x80);\n  byte v=SPI.transfer(0);\n  digitalWrite(CS,HIGH);\n  return v;\n}\nvoid loop(){Serial.println(readReg(0x0F),HEX);delay(500);}\n```\nSPI Mode 0: CPOL=0 CPHA=0, clock idles LOW, sample on rising edge."},
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def keyword_score(text: str) -> Tuple[int, int]:
    lower = text.lower()
    t1 = sum(1 for kw in TIER1_SET if kw in lower)
    t2 = sum(1 for kw in ALL_KEYWORDS if kw not in TIER1_SET and kw in lower)
    return t1, t2

def has_any_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ALL_KEYWORDS)

def passes_basic_length(inst: str, resp: str) -> bool:
    return (MIN_RESPONSE_CHARS <= len(resp) <= MAX_RESPONSE_CHARS
            and 10 <= len(inst) <= MAX_INSTRUCTION_CHARS)

def passes_quality_filter(inst: str, resp: str) -> bool:
    combined = inst + " " + resp
    if not has_any_keyword(combined): return False
    if REJECT_RE.search(combined): return False
    if not passes_basic_length(inst, resp): return False
    return True

def make_hash(inst: str, resp: str) -> str:
    return hashlib.md5((inst.strip() + resp[:200]).encode()).hexdigest()

def normalize_alpaca(row: dict, src: dict) -> Optional[Dict]:
    try:
        if src.get("special") == "conversations":
            convs = row.get("conversations", [])
            if len(convs) < 2: return None
            human = next((c["value"] for c in convs if c.get("from") == "human"), "")
            gpt   = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
            if not human or not gpt: return None
            return {"instruction": human.strip(), "input": "", "output": gpt.strip()}
        inst = str(row.get(src["inst_col"], "") or "").strip()
        ctx  = str(row.get(src["inp_col"], "") or "").strip() if src.get("inp_col") else ""
        resp = str(row.get(src["out_col"], "") or "").strip()
        if not inst or not resp: return None
        return {"instruction": inst, "input": ctx, "output": resp}
    except Exception:
        return None

def has_code_block(text: str) -> bool:
    return "```" in text

def quality_score(ex: dict) -> float:
    combined = ex["instruction"] + " " + ex["output"]
    t1, t2   = keyword_score(combined)
    code_bonus = 1.5 if has_code_block(ex["output"]) else 0.0
    resp_len   = len(ex["output"])
    len_score  = max(0.0, 1.0 - abs(resp_len - 600) / 1200)
    return t1 * 3.0 + t2 * 1.0 + code_bonus + len_score

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1
# ─────────────────────────────────────────────────────────────────────────────

def phase1_collect() -> List[Dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{Fore.CYAN}{'='*62}")
    print(f"  PHASE 1 — Raw Collection  (target ~{RAW_TARGET:,})")
    print(f"{'='*62}{Style.RESET_ALL}\n")

    all_raw: List[Dict] = []
    seen: set = set()

    print(f"{Fore.GREEN}  Injecting {len(SYNTHETIC_EXAMPLES)} synthetic seed examples...{Style.RESET_ALL}")
    for ex in SYNTHETIC_EXAMPLES:
        h = make_hash(ex["instruction"], ex["output"])
        seen.add(h)
        ex["_source"] = "synthetic_seed"
        all_raw.append(ex)

    for src in SOURCES:
        name = src["name"]
        print(f"\n  {Fore.YELLOW}► {name}  (scan up to {src['max_pull']:,}){Style.RESET_ALL}")
        try:
            ds = load_dataset(src["hf_id"], split=src["split"],
                              streaming=True, trust_remote_code=True)
            pulled = passed = 0
            with tqdm(total=src["max_pull"], desc="    Scanning", unit="rows", leave=False) as pbar:
                for row in ds:
                    if pulled >= src["max_pull"]: break
                    pulled += 1; pbar.update(1)
                    ex = normalize_alpaca(row, src)
                    if ex is None: continue
                    combined = ex["instruction"] + " " + ex["output"]
                    if not has_any_keyword(combined): continue
                    if not passes_basic_length(ex["instruction"], ex["output"]): continue
                    if REJECT_RE.search(combined): continue
                    h = make_hash(ex["instruction"], ex["output"])
                    if h in seen: continue
                    seen.add(h)
                    ex["_source"] = name
                    all_raw.append(ex)
                    passed += 1
            print(f"    ✓ {passed:,} passed  ({pulled:,} scanned)")
        except Exception as e:
            log.warning(f"  ✗ {name} failed: {e}")

    raw_path = RAW_DIR / "raw_50k.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for ex in all_raw:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n  {Fore.GREEN}✓ Phase 1 done — {len(all_raw):,} raw examples → {raw_path}{Style.RESET_ALL}")
    return all_raw

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2
# ─────────────────────────────────────────────────────────────────────────────

def phase2_clean(all_raw: List[Dict]) -> None:
    print(f"\n{Fore.CYAN}{'='*62}")
    print(f"  PHASE 2 — Quality Cleaning  ({len(all_raw):,} → {TARGET_FINAL:,})")
    print(f"{'='*62}{Style.RESET_ALL}\n")

    print("  [1/4] Strict quality filter...")
    filtered = [ex for ex in tqdm(all_raw, desc="    Filtering", leave=False)
                if passes_quality_filter(ex["instruction"], ex["output"])]
    print(f"        {len(all_raw):,} → {len(filtered):,}")

    print("  [2/4] Scoring...")
    for ex in tqdm(filtered, desc="    Scoring", leave=False):
        ex["_score"] = quality_score(ex)

    print("  [3/4] Selecting top examples...")
    seeds  = [ex for ex in filtered if ex.get("_source") == "synthetic_seed"]
    others = sorted([ex for ex in filtered if ex.get("_source") != "synthetic_seed"],
                    key=lambda x: x["_score"], reverse=True)
    selected = seeds + others[:TARGET_FINAL - len(seeds)]
    random.shuffle(selected)
    print(f"        Selected {len(selected):,} ({len(seeds)} seeds + {len(selected)-len(seeds):,} from sources)")

    print("  [4/4] Writing splits...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final = [{"instruction": ex["instruction"], "input": ex["input"], "output": ex["output"]}
             for ex in selected]
    split  = int(len(final) * (1 - VAL_SPLIT_RATIO))
    train, val = final[:split], final[split:]

    for path, data in [(OUTPUT_DIR/"train.jsonl", train), (OUTPUT_DIR/"val.jsonl", val)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    code_pct = round(sum(1 for ex in final if has_code_block(ex["output"])) / len(final) * 100, 1)
    src_dist: Dict[str, int] = defaultdict(int)
    for ex in selected: src_dist[ex.get("_source","?")] += 1

    stats = {"build_timestamp": datetime.now().isoformat(),
             "phase1_raw": len(all_raw), "phase2_filtered": len(filtered),
             "train": len(train), "val": len(val),
             "code_block_pct": code_pct, "sources": dict(src_dist)}
    with open(OUTPUT_DIR/"stats.json","w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{Fore.CYAN}{'='*62}")
    print(f"  Dataset Build Complete")
    print(f"{'='*62}{Style.RESET_ALL}")
    print(f"  Phase 1 raw     : {len(all_raw):>7,}")
    print(f"  After cleaning  : {len(selected):>7,}")
    print(f"  Train           : {len(train):>7,}  → {OUTPUT_DIR/'train.jsonl'}")
    print(f"  Val             : {len(val):>7,}  → {OUTPUT_DIR/'val.jsonl'}")
    print(f"  With code block : {code_pct}%")
    print(f"\n  Source breakdown:")
    for s, c in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"    {c:>5,}  {s}")
    print(f"\n  {Fore.YELLOW}SCP to Orin when ready:{Style.RESET_ALL}")
    print(f"  scp -r ./embedded_dataset/ user@<orin-ip>:/mnt/vlsi/embedded_tutor/datasets/")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = phase1_collect()
    phase2_clean(raw)

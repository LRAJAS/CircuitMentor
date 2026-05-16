"""
frontend.py — CircuitMentor Streamlit Frontend
===============================================
Run via: launch.py — do not run directly
Backend must be running on port 8000 before this starts.

Features:
  - Responsive (desktop / tablet / mobile)
  - Login, chat, exit flow
  - Admin dashboard
  - Easter eggs: 4x logo tap, konami code, idle popups, hidden commands
  - Session auto-cleanup on exit
  - Chat export as .txt
"""

import time
import json
import requests
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BACKEND   = "http://localhost:8000"
ADMIN_USR = "SungjinHo"
IDLE_WARN = 5 * 60      # 5 min → warning popup
IDLE_KICK = 10 * 60     # 10 min → auto logout warning

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CircuitMentor",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGO SVG (circuit board + glasses = teacher identity)
# ─────────────────────────────────────────────────────────────────────────────

LOGO_SVG = """
<svg width="64" height="64" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="80" height="80" rx="18" fill="#0C447C"/>
  <g stroke="#58A6FF" stroke-linecap="round" stroke-linejoin="round">
    <line x1="40" y1="14" x2="40" y2="26" stroke-width="2.5"/>
    <line x1="40" y1="54" x2="40" y2="66" stroke-width="2.5"/>
    <line x1="14" y1="40" x2="26" y2="40" stroke-width="2.5"/>
    <line x1="54" y1="40" x2="66" y2="40" stroke-width="2.5"/>
    <line x1="22" y1="22" x2="30" y2="30" stroke-width="2"/>
    <line x1="50" y1="50" x2="58" y2="58" stroke-width="2"/>
    <line x1="58" y1="22" x2="50" y2="30" stroke-width="2"/>
    <line x1="22" y1="58" x2="30" y2="50" stroke-width="2"/>
    <circle cx="40" cy="40" r="14" stroke-width="1.5"/>
    <circle cx="14" cy="40" r="2.5" fill="#0C447C" stroke-width="1.5"/>
    <circle cx="66" cy="40" r="2.5" fill="#0C447C" stroke-width="1.5"/>
    <circle cx="40" cy="14" r="2.5" fill="#0C447C" stroke-width="1.5"/>
    <circle cx="40" cy="66" r="2.5" fill="#0C447C" stroke-width="1.5"/>
  </g>
  <circle cx="14" cy="40" r="1.8" fill="#B5D4F4"/>
  <circle cx="66" cy="40" r="1.8" fill="#B5D4F4"/>
  <circle cx="40" cy="14" r="1.8" fill="#B5D4F4"/>
  <circle cx="40" cy="66" r="1.8" fill="#B5D4F4"/>
  <rect x="28" y="37" width="9" height="7" rx="3" fill="none" stroke="#B5D4F4" stroke-width="1.8"/>
  <rect x="43" y="37" width="9" height="7" rx="3" fill="none" stroke="#B5D4F4" stroke-width="1.8"/>
  <line x1="37" y1="40.5" x2="43" y2="40.5" stroke="#B5D4F4" stroke-width="1.8"/>
  <line x1="22" y1="40.5" x2="28" y2="40.5" stroke="#B5D4F4" stroke-width="1.5" stroke-dasharray="1 1"/>
  <line x1="52" y1="40.5" x2="58" y2="40.5" stroke="#B5D4F4" stroke-width="1.5" stroke-dasharray="1 1"/>
</svg>
"""

LOGO_SVG_SMALL = LOGO_SVG.replace('width="64"', 'width="36"').replace('height="64"', 'height="36"')

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.block-container { padding-top: 1rem; max-width: 100%; }
#MainMenu, footer, header { visibility: hidden; }

.cm-header {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px; border-bottom: 1px solid #21262d;
  background: #161b22; border-radius: 8px; margin-bottom: 16px;
}
.cm-title { font-size: 1.3rem; font-weight: 600; color: #58a6ff; margin: 0; }
.cm-sub   { font-size: 0.75rem; color: #8b949e; margin: 0; }
.cm-badge {
  font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
  background: #1f6feb22; border: 1px solid #1f6feb; color: #58a6ff;
}

.msg-user {
  background: #1c2128; border: 1px solid #30363d;
  border-radius: 12px 12px 3px 12px; padding: 12px 16px;
  margin: 8px 0 8px 40px; color: #e6edf3; font-size: 0.9rem; line-height: 1.6;
}
.msg-bot {
  background: #161b22; border: 1px solid #21262d;
  border-left: 3px solid #58a6ff; border-radius: 3px 12px 12px 12px;
  padding: 12px 16px; margin: 8px 40px 8px 0;
  color: #e6edf3; font-size: 0.9rem; line-height: 1.8;
}
.msg-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; margin-bottom: 5px; }
.lbl-user { color: #7c3aed; }
.lbl-bot  { color: #58a6ff; }

.msg-bot pre {
  background: #0d1117 !important; border: 1px solid #30363d;
  border-radius: 6px; padding: 12px; font-size: 0.82rem;
  overflow-x: auto; margin: 8px 0;
}

.stTextArea textarea {
  background: #161b22 !important; border: 1px solid #30363d !important;
  color: #e6edf3 !important; border-radius: 8px !important;
}
.stTextArea textarea:focus { border-color: #58a6ff !important; }
.stButton > button {
  border-radius: 8px !important; font-weight: 600 !important;
  transition: all 0.15s !important;
}

.stat-card {
  background: #161b22; border: 1px solid #21262d; border-radius: 8px;
  padding: 10px 14px; text-align: center;
}
.stat-n { font-size: 1.4rem; font-weight: 600; color: #58a6ff; }
.stat-l { font-size: 0.72rem; color: #8b949e; }

.chip {
  display: inline-block; background: #21262d; border: 1px solid #30363d;
  border-radius: 12px; padding: 2px 8px; font-size: 0.72rem; color: #8b949e; margin: 2px;
}

.join-card {
  max-width: 420px; margin: 60px auto; background: #161b22;
  border: 1px solid #30363d; border-radius: 16px; padding: 32px;
}
.join-logo { text-align: center; margin-bottom: 20px; }
.join-title { text-align: center; font-size: 1.5rem; font-weight: 600;
  color: #58a6ff; margin: 8px 0 4px; }
.join-sub { text-align: center; font-size: 0.82rem; color: #8b949e; margin-bottom: 24px; }

.credits-overlay {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.88); z-index: 9999;
  display: flex; align-items: center; justify-content: center;
}
.credits-card {
  background: #161b22; border: 1px solid #58a6ff; border-radius: 16px;
  padding: 40px 48px; text-align: center; max-width: 400px;
}
.credits-title { font-size: 1.1rem; color: #58a6ff; font-weight: 600; margin-bottom: 16px; }
.credits-name  { font-size: 1.3rem; color: #e6edf3; font-weight: 700; margin: 8px 0; }
.credits-under { font-size: 0.8rem; color: #8b949e; margin-bottom: 12px; }
.credits-sirs  { font-size: 1rem; color: #b5d4f4; font-weight: 500; }
.credits-meta  { font-size: 0.72rem; color: #5f6878; margin-top: 20px; line-height: 2; }

.full-screen-msg {
  text-align: center; padding: 80px 20px;
}
.full-screen-msg h2 { color: #e6edf3; font-size: 1.5rem; margin-bottom: 12px; }
.full-screen-msg p  { color: #8b949e; font-size: 0.9rem; }

.elapsed-tag { font-size: 0.68rem; color: #5f6878; margin-top: 6px; }

@media (max-width: 768px) {
  .msg-user { margin-left: 8px; }
  .msg-bot  { margin-right: 8px; }
  .join-card { margin: 20px; padding: 20px; }
  [data-testid="stSidebar"] { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "screen":         "join",      # join | chat | exit | admin | full
    "session_id":     None,
    "username":       None,
    "chat":           [],
    "logo_clicks":    0,
    "show_credits":   False,
    "last_activity":  time.time(),
    "admin_authed":   False,
    "admin_stats":    None,
    "input_key":      0,
    "pending_prompt": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs) -> dict:
    try:
        r = getattr(requests, method)(f"{BACKEND}{path}", timeout=120, **kwargs)
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Backend offline. Please tell the admin."}
    except Exception as e:
        return {"error": str(e)}


def do_join(username: str) -> str | None:
    r = api("post", "/session/join", json={"username": username})
    if "error" in r:
        return r["error"]
    if "detail" in r:
        return r["detail"]
    st.session_state.session_id = r["session_id"]
    st.session_state.username   = r["username"]
    st.session_state.chat       = []
    st.session_state.screen     = "chat"
    return None


def do_send(message: str, temperature: float, max_tokens: int):
    r = api("post", "/chat", json={
        "session_id":  st.session_state.session_id,
        "message":     message,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    })
    if "error" in r or "detail" in r:
        err = r.get("error") or r.get("detail", "Unknown error")
        st.session_state.chat.append({"role": "error", "content": err})
    else:
        st.session_state.chat.append({"role": "user",      "content": message})
        st.session_state.chat.append({"role": "assistant", "content": r["response"],
                                       "elapsed": r.get("elapsed", 0)})
    st.session_state.last_activity = time.time()


def do_leave() -> str:
    r = api("post", "/session/leave", json={"session_id": st.session_state.session_id})
    return r.get("chat_export", "")


def render_message(role: str, content: str, elapsed: float = 0):
    import re
    if role == "user":
        st.markdown(f"""
        <div class="msg-user">
          <div class="msg-label lbl-user">You</div>
          {content}
        </div>""", unsafe_allow_html=True)
    elif role == "error":
        st.error(content)
    else:
        def replace_code(m):
            lang = m.group(1) or "text"
            code = m.group(2).replace("<", "&lt;").replace(">", "&gt;")
            return f'<pre><code class="language-{lang}">{code}</code></pre>'
        html = re.sub(r"```(\w*)\n?(.*?)```", replace_code, content, flags=re.DOTALL)
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"`([^`]+)`",
                      r'<code style="background:#21262d;padding:1px 5px;border-radius:3px;font-size:0.85em">\1</code>',
                      html)
        html = html.replace("\n", "<br>")
        elapsed_tag = f'<div class="elapsed-tag">answered in {elapsed}s</div>' if elapsed else ""
        st.markdown(f"""
        <div class="msg-bot">
          <div class="msg-label lbl-bot">CircuitMentor</div>
          {html}
          {elapsed_tag}
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EASTER EGG JAVASCRIPT
# Handles: konami code, idle detection, logo clicks, beforeunload
# ─────────────────────────────────────────────────────────────────────────────

def inject_easter_egg_js():
    components.html(f"""
<script>
(function() {{
  // ── Konami code ─────────────────────────────────────────────────────────
  const KONAMI = [38,38,40,40,37,39,37,39,66,65];
  let ki = 0;
  document.addEventListener('keydown', e => {{
    ki = (e.keyCode === KONAMI[ki]) ? ki + 1 : 0;
    if (ki === KONAMI.length) {{
      ki = 0;
      // Binary rain effect
      const canvas = document.createElement('canvas');
      canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:99998;pointer-events:none';
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      document.body.appendChild(canvas);
      const ctx    = canvas.getContext('2d');
      const cols   = Math.floor(canvas.width / 16);
      const drops  = Array(cols).fill(1);
      const rain   = setInterval(() => {{
        ctx.fillStyle = 'rgba(13,17,23,0.08)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#58a6ff';
        ctx.font      = '14px monospace';
        drops.forEach((y, i) => {{
          ctx.fillText(Math.random() > 0.5 ? '1' : '0', i * 16, y * 16);
          drops[i] = (y * 16 > canvas.height && Math.random() > 0.95) ? 0 : y + 1;
        }});
      }}, 40);
      setTimeout(() => {{ clearInterval(rain); canvas.remove(); }}, 2500);
    }}
  }});

  // ── Idle detection ───────────────────────────────────────────────────────
  let idleTimer1, idleTimer2;
  const IDLE1 = {IDLE_WARN * 1000};
  const IDLE2 = {IDLE_KICK * 1000};

  function resetIdle() {{
    clearTimeout(idleTimer1);
    clearTimeout(idleTimer2);
    idleTimer1 = setTimeout(() => {{
      showToast("Still there? The circuit won't build itself..", 4000, "#f0982d");
    }}, IDLE1);
    idleTimer2 = setTimeout(() => {{
      showToast("You've been idle 10 minutes. Session closing soon!", 6000, "#f85149");
    }}, IDLE2);
  }}

  ['mousemove','keydown','click','scroll','touchstart'].forEach(e =>
    document.addEventListener(e, resetIdle, {{passive: true}})
  );
  resetIdle();

  // ── Toast notification ───────────────────────────────────────────────────
  function showToast(msg, duration, color) {{
    const t = document.createElement('div');
    t.innerText = msg;
    t.style.cssText = `
      position:fixed;bottom:32px;left:50%;transform:translateX(-50%);
      background:#161b22;border:1px solid ${{color}};color:#e6edf3;
      padding:12px 24px;border-radius:10px;font-size:14px;font-family:sans-serif;
      z-index:99999;box-shadow:0 4px 20px rgba(0,0,0,0.4);
      animation:fadeInUp 0.3s ease;
    `;
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeInUp {{
        from {{opacity:0;transform:translateX(-50%) translateY(12px)}}
        to   {{opacity:1;transform:translateX(-50%) translateY(0)}}
      }}
    `;
    document.head.appendChild(style);
    document.body.appendChild(t);
    setTimeout(() => t.remove(), duration);
  }}

  // ── Beforeunload: warn on tab close ─────────────────────────────────────
  window.addEventListener('beforeunload', e => {{
    e.preventDefault();
    e.returnValue = 'Your chat session will be deleted. Download your chat first?';
    // Beacon to backend to cleanup
    if (window._cm_session_id) {{
      navigator.sendBeacon(
        'http://localhost:8000/session/leave',
        JSON.stringify({{ session_id: window._cm_session_id }})
      );
    }}
  }});

}})();
</script>
""", height=0)


def inject_session_id(session_id: str):
    """Pass session_id to JS for beforeunload beacon."""
    components.html(f"""
<script>
window._cm_session_id = "{session_id}";
</script>
""", height=0)

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = {
    "Blink LED (Arduino)":    "Write Arduino code to blink an LED on pin 13 every 500ms.",
    "I2C read sensor":        "Explain I2C and show me how to read 2 bytes from a sensor at address 0x48 using Arduino.",
    "ESP32 + MQTT":           "Connect an ESP32 to WiFi and publish temperature to an MQTT broker using MicroPython.",
    "FreeRTOS tasks":         "Create two FreeRTOS tasks on ESP32 — one blinks an LED, one prints a heartbeat every 2 seconds.",
    "Watchdog timer fix":     "My ESP32 keeps resetting with a watchdog error. What causes it and how do I fix it?",
    "ATmega UART":            "Write bare-metal C to transmit 'Hello' over UART at 9600 baud on ATmega328P.",
    "PWM servo control":      "Explain PWM and write Arduino code to sweep a servo from 0 to 180 degrees.",
    "MicroPython ADC":        "Read an analog sensor on Raspberry Pi Pico with MicroPython and convert to voltage.",
}

HIDDEN_COMMANDS = {
    "who made you":      "credits",
    "hello world":       "hello_world",
    "sudo":              "sudo_joke",
    "verilog":           "vlsi_joke",
    "fpga":              "vlsi_joke",
}

HELLO_WORLD_RESPONSE = """Here is Hello World in 5 embedded languages — because anything worth doing is worth overdoing:

```c
// Embedded C — ATmega328P
#include <avr/io.h>
int main(void) {
    DDRB |= (1 << PB5);      // LED pin as output
    while(1) { PORTB ^= (1 << PB5); }  // blink = hello
}
```

```cpp
// Arduino C++
void setup() { Serial.begin(115200); }
void loop()  { Serial.println("Hello, World!"); delay(1000); }
```

```python
# MicroPython — ESP32
from machine import Pin
import time
led = Pin(2, Pin.OUT)
while True:
    print("Hello, World!")
    led.value(not led.value())
    time.sleep(1)
```

```c
// FreeRTOS — ESP32
void helloTask(void *p) {
    for(;;) {
        printf("Hello, World!\\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
```

```c
// Bare-metal C — Raspberry Pi Pico (RP2040)
#include "pico/stdlib.h"
int main() {
    stdio_init_all();
    while(true) { printf("Hello, World!\\n"); sleep_ms(1000); }
}
```

Welcome to CircuitMentor. What can I help you build today?"""

CREDITS_RESPONSE = """
**Built by:** Rajas L

**Under the guidance of:**
Pranav C Sir · Nikhil Z Sir

---

*CircuitMentor v1.0*
*Powered by Llama-3.1-8B (Fine-Tuned)*
*Running on NVIDIA Jetson AGX Orin 64GB*
*April 2026*
"""

SUDO_RESPONSE = "Permission denied. Try `sudo apt install patience` — it usually helps."
VLSI_RESPONSE = "That is my cousin's department — check the VLSI lab next door. I handle the stuff that runs on real hardware."

# ─────────────────────────────────────────────────────────────────────────────
# SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def screen_join():
    inject_easter_egg_js()

    # Check backend health
    health = api("get", "/health")
    if "error" in health:
        st.error("Cannot connect to CircuitMentor server. Please contact the admin.")
        return

    active = health.get("active_users", 0)
    max_u  = 10

    st.markdown(f"""
    <div class="join-card">
      <div class="join-logo">{LOGO_SVG}</div>
      <div class="join-title">CircuitMentor</div>
      <div class="join-sub">Embedded Systems AI Tutor · {active}/{max_u} users online</div>
    </div>
    """, unsafe_allow_html=True)

    if active >= max_u:
        st.error("Lab is full (10 users max). Please try again in a few minutes.")
        if st.button("Refresh"):
            st.rerun()
        return

    if not health.get("model_ready", False):
        st.warning("Model is warming up — please wait 1–2 minutes and refresh.")
        if st.button("Refresh"):
            st.rerun()
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Enter your name to start", placeholder="e.g. Riya, Arjun, Student01",
                                  max_chars=30, label_visibility="visible")
        if st.button("Join CircuitMentor", type="primary", use_container_width=True):
            if not username.strip():
                st.warning("Please enter your name.")
            else:
                err = do_join(username.strip())
                if err:
                    st.error(err)
                else:
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:#5f6878">Admin? Enter your credentials below.</div>',
                    unsafe_allow_html=True)
        if st.button("Admin Login", use_container_width=True):
            st.session_state.screen = "admin"
            st.rerun()


def screen_chat():
    inject_easter_egg_js()
    inject_session_id(st.session_state.session_id)

    # Credits overlay
    if st.session_state.show_credits:
        st.markdown(f"""
        <div class="credits-overlay">
          <div class="credits-card">
            <div class="credits-title">CircuitMentor</div>
            <div class="credits-name">Built by Rajas L</div>
            <div class="credits-under">Under the guidance of</div>
            <div class="credits-sirs">Pranav C Sir &nbsp;·&nbsp; Nikhil Z Sir</div>
            <div class="credits-meta">
              CircuitMentor v1.0<br>
              Llama-3.1-8B Fine-Tuned · LoRA r=16<br>
              NVIDIA Jetson AGX Orin 64GB<br>
              April 2026
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Close", key="close_credits"):
            st.session_state.show_credits = False
            st.rerun()
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo — click 4 times for easter egg
        logo_col, _ = st.columns([1, 3])
        with logo_col:
            st.markdown(LOGO_SVG_SMALL, unsafe_allow_html=True)
        st.markdown(f"**CircuitMentor** <span class='cm-badge'>v1.0</span>", unsafe_allow_html=True)
        st.markdown(f"<small style='color:#8b949e'>Welcome, **{st.session_state.username}**</small>",
                    unsafe_allow_html=True)

        if st.button("🔍 Tap logo 4x for surprise", key="logo_tap", use_container_width=True):
            st.session_state.logo_clicks += 1
            if st.session_state.logo_clicks >= 4:
                st.session_state.show_credits = True
                st.session_state.logo_clicks  = 0
                st.rerun()

        st.divider()

        st.markdown("#### Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        max_tokens  = st.slider("Max tokens", 128, 1024, 512, 64)

        st.divider()

        st.markdown("#### Quick examples")
        for label, prompt in EXAMPLES.items():
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

        st.divider()

        st.markdown("#### Session")
        msg_count = len([m for m in st.session_state.chat if m["role"] == "user"])
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="stat-card"><div class="stat-n">{msg_count}</div><div class="stat-l">Questions</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-n">200</div><div class="stat-l">Max msgs</div></div>',
                    unsafe_allow_html=True)

        # Export button
        if st.session_state.chat:
            chat_lines  = []
            for m in st.session_state.chat:
                role = "You" if m["role"] == "user" else "CircuitMentor"
                if m["role"] != "error":
                    chat_lines.append(f"{role}:\n{m['content']}\n\n{'─'*40}\n")
            chat_txt = f"CircuitMentor Chat — {st.session_state.username}\n{'='*40}\n\n" + "\n".join(chat_lines)
            st.download_button("💾 Download chat (.txt)", chat_txt,
                               file_name=f"circuitmentor_{st.session_state.username}.txt",
                               mime="text/plain", use_container_width=True)

        st.divider()
        if st.button("Exit session", use_container_width=True):
            st.session_state.screen = "exit"
            st.rerun()

        st.markdown(f"""
        <div style='font-size:0.7rem;color:#5f6878;line-height:1.8;margin-top:8px'>
          Llama-3.1-8B · LoRA r=16<br>
          bf16 · Jetson AGX Orin<br>
          No internet · On-device
        </div>""", unsafe_allow_html=True)

        chips = ["Arduino","ESP32","I2C","SPI","UART","MQTT","FreeRTOS","MicroPython","PWM","GPIO","ATmega","STM32"]
        st.markdown("#### Topics")
        st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

    # ── Main chat area ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="cm-header">
      {LOGO_SVG_SMALL}
      <div>
        <div class="cm-title">CircuitMentor <span class="cm-badge">Fine-Tuned</span></div>
        <div class="cm-sub">Edge-native · Running on Jetson AGX Orin · No internet required</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Render chat history
    if not st.session_state.chat:
        st.markdown("""
        <div style='text-align:center;padding:48px 20px;color:#8b949e'>
          <div style='font-size:2.5rem;margin-bottom:12px'>🔌</div>
          <div style='font-size:1rem;font-weight:600;color:#e6edf3'>Ask me anything about embedded systems</div>
          <div style='font-size:0.82rem;margin-top:8px'>Arduino · ESP32 · MicroPython · I2C · SPI · UART · FreeRTOS</div>
          <div style='font-size:0.75rem;margin-top:16px;color:#5f6878'>
            Hint: try the Konami code ↑↑↓↓←→←→BA for a surprise
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat:
            render_message(msg["role"], msg["content"], msg.get("elapsed", 0))

    st.divider()

    # Input
    user_input = st.text_area(
        "Your question",
        value=st.session_state.pending_prompt,
        placeholder="e.g. How do I configure I2C on an ESP32?",
        height=80, label_visibility="collapsed",
        key=f"chat_input_{st.session_state.input_key}",
    )

    col_send, col_clear = st.columns([5, 1])
    with col_send:
        send = st.button("Send to CircuitMentor", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.pending_prompt = ""
            st.session_state.input_key     += 1
            st.rerun()

    if send and user_input.strip():
        st.session_state.pending_prompt = ""
        st.session_state.input_key     += 1
        msg = user_input.strip().lower()

        # Easter egg hidden commands
        if "who made you" in msg or "who built you" in msg:
            st.session_state.chat.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat.append({"role": "assistant", "content": CREDITS_RESPONSE})
            st.rerun()
        elif "hello world" in msg:
            st.session_state.chat.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat.append({"role": "assistant", "content": HELLO_WORLD_RESPONSE})
            st.rerun()
        elif msg.startswith("sudo"):
            st.session_state.chat.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat.append({"role": "assistant", "content": SUDO_RESPONSE})
            st.rerun()
        elif "verilog" in msg or "fpga" in msg or "vhdl" in msg:
            st.session_state.chat.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat.append({"role": "assistant", "content": VLSI_RESPONSE})
            st.rerun()
        else:
            with st.spinner("CircuitMentor is thinking..."):
                do_send(user_input.strip(), temperature, max_tokens)
            st.rerun()

    elif send and not user_input.strip():
        st.warning("Type a question first.")


def screen_exit():
    st.markdown("""
    <div style='max-width:480px;margin:80px auto;text-align:center'>
      <div style='font-size:2rem;margin-bottom:12px'>👋</div>
      <h2 style='color:#e6edf3;margin-bottom:8px'>Before you go</h2>
      <p style='color:#8b949e;margin-bottom:24px'>
        Your session and chat history will be permanently deleted.<br>
        Would you like to download a copy first?
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Build export
    chat_lines = []
    for m in st.session_state.chat:
        if m["role"] != "error":
            role = "You" if m["role"] == "user" else "CircuitMentor"
            chat_lines.append(f"{role}:\n{m['content']}\n\n{'─'*40}\n")
    chat_txt = (
        f"CircuitMentor Chat Export\n"
        f"User: {st.session_state.username}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{'='*40}\n\n"
        + "\n".join(chat_lines)
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            "💾 Download chat as .txt",
            chat_txt,
            file_name=f"circuitmentor_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Exit and delete session", type="primary", use_container_width=True):
            do_leave()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state.screen = "join"
            st.rerun()

        if st.button("Go back to chat", use_container_width=True):
            st.session_state.screen = "chat"
            st.rerun()


def screen_admin():
    st.markdown(f"""
    <div class="cm-header">
      {LOGO_SVG_SMALL}
      <div>
        <div class="cm-title">Admin Dashboard</div>
        <div class="cm-sub">CircuitMentor · Jetson AGX Orin</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.admin_authed:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            pwd = st.text_input("Admin password", type="password")
            if st.button("Authenticate", type="primary", use_container_width=True):
                r = api("post", "/admin/stats", json={"username": ADMIN_USR, "password": pwd})
                if "error" in r or "detail" in r:
                    st.error("Access denied. Wrong password.")
                else:
                    st.session_state.admin_authed = True
                    st.session_state.admin_stats  = r
                    st.session_state.admin_pass   = pwd
                    st.rerun()
            if st.button("Back to login", use_container_width=True):
                st.session_state.screen = "join"
                st.rerun()
        return

    # Fetch fresh stats
    r = api("post", "/admin/stats", json={
        "username": ADMIN_USR,
        "password": st.session_state.get("admin_pass", "")
    })
    if "error" in r or "detail" in r:
        st.error("Session expired. Please re-authenticate.")
        st.session_state.admin_authed = False
        st.rerun()

    sys_info = r.get("system", {})
    sessions = r.get("active_sessions", [])

    # System stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active users",  len(sessions))
    col2.metric("Queue depth",   r.get("queue_depth", 0))
    col3.metric("RAM used",      f"{sys_info.get('ram_used_gb','?')} GB")
    col4.metric("GPU allocated", f"{sys_info.get('gpu_alloc_gb','?')} GB")

    st.divider()
    st.markdown("#### Active sessions")

    if not sessions:
        st.info("No active sessions.")
    else:
        for sess in sessions:
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            c1.markdown(f"**{sess['username']}**")
            c2.markdown(f"<small style='color:#8b949e'>{sess['created_at'][:16]}</small>",
                        unsafe_allow_html=True)
            c3.markdown(f"{sess['message_count']} msgs")
            if c4.button("Kick", key=f"kick_{sess['session_id']}"):
                api("post", f"/admin/kick/{sess['session_id']}", json={
                    "username": ADMIN_USR,
                    "password": st.session_state.get("admin_pass", "")
                })
                st.rerun()

    st.divider()
    col_r, col_c, col_b = st.columns(3)
    if col_r.button("Refresh stats"):
        st.rerun()
    if col_c.button("Clear ALL sessions", type="primary"):
        api("post", "/admin/clear_all", json={
            "username": ADMIN_USR,
            "password": st.session_state.get("admin_pass", "")
        })
        st.success("All sessions cleared.")
    if col_b.button("Back to login"):
        st.session_state.admin_authed = False
        st.session_state.screen       = "join"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

screen = st.session_state.screen

if screen == "join":
    screen_join()
elif screen == "chat":
    screen_chat()
elif screen == "exit":
    screen_exit()
elif screen == "admin":
    screen_admin()

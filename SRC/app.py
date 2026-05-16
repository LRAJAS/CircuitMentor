"""
app.py — CircuitMentor: Embedded Systems AI Tutor
==================================================
A production-grade Streamlit demo UI for your fine-tuned Llama-3 model.

Run:
    pip install streamlit transformers peft torch
    streamlit run app.py

    # With fine-tuned model:
    streamlit run app.py -- --model_path /mnt/vlsi/embedded_tutor/experiments/llama3_tutor_v1/final_model

    # For demo without trained model (uses base Llama 3):
    streamlit run app.py -- --model_path /mnt/vlsi/hf_cache/models/llama3-8b-instruct
"""

import argparse
import sys
import time
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CircuitMentor — Embedded AI Tutor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark professional theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── Header ── */
.cm-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a1a2e 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.cm-header h1 { margin: 0; font-size: 1.8rem; color: #58a6ff; }
.cm-header p  { margin: 4px 0 0; font-size: 0.9rem; color: #8b949e; }
.cm-badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-left: 8px;
}

/* ── Chat messages ── */
.user-msg {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    color: #e6edf3;
    line-height: 1.6;
}
.bot-msg {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 4px 12px 12px 12px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    color: #e6edf3;
    line-height: 1.8;
}
.msg-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.user-label { color: #7c3aed; }
.bot-label  { color: #58a6ff; }

/* ── Code blocks ── */
.bot-msg pre {
    background: #0d1117 !important;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    font-size: 0.85rem;
    line-height: 1.5;
    margin: 10px 0;
}
.bot-msg code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    color: #e6edf3;
}

/* ── Chip tags ── */
.topic-chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #8b949e;
    margin: 2px;
    cursor: default;
}

/* ── Stats bar ── */
.stats-bar {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    gap: 24px;
    font-size: 0.8rem;
    color: #8b949e;
    margin: 10px 0;
}
.stat-item strong { color: #58a6ff; }

/* ── Inputs ── */
.stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px #58a6ff22 !important;
}
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton button:hover { transform: translateY(-1px); }

/* ── Sidebar sections ── */
.sidebar-section {
    background: #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 10px 0;
}
.sidebar-section h4 { color: #58a6ff; margin: 0 0 8px; font-size: 0.85rem; }

/* ── Benchmark card ── */
.bench-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}
.bench-pass { color: #3fb950; font-weight: 600; }
.bench-fail { color: #f85149; font-weight: 600; }

/* ── Scrollable chat area ── */
.chat-container { max-height: 65vh; overflow-y: auto; padding-right: 4px; }

/* ── Thinking indicator ── */
@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
.thinking { animation: pulse 1.2s ease-in-out infinite; color: #58a6ff; font-size: 0.9rem; }

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    # Streamlit passes args after '--'
    try:
        idx = sys.argv.index("--")
        raw = sys.argv[idx+1:]
    except ValueError:
        raw = []
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/mnt/vlsi/embedded_tutor/experiments/llama3_tutor_v1/final_model")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", default="cuda")
    return p.parse_args(raw)

args = get_args()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached — only loads once)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are CircuitMentor, an expert embedded systems tutor for junior ECE students. "
    "You specialize in Arduino, ESP32, Raspberry Pi Pico, ATmega, STM32, MicroPython, "
    "Embedded C, C++, IoT protocols (MQTT, I2C, SPI, UART), and FreeRTOS. "
    "Always explain concepts clearly, provide working code with well-placed comments, "
    "and highlight common pitfalls. Format code in properly labelled code blocks."
)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pathlib import Path

    adapter_path = Path(model_path).parent / "final_lora_adapter"
    use_adapter  = adapter_path.exists() and not (Path(model_path) / "config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


def generate_response(
    tokenizer, model, user_message: str,
    history: list, temperature: float, max_tokens: int
) -> str:
    import torch

    # Build full conversation with Llama-3 chat template
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:   # Last 6 turns for context window safety
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": user_message})

    # Apply official Llama-3 chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0.01,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_message(role: str, content: str):
    """Render a chat message with proper code block formatting."""
    if role == "user":
        st.markdown(f"""
        <div class="user-msg">
            <div class="msg-label user-label">👤 You</div>
            {content}
        </div>""", unsafe_allow_html=True)
    else:
        # Convert markdown code blocks to styled HTML
        def code_replacer(match):
            lang = match.group(1) or "text"
            code = match.group(2).replace("<", "&lt;").replace(">", "&gt;")
            return f'<pre><code class="language-{lang}">{code}</code></pre>'

        html_content = re.sub(
            r"```(\w*)\n?(.*?)```",
            code_replacer,
            content,
            flags=re.DOTALL
        )
        # Bold **text**
        html_content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_content)
        # Inline `code`
        html_content = re.sub(r"`([^`]+)`", r'<code style="background:#21262d;padding:2px 6px;border-radius:4px">\1</code>', html_content)
        # Newlines
        html_content = html_content.replace("\n", "<br>")

        st.markdown(f"""
        <div class="bot-msg">
            <div class="msg-label bot-label">⚡ CircuitMentor</div>
            {html_content}
        </div>""", unsafe_allow_html=True)


def export_chat(history: list) -> str:
    """Export chat history as formatted markdown."""
    lines = [f"# CircuitMentor Chat Export\n*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n"]
    for i, turn in enumerate(history, 1):
        lines.append(f"### Question {i}\n{turn['user']}\n")
        lines.append(f"### Answer\n{turn['bot']}\n\n---\n")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE PROMPTS — shown as quick-start buttons
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = {
    "🔌 I2C Basics":        "Explain how I2C communication works and show me Arduino code to read from a sensor at address 0x48.",
    "📡 ESP32 + MQTT":      "How do I connect an ESP32 to WiFi and publish temperature data to an MQTT broker using MicroPython?",
    "⚡ Interrupts":        "Explain external interrupts on Arduino and write code to toggle an LED on a button press without polling.",
    "🔄 FreeRTOS Tasks":    "Create a FreeRTOS example on ESP32 with two tasks: one blinks an LED every 500ms, another logs a heartbeat every 2 seconds.",
    "🌡️ DS18B20 Sensor":   "Write MicroPython code for a Raspberry Pi Pico to read temperature from a DS18B20 sensor.",
    "🐛 Debug Serial":      "My Arduino Serial Monitor shows garbage characters. What are the possible causes and how do I fix each one?",
    "⚙️ PWM Servo":         "Explain PWM and write Arduino code to control a servo motor to sweep from 0° to 180°.",
    "🏃 Watchdog Timer":    "What causes an ESP32 watchdog timer reset? Show me how to prevent it during a long computation loop.",
}

TOPICS = [
    "Arduino", "ESP32", "Raspberry Pi Pico", "ATmega / AVR",
    "STM32", "MicroPython", "Embedded C", "FreeRTOS",
    "I2C", "SPI", "UART", "MQTT / IoT", "PWM", "Interrupts", "ADC/DAC",
]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "history"        not in st.session_state: st.session_state.history        = []
if "total_tokens"   not in st.session_state: st.session_state.total_tokens   = 0
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None
if "model_loaded"   not in st.session_state: st.session_state.model_loaded   = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 16px">
        <div style="font-size:2.5rem">⚡</div>
        <div style="font-size:1.1rem;font-weight:700;color:#58a6ff">CircuitMentor</div>
        <div style="font-size:0.75rem;color:#8b949e">Embedded Systems AI Tutor</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model Settings ────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                            help="Lower = more focused. Higher = more creative.")
    max_tokens  = st.slider("Max Tokens", 128, 1024, 512, 64)

    st.divider()

    # ── Quick Topics ──────────────────────────────────────────────────────────
    st.markdown("#### 📚 Topics Covered")
    chips_html = "".join(f'<span class="topic-chip">{t}</span>' for t in TOPICS)
    st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Questions", len(st.session_state.history))
    col2.metric("Tokens Out", st.session_state.total_tokens)

    st.divider()

    # ── Actions ───────────────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_tokens = 0
        st.rerun()

    if st.session_state.history:
        export_md = export_chat(st.session_state.history)
        st.download_button(
            "💾 Export Chat (.md)",
            export_md,
            file_name=f"circuitmentor_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    # ── Model Info ────────────────────────────────────────────────────────────
    st.markdown("#### ℹ️ Model Info")
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#8b949e;line-height:1.8">
    <b style="color:#e6edf3">Base:</b> Llama-3-8B-Instruct<br>
    <b style="color:#e6edf3">Fine-tune:</b> LoRA (r=16, α=32)<br>
    <b style="color:#e6edf3">Data:</b> 15k embedded examples<br>
    <b style="color:#e6edf3">Precision:</b> bf16<br>
    <b style="color:#e6edf3">Hardware:</b> Jetson AGX Orin 64GB<br>
    <b style="color:#e6edf3">Path:</b> <code style="font-size:0.7rem">{args.model_path[-40:]}</code>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="cm-header">
    <div style="font-size:2rem">⚡</div>
    <div>
        <h1>CircuitMentor<span class="cm-badge">Llama-3 Fine-Tuned</span></h1>
        <p>Edge-native AI tutor for Embedded Systems · Running on NVIDIA Jetson AGX Orin</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
if not st.session_state.model_loaded:
    with st.spinner("⚡ Loading CircuitMentor model onto Jetson GPU..."):
        try:
            tokenizer, model = load_model(args.model_path)
            st.session_state.model_loaded = True
            st.success(f"✅ Model loaded from `{args.model_path}`")
        except Exception as e:
            st.error(f"❌ Model load failed: {e}")
            st.info("Make sure the model path is correct and the venv is activated.")
            st.stop()
else:
    tokenizer, model = load_model(args.model_path)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_examples, tab_about = st.tabs(["💬 Chat", "🚀 Examples", "📋 About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB: CHAT
# ─────────────────────────────────────────────────────────────────────────────

with tab_chat:
    # Chat history display
    if st.session_state.history:
        chat_container = st.container()
        with chat_container:
            for turn in st.session_state.history:
                render_message("user", turn["user"])
                render_message("bot",  turn["bot"])
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#8b949e">
            <div style="font-size:3rem;margin-bottom:12px">🔌</div>
            <div style="font-size:1.1rem;font-weight:600;color:#e6edf3">Ask me anything about embedded systems</div>
            <div style="font-size:0.85rem;margin-top:8px">
                Arduino · ESP32 · MicroPython · I2C · SPI · UART · FreeRTOS · and more
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Input area
    user_input = st.text_area(
        "Your question",
        value=st.session_state.pending_prompt or "",
        placeholder="e.g. How do I read a DHT22 sensor with ESP32 using MicroPython?",
        height=90,
        label_visibility="collapsed",
        key="user_input",
    )

    col_send, col_clear = st.columns([5, 1])

    with col_send:
        send_clicked = st.button("⚡ Ask CircuitMentor", type="primary", use_container_width=True)

    with col_clear:
        if st.button("↺", help="Clear input", use_container_width=True):
            st.session_state.pending_prompt = None
            st.rerun()

    # Handle submission
    if send_clicked and user_input.strip():
        st.session_state.pending_prompt = None
        with st.spinner("🤔 CircuitMentor is thinking..."):
            try:
                response = generate_response(
                    tokenizer, model,
                    user_input.strip(),
                    st.session_state.history,
                    temperature, max_tokens,
                )
                st.session_state.history.append({
                    "user": user_input.strip(),
                    "bot":  response,
                    "time": datetime.now().strftime("%H:%M"),
                })
                st.session_state.total_tokens += len(response.split())
            except Exception as e:
                st.error(f"Generation error: {e}")
        st.rerun()

    elif send_clicked and not user_input.strip():
        st.warning("Please type a question first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB: EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

with tab_examples:
    st.markdown("### 🚀 Try These Examples")
    st.markdown("Click any example to instantly load it into the chat.")

    cols = st.columns(2)
    for i, (label, prompt) in enumerate(EXAMPLE_PROMPTS.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div class="bench-card">
                    <div style="font-weight:600;color:#e6edf3;margin-bottom:6px">{label}</div>
                    <div style="font-size:0.82rem;color:#8b949e;line-height:1.5">{prompt[:120]}...</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Try →", key=f"ex_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

with tab_about:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
**CircuitMentor** is an edge-native AI teaching assistant fine-tuned specifically
for junior ECE students learning embedded systems.

**What makes it different from ChatGPT:**
- Runs entirely **on-device** on an NVIDIA Jetson AGX Orin (no cloud, no API calls)
- Fine-tuned on **15,000 embedded-specific** instruction pairs
- Specialized in embedded C, C++, MicroPython, IoT protocols
- Designed to **explain, not just answer** — with working code and pitfall warnings
        """)

        st.markdown("### 📦 Tech Stack")
        st.markdown("""
| Layer | Technology |
|---|---|
| Base Model | Meta Llama-3-8B-Instruct |
| Fine-Tuning | LoRA (r=16, α=32) via PEFT |
| Precision | bf16 (Jetson CUDA 12.6) |
| Training | HuggingFace TRL SFTTrainer |
| Dataset | 15k embedded instruction pairs |
| UI | Streamlit |
| Hardware | Jetson AGX Orin 64GB |
        """)

    with col_b:
        st.markdown("### 📊 Training Details")
        st.markdown("""
| Parameter | Value |
|---|---|
| Dataset size | 15,000 examples |
| Code coverage | 81% with code blocks |
| Epochs | 2 |
| Batch size | 1 (+ grad_accum 16) |
| Learning rate | 2e-4 (cosine decay) |
| NEFTune noise | α = 5.0 |
| Training time | ~10–12 hours |
        """)

        st.markdown("### 🔧 Supported Topics")
        chips_html = "".join(
            f'<span class="topic-chip" style="margin:3px">{t}</span>' for t in TOPICS
        )
        st.markdown(f'<div style="line-height:2.2">{chips_html}</div>', unsafe_allow_html=True)

        st.markdown("### ⚠️ Out of Scope")
        st.markdown("""
- Deep RTOS kernel internals
- Advanced DSP mathematics
- PCB trace routing / signal integrity
- FPGA / Verilog / VHDL
        """)

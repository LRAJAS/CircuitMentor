#!/bin/bash
# CircuitMentor — Start Script
# Usage: bash scripts/start.sh

set -e

CIRCUITMENTOR_ROOT="/mnt/circuitmentor"
VENV="/mnt/vlsi/vlsi_env"
MODEL_PATH="$CIRCUITMENTOR_ROOT/experiments/llama3_tutor_v1/final_model"
LOG="$CIRCUITMENTOR_ROOT/logs/launch.log"
PID_FILE="$CIRCUITMENTOR_ROOT/circuitmentor.pid"

echo "⚡ Starting CircuitMentor..."

# Kill any existing processes
pkill -f "launch.py"  2>/dev/null || true
pkill -f "streamlit"  2>/dev/null || true
pkill -f "uvicorn"    2>/dev/null || true
pkill -f "ngrok"      2>/dev/null || true
sleep 3

# Activate venv
source "$VENV/bin/activate"

# Launch in background
mkdir -p "$CIRCUITMENTOR_ROOT/logs"
nohup python "$CIRCUITMENTOR_ROOT/app/launch.py" \
  --model-path "$MODEL_PATH" \
  --memory-cap-gb 50 \
  > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "  PID: $!"
echo "  Waiting 40 seconds for model to load..."
sleep 40

echo ""
echo "  CircuitMentor is LIVE"
cat "$CIRCUITMENTOR_ROOT/public_url.txt" 2>/dev/null || echo "  Check logs: tail -f $LOG"

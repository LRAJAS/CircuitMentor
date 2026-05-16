#!/bin/bash
# CircuitMentor — Stop Script
# Usage: bash scripts/stop.sh

echo "Stopping CircuitMentor..."

pkill -f "launch.py"  2>/dev/null || true
pkill -f "streamlit"  2>/dev/null || true
pkill -f "uvicorn"    2>/dev/null || true
pkill -f "ngrok"      2>/dev/null || true

rm -f /mnt/circuitmentor/circuitmentor.pid 2>/dev/null || true

echo "CircuitMentor stopped."

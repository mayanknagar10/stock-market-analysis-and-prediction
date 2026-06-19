#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "📈 StockPro Analytics"
if ! command -v python3 &>/dev/null; then echo "❌ Python 3.9+ required"; exit 1; fi
if [ ! -d "venv" ]; then echo "📦 Creating venv…"; python3 -m venv venv; fi
source venv/bin/activate
echo "📦 Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "🚀 Launching at http://localhost:8501"
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

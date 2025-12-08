#!/bin/bash
echo "🚀 Starting Sultan AI v12.7 (Live Edition)"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "📥 Fetching datasets..."
python backend/fetch_data.py
streamlit run frontend/Home.py

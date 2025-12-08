@echo off
echo 🚀 Starting Sultan AI v12.7 (Live Edition)
if not exist ".venv" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo 📥 Fetching datasets...
python backend/fetch_data.py
streamlit run frontend/Home.py





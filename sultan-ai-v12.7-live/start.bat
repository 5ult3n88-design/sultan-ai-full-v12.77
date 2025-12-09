@echo off
echo ========================================
echo    Sultan AI Trading Dashboard v12.7
echo ========================================
echo.

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Installing dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo.
echo Fetching latest market data...
python backend/fetch_data.py

echo.
echo Starting dashboard...
echo Opening browser at http://localhost:8501
start http://localhost:8501
cd frontend
streamlit run Home.py --server.headless true
pause

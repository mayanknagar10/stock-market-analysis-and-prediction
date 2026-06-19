@echo off
title StockPro Analytics
cd /d "%~dp0"
echo 📈 StockPro Analytics
python --version >nul 2>&1 || (echo ❌ Python required && pause && exit /b 1)
if not exist "venv\" python -m venv venv
call venv\Scripts\activate.bat
echo 📦 Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo 🚀 Launching at http://localhost:8501
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause

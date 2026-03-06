@echo off
echo ============================================================
echo  FINAL-MVP-CATTLE - Environment Setup
echo ============================================================
echo.

REM Check Python 3.11
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.11 is not installed!
    echo Please install Python 3.11.x from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment with Python 3.11...
py -3.11 -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: CUDA 12.4 PyTorch failed, trying CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo [4/4] Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo  Setup Complete! Verifying...
echo ============================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
python -c "import timm; import ultralytics; import streamlit; print('All packages OK!')"

echo.
echo ============================================================
echo  To run the app:
echo    1. venv\Scripts\activate.bat
echo    2. streamlit run app.py
echo ============================================================
pause

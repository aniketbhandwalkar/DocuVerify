@echo off
echo ===================================================
echo   DocumentVerify: One-Command Setup (Windows)
echo ===================================================

echo [1/4] Installing Root Dependencies...
npm install

echo.
echo [2/4] Installing Server Dependencies...
cd server
npm install

echo.
echo [3/4] Installing Client Dependencies...
cd ../client
npm install

echo.
echo [4/4] Setting up AI-ML Service (Python)...
cd ../ai-ml-service
echo Creating Virtual Environment...
python -m venv venv
call .\venv\Scripts\activate
echo Installing Python requirements (this may take a few minutes)...
pip install -r requirements.txt

echo.
echo ===================================================
echo   SETUP COMPLETE!
echo   Use 'run_all.bat' to start the system.
echo ===================================================
pause

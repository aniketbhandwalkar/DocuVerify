@echo off
echo ===================================================
echo   DocumentVerify: Starting All Services
echo ===================================================

echo Starting AI-ML Service in new window...
start "AI-ML Service" cmd /k "cd ai-ml-service && .\venv\Scripts\activate && python app.py"

echo Starting Backend Server in new window...
start "Backend Server" cmd /k "cd server && npm run dev"

echo Starting Frontend Client in new window...
start "Frontend Client" cmd /k "cd client && npm start"

echo.
echo All services are launching...
echo AI-ML: http://localhost:8000
echo Server: http://localhost:5000
echo Client: http://localhost:3000
echo.
echo Keep these windows open while using the application.
echo ===================================================

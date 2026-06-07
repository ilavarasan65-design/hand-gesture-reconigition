@echo off
rem Ensure you have Node.js installed and added to PATH.
cd /d "%~dp0"
echo Installing npm dependencies...
npm install
if errorlevel 1 (
  echo npm install failed. Ensure Node.js is installed and retry.
  pause
  exit /b 1
)
echo Starting the server...
npm start
pause

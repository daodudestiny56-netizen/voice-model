@echo off
cd /d "%~dp0"
echo Starting Voice Clone Studio... (opens http://127.0.0.1:7860)
".venv\Scripts\python.exe" app.py
pause

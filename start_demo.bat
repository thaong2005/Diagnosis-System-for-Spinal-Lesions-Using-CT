@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow docs\guides\windows_integration.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import PySide6, matplotlib, numpy, pydicom, SimpleITK"
if errorlevel 1 (
  echo Missing application dependencies. Run pip install -r requirements.txt in .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m apps.demo
if errorlevel 1 pause

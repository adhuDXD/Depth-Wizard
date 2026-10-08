@echo off
rem DepthWizard web app. Double-click to run.
rem First run installs everything (about 5 minutes); later runs start in seconds.
rem Extra arguments go to the server, e.g. run_local.bat --host 0.0.0.0 to serve the whole LAN.
cd /d "%~dp0"
title DepthWizard

where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed or not on PATH.
  echo Install Python 3.11 or newer from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" during setup, then run this file again.
  pause
  exit /b 1
)

fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul
if errorlevel 1 del ".venv\installed.ok" 2>nul

if not exist ".venv\installed.ok" (
  echo [1/3] Creating the Python environment...
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
  if errorlevel 1 goto fail
  echo [2/3] Installing packages. This takes a few minutes the first time...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  echo ok> ".venv\installed.ok"
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

if not exist "models\depth_anything_v2_vits.onnx" (
  echo [3/3] Downloading the AI depth model, about 99 MB...
  ".venv\Scripts\python.exe" scripts\download_model.py
  if errorlevel 1 echo Model download failed: the app will run with the non-AI fallback.
)

if not exist "data\geoid\us_nga_egm08_25.tif" (
  echo Downloading the EGM2008 geoid grid, about 81 MB, for CartoDEM datum conversion...
  ".venv\Scripts\python.exe" scripts\fetch_geoid.py
)

echo.
echo Starting DepthWizard at http://localhost:8000  -  keep this window open, close it to stop.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m depthwizard.server %*
pause
exit /b 0

:fail
echo.
echo Setup failed. Copy the messages above and ask for help.
pause
exit /b 1

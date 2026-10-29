@echo off
setlocal EnableExtensions
set "PROJECT=%~dp0.."
set "PY=%PROJECT%\.venv\Scripts\python.exe"
title DrugShift-UQ - Strengthening Existing Evidence (Resume Safe)
echo.
echo ============================================================
echo   DrugShift-UQ - ONE-CLICK INCREMENTAL STRENGTHENING
echo ============================================================
echo Original 22 jobs will NOT be retrained.
echo Uses the existing Python environment and downloaded GDSC2.
echo Works offline. If interrupted, double-click THIS same BAT again.
echo.
if not exist "%PY%" (
  echo ERROR: Existing virtualenv not found: "%PY%"
  echo Extract this entire folder INSIDE your original DrugShift-UQ_OneClick folder.
  pause
  exit /b 1
)
if not exist "%PROJECT%\results\LATEST_RUN.txt" (
  echo ERROR: Previous completed run not found. Do not install over a different project.
  pause
  exit /b 1
)
set "PYTHONPATH=%PROJECT%\src;%PYTHONPATH%"
cd /d "%PROJECT%"
"%PY%" -u "%~dp0strengthen.py" --project "%PROJECT%" --mode all
if errorlevel 1 (
  echo.
  echo The strengthening process stopped with an error.
  echo All existing jobs and successfully completed NEW jobs are preserved.
  echo Rerun this same BAT to resume. Do not run START_RESEARCH.bat.
  pause
  exit /b 1
)
for /f "usebackq delims=" %%I in ("%PROJECT%\results\LATEST_RUN.txt") do set "RID=%%I"
start "" "%PROJECT%\results\STRENGTHENED\%RID%\STRENGTHENED_REPORT.html"
echo.
echo DONE. Original experiments unchanged. Supplementary evidence completed.
pause

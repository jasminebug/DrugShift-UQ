@echo off
setlocal
cd /d "%~dp0"
if exist "results\LATEST_RUN.txt" (
 rem If final report already exists, do not need an automatic rerun.
 if exist "results\FINAL_REPORT.html" exit /b 0
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap.ps1" >> "%~dp0logs\autoresume_console.log" 2>&1

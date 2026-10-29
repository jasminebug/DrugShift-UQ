@echo off
setlocal
cd /d "%~dp0"
title DrugShift-UQ - One Click Research Pipeline
if not exist "logs" mkdir "logs"
echo.
echo  ============================================================
echo   DrugShift-UQ ^| Research pipeline - safe to resume
 echo  ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap.ps1"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
 echo Research pipeline completed. Open results\FINAL_REPORT.html
 if exist "results\FINAL_REPORT.html" start "" "%~dp0results\FINAL_REPORT.html"
) else (
 echo Pipeline ended with error %RC%. Read logs\pipeline.log and rerun this BAT.
 echo Completed jobs and epoch checkpoints will be reused.
)
echo.
pause
exit /b %RC%

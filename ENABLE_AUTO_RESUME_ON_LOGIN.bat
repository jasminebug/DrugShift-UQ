@echo off
setlocal
cd /d "%~dp0"
echo This optional step adds a startup shortcut for DrugShift-UQ under YOUR Windows account.
echo The project will resume on your next Windows sign-in, without administrator rights.
choice /M "Create login auto-resume shortcut"
if errorlevel 2 exit /b 0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\enable_autoresume.ps1"
echo You can disable this at any time with DISABLE_AUTO_RESUME.bat
pause

@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$link=Join-Path ([Environment]::GetFolderPath('Startup')) 'DrugShift-UQ Resume.lnk'; if(Test-Path $link){Remove-Item $link -Force; Write-Host 'Auto resume disabled.'}else{Write-Host 'Auto resume was not enabled.'}"
pause

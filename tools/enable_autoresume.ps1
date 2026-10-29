$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$startup=[Environment]::GetFolderPath('Startup')
$shortcut=Join-Path $startup 'DrugShift-UQ Resume.lnk'
$sh=New-Object -ComObject WScript.Shell
$link=$sh.CreateShortcut($shortcut)
$link.TargetPath=Join-Path $root 'AUTO_RESUME_QUIET.bat'
$link.WorkingDirectory=$root
$link.Description='Resume unfinished DrugShift-UQ pipeline after user signs in'
$link.WindowStyle=7
$link.Save()
Write-Host "Startup shortcut created: $shortcut"

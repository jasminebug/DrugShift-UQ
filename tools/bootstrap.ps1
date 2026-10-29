$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$root = (Get-Location).Path
$python = $null
$venv = Join-Path $root '.venv\Scripts\python.exe'

function Test-Python([string]$file) {
  try {
    $v = & $file -c 'import sys;print(sys.version_info.major, sys.version_info.minor)' 2>$null
    if ($LASTEXITCODE -eq 0 -and ($v -eq '3 11' -or $v -eq '3 12')) { return $true }
  } catch { }
  return $false
}
function Write-Status([string]$message) {
  Write-Host ('[DrugShift-UQ] ' + $message) -ForegroundColor Cyan
}
if (Test-Path $venv) {
  if (Test-Python $venv) { $python = $venv }
}
if ($null -eq $python) {
  $localpy = Join-Path $root '.python\python.exe'
  if (Test-Path $localpy) { if (Test-Python $localpy) { $python = $localpy } }
  if ($null -eq $python) {
    foreach ($version in @('3.11','3.12')) {
      try {
        $candidate = (& py ('-' + $version) -c 'import sys;print(sys.executable)' 2>$null)
        if ($LASTEXITCODE -eq 0 -and (Test-Python $candidate)) { $python = $candidate; break }
      } catch { }
    }
  }
  if ($null -eq $python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-Python $cmd.Source)) { $python = $cmd.Source }
  }
  if ($null -eq $python) {
    Write-Status 'Python 3.11/3.12 not found; downloading official Python 3.11.9 installer.'
    $installer = Join-Path $env:TEMP 'drugshift-python-3.11.9-amd64.exe'
    if (-not (Test-Path $installer)) {
      $ProgressPreference = 'SilentlyContinue'
      Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $installer -UseBasicParsing
    }
    $sig = Get-AuthenticodeSignature -FilePath $installer
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
      Remove-Item $installer -Force
      throw 'Official Python installer signature check failed; download stopped.'
    }
    $destination = Join-Path $root '.python'
    Write-Status 'Installing signed Python locally (no administrator permission required).'
    $p = Start-Process -FilePath $installer -ArgumentList @('/quiet','InstallAllUsers=0','PrependPath=0','Include_pip=1','Include_test=0',('TargetDir=' + $destination)) -PassThru -Wait
    if ($p.ExitCode -ne 0) { throw "Python installer exit code $($p.ExitCode)" }
    $python = Join-Path $destination 'python.exe'
    if (-not (Test-Python $python)) { throw 'Automatic Python installation did not produce a working interpreter.' }
  }
  if (-not (Test-Path $venv)) {
    Write-Status 'Creating project-isolated Python virtual environment.'
    & $python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
  }
  $python = $venv
}

$installedFlag = Join-Path $root '.venv\.drugshift_packages_ready'
if (-not (Test-Path $installedFlag)) {
  Write-Status 'Installing dependencies (this needs Internet on first launch).'
  & $python -m pip install --disable-pip-version-check --upgrade pip
  if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
  # CPU wheels avoid unwanted multi-gigabyte GPU CUDA downloads; CUDA will be used when an already compatible GPU torch install exists.
  & $python -m pip install --disable-pip-version-check --index-url 'https://download.pytorch.org/whl/cpu' 'torch>=2.3,<3'
  if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
  & $python -m pip install --disable-pip-version-check -e .
  if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
  & $python -m pip check
  if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict after install.' }
  Set-Content -Path $installedFlag -Value 'installed, verified with pip check'
}
Write-Status 'Running automatic data / training / evaluation / report stages.'
& $python -m drugshift.pipeline --root "$root"
if ($LASTEXITCODE -ne 0) { throw "Research runner failed with exit code $LASTEXITCODE" }
Write-Status 'Success. Results and reports have been generated.'

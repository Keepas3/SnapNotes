# Starts the dev server with the settings saved in backend\.env. Normally launched by ..\start.ps1.
Set-Location $PSScriptRoot
foreach ($line in Get-Content (Join-Path $PSScriptRoot '.env')) {
  $i = $line.IndexOf('=')
  if ($i -gt 0 -and -not $line.StartsWith('#')) {
    [Environment]::SetEnvironmentVariable($line.Substring(0, $i).Trim(), $line.Substring($i + 1).Trim(), 'Process')
  }
}
.\.venv\Scripts\python.exe -m app.admin check
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --host 0.0.0.0

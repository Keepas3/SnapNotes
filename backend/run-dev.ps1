# Starts the dev server with the settings saved in backend\.env. Normally launched by ..\start.ps1.
Set-Location $PSScriptRoot
foreach ($line in Get-Content (Join-Path $PSScriptRoot '.env')) {
  $i = $line.IndexOf('=')
  if ($i -gt 0 -and -not $line.StartsWith('#')) {
    [Environment]::SetEnvironmentVariable($line.Substring(0, $i).Trim(), $line.Substring($i + 1).Trim(), 'Process')
  }
}
.\.venv\Scripts\python.exe -m app.admin check
# Listen on this computer only (the Android emulator reaches it as 10.0.2.2). Everyone on your Wi-Fi could otherwise try to use the server,
# and the token would travel unencrypted across it. A real phone needs the network address: set SNAPNOTES_DEV_URL in .env to use it.
$bind = if ($env:SNAPNOTES_DEV_URL) { '0.0.0.0' } else { '127.0.0.1' }
Write-Host "Listening on $bind" -ForegroundColor DarkGray
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --host $bind

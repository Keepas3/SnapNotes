<#
  One command to start SnapNotes for development: .\start.ps1

  First run: makes a token and asks which AI to use, saved in backend\.env (git ignores it).
  After that: opens the server window and the app window with the same settings every time.

    .\start.ps1 -Reset    forget the saved settings and ask again (makes a new token)
#>
param([switch]$Reset)

$root = $PSScriptRoot
$python = Join-Path $root 'backend\.venv\Scripts\python.exe'
$envFile = Join-Path $root 'backend\.env'

if (-not (Test-Path $python)) { throw "Backend is not set up: $python is missing. See backend\README.md." }
if (-not (Test-Path (Join-Path $root 'mobile\node_modules'))) { throw "Run 'npm install' in the mobile folder first." }

if ($Reset -or -not (Test-Path $envFile)) {
  $token = & $python -c "import secrets; print(secrets.token_urlsafe(32))"
  $lines = @("SNAPNOTES_TOKEN=$token")
  do {
    $key = (Read-Host 'Paste your Gemini API key, or type demo for the fake canned AI').Trim().Trim('"', "'")
  } while (-not $key)   # an empty answer asks again, so a missed paste can no longer turn on demo mode
  if ($key -eq 'demo') {
    $lines += 'SNAPNOTES_DEMO=1'
  } else {
    $lines += 'SNAPNOTES_PROVIDER=gemini'
    $lines += "GEMINI_API_KEY=$key"
  }
  [IO.File]::WriteAllLines($envFile, $lines)
  $mode = if ($key -eq 'demo') { 'DEMO (canned cell-biology cards for any upload)' } else { "Gemini (key of $($key.Length) characters)" }
  Write-Host "Saved to backend\.env (private, not committed). AI: $mode" -ForegroundColor Green
}

# The server window reads backend\.env itself, so nothing is retyped and no key appears on screen.
Start-Process powershell -WorkingDirectory (Join-Path $root 'backend') -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', 'run-dev.ps1'

# The app window inherits these, so a development build connects to this server with this token by itself.
# (--clear makes Metro rebuild so a new token is never served from its cache.) Set SNAPNOTES_DEV_URL in
# backend\.env to use a real phone's address instead of the emulator's.
$saved = (Get-Content $envFile | Where-Object { $_ -like 'SNAPNOTES_TOKEN=*' }) -replace '^SNAPNOTES_TOKEN=', ''
$devUrl = (Get-Content $envFile | Where-Object { $_ -like 'SNAPNOTES_DEV_URL=*' }) -replace '^SNAPNOTES_DEV_URL=', ''
if (-not $devUrl) { $devUrl = 'http://10.0.2.2:8000' }
$env:EXPO_PUBLIC_SNAPNOTES_DEV_TOKEN = $saved
$env:EXPO_PUBLIC_SNAPNOTES_DEV_URL = $devUrl
Start-Process powershell -WorkingDirectory (Join-Path $root 'mobile') -ArgumentList '-NoExit', '-Command', 'npx expo start --clear'
# The app window has its own copy now. Do not leave the token in this terminal, where a later "expo export" or release build run from it
# would copy it into an app you might share (Expo puts every EXPO_PUBLIC_ variable it sees into the bundle).
Remove-Item Env:EXPO_PUBLIC_SNAPNOTES_DEV_TOKEN, Env:EXPO_PUBLIC_SNAPNOTES_DEV_URL -ErrorAction SilentlyContinue

Write-Host ''
Write-Host 'Two windows opened: the server and the app. When the app window is ready, press a for the emulator.'
Write-Host "The app fills in the server ($devUrl) and the token itself; nothing to type in Settings."

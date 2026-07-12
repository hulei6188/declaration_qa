@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo [1/4] Stopping API, worker, and frontend...
call "%~dp0stop-app.bat"

echo [2/4] Cleaning stale listeners on ports 8010 and 5173...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "$ports = @(8010, 5173);" ^
  "$processes = @(Get-CimInstance Win32_Process);" ^
  "$roots = @($processes | Where-Object {" ^
  "  ($_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'start-app\.bat.? (api|worker|frontend)') -or" ^
  "  ($_.Name -match '^python(.exe)?$' -and $_.CommandLine -match 'declaration_qa\.api\.server\.app:app|-m declaration_qa\.indexing\.worker') -or" ^
  "  ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'declaration_qa[\\/]frontend.*vite')" ^
  "});" ^
  "$queue = @($roots | ForEach-Object { [int]$_.ProcessId });" ^
  "$targets = @{};" ^
  "while ($queue.Count -gt 0) {" ^
  "  $currentPid = [int]$queue[0];" ^
  "  $queue = @($queue | Select-Object -Skip 1);" ^
  "  if ($targets.ContainsKey($currentPid)) { continue };" ^
  "  $targets[$currentPid] = $true;" ^
  "  $queue += @($processes | Where-Object { [int]$_.ParentProcessId -eq $currentPid } | ForEach-Object { [int]$_.ProcessId });" ^
  "};" ^
  "@($targets.Keys) | Sort-Object -Descending | ForEach-Object { Stop-Process -Id $_ -Force };" ^
  "Start-Sleep -Seconds 1;" ^
  "$listeners = @(Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in $ports });" ^
  "foreach ($listener in $listeners) {" ^
  "  $ownerPid = [int]$listener.OwningProcess;" ^
  "  $children = @($processes | Where-Object { [int]$_.ParentProcessId -eq $ownerPid });" ^
  "  foreach ($child in $children) { Stop-Process -Id ([int]$child.ProcessId) -Force };" ^
  "  Stop-Process -Id $ownerPid -Force;" ^
  "};" ^
  "Start-Sleep -Seconds 2;" ^
  "$remaining = @(Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in $ports });" ^
  "if ($remaining) {" ^
  "  $remaining | ForEach-Object { Write-Host ('Port {0} is still occupied by PID {1}.' -f $_.LocalPort, $_.OwningProcess) };" ^
  "  exit 2;" ^
  "}"

if errorlevel 1 (
    echo Failed to release application ports. Close the reported process and retry.
    exit /b 1
)

echo [3/4] Starting application...
set "NO_PROXY=127.0.0.1,localhost,127.0.0.0/8"
call "%~dp0start-app.bat"
if errorlevel 1 exit /b %errorlevel%

echo [4/4] Waiting for API health check...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ready = $false;" ^
  "for ($i = 0; $i -lt 30; $i++) {" ^
  "  try {" ^
  "    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/api/declaration-qa/health' -TimeoutSec 2;" ^
  "    if ($health.ok) { $ready = $true; break }" ^
  "  } catch {}" ^
  "  Start-Sleep -Seconds 1;" ^
  "};" ^
  "if (-not $ready) { Write-Host 'API health check timed out.'; exit 3 };" ^
  "Write-Host 'API is healthy: http://127.0.0.1:8010';" ^
  "Write-Host 'Frontend: http://127.0.0.1:5173'"

exit /b %errorlevel%

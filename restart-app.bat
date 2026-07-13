@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%\"
set "CONDA_ENV=langchain-agent"
set "MINERU_MODEL_SOURCE=modelscope"

if /i "%~1"=="api" goto run_api
if /i "%~1"=="worker" goto run_worker
if /i "%~1"=="frontend" goto run_frontend

echo [1/4] Stopping API, worker, and frontend...
call "%~dp0stop-app.bat"

echo [2/4] Cleaning stale listeners on ports 8010 and 5173...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "$ports = @(8010, 5173);" ^
  "$processes = @(Get-CimInstance Win32_Process);" ^
  "$roots = @($processes | Where-Object {" ^
  "  ($_.Name -eq 'cmd.exe' -and $_.CommandLine -match '(start-app|restart-app)\.bat.? (api|worker|frontend)') -or" ^
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
call :activate_conda_env
if errorlevel 1 exit /b %errorlevel%

python -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo Python dependencies were not found in conda environment "%CONDA_ENV%".
    echo Run: conda activate %CONDA_ENV%
    echo Then: pip install -r requirements-dev.txt ^&^& pip install -e .
    exit /b 1
)
where npm >nul 2>nul || (
    echo npm was not found. Install Node.js and try again.
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example. Configure API keys before using remote models.
)

netstat -ano | findstr /R /C:":8010 .*LISTENING" >nul
if errorlevel 1 (
    start "Declaration QA API" cmd /k call "%~f0" api
) else (
    echo API is already listening on port 8010.
)

powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(.exe)?$' -and $_.CommandLine -match '-m declaration_qa.indexing.worker' }; if ($p) { exit 0 } else { exit 1 }" >nul 2>nul
if errorlevel 1 (
    start "Declaration QA Worker" cmd /k call "%~f0" worker
) else (
    echo Index worker is already running.
)

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd "frontend"
    call npm install || (popd & exit /b 1)
    popd
)

netstat -ano | findstr /R /C:":5173 .*LISTENING" >nul
if errorlevel 1 (
    start "Declaration QA Frontend" cmd /k call "%~f0" frontend
) else (
    echo Frontend is already listening on port 5173.
)

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

if errorlevel 1 exit /b %errorlevel%
start "" "http://127.0.0.1:5173"
exit /b 0

:run_api
title Declaration QA API
call :activate_conda_env
if errorlevel 1 exit /b %errorlevel%
echo Starting API with conda environment "%CONDA_ENV%" at http://127.0.0.1:8010 ...
python -m uvicorn declaration_qa.api.server.app:app --app-dir src --host 127.0.0.1 --port 8010 --reload
exit /b %errorlevel%

:run_worker
title Declaration QA Worker
call :activate_conda_env
if errorlevel 1 exit /b %errorlevel%
set "PYTHONPATH=%PROJECT_ROOT%src"
echo Starting RQ index worker with conda environment "%CONDA_ENV%" ...
python -m declaration_qa.indexing.worker
exit /b %errorlevel%

:run_frontend
title Declaration QA Frontend
call :activate_conda_env
if errorlevel 1 exit /b %errorlevel%
cd /d "%PROJECT_ROOT%frontend"
echo Starting frontend at http://127.0.0.1:5173 ...
call npm run dev
exit /b %errorlevel%

:activate_conda_env
if /i "%CONDA_DEFAULT_ENV%"=="%CONDA_ENV%" exit /b 0
where conda >nul 2>nul || (
    echo Conda was not found. Install Miniconda or Anaconda and try again.
    exit /b 1
)
call conda activate "%CONDA_ENV%" >nul 2>nul
if errorlevel 1 (
    echo Failed to activate conda environment "%CONDA_ENV%".
    echo Run: conda env list
    exit /b 1
)
exit /b 0

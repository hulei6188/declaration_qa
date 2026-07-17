@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%\"
set "CONDA_ENV=langchain-agent"
set "MINERU_MODEL_SOURCE=modelscope"
set "PYTHON_EXE="
set "CONDA_BASE="

for /f "delims=" %%I in ('conda info --base 2^>nul') do if not defined CONDA_BASE set "CONDA_BASE=%%I"
if defined CONDA_BASE if exist "%CONDA_BASE%\envs\%CONDA_ENV%\python.exe" set "PYTHON_EXE=%CONDA_BASE%\envs\%CONDA_ENV%\python.exe"

if not defined PYTHON_EXE (
    echo Conda environment "%CONDA_ENV%" was not found.
    echo Run: conda env list
    exit /b 1
)

if /i "%~1"=="api" goto run_api
if /i "%~1"=="mineru" goto run_mineru
if /i "%~1"=="worker" goto run_worker
if /i "%~1"=="eval-worker" goto run_eval_worker
if /i "%~1"=="frontend" goto run_frontend

echo [1/4] Stopping existing application processes...
echo [2/4] Cleaning listeners on ports 8010, 8011, and 5173...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "$ports = @(8010, 8011, 5173);" ^
  "$processes = @(Get-CimInstance Win32_Process);" ^
  "$roots = @($processes | Where-Object {" ^
  "  ($_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'restart-app\.bat.? (api|mineru|worker|eval-worker|frontend)') -or" ^
  "  ($_.Name -match '^python(.exe)?$' -and $_.CommandLine -match 'gov_service_agent\.api\.server\.app:app|-m gov_service_agent\.indexing\.worker|-m gov_service_agent\.application\.evals\.worker|-m mineru\.cli\.fast_api') -or" ^
  "  ($_.Name -eq 'node.exe' -and $_.CommandLine -match '(declaration-qa|gov-service-agent)[\\/]frontend.*vite')" ^
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

"%PYTHON_EXE%" -c "import chonkie, fastapi, langmem, mineru, uvicorn; from langgraph.checkpoint.postgres import PostgresSaver; from langgraph.store.postgres import PostgresStore" >nul 2>nul
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
    start "Gov Service Agent API" cmd /k call "%~f0" api
) else (
    echo API is already listening on port 8010.
)

netstat -ano | findstr /R /C:":8011 .*LISTENING" >nul
if errorlevel 1 (
    start "MinerU API" cmd /k call "%~f0" mineru
) else (
    echo MinerU API is already listening on port 8011.
)

powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(.exe)?$' -and $_.CommandLine -match '-m gov_service_agent.indexing.worker' }; if ($p) { exit 0 } else { exit 1 }" >nul 2>nul
if errorlevel 1 (
    start "Gov Service Agent Worker" cmd /k call "%~f0" worker
) else (
    echo Index worker is already running.
)

findstr /I /X "GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND=rq" .env >nul 2>nul
if not errorlevel 1 (
    powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(.exe)?$' -and $_.CommandLine -match '-m gov_service_agent.application.evals.worker' }; if ($p) { exit 0 } else { exit 1 }" >nul 2>nul
    if errorlevel 1 (
        start "Gov Service Agent Evaluation Worker" cmd /k call "%~f0" eval-worker
    ) else (
        echo Evaluation worker is already running.
    )
)

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd "frontend"
    call npm install || (popd & exit /b 1)
    popd
)

netstat -ano | findstr /R /C:":5173 .*LISTENING" >nul
if errorlevel 1 (
    start "Gov Service Agent Frontend" cmd /k call "%~f0" frontend
) else (
    echo Frontend is already listening on port 5173.
)

echo [4/4] Waiting for API health check...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ready = $false;" ^
  "$mineruReady = $false;" ^
  "for ($i = 0; $i -lt 60; $i++) {" ^
  "  try {" ^
  "    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/api/declaration-qa/health' -TimeoutSec 2;" ^
  "    if ($health.ok) { $ready = $true; break }" ^
  "  } catch {}" ^
  "  Start-Sleep -Seconds 1;" ^
  "};" ^
  "if (-not $ready) { Write-Host 'API health check timed out.'; exit 3 };" ^
  "for ($i = 0; $i -lt 60; $i++) {" ^
  "  try {" ^
  "    $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8011/docs' -UseBasicParsing -TimeoutSec 2;" ^
  "    if ($response.StatusCode -eq 200) { $mineruReady = $true; break }" ^
  "  } catch {}" ^
  "  Start-Sleep -Seconds 1;" ^
  "};" ^
  "if (-not $mineruReady) { Write-Host 'MinerU API health check timed out.'; exit 4 };" ^
  "Write-Host 'API is healthy: http://127.0.0.1:8010';" ^
  "Write-Host 'MinerU API is healthy: http://127.0.0.1:8011';" ^
  "Write-Host 'Frontend: http://127.0.0.1:5173'"

if errorlevel 1 exit /b %errorlevel%
start "" "http://127.0.0.1:5173"
exit /b 0

:run_api
title Gov Service Agent API
echo Starting API with conda environment "%CONDA_ENV%" at http://127.0.0.1:8010 ...
"%PYTHON_EXE%" -m uvicorn gov_service_agent.api.server.app:app --app-dir src --host 127.0.0.1 --port 8010 --reload
exit /b %errorlevel%

:run_mineru
title MinerU API
echo Starting MinerU API with conda environment "%CONDA_ENV%" at http://127.0.0.1:8011 ...
"%PYTHON_EXE%" -m mineru.cli.fast_api --host 127.0.0.1 --port 8011
exit /b %errorlevel%

:run_worker
title Gov Service Agent Worker
set "PYTHONPATH=%PROJECT_ROOT%src"
echo Starting RQ index worker with conda environment "%CONDA_ENV%" ...
"%PYTHON_EXE%" -m gov_service_agent.indexing.worker
exit /b %errorlevel%

:run_frontend
title Gov Service Agent Frontend
cd /d "%PROJECT_ROOT%frontend"
echo Starting frontend at http://127.0.0.1:5173 ...
call npm run dev
exit /b %errorlevel%

:run_eval_worker
title Gov Service Agent Evaluation Worker
set "PYTHONPATH=%PROJECT_ROOT%src"
echo Starting RQ evaluation worker with conda environment "%CONDA_ENV%" ...
"%PYTHON_EXE%" -m gov_service_agent.application.evals.worker
exit /b %errorlevel%

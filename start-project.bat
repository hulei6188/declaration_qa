@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%\"
set "PYTHON_EXE="
set "CONDA_BASE="

for /f "delims=" %%I in ('conda info --base 2^>nul') do if not defined CONDA_BASE set "CONDA_BASE=%%I"
if defined CONDA_BASE if exist "%CONDA_BASE%\python.exe" (
    "%CONDA_BASE%\python.exe" -c "import fastapi, uvicorn" >nul 2>nul
    if !errorlevel! equ 0 set "PYTHON_EXE=%CONDA_BASE%\python.exe"
)

if /i "%~1"=="api" goto run_api
if /i "%~1"=="worker" goto run_worker
if /i "%~1"=="frontend" goto run_frontend

echo [1/4] Checking prerequisites...
if not defined PYTHON_EXE (
    echo Conda base was not found or its project dependencies are missing.
    echo Run: conda activate base
    echo Then: pip install -r requirements-dev.txt ^&^& pip install -e .
    exit /b 1
)
where docker >nul 2>nul || (
    echo Docker was not found. Install or start Docker Desktop and try again.
    exit /b 1
)
where npm >nul 2>nul || (
    echo npm was not found. Install Node.js and try again.
    exit /b 1
)
docker info >nul 2>nul || (
    echo Docker Desktop is not running.
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example. Configure API keys before using remote models.
)

echo [2/4] Starting Docker infrastructure...
docker compose up -d || exit /b 1

echo [3/4] Starting application processes...
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

echo [4/4] Startup commands issued.
echo Frontend: http://127.0.0.1:5173
echo API docs: http://127.0.0.1:8010/docs
echo Kibana:   http://127.0.0.1:5601
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5173"
exit /b 0

:run_api
title Declaration QA API
echo Starting API at http://127.0.0.1:8010 ...
"%PYTHON_EXE%" -m uvicorn declaration_qa.api.server.app:app --app-dir src --host 127.0.0.1 --port 8010 --reload
exit /b %errorlevel%

:run_worker
title Declaration QA Worker
set "PYTHONPATH=%PROJECT_ROOT%src"
echo Starting RQ index worker ...
"%PYTHON_EXE%" -m declaration_qa.indexing.worker
exit /b %errorlevel%

:run_frontend
title Declaration QA Frontend
cd /d "%PROJECT_ROOT%frontend"
echo Starting frontend at http://127.0.0.1:5173 ...
call npm run dev
exit /b %errorlevel%

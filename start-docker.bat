@echo off
setlocal EnableExtensions

cd /d "%~dp0"

where docker >nul 2>nul || (
    echo Docker was not found. Install or start Docker Desktop and try again.
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

echo Starting Docker infrastructure...
docker compose up -d
if errorlevel 1 (
    echo Failed to start Docker infrastructure.
    exit /b 1
)

echo Docker infrastructure started.
exit /b 0

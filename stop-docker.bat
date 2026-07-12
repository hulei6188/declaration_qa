@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo Stopping Docker infrastructure...
docker compose stop
if errorlevel 1 (
    echo Failed to stop Docker infrastructure.
    exit /b 1
)

echo Docker containers stopped. Volumes and indexed data were preserved.
exit /b 0

@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo [1/2] Stopping API, worker, and frontend...
powershell -NoProfile -Command "$targets = Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^python(.exe)?$' -and $_.CommandLine -match 'declaration_qa.api.server.app:app|declaration_qa.indexing.worker') -or ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'declaration_qa[\\/]frontend.*vite') }; $targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul

taskkill /FI "WINDOWTITLE eq Declaration QA API*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq Declaration QA Worker*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq Declaration QA Frontend*" /T /F >nul 2>nul

echo [2/2] Stopping Docker infrastructure...
docker compose stop

echo Project stopped. Docker volumes and indexed data were preserved.
exit /b 0

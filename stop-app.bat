@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo Stopping API, MinerU, worker, and frontend...
powershell -NoProfile -Command "$targets = Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^python(.exe)?$' -and $_.CommandLine -match 'declaration_qa.api.server.app:app|declaration_qa.indexing.worker|mineru.cli.fast_api') -or ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'declaration_qa[\\/]frontend.*vite') }; $targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul

taskkill /FI "WINDOWTITLE eq Declaration QA API*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq MinerU API*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq Declaration QA Worker*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq Declaration QA Frontend*" /T /F >nul 2>nul

echo Application processes stopped. Docker containers are still running.
exit /b 0

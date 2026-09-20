@echo off
rem Anvira Runtime launcher for a source checkout (Windows). Run from this folder:
rem     anvira                 opens the live dashboard (starts the runtime if it is not running)
rem     anvira status | doctor | model list | orcha run "..." | --help
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set "PYTHONPATH=%ROOT%runtime;%ROOT%sdk\python;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>&1
"%PY%" -m anvira_runtime %*
exit /b %ERRORLEVEL%

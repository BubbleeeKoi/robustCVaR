@echo off
setlocal
cd /d "%~dp0..\.."
set PY=C:\Users\Chen\anaconda3\envs\portfolio\python.exe
set LOG=robust_cvar_portfolio\outputs\v3_run_log.txt
echo [%date% %time%] V3 start >> "%LOG%"
"%PY%" -u robust_cvar_portfolio\experiments\run_v3_experiment.py >> "%LOG%" 2>&1
echo [%date% %time%] V3 exit %ERRORLEVEL% >> "%LOG%"

@echo off
REM ============================================================================
REM  run-api.bat  —  Start the ImmunoSense API cleanly.
REM  Run from project root in an OPEN cmd window:  run-api.bat
REM ============================================================================
cd /d C:\Projects\immunosense

if not exist env.local.bat (
  echo [ERROR] env.local.bat not found in C:\Projects\immunosense
  echo Copy it here and fill in your SUPABASE_SERVICE_ROLE_KEY.
  pause
  goto :eof
)

REM --- load secrets/config ---
call env.local.bat

REM --- check the key was filled in (robust: compare to placeholder text) ---
if "%SUPABASE_SERVICE_ROLE_KEY%"=="<paste-your-service-role-key-here>" (
  echo [ERROR] SUPABASE_SERVICE_ROLE_KEY is still the placeholder.
  echo Edit env.local.bat and paste your real service_role key, then retry.
  pause
  goto :eof
)
if "%SUPABASE_SERVICE_ROLE_KEY%"=="" (
  echo [ERROR] SUPABASE_SERVICE_ROLE_KEY is empty. Check env.local.bat.
  pause
  goto :eof
)

REM --- free port 8000 if a process is listening on it ---
echo Checking port 8000...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000"') do (
  echo Killing stale PID %%P on port 8000
  taskkill /PID %%P /F >nul 2>&1
)

echo.
echo Starting API on http://127.0.0.1:8000   (press Ctrl+C to stop)
echo.
venv\Scripts\python.exe -m uvicorn server.api.app:app --port 8000

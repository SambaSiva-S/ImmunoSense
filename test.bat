@echo off
REM ============================================================================
REM  test.bat  —  Run the backend test suite with a CLEAN environment.
REM  Clears the ambient vars that have caused false failures (DATABASE_URL,
REM  ENABLE_DEBUG_ENDPOINT, etc.) so tests use their own sqlite + defaults.
REM  Run from project root:  test.bat            (runs everything)
REM                          test.bat server     (just server tests)
REM ============================================================================
setlocal

cd /d C:\Projects\immunosense

REM --- clear ambient vars that pollute tests ---
set "DATABASE_URL="
set "DEV_AUTH="
set "ENABLE_DEBUG_ENDPOINT="
set "CORS_ORIGINS="
set "SUPABASE_SERVICE_ROLE_KEY="
set "SUPABASE_URL="
set "DIETARY_DENSITY_CACHE="
set "DIETARY_FOOD_INDEX_CACHE="

if "%1"=="server" (
  venv\Scripts\python.exe -m pytest server\tests -q
) else if "%1"=="" (
  venv\Scripts\python.exe -m pytest server\tests tests\conductor -q
) else (
  venv\Scripts\python.exe -m pytest %1 -q
)

endlocal

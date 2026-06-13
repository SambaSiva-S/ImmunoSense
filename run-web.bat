@echo off
REM ============================================================================
REM  run-web.bat  —  Start the ImmunoSense web app dev server.
REM  Canonical web folder: C:\Projects\immunosense\web
REM  Run from project root in cmd:  run-web.bat
REM ============================================================================
set "WEBDIR=C:\Projects\immunosense\web"

if not exist "%WEBDIR%\package.json" goto :nofolder

cd /d "%WEBDIR%"

if exist .env.local goto :haveenv
if exist .env.example copy .env.example .env.local >nul
:haveenv

if exist node_modules goto :havedeps
echo Installing dependencies (first run, may take a minute)...
call npm install
:havedeps

echo Starting web dev server (Vite)...
call npm run dev
goto :eof

:nofolder
echo [ERROR] No web app at %WEBDIR% (no package.json).
echo Copy your web app there first, e.g.:
echo   robocopy "C:\temp\Photo\web" "%WEBDIR%" /E /XD node_modules dist
pause

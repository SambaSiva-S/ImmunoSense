@echo off
REM ============================================================================
REM  commit.bat  —  Safely commit + push.   Usage: commit.bat "message"
REM ============================================================================
cd /d C:\Projects\immunosense

if "%~1"=="" (
  echo Usage: commit.bat "your commit message"
  goto :eof
)

git add server web SECURITY.md 2>nul

REM --- secret scan: write staged names to a temp file, search it ---
git diff --cached --name-only > "%TEMP%\_staged.txt" 2>nul
findstr /I /C:"env.local" /C:"token.txt" /C:"node_modules" "%TEMP%\_staged.txt" >nul
if not errorlevel 1 (
  echo [ABORT] A secret or node_modules file is staged:
  findstr /I /C:"env.local" /C:"token.txt" /C:"node_modules" "%TEMP%\_staged.txt"
  echo Unstage it:  git restore --staged ^<file^>
  del "%TEMP%\_staged.txt" >nul 2>&1
  goto :eof
)
del "%TEMP%\_staged.txt" >nul 2>&1

echo Staged files:
git diff --cached --name-only
echo.
git commit -m "%~1"
git push

@echo off
REM ============================================================================
REM  apply.bat  —  Apply an extracted package into the project, then test.
REM  Usage:  apply.bat C:\path\to\extracted\package
REM ============================================================================
set "SRC=%~1"
if "%SRC%"=="" goto :usage
if not exist "%SRC%" goto :notfound

REM --- copy server\ if present anywhere under SRC ---
if exist "%SRC%\server" call :copyserver "%SRC%\server"
for /f "delims=" %%D in ('dir /s /b /ad "%SRC%\server" 2^>nul') do call :copyserver "%%D"

REM --- copy web\src if present ---
if exist "%SRC%\web\src" xcopy /E /I /Y "%SRC%\web\src" "C:\Projects\immunosense\web\src" >nul

echo.
echo Running server tests to verify...
call C:\Projects\immunosense\test.bat server
goto :eof

:copyserver
echo Copying backend from %~1
xcopy /E /I /Y "%~1" "C:\Projects\immunosense\server" >nul
goto :eof

:usage
echo Usage: apply.bat ^<path-to-extracted-package^>
echo Example: apply.bat C:\temp\Photo
goto :eof

:notfound
echo [ERROR] Folder not found: %SRC%
goto :eof

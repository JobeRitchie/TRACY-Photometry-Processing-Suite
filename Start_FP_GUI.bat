@echo off
REM Fiber Photometry Analysis GUI Launcher
REM This batch file starts the FP Analysis GUI

REM Change to the script directory (pushd also works from a network share)
pushd "%~dp0"

REM Find a working Python 3.9+ with tkinter. "python" on PATH is not trusted
REM on its own: Windows ships a Microsoft Store stub named python.exe that
REM fails every call, so a real install that is missing from PATH (or sits
REM behind the stub) would otherwise look like no Python at all.
set "PYEXE="
call :probe python
if not defined PYEXE call :probe py -3
if not defined PYEXE for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if not defined PYEXE call :probe "%%d\python.exe"
if not defined PYEXE for /d %%d in ("%ProgramFiles%\Python3*") do if not defined PYEXE call :probe "%%d\python.exe"
for %%d in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do if not defined PYEXE call :probe "%%~d\python.exe"

if not defined PYEXE (
    echo ERROR: No working Python 3.9 or newer was found.
    echo.
    echo Install Python from https://www.python.org/downloads/ and on the first
    echo installer screen tick "Add python.exe to PATH". Keep the default
    echo "tcl/tk and IDLE" option, which TRACY needs for its window.
    echo.
    echo If Python is already installed, open Windows Settings ^> Apps ^>
    echo Advanced app settings ^> App execution aliases and turn OFF both
    echo "App Installer python.exe" and "App Installer python3.exe".
    echo.
    pause
    popd
    exit /b 1
)

REM Read the app version from the source so this prompt always matches APP_VERSION
set "APP_VERSION=unknown"
for /f "usebackq delims=" %%v in (`call "%PYEXE%" -c "import re;q=chr(34);d=open('fp_analysis_gui.py',encoding='utf-8').read();m=re.search('APP_VERSION\\s*=\\s*'+q+'([^'+q+']+)'+q,d);print(m.group(1) if m else 'unknown')"`) do set "APP_VERSION=%%v"

echo ========================================
echo  Fiber Photometry Analysis GUI v%APP_VERSION%
echo ========================================
echo.

echo Using Python: %PYEXE%
echo Checking dependencies...
echo.

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo WARNING: requirements.txt not found
    echo Attempting to start GUI anyway...
    echo.
    goto :startgui
)

REM Install/upgrade dependencies silently
echo Installing required packages...
"%PYEXE%" -m pip install --quiet --upgrade pip
"%PYEXE%" -m pip install --quiet -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may not have installed correctly
    echo The GUI will attempt to start anyway...
    echo.
) else (
    echo Dependencies installed successfully!
    echo.
)

:startgui
echo Starting Fiber Photometry Analysis GUI v%APP_VERSION%...
echo.

REM Run the Python GUI
"%PYEXE%" fp_analysis_gui.py

REM Pause to see any error messages if the GUI fails
if errorlevel 1 (
    echo.
    echo ========================================
    echo  ERROR: Failed to start the GUI
    echo ========================================
    echo.
    echo Please check the error message above.
    echo You can also try installing requirements manually:
    echo    "%PYEXE%" -m pip install -r requirements.txt
    echo.
    pause
)
popd
exit /b

REM Sets PYEXE to the full interpreter path if the given command runs a real
REM Python 3.9+ that can import tkinter; leaves it unset otherwise.
:probe
set "PROBED="
for /f "usebackq delims=" %%p in (`call %* -c "import sys,tkinter;assert sys.version_info>=(3,9);print(sys.executable)" 2^>nul`) do set "PROBED=%%p"
if defined PROBED if exist "%PROBED%" set "PYEXE=%PROBED%"
exit /b

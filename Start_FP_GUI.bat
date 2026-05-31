@echo off
REM Fiber Photometry Analysis GUI Launcher
REM This batch file starts the FP Analysis GUI

echo ========================================
echo  Fiber Photometry Analysis GUI v2.7
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo Python found. Checking dependencies...
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
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

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
echo Starting Fiber Photometry Analysis GUI v2.7...
echo.

REM Run the Python GUI
python fp_analysis_gui.py

REM Pause to see any error messages if the GUI fails
if errorlevel 1 (
    echo.
    echo ========================================
    echo  ERROR: Failed to start the GUI
    echo ========================================
    echo.
    echo Please check the error message above.
    echo You can also try installing requirements manually:
    echo    pip install -r requirements.txt
    echo.
    pause
)

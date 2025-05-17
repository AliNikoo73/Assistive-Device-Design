@echo off
echo Installing GaitSim Assist...

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.9 or higher.
    exit /b 1
)

REM Check Python version
for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(sys.version.split()[0])"') do (
    set python_version=%%a
)
if %python_version% LSS 9 (
    echo Python version is less than the required version 3.9
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Install GaitSim Assist in development mode
echo Installing GaitSim Assist...
pip install -e .

echo Installation complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat 
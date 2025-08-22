@echo off
echo Setting up virtual environment for ICE dataset download...
echo.

REM Check if virtual environment exists, if not create it
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required dependencies...
pip install -r requirements.txt

echo.
echo Running dataset download script...
python download_datasets.py

echo.
echo Deactivating virtual environment...
deactivate

echo.
echo Dataset download process completed!
pause
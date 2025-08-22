@echo off
setlocal

echo Creating virtual environment...
python -m venv dataset_venv

echo Activating virtual environment...
call dataset_venv\Scripts\activate.bat

echo Installing required packages...
pip install datasets

echo Running dataset download...
python download_datasets.py

echo Deactivating virtual environment...
deactivate

echo Cleaning up virtual environment...
rd /s /q dataset_venv

echo Done!
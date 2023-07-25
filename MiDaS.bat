@echo off
set "venv_folder=env"

rem Check if the 'env' folder exists
if not exist %venv_folder% (
    rem Create a virtual environment named 'env'
    python -m venv %venv_folder%
    echo Virtual environment 'env' created successfully.
) 

call env\Scripts\activate.bat

pip install -r requirements.txt

python final.py

pause

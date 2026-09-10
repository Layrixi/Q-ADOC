@echo off
REM Double-click convenience wrapper around run_dev.py.
REM All the actual logic (process management, clean shutdown) lives in the
REM Python script - this just launches it with your venv's Python.
REM Remember to run ollama as well, the app doesn't run it on itself.
REM Ollama should be run on port 11434. Otherwise app won't work properly since it wont have access to models

REM EDIT THIS if your virtual environment folder has a different name/path.
set VENV_PATH=.venv

if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [launcher] No venv found at %VENV_PATH%, using system Python instead.
    echo [launcher] Edit VENV_PATH in this file if your venv has a different name.
)

python run_dev.py

pause

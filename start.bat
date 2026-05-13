@echo off

:: ============================================================
::  Inpaint Anything - Launch Web UI
::  Run install.bat first if environment is not set up
:: ============================================================

set "ENV_NAME=inpaint-anything"
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

:: --- Find and activate conda ---
where conda >nul 2>&1 && goto :conda_ok
for %%P in (
    "%USERPROFILE%\anaconda3"  "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Anaconda3"  "%USERPROFILE%\Miniconda3"
    "C:\tools\Anaconda3"       "C:\tools\Miniconda3"
    "C:\Anaconda3"             "C:\Miniconda3"
    "C:\ProgramData\Anaconda3" "C:\ProgramData\Miniconda3"
    "D:\Anaconda3"             "D:\Miniconda3"
) do (
    if exist "%%~P\Scripts\activate.bat" (
        call "%%~P\Scripts\activate.bat" "%%~P"
        goto :conda_ok
    )
)
echo [ERROR] conda not found. Install Anaconda/Miniconda first.
pause & exit /b 1

:conda_ok
call conda activate %ENV_NAME%

echo ============================================================
echo   Inpaint Anything Web UI
echo   http://localhost:7860
echo   Ctrl+C to stop
echo ============================================================

cd /d "%SCRIPT_DIR%\app"
python app.py --lama_config "%SCRIPT_DIR%\lama\configs\prediction\default.yaml" --lama_ckpt "%SCRIPT_DIR%\pretrained_models\big-lama" --sam_ckpt "%SCRIPT_DIR%\pretrained_models\sam_vit_h_4b8939.pth"

pause

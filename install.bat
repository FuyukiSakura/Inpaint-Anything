@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  Inpaint Anything - One-time Environment Setup
::  Run this once, then use start.bat to launch
:: ============================================================

set "ENV_NAME=inpaint-anything"
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo   Inpaint Anything - Environment Setup
echo ============================================================
echo.

:: --- Find conda ---
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
echo [INFO] Using conda:
call conda --version
echo.

:: --- Create env ---
echo [1/4] Creating conda environment ...
call conda create -y -n %ENV_NAME% python=3.10
call conda activate %ENV_NAME%

:: --- PyTorch + CUDA via pip ---
echo.
echo [2/4] Installing PyTorch with CUDA 12.1 ...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

:: --- Segment Anything ---
echo.
echo [3/4] Installing Segment Anything ...
pip install -e "%SCRIPT_DIR%\segment_anything"

:: --- All other deps from requirements.txt ---
echo.
echo [4/4] Installing dependencies from requirements.txt ...
pip install -r "%SCRIPT_DIR%\requirements.txt"

:: --- SAM checkpoint ---
echo.
if not exist "%SCRIPT_DIR%\pretrained_models" mkdir "%SCRIPT_DIR%\pretrained_models"
if not exist "%SCRIPT_DIR%\pretrained_models\sam_vit_h_4b8939.pth" (
    echo Downloading SAM ViT-H checkpoint ~2.4 GB ...
    python -c "import urllib.request,os;urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',os.path.join(r'%SCRIPT_DIR%','pretrained_models','sam_vit_h_4b8939.pth'),reporthook=lambda b,bs,ts:print(f'\r  {b*bs//1000000}/{ts//1000000} MB',end=''))"
    echo.
) else (
    echo [INFO] SAM checkpoint already present.
)

if not exist "%SCRIPT_DIR%\pretrained_models\big-lama\models\best.ckpt" (
    echo.
    echo !! big-lama checkpoint NOT FOUND !!
    echo Download from: https://disk.yandex.ru/d/ouP6l8VJ0HpMZg
    echo Extract to:    %SCRIPT_DIR%\pretrained_models\big-lama\
)

echo.
echo ============================================================
echo   Setup complete! Run start.bat to launch the web UI.
echo ============================================================
pause

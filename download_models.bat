
@echo off
setlocal

echo =====================================
echo      PhishGuard Model Downloader
echo =====================================

where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed.
    pause
    exit /b 1
)

git lfs version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git LFS is not installed.
    pause
    exit /b 1
)

set "REPO=https://github.com/imtiaz33010-rgb/PhishGuard-Application.git"
set "TEMP_REPO=%TEMP%\PhishGuard-LFS"

if exist "%TEMP_REPO%" rmdir /s /q "%TEMP_REPO%"

echo Downloading models from GitHub...

git clone "%REPO%" "%TEMP_REPO%"
if errorlevel 1 goto failed

git -C "%TEMP_REPO%" lfs pull
if errorlevel 1 goto failed

if not exist "models" mkdir "models"

copy /Y "%TEMP_REPO%\models\*.joblib" "models\"
if errorlevel 1 goto failed

rmdir /s /q "%TEMP_REPO%"

echo.
echo SUCCESS! PhishGuard models are ready.
pause
exit /b 0

:failed
echo.
echo ERROR: Model download failed.
echo Check your internet connection and Git LFS installation.
if exist "%TEMP_REPO%" rmdir /s /q "%TEMP_REPO%"
pause
exit /b 1

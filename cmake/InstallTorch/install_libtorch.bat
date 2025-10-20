@echo off
setlocal enabledelayedexpansion

:: ==========================
:: Variables
:: ==========================
set TORCH_VERSION=2.0.1
set ROOT_INSTALL_DIR=C:\ProgramData
set RELEASE_DIR=%ROOT_INSTALL_DIR%\libtorch-%TORCH_VERSION%-Release
set RELEASE_URL=https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.1%%2Bcpu.zip
set TEMP_ZIP=%TEMP%\libtorch-%TORCH_VERSION%.zip

set ZIP_FILE=%CD%\CMake\InstallTorch\torch_win_2.0.1.zip
set TEMP_EXTRACT_DIR=%TEMP%\torch_extracted

:: ==========================
:: Step 1: Download & Install LibTorch if not already installed (THIS IS FOR DEVELOPMENT PURPOSES ONLY)
:: ==========================
if not exist "%RELEASE_DIR%" (
    echo [INFO] LibTorch not found. Downloading...
    curl --ssl-no-revoke -L -o "%TEMP_ZIP%" "%RELEASE_URL%"

    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to download LibTorch.
        exit /b 1
    )

    :: Extract to ProgramData
    echo [INFO] Extracting LibTorch...
    powershell -Command "Expand-Archive -Path '%TEMP_ZIP%' -DestinationPath '%ROOT_INSTALL_DIR%' -Force"

    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Extraction failed.
        exit /b 1
    )

    :: Rename extracted folder to match version
    if exist "%ROOT_INSTALL_DIR%\libtorch" (
        rename "%ROOT_INSTALL_DIR%\libtorch" "libtorch-%TORCH_VERSION%-Release"
    )

    echo [INFO] LibTorch installed successfully to %RELEASE_DIR%
) else (
    echo [INFO] LibTorch is already installed. Skipping download.
)

:: ==========================
:: Step 2: Extract torch_win_2.0.1.zip & Move to System32 (THESE ARE THE FILES FOR RUNTIME USAGE)
:: ==========================
:: Ensure ZIP file exists
if not exist "%ZIP_FILE%" (
    echo [ERROR] Zip file not found: %ZIP_FILE%
    exit /b 1
)

:: Create temporary extraction directory
if not exist "%TEMP_EXTRACT_DIR%" (
    mkdir "%TEMP_EXTRACT_DIR%"
)

:: Extract torch_win_2.0.1.zip
echo [INFO] Extracting torch_win_2.0.1.zip...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_EXTRACT_DIR%' -Force"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Extraction failed.
    exit /b 1
)

:: Move all extracted files to System32
echo [INFO] Moving extracted files to C:\Windows\System32...
set FILE_COUNT=0

for /r "%TEMP_EXTRACT_DIR%" %%f in (*) do (
    echo [MOVING] %%~nxf
    move /y "%%f" "%SystemRoot%\System32"
    set /a FILE_COUNT+=1
)

if %FILE_COUNT% EQU 0 (
    echo [ERROR] No files found to move. Exiting...
    exit /b 1
)

:: ==========================
:: Step 3: Clean up extracted files (Cleanup)
:: ==========================
echo [INFO] Cleaning up temporary files...
rd /s /q "%TEMP_EXTRACT_DIR%"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to delete temporary files.
) else (
    echo [INFO] Cleanup complete.
)

:: Success message
echo [SUCCESS] All files moved to C:\Windows\System32, and LibTorch is installed in %RELEASE_DIR%.
exit /b 0

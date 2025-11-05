@echo off
echo === Building TensorRT LinalgSolve Plugin ===

cd /d "%~dp0"

if exist build rmdir /s /q build
mkdir build
cd build

echo.
echo === Running CMake Configuration ===
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo === Building Release ===
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo === Build Successful ===
echo Plugin DLL: build\Release\linalgSolvePlugin.dll
pause
@echo off
setlocal
set "UV_CACHE_DIR=%TEMP%\MoonTranslate-build\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%LOCALAPPDATA%\MoonTranslate\python"
set "UV_PYTHON_BIN_DIR=%LOCALAPPDATA%\MoonTranslate\bin"
set "MOONTRANSLATE_DATA=%LOCALAPPDATA%\MoonTranslate"
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" (
  echo Install uv from https://docs.astral.sh/uv/getting-started/installation/ then run this file again.
  pause
  exit /b 1
)
"%UV%" python install 3.12
if errorlevel 1 goto failed
"%UV%" venv --python 3.12 "%MOONTRANSLATE_DATA%\venv"
if errorlevel 1 goto failed
"%UV%" pip install --python "%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
"%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" "%~dp0prepare_engine.py"
if errorlevel 1 goto failed
"%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" "%~dp0prepare_inpainting.py"
if errorlevel 1 goto failed
"%UV%" pip install --python "%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" onnx==1.22.0
if errorlevel 1 goto gpu_skipped
"%UV%" pip install --python "%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" --target "%MOONTRANSLATE_DATA%\acceleration\directml" --no-deps onnxruntime-directml==1.24.4
if errorlevel 1 goto gpu_skipped
"%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" "%~dp0prepare_gpu.py"
if errorlevel 1 goto gpu_skipped
goto gpu_done
:gpu_skipped
echo Optional GPU acceleration unavailable. CPU inpainting remains available.
:gpu_done
"%MOONTRANSLATE_DATA%\venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, sys.argv[1]); from core import Reader; Reader()" "%~dp0."
if errorlevel 1 goto failed
echo Setup complete. Open Start.cmd.
pause
exit /b 0
:failed
echo Setup failed. Keep this window open to read the error.
pause
exit /b 1

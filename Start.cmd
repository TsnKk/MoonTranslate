@echo off
setlocal
set "MOONTRANSLATE_DATA=%LOCALAPPDATA%\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\MoonTranslate"
if not exist "%MOONTRANSLATE_DATA%\venv\Scripts\pythonw.exe" set "MOONTRANSLATE_DATA=%LOCALAPPDATA%\MoonTranslate"
if not exist "%MOONTRANSLATE_DATA%\venv\Scripts\pythonw.exe" (
  echo Please run Setup.cmd first.
  pause
  exit /b 1
)
start "MoonTranslate" "%MOONTRANSLATE_DATA%\venv\Scripts\pythonw.exe" "%~dp0launcher.py"
if errorlevel 1 (
  echo MoonTranslate could not start. See %%TEMP%%\MoonTranslate-startup.log
  pause
)

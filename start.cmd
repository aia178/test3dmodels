@echo off
rem RoomFit launcher. ASCII only on purpose: Japanese text in a .cmd file is misread by cmd.exe (code page 932).
rem Needs: Node.js 22+, uv (https://docs.astral.sh/uv/). See README.md for details.
setlocal
cd /d "%~dp0"

if not exist "web\node_modules" (
  echo [1/3] Installing web dependencies...
  call npm --prefix web ci || goto :error
)
if not exist "web\dist\index.html" (
  echo [2/3] Building the web app...
  call npm --prefix web run build || goto :error
)
echo [3/3] Starting RoomFit. Open http://127.0.0.1:8787 in your browser.
echo       Press Ctrl+C in this window to stop.
rem To use from a phone on the same Wi-Fi, change 127.0.0.1 below to 0.0.0.0 (read the note in README.md first).
uv run --directory server uvicorn roomfit.app:app --host 127.0.0.1 --port 8787
goto :eof

:error
echo Setup failed. Check the messages above.
pause
exit /b 1

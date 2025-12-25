@echo off
title Lorebary Proxy Launcher
color 0A

echo ========================================
echo    LOREBARY DEEPSEEK V3.2 PROXY
echo ========================================
echo.

REM Automatically change to the directory where this script is located
cd /d "%~dp0"

echo Current directory: %CD%
echo.
echo Starting proxy server...
echo.

REM Start the Node.js server in a new window
start "Lorebary Proxy Server" cmd /k "npm start"

echo Waiting for server to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting Cloudflare tunnel...
echo.

REM Start Cloudflare tunnel in a separate window (stays open)
start "Cloudflare Tunnel - CHECK THIS WINDOW FOR URL" cmd /k "echo Starting Cloudflare Tunnel... && echo. && echo COPY THE URL THAT APPEARS BELOW: && echo. && cloudflared tunnel --url http://localhost:3000"

echo.
echo ========================================
echo    PROXY IS NOW RUNNING!
echo ========================================
echo.
echo IMPORTANT:
echo 1. Look at the "Cloudflare Tunnel" window
echo 2. Copy the URL that looks like:
echo    https://something-random-1234.trycloudflare.com
echo 3. Use that URL in Lorebary settings
echo.
echo Control Panel: http://localhost:3001
echo.
echo ========================================
echo.
timeout /t 3 /nobreak >nul

REM Open control panel in browser
start http://localhost:3001

echo Press any key to minimize this window...
pause >nul
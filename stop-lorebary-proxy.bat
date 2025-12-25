@echo off
title Stop Lorebary Proxy
color 0C

echo ========================================
echo    STOPPING LOREBARY PROXY
echo ========================================
echo.

echo Stopping Node.js server...
taskkill /F /IM node.exe /T >nul 2>&1

echo Stopping Cloudflare tunnel...
taskkill /F /IM cloudflared.exe /T >nul 2>&1

echo.
echo ========================================
echo    PROXY STOPPED!
echo ========================================
echo.

REM Clean up temp file
if exist "C:\Users\breno\OneDrive\Documentos\important\lorebary-proxy\tunnel-url.txt" (
    del "C:\Users\breno\OneDrive\Documentos\important\lorebary-proxy\tunnel-url.txt"
)

echo All processes terminated.
echo.
timeout /t 3
exit
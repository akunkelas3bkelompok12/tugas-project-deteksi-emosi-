@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "NODE_ROOT=%PROJECT_ROOT%.tools\node"
set "NETLIFY_JS=%PROJECT_ROOT%.tools\netlify\node_modules\netlify-cli\bin\run.js"

if not exist "%NODE_ROOT%\node.exe" (
  echo Node portable belum ditemukan di .tools\node.
  exit /b 1
)

if not exist "%NETLIFY_JS%" (
  echo Netlify CLI belum ditemukan di .tools\netlify.
  exit /b 1
)

set "PATH=%NODE_ROOT%;%PATH%"
"%NODE_ROOT%\node.exe" "%NETLIFY_JS%" %*

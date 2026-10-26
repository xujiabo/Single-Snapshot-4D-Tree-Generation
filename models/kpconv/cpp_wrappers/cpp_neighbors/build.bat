@echo off
setlocal
pushd "%~dp0" || exit /b 1
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" setup.py build_ext --inplace
set "build_status=%ERRORLEVEL%"
popd
exit /b %build_status%

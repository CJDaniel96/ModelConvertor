@ECHO off
REM Set the path to the location of the Python Libraries
set PYTHONPATH = %~dp0libs
REM Start the Python program
%~dp0libs\python.exe -m streamlit run %~dp0main.py
if %ERRORLEVEL% NEQ 0 pause

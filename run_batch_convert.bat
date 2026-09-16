@echo off
cd /d D:\src\waymo2mcap
"D:\CondaData\envs\waymo\python.exe" auto_convert.py > convert_batch.log 2> convert_batch.err.log
echo EXIT_CODE=%ERRORLEVEL% >> convert_batch.log

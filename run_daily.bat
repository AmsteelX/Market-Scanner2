@echo off
REM สคริปต์สำหรับ Windows Task Scheduler
REM ตั้งเวลา: Task Scheduler > Create Basic Task > Daily > Start a program > เลือกไฟล์นี้
cd /d "%~dp0"
python scanner.py --outdir reports --top 20
start "" "reports\latest.html"

#!/usr/bin/env bash
# สคริปต์สำหรับตั้ง cron บน macOS / Linux
# ตัวอย่างตั้ง cron ให้รันทุกวันจันทร์-ศุกร์ 05:30 น. เวลาไทย:
#   crontab -e   แล้วใส่บรรทัด:
#   30 5 * * 1-5 /path/to/market_scanner/run_daily.sh >> /path/to/market_scanner/cron.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" scanner.py --outdir reports --top 20
# เปิดรายงานอัตโนมัติ (ลบบรรทัดนี้ถ้าไม่ต้องการ)
if command -v open >/dev/null 2>&1; then open reports/latest.html
elif command -v xdg-open >/dev/null 2>&1; then xdg-open reports/latest.html
fi

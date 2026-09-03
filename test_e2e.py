"""รัน main() แบบครบวงจรด้วยข้อมูลจำลอง (ไม่แตะเน็ต) เพื่อจับ error ตอนรันจริง"""
import datetime as dt
import glob
import os
import shutil
import sys

import pandas as pd

import earnings as ea
import extended as ex
import scanner as sc
import test_scanner as ts     # ใช้หุ้นจำลองชุดเดิม

fake = {"LONGA": ts.brk, "LONGB": ts.spike, "LONGC": ts.pull,
        "SHRTA": ts.bkd, "SHRTB": ts.dist, "SHRTC": ts.fade, "SPY": ts.brk}
sc.download_prices = lambda tickers, days, chunk=100: fake
sc.get_universe = lambda *a, **k: list(fake)

# จำลองวันประกาศงบ: LONGB งบพรุ่งนี้ (ต้องโดนตัด), SHRTC เพิ่งประกาศเมื่อวาน (ติดป้าย)
TODAY = dt.date.today()
FAKE_EARN = {
    "LONGA": (TODAY + dt.timedelta(days=40), TODAY - dt.timedelta(days=50)),
    "LONGB": (TODAY + dt.timedelta(days=1),  TODAY - dt.timedelta(days=89)),
    "LONGC": (TODAY + dt.timedelta(days=30), TODAY - dt.timedelta(days=60)),
    "SHRTA": (TODAY + dt.timedelta(days=25), TODAY - dt.timedelta(days=65)),
    "SHRTB": (TODAY + dt.timedelta(days=3),  TODAY - dt.timedelta(days=87)),
    "SHRTC": (TODAY + dt.timedelta(days=88), TODAY - dt.timedelta(days=1)),
}
ea._fetch_one = lambda t, today: FAKE_EARN.get(t, (None, None))

# จำลองราคานอกเวลาทำการ (คิดเป็น % จากราคาปิด)
FAKE_EXT_PCT = {"LONGA": +7.0, "SHRTA": +6.0, "LONGC": -0.4, "SHRTC": +1.1}
_TS = pd.Timestamp("2026-09-03 17:30", tz="America/New_York")   # after-hours


def _fake_quotes(tickers, chunk=40, verbose=True):
    out = {}
    for t in tickers:
        if t in FAKE_EXT_PCT and t in fake:
            base = float(fake[t]["Close"].iloc[-1])
            out[t] = {"price": base * (1 + FAKE_EXT_PCT[t] / 100),
                      "session": "post", "ts": _TS, "volume": 250_000.0}
    return out


sc.get_extended_quotes = _fake_quotes
sc.current_session = lambda now=None: "post"

shutil.rmtree("/tmp/rep", ignore_errors=True)
sys.argv = ["scanner.py", "--outdir", "/tmp/rep", "--account", "25000", "--risk", "0.75"]
assert sc.main() == 0, "main() ไม่สำเร็จ"

csv = glob.glob("/tmp/rep/signals_*.csv")
assert csv and glob.glob("/tmp/rep/report_*.html") and os.path.exists("/tmp/rep/latest.html")
assert os.path.exists("/tmp/rep/.earnings_cache.json"), "ไม่ได้เขียน cache วันงบ"
d = pd.read_csv(csv[0])

print("\n" + "=" * 70)
print(d[["side", "setup", "ticker", "score", "entry", "stop", "target",
         "shares", "earnings", "ext_chg_%", "ext_flag"]].to_string(index=False))
print("=" * 70)

fails = []
def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)

check("หุ้นที่งบออกพรุ่งนี้โดนตัดออกจากผล", "LONGB" not in set(d["ticker"]),
      f"เหลือ {sorted(set(d['ticker']))}")
check("หุ้นงบอีก 3 วันก็โดนตัดด้วย", "SHRTB" not in set(d["ticker"]))
check("หุ้นที่งบยังอีกไกลยังอยู่", {"LONGA", "LONGC", "SHRTA"} <= set(d["ticker"]))
check("หุ้นเพิ่งประกาศงบยังอยู่แต่ติดป้าย",
      "SHRTC" in set(d["ticker"]) and
      d.loc[d["ticker"] == "SHRTC", "earnings"].str.contains("เพิ่งประกาศ").all())
check("มีสัญญาณทั้งสองฝั่ง", set(d["side"]) == {"long", "short"}, str(sorted(set(d["side"]))))

lg, sh = d[d["side"] == "long"], d[d["side"] == "short"]
check("ฝั่งซื้อ: stop < entry < target ทุกแถว",
      ((lg["stop"] < lg["entry"]) & (lg["entry"] < lg["target"])).all())
check("ฝั่งชอร์ต: target < entry < stop ทุกแถว",
      ((sh["target"] < sh["entry"]) & (sh["entry"] < sh["stop"])).all())
check("R:R = 2 ทุกแถว", (d["rr"].round(2) == 2.0).all(), str(d["rr"].unique()))
risk = (d["entry"] - d["stop"]).abs() * d["shares"]
check("ขนาดไม้เคารพ --account 25000 --risk 0.75 (= $187.5)", (risk <= 187.5 * 1.01).all(),
      str(risk.round(2).tolist()))

# ── ราคานอกเวลาทำการ ──
la = d[d["ticker"] == "LONGA"]
sa = d[d["ticker"] == "SHRTA"]
check("LONGA (ซื้อ) วิ่งขึ้น 7% นอกเวลา = ran",
      (la["ext_flag"] == "ran").all() and (la["ext_chg_%"].round(1) == 7.0).all(),
      str(la[["ext_chg_%", "ext_flag"]].values.tolist()))
check("SHRTA (ชอร์ต) วิ่งขึ้น 6% นอกเวลา = against",
      (sa["ext_flag"] == "against").all(),
      str(sa[["ext_chg_%", "ext_flag"]].values.tolist()))
check("หุ้นที่ขยับน้อยได้ flag ok",
      (d.loc[d["ticker"] == "LONGC", "ext_flag"] == "ok").all())
check("หุ้นที่ไม่มีราคานอกเวลาไม่พัง (unknown)",
      (d.loc[d["ticker"] == "SPY", "ext_flag"] == "unknown").all(),
      str(d.loc[d["ticker"] == "SPY", "ext_flag"].unique()))
check("ทุกแถวมีคอลัมน์ ext_session เมื่อมีข้อมูล",
      (d.loc[d["ext_flag"] != "unknown", "ext_session"] == "post").all())

html_txt = open("/tmp/rep/latest.html", encoding="utf-8").read()
check("รายงานมีกล่องเตือน gap", "เตือน gap นอกเวลา" in html_txt)
check("รายงานบอกว่าเป็นรอบหลังตลาดปิด", "after-hours" in html_txt or "หลังตลาดปิด" in html_txt)
check("รายงานมีคอลัมน์นอกเวลา", "<th>นอกเวลา</th>" in html_txt)

# ── ปิดการเช็กนอกเวลา ──
shutil.rmtree("/tmp/rep4", ignore_errors=True)
sys.argv = ["scanner.py", "--outdir", "/tmp/rep4", "--extended", "off", "--earnings", "off"]
assert sc.main() == 0
d4 = pd.read_csv(glob.glob("/tmp/rep4/signals_*.csv")[0])
check("--extended off ไม่ดึงราคานอกเวลา", (d4["ext_flag"] == "unknown").all())

# ── ปรับเกณฑ์เตือนเป็น 10% แล้ว 7% ต้องไม่เตือน ──
shutil.rmtree("/tmp/rep5", ignore_errors=True)
sys.argv = ["scanner.py", "--outdir", "/tmp/rep5", "--gap-alert", "10", "--earnings", "off"]
assert sc.main() == 0
d5 = pd.read_csv(glob.glob("/tmp/rep5/signals_*.csv")[0])
check("--gap-alert 10 ทำให้ 7% ไม่ถูกเตือน",
      (d5.loc[d5["ticker"] == "LONGA", "ext_flag"] == "ok").all(),
      str(d5.loc[d5["ticker"] == "LONGA", "ext_flag"].unique()))

# ── โหมด flag ต้องไม่ตัดใครออก ──
shutil.rmtree("/tmp/rep2", ignore_errors=True)
sys.argv = ["scanner.py", "--outdir", "/tmp/rep2", "--earnings", "flag"]
assert sc.main() == 0
d2 = pd.read_csv(glob.glob("/tmp/rep2/signals_*.csv")[0])
check("โหมด flag เก็บหุ้นใกล้วันงบไว้ด้วย", "LONGB" in set(d2["ticker"]),
      f"{sorted(set(d2['ticker']))}")

# ── โหมด long อย่างเดียว ──
shutil.rmtree("/tmp/rep3", ignore_errors=True)
sys.argv = ["scanner.py", "--outdir", "/tmp/rep3", "--sides", "long", "--earnings", "off"]
assert sc.main() == 0
d3 = pd.read_csv(glob.glob("/tmp/rep3/signals_*.csv")[0])
check("--sides long ให้เฉพาะฝั่งซื้อ", set(d3["side"]) == {"long"}, str(set(d3["side"])))
check("--earnings off ไม่สร้าง cache", not os.path.exists("/tmp/rep3/.earnings_cache.json"))

print()
if fails:
    print(f"ไม่ผ่าน {len(fails)} ข้อ: {fails}")
    raise SystemExit(1)
print("ผ่านทุกข้อ")

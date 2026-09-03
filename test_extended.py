"""ทดสอบตรรกะราคานอกเวลาทำการ (ไม่ต้องต่อเน็ต)"""
import pandas as pd

import extended as ex

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


def et(s):
    """สร้าง timestamp ตามเวลานิวยอร์ก"""
    return pd.Timestamp(s, tz="America/New_York")


print("— แยกช่วงเวลาซื้อขาย —")
cases = [
    ("2026-09-03 03:59", "closed", "ก่อน 04:00 ยังไม่เปิด pre-market"),
    ("2026-09-03 04:00", "pre", "04:00 เริ่ม pre-market พอดี"),
    ("2026-09-03 09:29", "pre", "ก่อนระฆังหนึ่งนาที"),
    ("2026-09-03 09:30", "regular", "09:30 ตลาดเปิด"),
    ("2026-09-03 15:59", "regular", "ก่อนปิดหนึ่งนาที"),
    ("2026-09-03 16:00", "post", "16:00 เข้า after-hours"),
    ("2026-09-03 19:59", "post", "ก่อน 20:00"),
    ("2026-09-03 20:00", "closed", "20:00 ปิดสนิท"),
    ("2026-09-04 11:00", "regular", "ศุกร์ยังเป็นวันทำการ"),
    ("2026-09-05 11:00", "closed", "เสาร์ปิด"),
    ("2026-09-06 11:00", "closed", "อาทิตย์ปิด"),
]
for stamp, want, why in cases:
    got = ex.session_of(et(stamp))
    check(f"{stamp} → {want}", got == want, f"ได้ {got} · {why}")

# timestamp ที่เป็น UTC ต้องแปลงเป็นเวลานิวยอร์กก่อนตัดสิน
# 2026-09-03 20:30 UTC = 16:30 ET (EDT) = after-hours
check("แปลง UTC เป็นเวลานิวยอร์กถูกต้อง",
      ex.session_of(pd.Timestamp("2026-09-03 20:30", tz="UTC")) == "post",
      f"ได้ {ex.session_of(pd.Timestamp('2026-09-03 20:30', tz='UTC'))}")
# 2026-09-03 13:00 UTC = 09:00 ET = pre-market (รอบเช้าที่เราจะตั้ง)
check("รอบ 13:00 UTC ตกในช่วง pre-market",
      ex.session_of(pd.Timestamp("2026-09-03 13:00", tz="UTC")) == "pre")
# 2026-09-03 22:30 UTC = 18:30 ET = after-hours (รอบเย็นเดิม)
check("รอบ 22:30 UTC ตกในช่วง after-hours",
      ex.session_of(pd.Timestamp("2026-09-03 22:30", tz="UTC")) == "post")
check("ts เป็น None ถือว่าปิด", ex.session_of(None) == "closed")

print("\n— จัดประเภท gap —")
g = ex.classify_gap
check("ซื้อ + ร่วง 6% = สวนทาง", g(-6.0, "long", 3)[0] == "against", g(-6.0, "long", 3)[1])
check("ซื้อ + ขึ้น 6% = วิ่งหนี", g(6.0, "long", 3)[0] == "ran", g(6.0, "long", 3)[1])
check("ชอร์ต + ขึ้น 6% = สวนทาง", g(6.0, "short", 3)[0] == "against", g(6.0, "short", 3)[1])
check("ชอร์ต + ร่วง 6% = วิ่งหนี", g(-6.0, "short", 3)[0] == "ran", g(-6.0, "short", 3)[1])
check("ขยับ 1% = ปกติ (ทั้งสองฝั่ง)",
      g(1.0, "long", 3)[0] == "ok" and g(-1.0, "short", 3)[0] == "ok")
check("ขยับ 2.9% ยังไม่เตือน (ต่ำกว่าเกณฑ์ 3)", g(-2.9, "long", 3)[0] == "ok")
check("ขยับ 3.0% เตือนพอดี (ขอบ)", g(-3.0, "long", 3)[0] == "against")
check("ปรับเกณฑ์เป็น 5 แล้ว 4% ไม่เตือน", g(-4.0, "long", 5)[0] == "ok")
check("ไม่มีข้อมูล = unknown", g(None, "long", 3)[0] == "unknown")
check("NaN = unknown", g(float("nan"), "long", 3)[0] == "unknown")

print("\n— ผูกเข้ากับผลสัญญาณ —")
hits = [
    {"ticker": "AAA", "close": 100.0, "plan": {"side": "long"}},    # +6% วิ่งหนี
    {"ticker": "BBB", "close": 50.0, "plan": {"side": "long"}},     # -8% สวนทาง
    {"ticker": "CCC", "close": 200.0, "plan": {"side": "short"}},   # +7% สวนทาง
    {"ticker": "DDD", "close": 80.0, "plan": {"side": "long"}},     # +0.5% ปกติ
    {"ticker": "EEE", "close": 30.0, "plan": {"side": "long"}},     # ไม่มีข้อมูล
]
quotes = {
    "AAA": {"price": 106.0, "session": "post", "ts": et("2026-09-03 17:00"), "volume": 5e5},
    "BBB": {"price": 46.0, "session": "post", "ts": et("2026-09-03 17:00"), "volume": 9e5},
    "CCC": {"price": 214.0, "session": "pre", "ts": et("2026-09-04 08:00"), "volume": 2e5},
    "DDD": {"price": 80.4, "session": "post", "ts": et("2026-09-03 17:00"), "volume": 1e5},
}
n = ex.attach_to_signals(hits, quotes, 3.0)
by = {h["ticker"]: h for h in hits}
check("นับจำนวนที่ต้องเตือนถูก (3 ตัว)", n == 3, f"ได้ {n}")
check("AAA ซื้อ +6% = ran", by["AAA"]["ext_status"] == "ran", by["AAA"]["ext_text"])
check("BBB ซื้อ -8% = against", by["BBB"]["ext_status"] == "against", by["BBB"]["ext_text"])
check("CCC ชอร์ต +7% = against", by["CCC"]["ext_status"] == "against", by["CCC"]["ext_text"])
check("DDD ขยับน้อย = ok", by["DDD"]["ext_status"] == "ok", by["DDD"]["ext_text"])
check("EEE ไม่มีราคานอกเวลา = unknown", by["EEE"]["ext_status"] == "unknown")
check("คำนวณ % เปลี่ยนแปลงถูก", abs(by["AAA"]["ext_chg"] - 6.0) < 1e-9,
      f"{by['AAA']['ext_chg']:.3f}")
check("เก็บช่วงเวลาไว้ด้วย", by["CCC"]["ext_session"] == "pre")
check("ตัวที่ไม่มีข้อมูลไม่ทำให้พัง", by["EEE"]["ext_chg"] is None)

print()
if failures:
    print(f"ไม่ผ่าน {len(failures)} ข้อ: {failures}")
    raise SystemExit(1)
print("ผ่านทุกข้อ")

"""ตรวจสูตรอินดิเคเตอร์เทียบกับค่าอ้างอิงที่รู้ผลล่วงหน้า (ไม่ต้องต่อเน็ต)"""
import numpy as np
import pandas as pd

import indicators as ind

# ชุดข้อมูลมาตรฐานของ Wilder/StockCharts สำหรับทดสอบ RSI(14)
CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]
EXPECTED_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- RSI
close = pd.Series(CLOSES)
got = ind.rsi(close, 14).dropna().to_numpy()
exp = np.array(EXPECTED_RSI)
# ตารางอ้างอิงที่เผยแพร่ปัดทศนิยมในขั้นตอน seed ทำให้ต่างกันราว 0.05-0.07 จุด
# แล้วค่อย ๆ ลู่เข้าหากัน (เห็นชัดว่า diff ลดลงเรื่อย ๆ) จึงตั้ง tolerance ที่ 0.1
check("RSI(14) ตรงกับค่าอ้างอิง StockCharts", len(got) == len(exp) and np.allclose(got, exp, atol=0.1),
      f"maxdiff={np.abs(got[:len(exp)] - exp).max():.4f}" if len(got) >= len(exp) else f"len {len(got)} vs {len(exp)}")

# ขอบ: ราคาขึ้นอย่างเดียว -> RSI = 100 ; ลงอย่างเดียว -> RSI = 0
up_only = pd.Series(np.arange(1, 40, dtype=float))
check("RSI ขาขึ้นล้วน = 100", abs(ind.rsi(up_only, 14).iloc[-1] - 100.0) < 1e-9)
down_only = pd.Series(np.arange(40, 1, -1, dtype=float))
check("RSI ขาลงล้วน = 0", abs(ind.rsi(down_only, 14).iloc[-1] - 0.0) < 1e-9)

# ---------------------------------------------------------------- Wilder / ATR
# ถ้า TR คงที่ทุกวัน ATR ต้องเท่ากับค่านั้นเป๊ะ
n = 30
h = pd.Series(np.full(n, 11.0)); l = pd.Series(np.full(n, 10.0)); c = pd.Series(np.full(n, 10.5))
check("ATR ของแท่งกว้าง 1.0 คงที่ = 1.0", abs(ind.atr(h, l, c, 14).iloc[-1] - 1.0) < 1e-9)

# seed ของ Wilder ต้องเป็นค่าเฉลี่ยธรรมดาของ 14 ค่าแรก
s = pd.Series([float(i) for i in range(1, 31)])
w = ind.wilder(s, 14)
check("Wilder seed = SMA ของ 14 ค่าแรก", abs(w.iloc[13] - s.iloc[:14].mean()) < 1e-9,
      f"{w.iloc[13]:.4f} vs {s.iloc[:14].mean():.4f}")
check("Wilder ก่อนครบ n เป็น NaN", bool(w.iloc[:13].isna().all()))
# ค่าถัดไป = prev + (x - prev)/n
expected_next = w.iloc[13] + (s.iloc[14] - w.iloc[13]) / 14
check("Wilder recursion ถูกต้อง", abs(w.iloc[14] - expected_next) < 1e-9)

# ---------------------------------------------------------------- ADX
# เทรนด์ขึ้นแรงสม่ำเสมอ -> ADX ต้องสูง (>40)
k = 80
base = np.arange(k, dtype=float)
h2 = pd.Series(base + 1.0); l2 = pd.Series(base); c2 = pd.Series(base + 0.8)
adx_trend = ind.adx(h2, l2, c2, 14).iloc[-1]
check("ADX เทรนด์ขึ้นชัด > 40", adx_trend > 40, f"ADX={adx_trend:.1f}")

# ตลาด sideways สุ่ม -> ADX ต้องต่ำกว่าเทรนด์ชัดเจน
rng = np.random.default_rng(7)
noise = 100 + np.cumsum(rng.normal(0, 0.3, 200))
c3 = pd.Series(noise); h3 = c3 + 0.5; l3 = c3 - 0.5
adx_side = ind.adx(h3, l3, c3, 14).iloc[-1]
check("ADX sideways ต่ำกว่า ADX เทรนด์", adx_side < adx_trend, f"{adx_side:.1f} < {adx_trend:.1f}")

# ---------------------------------------------------------------- Bollinger
c4 = pd.Series(rng.normal(100, 2, 300))
pctb = ind.bollinger_pctb(c4, 20, 2.0)
mid = ind.sma(c4, 20)
sd = c4.rolling(20).std(ddof=0)
manual = (c4 - (mid - 2 * sd)) / (4 * sd)
check("%B ตรงกับสูตรมือ", np.allclose(pctb.dropna(), manual.dropna(), atol=1e-9))
check("%B ส่วนใหญ่อยู่ในช่วง 0-1", (pctb.dropna().between(-0.5, 1.5)).mean() > 0.98)

# ปิดที่แบนด์กลางพอดี -> %B = 0.5
flat = pd.Series(np.concatenate([np.full(19, 100.0), [100.0]] * 3))
check("%B ของราคานิ่ง = NaN (sd=0, ไม่หารศูนย์)", bool(np.isnan(ind.bollinger_pctb(flat, 20).iloc[-1])))

# ---------------------------------------------------------------- RVOL
vol = pd.Series([100.0] * 25 + [300.0])
r = ind.rvol(vol, 20)
check("RVOL วอลุ่ม 3 เท่า = 3.0", abs(r.iloc[-1] - 3.0) < 1e-9, f"{r.iloc[-1]:.3f}")
check("RVOL ไม่นับวันนี้ในค่าเฉลี่ย", abs(ind.rvol(pd.Series([100.0] * 30), 20).iloc[-1] - 1.0) < 1e-9)

# ---------------------------------------------------------------- close location
h5 = pd.Series([10.0, 10.0, 10.0]); l5 = pd.Series([8.0, 8.0, 8.0]); c5 = pd.Series([10.0, 8.0, 9.0])
cl = ind.close_location(h5, l5, c5)
check("close_location: ปิดที่ high=1, low=0, กลาง=0.5",
      abs(cl.iloc[0] - 1) < 1e-9 and abs(cl.iloc[1]) < 1e-9 and abs(cl.iloc[2] - 0.5) < 1e-9)

# ---------------------------------------------------------------- rolling high/low
c6 = pd.Series([1.0, 5.0, 3.0, 2.0, 9.0, 4.0])
check("rolling_high(3) ถูกต้อง",
      ind.rolling_high(c6, 3).tolist()[2:] == [5.0, 5.0, 9.0, 9.0])

print()
if failures:
    print(f"เทสต์ไม่ผ่าน {len(failures)} ข้อ: {failures}")
    raise SystemExit(1)
print("ผ่านทุกข้อ")

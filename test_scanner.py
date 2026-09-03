"""ทดสอบตรรกะสัญญาณด้วยหุ้นจำลองที่ 'รู้คำตอบอยู่แล้ว' (ไม่ต้องต่อเน็ต)"""
import datetime as dt

import numpy as np
import pandas as pd

import earnings as ea
import scanner as sc

rng = np.random.default_rng(42)
IDX = pd.bdate_range("2024-01-01", periods=300)
failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


def make(close, vol=None, hl_pct=0.015, cloc_last=None):
    """สร้าง OHLCV จาก series ราคาปิด"""
    close = np.asarray(close, dtype=float)
    n = len(close)
    rngm = close * hl_pct
    high, low = close + rngm * 0.6, close - rngm * 0.6
    if cloc_last is not None:                      # บังคับตำแหน่งปิดของแท่งสุดท้าย
        close = close.copy()
        close[-1] = low[-1] + cloc_last * (high[-1] - low[-1])
    openp = np.concatenate([[close[0]], close[:-1]])
    if vol is None:
        vol = np.full(n, 5_000_000.0)
    return pd.DataFrame(
        {"Open": openp, "High": np.maximum.reduce([high, openp, close]),
         "Low": np.minimum.reduce([low, openp, close]),
         "Close": close, "Volume": np.asarray(vol, dtype=float)},
        index=IDX[:n],
    )


n = 300

# ══ ฝั่งซื้อ ═══════════════════════════════════════════════════════
# 1. BREAKOUT: เทรนด์ขึ้นชัด + วันสุดท้ายทะลุ high 20 วันด้วยวอลุ่ม 3 เท่า
trend = 100 * np.exp(np.linspace(0, 0.55, n)) + rng.normal(0, 0.4, n)
trend[-1] = trend[-2] * 1.045
v1 = np.full(n, 6e6); v1[-1] = 2.1e7
brk = make(trend, v1, cloc_last=0.95)

# 2. VOLSPIKE: ขึ้นช้า ๆ แล้ววอลุ่มพุ่ง 4 เท่า +5% วันเดียว
slow = 60 + np.linspace(0, 9, n) + rng.normal(0, 0.3, n)
slow[-1] = slow[-2] * 1.055
v2 = np.full(n, 4e6); v2[-1] = 1.7e7
spike = make(slow, v2, cloc_last=0.88)

# 3. PULLBACK: ขาขึ้นยาว แล้วย่อแรง 5 วันสุดท้าย แต่ยังเหนือ SMA200
up = 80 * np.exp(np.linspace(0, 0.75, n - 5)) + rng.normal(0, 0.3, n - 5)
pull = make(np.concatenate([up, up[-1] * np.array([.975, .955, .935, .925, .915])]),
            np.full(n, 7e6), cloc_last=0.3)

# ══ ฝั่งชอร์ต (กลับด้านของข้างบน) ══════════════════════════════════
# 4. BREAKDOWN: เทรนด์ลงชัด + วันสุดท้ายหลุด low 20 วันด้วยวอลุ่ม 3 เท่า
dtrend = 200 * np.exp(np.linspace(0, -0.55, n)) + rng.normal(0, 0.4, n)
dtrend[-1] = dtrend[-2] * 0.955
v4 = np.full(n, 6e6); v4[-1] = 2.1e7
bkd = make(dtrend, v4, cloc_last=0.05)

# 5. DISTRIBUTION: ลงช้า ๆ แล้ววอลุ่มพุ่ง 4 เท่า -5% วันเดียว ปิดที่ก้นแท่ง
dslow = 120 - np.linspace(0, 18, n) + rng.normal(0, 0.3, n)
dslow[-1] = dslow[-2] * 0.945
v5 = np.full(n, 4e6); v5[-1] = 1.7e7
dist = make(dslow, v5, cloc_last=0.12)

# 6. RALLYFADE: ขาลงยาว แล้วเด้งแรง 5 วันสุดท้าย แต่ยังใต้ SMA200
dn_leg = 300 * np.exp(np.linspace(0, -0.85, n - 5)) + rng.normal(0, 0.3, n - 5)
fade = make(np.concatenate([dn_leg, dn_leg[-1] * np.array([1.03, 1.055, 1.075, 1.09, 1.10])]),
            np.full(n, 7e6), cloc_last=0.7)

# ══ ตัวควบคุม ══════════════════════════════════════════════════════
thin = make(trend, np.full(n, 20_000.0), cloc_last=0.95)         # สภาพคล่องต่ำ
flat = make(np.full(n, 50.0) + rng.normal(0, .02, n), np.full(n, 9e6), hl_pct=0.001)  # ไม่แกว่ง
chop = make(100 + np.cumsum(rng.normal(0, 0.5, n)), np.full(n, 8e6))  # sideways

cases = {"BRK": brk, "SPK": spike, "PUL": pull, "BKD": bkd, "DST": dist,
         "FADE": fade, "THIN": thin, "FLAT": flat, "CHOP": chop}
mets = {k: sc.compute_metrics(k, v) for k, v in cases.items()}
check("compute_metrics คำนวณได้ทุกตัว", all(m is not None for m in mets.values()))


def fired(m):
    return {name for name, fn in sc.SIGNALS if fn(m)}


# ── สัญญาณติดถูกตัว ──
print("\n— ฝั่งซื้อ —")
check("หุ้นทะลุแนวต้านติด BREAKOUT", "BREAKOUT" in fired(mets["BRK"]),
      f"ติด={fired(mets['BRK'])}")
check("หุ้นวอลุ่มพุ่งปิดแข็งติด VOLSPIKE", "VOLSPIKE" in fired(mets["SPK"]),
      f"ติด={fired(mets['SPK'])}")
check("หุ้นย่อในขาขึ้นติด PULLBACK", "PULLBACK" in fired(mets["PUL"]),
      f"ติด={fired(mets['PUL'])} rsi={mets['PUL']['rsi14']:.0f}")

print("\n— ฝั่งชอร์ต —")
check("หุ้นหลุดแนวรับติด BREAKDOWN", "BREAKDOWN" in fired(mets["BKD"]),
      f"ติด={fired(mets['BKD'])} rvol={mets['BKD']['rvol']:.1f} adx={mets['BKD']['adx14']:.0f}")
check("หุ้นวอลุ่มพุ่งปิดอ่อนติด DISTRIBUTION", "DISTRIBUTION" in fired(mets["DST"]),
      f"ติด={fired(mets['DST'])} rvol={mets['DST']['rvol']:.1f} ret1d={mets['DST']['ret1d']:.1f}")
check("หุ้นขาลงเด้งติด RALLYFADE", "RALLYFADE" in fired(mets["FADE"]),
      f"ติด={fired(mets['FADE'])} rsi={mets['FADE']['rsi14']:.0f} ret5d={mets['FADE']['ret5d']:.1f}")

print("\n— ไม่ข้ามฝั่ง —")
check("หุ้นขาขึ้นไม่ติดสัญญาณชอร์ตเลย",
      not any(sc.SETUP_SIDE[s] == "short" for s in fired(mets["BRK"]) | fired(mets["PUL"])),
      f"BRK={fired(mets['BRK'])} PUL={fired(mets['PUL'])}")
check("หุ้นขาลงไม่ติดสัญญาณซื้อเลย",
      not any(sc.SETUP_SIDE[s] == "long" for s in fired(mets["BKD"]) | fired(mets["FADE"])),
      f"BKD={fired(mets['BKD'])} FADE={fired(mets['FADE'])}")
check("หุ้น sideways ไม่ติดสัญญาณใดเลย", fired(mets["CHOP"]) == set(),
      f"ติด={fired(mets['CHOP'])}")

print("\n— ตัวกรองพื้นฐาน —")
cfg = dict(sc.CONFIG)
check("ตัดหุ้นวอลุ่มบาง", not sc.passes_liquidity(mets["THIN"], cfg),
      f"$vol={mets['THIN']['avg_dollar_vol']:,.0f}")
check("ตัดหุ้นที่แทบไม่แกว่ง", not sc.passes_liquidity(mets["FLAT"], cfg),
      f"ATR%={mets['FLAT']['atr_pct']:.2f}")
check("หุ้นสภาพคล่องดีผ่านตัวกรอง",
      all(sc.passes_liquidity(mets[k], cfg) for k in ("BRK", "SPK", "PUL", "BKD", "DST", "FADE")))

print("\n— แผนเทรด —")
pl = sc.trade_plan(mets["BRK"], "BREAKOUT", cfg)
check("LONG: stop ใต้ราคาเข้า, target เหนือราคาเข้า",
      pl["stop"] < pl["entry"] < pl["target"], f"{pl['stop']:.2f} < {pl['entry']:.2f} < {pl['target']:.2f}")
ps = sc.trade_plan(mets["BKD"], "BREAKDOWN", cfg)
check("SHORT: stop เหนือราคาเข้า, target ใต้ราคาเข้า",
      ps["target"] < ps["entry"] < ps["stop"], f"{ps['target']:.2f} < {ps['entry']:.2f} < {ps['stop']:.2f}")
check("ทั้งสองฝั่งได้ R:R = 2.0 เท่ากัน",
      abs(pl["rr"] - 2.0) < 1e-9 and abs(ps["rr"] - 2.0) < 1e-9,
      f"long={pl['rr']:.3f} short={ps['rr']:.3f}")
check("stop_pct ติดลบทั้งสองฝั่ง (= ขาดทุน)", pl["stop_pct"] < 0 and ps["stop_pct"] < 0,
      f"long={pl['stop_pct']:.2f}% short={ps['stop_pct']:.2f}%")
check("target_pct เป็นบวกทั้งสองฝั่ง (= กำไร)", pl["target_pct"] > 0 and ps["target_pct"] > 0,
      f"long=+{pl['target_pct']:.2f}% short=+{ps['target_pct']:.2f}%")

pf = sc.trade_plan(mets["FADE"], "RALLYFADE", cfg)
check("stop ของ RALLYFADE อยู่เหนือ high 5 วัน", pf["stop"] > mets["FADE"]["high5"],
      f"{pf['stop']:.2f} > {mets['FADE']['high5']:.2f}")
pp = sc.trade_plan(mets["PUL"], "PULLBACK", cfg)
check("stop ของ PULLBACK อยู่ใต้ low 5 วัน", pp["stop"] < mets["PUL"]["low5"],
      f"{pp['stop']:.2f} < {mets['PUL']['low5']:.2f}")

budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100
for nm, p in (("long", pl), ("short", ps), ("fade", pf), ("pull", pp)):
    risk = p["shares"] * abs(p["entry"] - p["stop"])
    check(f"ขนาดไม้ ({nm}) ไม่เกินงบความเสี่ยง", risk <= budget + 1e-6, f"${risk:.2f} <= ${budget:.2f}")

print("\n— คะแนน —")
all_scores = [(k, nmm, fn(m)[1]) for k, m in mets.items() for nmm, fn in sc.SIGNALS if fn(m)]
check("คะแนนทุกตัวอยู่ในช่วง 0-100", all(0 <= s <= 100 for _, _, s in all_scores),
      str([(k, nm2, round(s, 1)) for k, nm2, s in all_scores]))

print("\n— ตัวกรองวันประกาศงบ —")
TODAY = dt.date(2026, 9, 3)
check("งบพรุ่งนี้ = blackout",
      ea.earnings_verdict({"days_to": 1, "next": dt.date(2026, 9, 4), "days_since": None,
                           "last": None})[0] == "blackout")
check("งบวันนี้ = blackout",
      ea.earnings_verdict({"days_to": 0, "next": TODAY, "days_since": None, "last": None})[0] == "blackout")
check("งบอีก 5 วัน = blackout (ขอบพอดี)",
      ea.earnings_verdict({"days_to": 5, "next": dt.date(2026, 9, 8), "days_since": None,
                           "last": None})[0] == "blackout")
check("งบอีก 6 วัน = ผ่าน",
      ea.earnings_verdict({"days_to": 6, "next": dt.date(2026, 9, 9), "days_since": None,
                           "last": None})[0] == "clear")
check("เพิ่งประกาศงบเมื่อวาน = post",
      ea.earnings_verdict({"days_to": 90, "next": dt.date(2026, 12, 1), "days_since": 1,
                           "last": dt.date(2026, 9, 2)})[0] == "post")
check("ไม่มีข้อมูลงบ = unknown", ea.earnings_verdict(None)[0] == "unknown")
check("ปรับ blackout_days เป็น 10 แล้วงบอีก 8 วันโดนตัด",
      ea.earnings_verdict({"days_to": 8, "next": dt.date(2026, 9, 11), "days_since": None,
                           "last": None}, blackout_days=10)[0] == "blackout")

# cache ต้องอ่าน/เขียนได้และไม่ยิงเน็ตซ้ำ
import json, os, tempfile
tmp = os.path.join(tempfile.mkdtemp(), "c.json")
json.dump({"XYZ": {"fetched": TODAY.isoformat(), "next": "2026-10-01", "last": "2026-07-01"}},
          open(tmp, "w"))
calls = []
ea._fetch_one = lambda t, today: (calls.append(t), (None, None))[1]
info = ea.get_earnings_info(["XYZ"], today=TODAY, cache_path=tmp, verbose=False)
check("cache ที่ยังไม่หมดอายุไม่ยิงเน็ตซ้ำ", calls == [] and info["XYZ"]["days_to"] == 28,
      f"calls={calls} days_to={info['XYZ']['days_to']}")
json.dump({"XYZ": {"fetched": "2026-01-01", "next": None, "last": None}}, open(tmp, "w"))
ea.get_earnings_info(["XYZ"], today=TODAY, cache_path=tmp, verbose=False)
check("cache หมดอายุแล้วยิงใหม่", calls == ["XYZ"], f"calls={calls}")

print("\n— รายงาน —")
rows = [m for m in mets.values() if m]
reg = sc.market_regime({**cases, "SPY": brk}, rows)
check("market_regime คืนค่าครบ", all(k in reg for k in ("note", "spy_trend", "breadth", "favors")),
      f"trend={reg['spy_trend']} favors={reg['favors']}")
reg_bear = sc.market_regime({**cases, "SPY": bkd}, rows)
check("ตลาดขาลง regime บอกว่าเอื้อฝั่งชอร์ต", reg_bear["favors"] == "ฝั่งชอร์ต",
      f"{reg_bear['spy_trend']} -> {reg_bear['favors']}")

results = {k: [] for k, _ in sc.SIGNALS}
for k, m in mets.items():
    for nm2, fn in sc.SIGNALS:
        r = fn(m)
        if r:
            rr = dict(m)
            rr.update(setup=nm2, side=sc.SETUP_SIDE[nm2], score=r[1], why=r[2],
                      plan=sc.trade_plan(m, nm2, cfg),
                      earn_status="clear", earn_text="งบอีก 30 วัน")
            results[nm2].append(rr)
page = sc.build_html(results, reg, len(rows), 6, cfg, "2026-09-02", excluded=3)
check("HTML สร้างได้และมีทั้งสองฝั่ง",
      "ฝั่งซื้อ (Long)" in page and "ฝั่งขายชอร์ต (Short)" in page and len(page) > 6000,
      f"{len(page):,} ตัวอักษร")
open("/tmp/preview.html", "w").write(page)

print()
if failures:
    print(f"ไม่ผ่าน {len(failures)} ข้อ: {failures}")
    raise SystemExit(1)
print(f"ผ่านทุกข้อ ({len(all_scores)} สัญญาณที่ตรวจ)")

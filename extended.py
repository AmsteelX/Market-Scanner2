"""
extended.py - ราคานอกเวลาทำการ (pre-market / after-hours)

ทำไมต้องมี: รายงานคำนวณจากราคาปิดของวันทำการล่าสุด แต่ระหว่างที่คุณนอน
ราคาอาจวิ่งไปไกลแล้วในช่วงนอกเวลา ถ้าหุ้นที่ติดสัญญาณซื้อร่วง 6% หลังปิด
แผนเทรดทั้งแผนก็ใช้ไม่ได้แล้ว — โมดูลนี้เอาราคาล่าสุดจริง ๆ มาเทียบให้

ช่วงเวลาซื้อขายของตลาดสหรัฐ (เวลานิวยอร์ก):
    04:00-09:30  pre-market      ก่อนตลาดเปิด
    09:30-16:00  regular         เวลาทำการปกติ
    16:00-20:00  after-hours     หลังตลาดปิด
    นอกจากนี้    closed          ปิดสนิท ใช้ราคาล่าสุดที่มี

ดึงเฉพาะหุ้นที่ติดสัญญาณ (ไม่กี่สิบตัว) เพราะข้อมูลรายนาทีหนักกว่ารายวันมาก
"""
from __future__ import annotations

import datetime as dt

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - สำรองไว้เผื่อระบบไม่มีฐานข้อมูลไทม์โซน
    ET = None

# ขอบเขตของแต่ละช่วง (ชั่วโมง, นาที) ตามเวลานิวยอร์ก
PRE_OPEN = (4, 0)
REG_OPEN = (9, 30)
REG_CLOSE = (16, 0)
POST_CLOSE = (20, 0)

SESSION_TH = {
    "pre": "ก่อนเปิดตลาด",
    "regular": "ในเวลาทำการ",
    "post": "หลังตลาดปิด",
    "closed": "ตลาดปิดสนิท",
}


def _minutes(hm) -> int:
    return hm[0] * 60 + hm[1]


def session_of(ts) -> str:
    """บอกว่าเวลานี้อยู่ในช่วงไหนของวันซื้อขาย (รับ timestamp ที่มีไทม์โซน)"""
    if ts is None:
        return "closed"
    try:
        et = ts.tz_convert(ET) if ET is not None else ts
    except Exception:  # noqa: BLE001 - timestamp ไม่มีไทม์โซน
        et = ts

    if et.weekday() >= 5:
        return "closed"
    mins = et.hour * 60 + et.minute
    if _minutes(PRE_OPEN) <= mins < _minutes(REG_OPEN):
        return "pre"
    if _minutes(REG_OPEN) <= mins < _minutes(REG_CLOSE):
        return "regular"
    if _minutes(REG_CLOSE) <= mins < _minutes(POST_CLOSE):
        return "post"
    return "closed"


def current_session(now=None) -> str:
    """ตอนนี้ตลาดสหรัฐอยู่ในช่วงไหน — ใช้ตั้งชื่อรอบการสแกน"""
    import pandas as pd

    now = now or pd.Timestamp.now(tz="UTC")
    if getattr(now, "tzinfo", None) is None:
        now = pd.Timestamp(now).tz_localize("UTC")
    return session_of(now)


def get_extended_quotes(tickers, chunk: int = 40, verbose: bool = True) -> dict:
    """ดึงราคาล่าสุดรวมช่วงนอกเวลา
    คืน dict[ticker] -> {'price', 'session', 'ts', 'volume'} (ข้ามตัวที่ดึงไม่ได้)
    """
    import pandas as pd
    import yfinance as yf

    tickers = list(tickers)
    out: dict[str, dict] = {}

    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        try:
            raw = yf.download(
                batch, period="2d", interval="1m", prepost=True,
                group_by="ticker", progress=False, threads=True,
                auto_adjust=False, timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ! ดึงราคานอกเวลาไม่สำเร็จ: {e}")
            continue
        if raw is None or len(raw) == 0:
            continue

        for t in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.get_level_values(0):
                        df = raw[t]
                    elif t in raw.columns.get_level_values(1):
                        df = raw.xs(t, axis=1, level=1)
                    else:
                        continue
                else:
                    if len(batch) != 1:
                        continue
                    df = raw

                df = df.rename(columns={c: str(c).title() for c in df.columns})
                if "Close" not in df.columns:
                    continue
                s = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if len(s) == 0:
                    continue

                ts = s.index[-1]
                vol = 0.0
                if "Volume" in df.columns:
                    v = pd.to_numeric(df["Volume"], errors="coerce")
                    sess_now = session_of(ts)
                    # รวมวอลุ่มเฉพาะแท่งที่อยู่ในช่วงเดียวกันกับแท่งล่าสุด
                    same = [ix for ix in v.index[-480:] if session_of(ix) == sess_now]
                    vol = float(v.loc[same].fillna(0).sum()) if same else 0.0

                out[t] = {
                    "price": float(s.iloc[-1]),
                    "session": session_of(ts),
                    "ts": ts,
                    "volume": vol,
                }
            except Exception:  # noqa: BLE001
                continue

    if verbose:
        print(f"  ได้ราคานอกเวลา {len(out)}/{len(tickers)} ตัว")
    return out


def classify_gap(chg_pct, side: str, threshold: float = 3.0) -> tuple[str, str]:
    """ตัดสินว่าการเคลื่อนไหวนอกเวลากระทบแผนเทรดยังไง
    คืน (สถานะ, ข้อความ) — สถานะ: 'ok' | 'against' | 'ran' | 'unknown'
      against = วิ่งสวนทางที่เราจะเทรด (แผนเสีย ควรข้าม)
      ran     = วิ่งไปทางเดียวกับเราแรงเกิน (ราคาเข้าเดิมใช้ไม่ได้ ไล่ราคาเสี่ยง)
    """
    if chg_pct is None or chg_pct != chg_pct:  # None หรือ NaN
        return "unknown", "—"

    favorable = chg_pct > 0 if side == "long" else chg_pct < 0
    size = abs(chg_pct)

    if size < threshold:
        return "ok", f"{chg_pct:+.1f}%"
    if favorable:
        return "ran", f"{chg_pct:+.1f}% วิ่งหนีไปแล้ว"
    return "against", f"{chg_pct:+.1f}% สวนทาง"


def attach_to_signals(hits, quotes: dict, threshold: float = 3.0) -> int:
    """ใส่ข้อมูลนอกเวลาลงในผลสัญญาณแต่ละรายการ คืนจำนวนที่ต้องเตือน"""
    alerts = 0
    for r in hits:
        q = quotes.get(r["ticker"])
        if not q or not r.get("close"):
            r["ext_price"] = None
            r["ext_chg"] = None
            r["ext_session"] = None
            r["ext_status"], r["ext_text"] = "unknown", "—"
            continue

        chg = 100.0 * (q["price"] - r["close"]) / r["close"]
        status, text = classify_gap(chg, r["plan"]["side"], threshold)
        r["ext_price"] = q["price"]
        r["ext_chg"] = chg
        r["ext_session"] = q["session"]
        r["ext_volume"] = q.get("volume", 0.0)
        r["ext_status"], r["ext_text"] = status, text
        if status in ("against", "ran"):
            alerts += 1
    return alerts

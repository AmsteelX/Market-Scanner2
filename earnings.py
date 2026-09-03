"""
earnings.py - หาวันประกาศงบของแต่ละหุ้น เพื่อกันไม่ให้เข้าไม้ก่อนงบ

ทำไมต้องกรอง: หุ้นที่วอลุ่มพุ่งผิดปกติส่วนใหญ่คือวันประกาศงบหรือวันก่อนหน้า
การถือข้ามคืนช่วงงบ = เดิมพันกับ gap ที่จุดตัดขาดทุนคุมไม่ได้เลย
(ตื่นมาราคาเปิดต่ำกว่าจุดตัดขาดทุน 15% ก็เกิดขึ้นบ่อย)

ดึงเฉพาะหุ้นที่ติดสัญญาณเท่านั้น (ไม่กี่สิบตัว) ไม่ได้ดึงทั้ง 600 ตัว
เพื่อไม่ให้โดน rate limit ของ Yahoo และเก็บ cache ไว้ 3 วัน
"""
from __future__ import annotations

import datetime as dt
import json
import os

CACHE_FILE = ".earnings_cache.json"
CACHE_TTL_DAYS = 3


def _load_cache(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(path: str, cache: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
    except Exception:  # noqa: BLE001
        pass


def _to_date(x):
    """แปลงค่าที่ yfinance คืนมาหลายรูปแบบให้เป็น datetime.date"""
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    try:
        import pandas as pd

        ts = pd.Timestamp(x)
        return None if pd.isna(ts) else ts.date()
    except Exception:  # noqa: BLE001
        return None


def _fetch_one(ticker: str, today: dt.date):
    """คืนวันประกาศงบครั้งถัดไป และครั้งล่าสุดที่ผ่านมา (อาจเป็น None ถ้าหาไม่เจอ)"""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    dates: list[dt.date] = []

    # ทางที่ 1: calendar (เร็วที่สุด มีเฉพาะงบครั้งถัดไป)
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("Earnings Date High")
            if raw is not None:
                for d in (raw if isinstance(raw, (list, tuple)) else [raw]):
                    d = _to_date(d)
                    if d:
                        dates.append(d)
        elif cal is not None and hasattr(cal, "loc"):  # DataFrame แบบเก่า
            for d in list(cal.loc["Earnings Date"]) if "Earnings Date" in cal.index else []:
                d = _to_date(d)
                if d:
                    dates.append(d)
    except Exception:  # noqa: BLE001
        pass

    # ทางที่ 2: get_earnings_dates (มีทั้งอดีตและอนาคต)
    try:
        df = tk.get_earnings_dates(limit=12)
        if df is not None and len(df) > 0:
            for idx in df.index:
                d = _to_date(idx)
                if d:
                    dates.append(d)
    except Exception:  # noqa: BLE001
        pass

    if not dates:
        return None, None
    dates = sorted(set(dates))
    future = [d for d in dates if d >= today]
    past = [d for d in dates if d < today]
    return (future[0] if future else None), (past[-1] if past else None)


def get_earnings_info(tickers, today: dt.date | None = None,
                      cache_path: str = CACHE_FILE, verbose: bool = True) -> dict:
    """คืน dict[ticker] -> {'next': date|None, 'last': date|None, 'days_to': int|None}"""
    today = today or dt.date.today()
    cache = _load_cache(cache_path)
    out: dict[str, dict] = {}
    fetched = 0

    for t in tickers:
        entry = cache.get(t)
        fresh = False
        if entry:
            try:
                age = (today - dt.date.fromisoformat(entry["fetched"])).days
                fresh = 0 <= age <= CACHE_TTL_DAYS
            except Exception:  # noqa: BLE001
                fresh = False

        if fresh:
            nxt = dt.date.fromisoformat(entry["next"]) if entry.get("next") else None
            lst = dt.date.fromisoformat(entry["last"]) if entry.get("last") else None
        else:
            try:
                nxt, lst = _fetch_one(t, today)
            except Exception:  # noqa: BLE001
                nxt, lst = None, None
            fetched += 1
            cache[t] = {
                "fetched": today.isoformat(),
                "next": nxt.isoformat() if nxt else None,
                "last": lst.isoformat() if lst else None,
            }

        out[t] = {
            "next": nxt,
            "last": lst,
            "days_to": (nxt - today).days if nxt else None,
            "days_since": (today - lst).days if lst else None,
        }

    _save_cache(cache_path, cache)
    if verbose and fetched:
        print(f"  ดึงวันประกาศงบใหม่ {fetched} ตัว (ที่เหลือใช้ cache)")
    return out


def earnings_verdict(info: dict | None, blackout_days: int = 5,
                     post_days: int = 1) -> tuple[str, str]:
    """ตัดสินว่าหุ้นตัวนี้ควรเข้าไม้ไหม
    คืน (สถานะ, ข้อความ) โดยสถานะเป็น 'clear' | 'blackout' | 'post' | 'unknown'
    """
    if not info:
        return "unknown", "ไม่ทราบวันงบ"

    d = info.get("days_to")
    ds = info.get("days_since")

    if d is not None and 0 <= d <= blackout_days:
        when = "วันนี้" if d == 0 else f"อีก {d} วัน"
        return "blackout", f"งบ{when} ({info['next']:%d %b})"
    if ds is not None and 0 <= ds <= post_days:
        return "post", f"เพิ่งประกาศงบ ({info['last']:%d %b})"
    if d is not None:
        return "clear", f"งบอีก {d} วัน"
    return "unknown", "ไม่ทราบวันงบ"

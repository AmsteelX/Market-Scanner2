#!/usr/bin/env python3
"""
scanner.py - Market Signal Scanner สำหรับหุ้นสหรัฐ (เทรดสั้น 1-10 วัน)

สแกน 6 รูปแบบสัญญาณ แยกเป็นฝั่งซื้อและฝั่งขายชอร์ต

  ฝั่งซื้อ (LONG)
    A) BREAKOUT      - ทะลุ high 20 วันพร้อมวอลุ่มและเทรนด์หนุน
    B) VOLSPIKE      - วอลุ่มพุ่งผิดปกติพร้อมราคาปิดแข็ง
    C) PULLBACK      - หุ้นขาขึ้นย่อลึกชั่วคราว (mean reversion)

  ฝั่งขายชอร์ต (SHORT)
    D) BREAKDOWN     - หลุด low 20 วันพร้อมวอลุ่มและเทรนด์ลงหนุน
    E) DISTRIBUTION  - วอลุ่มพุ่งผิดปกติพร้อมราคาปิดอ่อน (แรงขาย)
    F) RALLYFADE     - หุ้นขาลงเด้งจนซื้อมากเกินไป (mean reversion ฝั่งชอร์ต)

กรองวันประกาศงบให้อัตโนมัติ (earnings.py)
เทียบกับราคานอกเวลาทำการ pre-market / after-hours และเตือนเมื่อ gap แรง (extended.py)

ผลลัพธ์: ไฟล์ CSV + หน้า HTML dashboard + สรุปบนหน้าจอ

รันแบบง่ายสุด:   python scanner.py
ดูตัวเลือกทั้งหมด: python scanner.py --help

*** ไม่ใช่คำแนะนำการลงทุน — เป็นเครื่องมือคัดกรองเพื่อไปดูกราฟต่อเท่านั้น ***
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import sys
import warnings

import numpy as np
import pandas as pd

import indicators as ind
from earnings import earnings_verdict, get_earnings_info
from extended import SESSION_TH, attach_to_signals, current_session, get_extended_quotes
from universe import get_universe

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  ตั้งค่าได้ตรงนี้
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "universe": "both",        # 'sp500' | 'nasdaq100' | 'both' | 'fallback'
    "history_days": 400,       # ดึงย้อนหลังกี่วันปฏิทิน (ต้อง >300 เพื่อคำนวณ SMA200)
    "min_price": 5.0,          # ตัดหุ้นราคาต่ำ (สเปรดกว้าง ปั่นง่าย)
    "max_price": 2000.0,
    "min_dollar_vol": 20e6,    # มูลค่าซื้อขายเฉลี่ย 20 วัน ต้อง >= 20 ล้านดอลลาร์
    "min_atr_pct": 1.2,        # ต้องแกว่งพอให้เทรดสั้นมีกำไร (ATR >= 1.2% ของราคา)
    "max_atr_pct": 15.0,       # แต่ไม่บ้าคลั่งเกินไป
    "top_n": 15,               # แสดงกี่ตัวต่อสัญญาณ
    "sides": "both",           # 'long' | 'short' | 'both'
    # ── ตัวกรองวันประกาศงบ ──
    "earnings_mode": "exclude",   # 'exclude' = ตัดทิ้ง | 'flag' = แสดงแต่ติดป้าย | 'off' = ไม่เช็ก
    "earnings_blackout_days": 5,  # ห้ามเข้าไม้ถ้างบจะออกภายในกี่วัน
    "earnings_post_days": 1,      # ติดป้าย "เพิ่งประกาศงบ" ถ้างบออกไปแล้วไม่เกินกี่วัน
    "earnings_max_lookup": 150,   # จำกัดจำนวนการเรียก API กันโดน rate limit
    # ── ราคานอกเวลาทำการ (pre-market / after-hours) ──
    "extended": True,             # ดึงราคานอกเวลามาเทียบกับราคาปิดไหม
    "gap_alert_pct": 3.0,         # ขยับเกินกี่ % ถึงจะเตือน
    # ── การจัดการความเสี่ยง ──
    "account_size": 10000.0,
    "risk_per_trade_pct": 1.0,  # เสี่ยงกี่ % ของพอร์ตต่อไม้
    "stop_atr_mult": 1.5,
    "target_atr_mult": 3.0,
}

SETUP_LABEL = {
    "BREAKOUT":     "A · Breakout — ทะลุแนวต้าน 20 วัน",
    "VOLSPIKE":     "B · Volume Spike — วอลุ่มพุ่ง ปิดแข็ง",
    "PULLBACK":     "C · Pullback — ย่อในขาขึ้น (mean reversion)",
    "BREAKDOWN":    "D · Breakdown — หลุดแนวรับ 20 วัน",
    "DISTRIBUTION": "E · Distribution — วอลุ่มพุ่ง ปิดอ่อน (แรงขาย)",
    "RALLYFADE":    "F · Rally Fade — เด้งจนซื้อมากเกินในขาลง",
}
SETUP_SIDE = {
    "BREAKOUT": "long", "VOLSPIKE": "long", "PULLBACK": "long",
    "BREAKDOWN": "short", "DISTRIBUTION": "short", "RALLYFADE": "short",
}


# ═══════════════════════════════════════════════════════════════════
#  ดึงข้อมูล
# ═══════════════════════════════════════════════════════════════════
def download_prices(tickers: list[str], days: int, chunk: int = 100) -> dict[str, pd.DataFrame]:
    """ดึง OHLCV รายวันเป็นชุด ๆ คืน dict[ticker] -> DataFrame"""
    import yfinance as yf

    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days)
    data: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        print(f"  ดึงข้อมูล {i + 1}-{i + len(batch)} จาก {len(tickers)} ...", flush=True)
        try:
            raw = yf.download(
                batch, start=start, end=end, interval="1d",
                group_by="ticker", auto_adjust=False, threads=True,
                progress=False, timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ! ชุดนี้ดึงไม่สำเร็จ: {e}")
            continue
        if raw is None or raw.empty:
            continue

        for t in batch:
            df = _extract(raw, t, single=len(batch) == 1)
            if df is not None and len(df) >= 220:
                data[t] = df

    return data


def _extract(raw: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame | None:
    """ดึง OHLCV ของหุ้นตัวเดียวออกจากผลลัพธ์ของ yf.download (รองรับหลายรูปแบบคอลัมน์)"""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                df = raw[ticker].copy()
            elif ticker in raw.columns.get_level_values(1):
                df = raw.xs(ticker, axis=1, level=1).copy()
            else:
                return None
        else:
            if not single:
                return None
            df = raw.copy()

        df.columns = [str(c).title() for c in df.columns]
        need = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in need):
            return None
        df = df[need].apply(pd.to_numeric, errors="coerce").dropna(how="any")
        return df if len(df) > 0 else None
    except Exception:  # noqa: BLE001
        return None


# ═══════════════════════════════════════════════════════════════════
#  คำนวณตัวชี้วัดของหุ้นแต่ละตัว (เอาเฉพาะแถวล่าสุด)
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(t: str, df: pd.DataFrame) -> dict | None:
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    if len(df) < 220:
        return None

    ema20, ema50, sma200 = ind.ema(c, 20), ind.ema(c, 50), ind.sma(c, 200)
    m = {
        "ticker": t,
        "date": df.index[-1].date().isoformat(),
        "close": c.iloc[-1],
        "open": o.iloc[-1],
        "high": h.iloc[-1],
        "low": l.iloc[-1],
        "volume": v.iloc[-1],
        "ema20": ema20.iloc[-1],
        "ema50": ema50.iloc[-1],
        "sma200": sma200.iloc[-1],
        "rsi14": ind.rsi(c, 14).iloc[-1],
        "adx14": ind.adx(h, l, c, 14).iloc[-1],
        "atr14": ind.atr(h, l, c, 14).iloc[-1],
        "pctb": ind.bollinger_pctb(c, 20, 2.0).iloc[-1],
        "bbw": ind.bollinger_bandwidth(c, 20, 2.0).iloc[-1],
        "rvol": ind.rvol(v, 20).iloc[-1],
        "cloc": ind.close_location(h, l, c).iloc[-1],
        "ret1d": ind.pct_change_n(c, 1).iloc[-1],
        "ret5d": ind.pct_change_n(c, 5).iloc[-1],
        "ret20d": ind.pct_change_n(c, 20).iloc[-1],
        "high20": ind.rolling_high(h, 20).iloc[-1],
        "low20": ind.rolling_low(l, 20).iloc[-1],
        "high5": ind.rolling_high(h, 5).iloc[-1],
        "low5": ind.rolling_low(l, 5).iloc[-1],
        "high52w": ind.rolling_high(h, 252).iloc[-1],
        "low52w": ind.rolling_low(l, 252).iloc[-1],
        "avg_dollar_vol": (c * v).rolling(20).mean().iloc[-1],
    }

    if not np.isfinite(m["close"]) or m["close"] <= 0:
        return None

    m["atr_pct"] = 100.0 * m["atr14"] / m["close"]
    m["dist_high20_pct"] = 100.0 * (m["high20"] - m["close"]) / m["close"]
    m["dist_52wh_pct"] = 100.0 * (m["high52w"] - m["close"]) / m["close"]
    m["dist_52wl_pct"] = 100.0 * (m["close"] - m["low52w"]) / m["close"]
    m["above_sma200"] = bool(m["close"] > m["sma200"]) if np.isfinite(m["sma200"]) else False
    m["above_ema50"] = bool(m["close"] > m["ema50"]) if np.isfinite(m["ema50"]) else False
    _bbw_rank = ind.bollinger_bandwidth(c, 20, 2.0).tail(126).rank(pct=True).iloc[-1]
    m["bbw_pctile"] = float(_bbw_rank * 100) if np.isfinite(_bbw_rank) else np.nan
    m["gap_pct"] = 100.0 * (m["open"] - c.iloc[-2]) / c.iloc[-2]
    return m


def passes_liquidity(m: dict, cfg: dict) -> bool:
    return (
        cfg["min_price"] <= m["close"] <= cfg["max_price"]
        and np.isfinite(m["avg_dollar_vol"])
        and m["avg_dollar_vol"] >= cfg["min_dollar_vol"]
        and np.isfinite(m["atr_pct"])
        and cfg["min_atr_pct"] <= m["atr_pct"] <= cfg["max_atr_pct"]
    )


# ═══════════════════════════════════════════════════════════════════
#  ตรรกะสัญญาณ — ฝั่งซื้อ
# ═══════════════════════════════════════════════════════════════════
def _clamp(x, lo=0.0, hi=1.0):
    if x is None or not np.isfinite(x):
        return lo
    return float(min(max(x, lo), hi))


def signal_breakout(m: dict):
    """ทะลุ high 20 วัน + เทรนด์เรียงตัว + วอลุ่มหนุน"""
    ok = (
        m["close"] >= m["high20"] * 0.995
        and m["close"] > m["ema20"] > m["ema50"]
        and m["above_sma200"]
        and m["rvol"] >= 1.3
        and m["adx14"] >= 20
        and m["ret20d"] > 0
        and m["cloc"] >= 0.5
    )
    if not ok:
        return None

    score = (
        25 * _clamp((m["rvol"] - 1.0) / 2.0)           # แรงวอลุ่ม
        + 25 * _clamp(m["adx14"] / 40.0)                # ความแรงเทรนด์
        + 20 * _clamp(m["ret20d"] / 25.0)               # โมเมนตัม 1 เดือน
        + 15 * _clamp(m["cloc"])                        # ปิดใกล้ high ของวัน
        + 15 * _clamp(1.0 - m["dist_52wh_pct"] / 20.0)  # ใกล้จุดสูงสุด 52 สัปดาห์
    )
    if m["ret20d"] > 40:   # วิ่งมาไกลเกินไปแล้ว ไล่ราคาเสี่ยง
        score *= 0.8
    if m["rsi14"] > 80:
        score *= 0.85

    why = (f"ปิดที่/เหนือ high 20 วัน · RVOL {m['rvol']:.1f}x · ADX {m['adx14']:.0f} "
           f"· +{m['ret20d']:.1f}% ใน 20 วัน")
    return True, score, why


def signal_volspike(m: dict):
    """วอลุ่มพุ่งผิดปกติพร้อมราคาปิดแข็ง"""
    ok = (
        m["rvol"] >= 2.0
        and m["ret1d"] > 1.0
        and m["cloc"] >= 0.6
        and m["close"] * m["volume"] >= 30e6
        and m["above_ema50"]
    )
    if not ok:
        return None

    score = (
        35 * _clamp((m["rvol"] - 2.0) / 4.0)
        + 20 * _clamp(m["cloc"])
        + 20 * _clamp(m["ret1d"] / 8.0)
        + 15 * _clamp(m["adx14"] / 35.0)
        + 10 * (1.0 if m["above_sma200"] else 0.0)
    )
    if m["ret5d"] > 20:   # เพิ่งวิ่งไปหลายวันแล้ว ของใหม่น้อยลง
        score *= 0.85

    why = (f"วอลุ่ม {m['rvol']:.1f} เท่าปกติ · +{m['ret1d']:.1f}% วันเดียว "
           f"· ปิดที่ {m['cloc'] * 100:.0f}% ของแท่ง")
    return True, score, why


def signal_pullback(m: dict):
    """หุ้นขาขึ้นย่อลึกชั่วคราว — เล่นเด้ง"""
    oversold = (m["rsi14"] <= 38) or (np.isfinite(m["pctb"]) and m["pctb"] <= 0.10)
    ok = (
        m["above_sma200"]
        and m["dist_52wh_pct"] <= 30
        and oversold
        and m["ret5d"] < 0
    )
    if not ok:
        return None

    score = (
        30 * _clamp((40.0 - m["rsi14"]) / 25.0)                        # ยิ่ง oversold ยิ่งได้คะแนน
        + 25 * _clamp((0.15 - (m["pctb"] if np.isfinite(m["pctb"]) else 0.15)) / 0.25)
        + 25 * _clamp((m["close"] / m["sma200"] - 1.0) / 0.25)         # เทรนด์ใหญ่ยังแข็ง
        + 20 * _clamp(1.0 - m["dist_52wh_pct"] / 30.0)                 # ไม่ได้พังลงมาไกล
    )
    if m["ret5d"] < -15:   # ย่อแรงเกินไป อาจเป็นข่าวร้ายไม่ใช่ย่อปกติ
        score *= 0.7

    why = f"RSI {m['rsi14']:.0f} · %B {m['pctb']:.2f} · {m['ret5d']:.1f}% ใน 5 วัน แต่ยังเหนือ SMA200"
    return True, score, why


# ═══════════════════════════════════════════════════════════════════
#  ตรรกะสัญญาณ — ฝั่งขายชอร์ต (กลับด้านของฝั่งซื้อ)
# ═══════════════════════════════════════════════════════════════════
def signal_breakdown(m: dict):
    """หลุด low 20 วัน + เทรนด์เรียงลง + วอลุ่มหนุน"""
    ok = (
        m["close"] <= m["low20"] * 1.005
        and m["close"] < m["ema20"] < m["ema50"]
        and not m["above_sma200"]
        and m["rvol"] >= 1.3
        and m["adx14"] >= 20
        and m["ret20d"] < 0
        and m["cloc"] <= 0.5
    )
    if not ok:
        return None

    score = (
        25 * _clamp((m["rvol"] - 1.0) / 2.0)
        + 25 * _clamp(m["adx14"] / 40.0)
        + 20 * _clamp(-m["ret20d"] / 25.0)
        + 15 * _clamp(1.0 - m["cloc"])                   # ปิดใกล้ low ของวัน
        + 15 * _clamp(1.0 - m["dist_52wl_pct"] / 20.0)   # ใกล้จุดต่ำสุด 52 สัปดาห์
    )
    if m["ret20d"] < -40:   # ร่วงมาเยอะแล้ว เสี่ยงเด้งแรง
        score *= 0.8
    if m["rsi14"] < 20:
        score *= 0.85

    why = (f"ปิดที่/ใต้ low 20 วัน · RVOL {m['rvol']:.1f}x · ADX {m['adx14']:.0f} "
           f"· {m['ret20d']:.1f}% ใน 20 วัน")
    return True, score, why


def signal_distribution(m: dict):
    """วอลุ่มพุ่งผิดปกติพร้อมราคาปิดอ่อน — แรงขายเข้ามาจริง"""
    ok = (
        m["rvol"] >= 2.0
        and m["ret1d"] < -1.0
        and m["cloc"] <= 0.4
        and m["close"] * m["volume"] >= 30e6
        and not m["above_ema50"]
    )
    if not ok:
        return None

    score = (
        35 * _clamp((m["rvol"] - 2.0) / 4.0)
        + 20 * _clamp(1.0 - m["cloc"])
        + 20 * _clamp(-m["ret1d"] / 8.0)
        + 15 * _clamp(m["adx14"] / 35.0)
        + 10 * (0.0 if m["above_sma200"] else 1.0)
    )
    if m["ret5d"] < -20:   # ร่วงมาหลายวันแล้ว เสี่ยงเด้ง
        score *= 0.85

    why = (f"วอลุ่ม {m['rvol']:.1f} เท่าปกติ · {m['ret1d']:.1f}% วันเดียว "
           f"· ปิดที่ {m['cloc'] * 100:.0f}% ของแท่ง")
    return True, score, why


def signal_rallyfade(m: dict):
    """หุ้นขาลงเด้งจนซื้อมากเกินไป — ชอร์ตตอนเด้ง ไม่ใช่ไล่ชอร์ตตอนร่วง"""
    overbought = (m["rsi14"] >= 62) or (np.isfinite(m["pctb"]) and m["pctb"] >= 0.90)
    ok = (
        not m["above_sma200"]
        and m["dist_52wl_pct"] <= 30      # ยังอยู่ใกล้จุดต่ำสุด = ยังอ่อนแอจริง
        and overbought
        and m["ret5d"] > 0
    )
    if not ok:
        return None

    pctb = m["pctb"] if np.isfinite(m["pctb"]) else 0.85
    score = (
        30 * _clamp((m["rsi14"] - 60.0) / 25.0)                        # ยิ่ง overbought ยิ่งได้คะแนน
        + 25 * _clamp((pctb - 0.85) / 0.25)
        + 25 * _clamp((1.0 - m["close"] / m["sma200"]) / 0.25)         # เทรนด์ใหญ่ยังอ่อน
        + 20 * _clamp(1.0 - m["dist_52wl_pct"] / 30.0)
    )
    if m["ret5d"] > 15:   # เด้งแรงเกินไป อาจเป็นการกลับตัวจริง
        score *= 0.7

    why = f"RSI {m['rsi14']:.0f} · %B {m['pctb']:.2f} · +{m['ret5d']:.1f}% ใน 5 วัน แต่ยังใต้ SMA200"
    return True, score, why


SIGNALS = [
    ("BREAKOUT", signal_breakout),
    ("VOLSPIKE", signal_volspike),
    ("PULLBACK", signal_pullback),
    ("BREAKDOWN", signal_breakdown),
    ("DISTRIBUTION", signal_distribution),
    ("RALLYFADE", signal_rallyfade),
]


# ═══════════════════════════════════════════════════════════════════
#  แผนเทรด (รองรับทั้งสองฝั่ง)
# ═══════════════════════════════════════════════════════════════════
def trade_plan(m: dict, setup: str, cfg: dict) -> dict:
    """จุดเข้า/ตัดขาดทุน/เป้าหมาย และขนาดโพสิชันตามความเสี่ยงที่ยอมรับได้
    ฝั่งชอร์ตกลับด้าน: stop อยู่เหนือราคา เป้าหมายอยู่ใต้ราคา
    """
    entry = m["close"]
    atr = m["atr14"]
    side = SETUP_SIDE[setup]
    k, tmul = cfg["stop_atr_mult"], cfg["target_atr_mult"]

    if side == "long":
        if setup == "PULLBACK":
            # ย่อในขาขึ้น: วางสต็อปใต้ low 5 วัน เผื่อ noise ครึ่ง ATR
            stop = min(m["low5"] - 0.5 * atr, entry - k * atr)
        else:
            stop = entry - k * atr
        target = entry + tmul * atr
        risk_per_share = max(entry - stop, 0.01)
        reward = target - entry
    else:
        if setup == "RALLYFADE":
            # เด้งในขาลง: วางสต็อปเหนือ high 5 วัน เผื่อ noise ครึ่ง ATR
            stop = max(m["high5"] + 0.5 * atr, entry + k * atr)
        else:
            stop = entry + k * atr
        target = entry - tmul * atr
        risk_per_share = max(stop - entry, 0.01)
        reward = entry - target

    risk_budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100.0
    shares = int(risk_budget // risk_per_share)

    return {
        "side": side,
        "entry": entry,
        "stop": stop,
        "target": target,
        "stop_pct": -100.0 * risk_per_share / entry,   # ติดลบเสมอ = ขาดทุนถ้าโดนสต็อป
        "target_pct": 100.0 * reward / entry,          # เป็นบวกเสมอ = กำไรถ้าถึงเป้า
        "rr": reward / risk_per_share,
        "shares": shares,
        "position_value": shares * entry,
    }


# ═══════════════════════════════════════════════════════════════════
#  สภาพตลาดโดยรวม (regime)
# ═══════════════════════════════════════════════════════════════════
def market_regime(data: dict[str, pd.DataFrame], rows: list[dict]) -> dict:
    reg = {"note": "ไม่มีข้อมูล SPY", "spy_trend": "n/a", "breadth": np.nan,
           "breadth_200": np.nan, "risk_on": None, "favors": "ทั้งสองฝั่ง"}

    pool = [r for r in rows if np.isfinite(r.get("ema50", np.nan))]
    if pool:
        reg["breadth"] = 100.0 * sum(1 for r in pool if r["above_ema50"]) / len(pool)
        reg["breadth_200"] = 100.0 * sum(1 for r in pool if r["above_sma200"]) / len(pool)

    spy = data.get("SPY")
    if spy is not None and len(spy) >= 220:
        c = spy["Close"]
        e50, s200 = ind.ema(c, 50).iloc[-1], ind.sma(c, 200).iloc[-1]
        px = c.iloc[-1]
        above50, above200 = px > e50, px > s200
        reg["spy_close"] = px
        reg["spy_above_50"] = bool(above50)
        reg["spy_above_200"] = bool(above200)
        reg["spy_ret20"] = ind.pct_change_n(c, 20).iloc[-1]
        if above50 and above200:
            reg["spy_trend"], reg["risk_on"] = "ขาขึ้น", True
        elif above200:
            reg["spy_trend"], reg["risk_on"] = "ขาขึ้นแต่ย่อ (ใต้ EMA50)", None
        else:
            reg["spy_trend"], reg["risk_on"] = "อ่อนแอ (ใต้ SMA200)", False

    br = reg.get("breadth", np.nan)
    if reg["risk_on"] is True and np.isfinite(br) and br >= 50:
        reg["favors"] = "ฝั่งซื้อ"
        reg["note"] = ("ตลาดหนุนฝั่งซื้อ — สัญญาณ breakout มีโอกาสไปต่อสูงกว่าปกติ "
                       "ส่วนสัญญาณชอร์ตในตลาดแบบนี้โดนบีบกลับบ่อย ควรลดขนาดหรือข้าม")
    elif reg["risk_on"] is False:
        reg["favors"] = "ฝั่งชอร์ต"
        reg["note"] = ("ตลาดอ่อนแอ — breakout ฝั่งซื้อล้มเหลวบ่อย "
                       "สัญญาณฝั่งชอร์ตมีโอกาสทำงานดีกว่า แต่ตลาดขาลงเด้งแรงและเร็ว ต้องคุมสต็อปเคร่ง")
    else:
        reg["favors"] = "ทั้งสองฝั่ง (ระวัง)"
        reg["note"] = ("ตลาดก้ำกึ่ง — ทั้งสองฝั่งมีสัญญาณหลอกเยอะ "
                       "เลือกเฉพาะคะแนนสูงและลดขนาดไม้ลง")
    return reg


# ═══════════════════════════════════════════════════════════════════
#  รายงาน
# ═══════════════════════════════════════════════════════════════════
def _fmt(x, nd=2, dash="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:,.{nd}f}"


_EARN_CLASS = {"blackout": "warn", "post": "warn", "clear": "muted", "unknown": "muted"}
_EXT_CLASS = {"against": "bad", "ran": "warn", "ok": "muted", "unknown": "muted"}


def build_html(results: dict[str, list[dict]], reg: dict, scanned: int, passed: int,
               cfg: dict, asof: str, excluded: int = 0, sess: str = "closed",
               alerts: int = 0) -> str:
    def rows_html(rows):
        if not rows:
            return '<tr><td colspan="13" class="empty">ไม่มีหุ้นเข้าเงื่อนไขนี้วันนี้</td></tr>'
        out = []
        for r in rows:
            p = r["plan"]
            est, etxt = r.get("earn_status", "unknown"), r.get("earn_text", "—")
            out.append(f"""<tr>
  <td class="tk">{html.escape(r['ticker'])}</td>
  <td><span class="score" style="--w:{min(r['score'], 100):.0f}%">{r['score']:.0f}</span></td>
  <td class="num">{_fmt(r['close'])}</td>
  <td class="num {'up' if r['ret1d'] >= 0 else 'dn'}">{r['ret1d']:+.1f}%</td>
  <td class="num">{_fmt(r['rvol'], 1)}x</td>
  <td class="num">{_fmt(r['rsi14'], 0)}</td>
  <td class="num">{_fmt(r['atr_pct'], 1)}%</td>
  <td class="num">{_fmt(p['entry'])}</td>
  <td class="num dn">{_fmt(p['stop'])}<span class="sub">{p['stop_pct']:.1f}%</span></td>
  <td class="num up">{_fmt(p['target'])}<span class="sub">+{p['target_pct']:.1f}%</span></td>
  <td class="num">{p['shares']:,}</td>
  <td class="earn {_EARN_CLASS.get(est, 'muted')}">{html.escape(etxt)}</td>
  <td class="ext {_EXT_CLASS.get(r.get('ext_status', 'unknown'), 'muted')}">{html.escape(r.get('ext_text', '—'))}
    {f'<span class="sub">{html.escape(SESSION_TH.get(r.get("ext_session") or "", ""))}</span>' if r.get('ext_session') else ''}</td>
</tr>""")
        return "\n".join(out)

    def section(key):
        rows = results.get(key, [])
        return f"""
<section class="card">
  <h2>{html.escape(SETUP_LABEL[key])} <span class="count">{len(rows)} ตัว</span></h2>
  <div class="scroll"><table>
    <thead><tr>
      <th>หุ้น</th><th>คะแนน</th><th>ราคา</th><th>1 วัน</th><th>RVOL</th><th>RSI</th>
      <th>ATR%</th><th>จุดเข้า</th><th>ตัดขาดทุน</th><th>เป้าหมาย</th><th>จำนวนหุ้น</th>
      <th>งบ</th><th>นอกเวลา</th>
    </tr></thead>
    <tbody>{rows_html(rows)}</tbody>
  </table></div>
</section>"""

    longs = [k for k in SETUP_LABEL if SETUP_SIDE[k] == "long" and k in results]
    shorts = [k for k in SETUP_LABEL if SETUP_SIDE[k] == "short" and k in results]

    blocks = ""
    if longs:
        n = sum(len(results[k]) for k in longs)
        blocks += (f'<h3 class="side up">▲ ฝั่งซื้อ (Long) <span class="count">{n} สัญญาณ</span></h3>'
                   + "".join(section(k) for k in longs))
    if shorts:
        n = sum(len(results[k]) for k in shorts)
        blocks += (f'<h3 class="side dn">▼ ฝั่งขายชอร์ต (Short) <span class="count">{n} สัญญาณ</span></h3>'
                   + "".join(section(k) for k in shorts))

    br, br200 = reg.get("breadth", float("nan")), reg.get("breadth_200", float("nan"))
    regime_class = {True: "good", False: "bad", None: "warn"}[reg.get("risk_on")]
    sess_label = {
        "pre": "รอบก่อนตลาดเปิด · ราคานอกเวลาเป็นของ pre-market",
        "regular": "รอบระหว่างเวลาทำการ · ราคานอกเวลาคือราคาสด",
        "post": "รอบหลังตลาดปิด · ราคานอกเวลาเป็นของ after-hours",
        "closed": "รอบนอกเวลาซื้อขาย · ใช้ราคาล่าสุดที่มี",
    }.get(sess, "")

    if not cfg["extended"]:
        gap_box = '<div class="note muted">ราคานอกเวลาทำการ: ไม่ได้เช็ก</div>'
    else:
        seen, flagged = set(), []          # หุ้นตัวเดียวอาจติดหลายสัญญาณ นับครั้งเดียวพอ
        for rs in results.values():
            for r in rs:
                if r.get("ext_status") in ("against", "ran") and r["ticker"] not in seen:
                    seen.add(r["ticker"])
                    flagged.append(r)
        flagged.sort(key=lambda r: abs(r.get("ext_chg") or 0), reverse=True)
        if flagged:
            chips = " · ".join(
                f"<b>{html.escape(r['ticker'])}</b> {r['ext_chg']:+.1f}%"
                for r in flagged[:12])
            gap_box = (f'<div class="note warn"><b>เตือน gap นอกเวลา ({len(flagged)} ตัว)</b> — '
                       f'ราคาขยับเกิน {cfg["gap_alert_pct"]:.0f}% หลังราคาปิดที่ใช้คำนวณ '
                       f'แผนเทรดของตัวเหล่านี้ต้องคิดใหม่:<br>{chips}</div>')
        else:
            gap_box = ('<div class="note muted">ราคานอกเวลาทำการ: '
                       f'ไม่มีตัวไหนขยับเกิน {cfg["gap_alert_pct"]:.0f}% — แผนเทรดในตารางยังใช้ได้</div>')

    earn_note = {
        "exclude": f"ตัดหุ้นที่จะประกาศงบภายใน {cfg['earnings_blackout_days']} วันออกแล้ว "
                   f"({excluded} สัญญาณ)",
        "flag": "แสดงทุกสัญญาณ แต่ติดป้ายเตือนหุ้นที่ใกล้วันงบ",
        "off": "ไม่ได้เช็กวันประกาศงบ — ตรวจเองก่อนเข้าไม้ทุกครั้ง",
    }[cfg["earnings_mode"]]

    return f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Scanner — {html.escape(asof)}</title>
<style>
:root {{
  --bg:#f6f7f9; --panel:#fff; --ink:#16191d; --muted:#6b7280; --line:#e5e7eb;
  --up:#0f8a4c; --dn:#c62b32; --accent:#2b5cd9; --good:#0f8a4c; --bad:#c62b32; --warn:#b4740b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0f1216; --panel:#171b21; --ink:#e8eaed; --muted:#9aa3af; --line:#2a3038;
    --up:#3ddc84; --dn:#ff6b6b; --accent:#7aa2ff; --good:#3ddc84; --bad:#ff6b6b; --warn:#e0a94a; }}
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans Thai",Roboto,sans-serif; }}
.wrap {{ max-width:1340px; margin:0 auto; padding:24px 16px 64px }}
header h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:-.01em }}
header .sub {{ color:var(--muted); font-size:13px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:16px; margin:12px 0 }}
.card h2 {{ margin:0 0 12px; font-size:15px; font-weight:650 }}
h3.side {{ margin:28px 0 4px; font-size:15px; font-weight:700; letter-spacing:.01em }}
.count {{ color:var(--muted); font-weight:400; font-size:13px; margin-left:6px }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px }}
.stat {{ padding:10px 12px; border:1px solid var(--line); border-radius:9px }}
.stat b {{ display:block; font-size:19px; font-weight:650; margin-top:2px }}
.stat span {{ color:var(--muted); font-size:12px }}
.good {{ color:var(--good) }} .bad {{ color:var(--bad) }} .warn {{ color:var(--warn) }}
.muted {{ color:var(--muted) }}
.scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch }}
table {{ border-collapse:collapse; width:100%; min-width:1000px; font-size:13px }}
th {{ text-align:left; color:var(--muted); font-weight:550; font-size:11.5px;
  text-transform:uppercase; letter-spacing:.04em; padding:6px 9px;
  border-bottom:1px solid var(--line); white-space:nowrap }}
td {{ padding:8px 9px; border-bottom:1px solid var(--line); vertical-align:top }}
tr:last-child td {{ border-bottom:0 }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap }}
.tk {{ font-weight:650; letter-spacing:.02em }}
.up {{ color:var(--up) }} .dn {{ color:var(--dn) }}
.sub {{ display:block; font-size:11px; color:var(--muted) }}
.earn, .ext {{ font-size:12px; white-space:nowrap }}
.ext {{ font-weight:550 }}
.note.warn {{ border-color:var(--warn); background:color-mix(in srgb,var(--warn) 10%,transparent) }}
.empty {{ color:var(--muted); text-align:center; padding:22px }}
.score {{ display:inline-block; min-width:38px; text-align:center; padding:2px 7px; border-radius:6px;
  font-weight:650; font-size:12px; background:linear-gradient(90deg,
  color-mix(in srgb,var(--accent) 26%,transparent) var(--w), transparent var(--w)); }}
.note {{ padding:11px 13px; border-radius:9px; border:1px solid var(--line); margin-top:12px;
  font-size:13px }}
footer {{ color:var(--muted); font-size:12px; margin-top:28px; line-height:1.75 }}
</style></head><body><div class="wrap">

<header>
  <h1>Market Signal Scanner</h1>
  <div class="sub">หุ้นสหรัฐ · ข้อมูลปิดวันที่ {html.escape(asof)} ·
    สแกน {scanned:,} ตัว ผ่านตัวกรองสภาพคล่อง {passed:,} ตัว<br>{html.escape(sess_label)}</div>
</header>

<section class="card">
  <h2>สภาพตลาดวันนี้</h2>
  <div class="stats">
    <div class="stat"><span>เทรนด์ SPY</span><b class="{regime_class}">{html.escape(str(reg.get('spy_trend', 'n/a')))}</b></div>
    <div class="stat"><span>ตลาดเอื้อฝั่ง</span><b class="{regime_class}">{html.escape(str(reg.get('favors', '—')))}</b></div>
    <div class="stat"><span>หุ้นเหนือ EMA50</span><b>{_fmt(br, 0)}%</b></div>
    <div class="stat"><span>หุ้นเหนือ SMA200</span><b>{_fmt(br200, 0)}%</b></div>
    <div class="stat"><span>สัญญาณทั้งหมด</span><b>{sum(len(v) for v in results.values())}</b></div>
  </div>
  <div class="note {regime_class}">{html.escape(reg['note'])}</div>
  <div class="note muted">ตัวกรองวันประกาศงบ: {html.escape(earn_note)}</div>
  {gap_box}
</section>

{blocks}

<footer>
  <b>วิธีอ่าน</b> — คะแนนเต็ม 100 คิดจากแรงวอลุ่ม ความแรงเทรนด์ โมเมนตัม และตำแหน่งราคา
  ยิ่งสูงยิ่งเข้าเงื่อนไขครบ ไม่ได้แปลว่ากำไรแน่นอน ·
  คอลัมน์ "ตัดขาดทุน" ของฝั่งชอร์ตจะอยู่<i>เหนือ</i>ราคาเข้า และ "เป้าหมาย" อยู่<i>ใต้</i>ราคาเข้า<br>
  จุดตัดขาดทุน = {cfg['stop_atr_mult']}×ATR · เป้าหมาย = {cfg['target_atr_mult']}×ATR (อัตราส่วน {cfg['target_atr_mult'] / cfg['stop_atr_mult']:.0f}:1)
  · จำนวนหุ้นคำนวณจากพอร์ต ${cfg['account_size']:,.0f} เสี่ยง {cfg['risk_per_trade_pct']}% ต่อไม้<br><br>
  <b>คอลัมน์ "นอกเวลา"</b> — ราคาล่าสุดในช่วง pre-market หรือ after-hours เทียบกับราคาปิดที่ใช้คำนวณแผน
  <span class="bad">แดง = วิ่งสวนทางที่จะเทรด</span> (แผนเสีย ข้ามไป) ·
  <span class="warn">เหลือง = วิ่งหนีไปทางเดียวกับเราแล้ว</span> (เข้าที่ราคาเดิมไม่ได้ ไล่ราคาเสี่ยง) ·
  เทา = ขยับน้อย แผนยังใช้ได้<br>
  <b>ระวัง</b> — นอกเวลาทำการวอลุ่มบางมาก ราคาขยับได้ด้วยไม้เล็ก ๆ และมักกลับทางตอนตลาดเปิดจริง
  ใช้เป็นสัญญาณเตือนว่า "ต้องดูใหม่" ไม่ใช่ราคาที่จะได้จริง<br><br>
  <b>เรื่องการชอร์ต</b> — ต้องมีบัญชี margin และหุ้นต้องมีให้ยืม (บางตัวยืมไม่ได้หรือค่ายืมแพงมาก)
  ขาดทุนฝั่งชอร์ตไม่มีเพดาน และการบีบชอร์ต (short squeeze) เกิดเร็วมาก
  สคริปต์ไม่ได้เช็กสถานะการยืมหุ้นให้ ต้องดูในโบรกเกอร์เอง<br><br>
  <b>ข้อจำกัด</b> — เป็นเครื่องมือ<i>คัดกรอง</i>เพื่อย่นเวลาหาหุ้นไปดูกราฟต่อ ไม่ใช่คำแนะนำการลงทุน
  ไม่ได้กรองข่าวหรือเหตุการณ์พิเศษ วันประกาศงบดึงจาก Yahoo ซึ่งบางครั้งไม่อัปเดต
  ควรตรวจกราฟและปฏิทินข่าวก่อนตัดสินใจทุกครั้ง และทดสอบด้วยเงินจำลองก่อนใช้เงินจริง
</footer>
</div></body></html>"""


# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="Market signal scanner สำหรับหุ้นสหรัฐ")
    ap.add_argument("--universe", default=CONFIG["universe"],
                    choices=["sp500", "nasdaq100", "both", "fallback"])
    ap.add_argument("--tickers", default=None, help="ระบุหุ้นเองคั่นด้วยจุลภาค เช่น AAPL,NVDA,TSLA")
    ap.add_argument("--sides", default=CONFIG["sides"], choices=["long", "short", "both"],
                    help="สแกนฝั่งไหน (ค่าเริ่มต้น: both)")
    ap.add_argument("--earnings", default=CONFIG["earnings_mode"],
                    choices=["exclude", "flag", "off"],
                    help="จัดการหุ้นใกล้วันประกาศงบ (ค่าเริ่มต้น: exclude)")
    ap.add_argument("--earnings-days", type=int, default=CONFIG["earnings_blackout_days"],
                    help="ห้ามเข้าไม้ถ้างบออกภายในกี่วัน (ค่าเริ่มต้น: 5)")
    ap.add_argument("--extended", default="on", choices=["on", "off"],
                    help="ดึงราคานอกเวลาทำการมาเทียบ (ค่าเริ่มต้น: on)")
    ap.add_argument("--gap-alert", type=float, default=CONFIG["gap_alert_pct"],
                    help="ขยับนอกเวลาเกินกี่ %% ถึงเตือน (ค่าเริ่มต้น: 3.0)")
    ap.add_argument("--top", type=int, default=CONFIG["top_n"])
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--account", type=float, default=CONFIG["account_size"])
    ap.add_argument("--risk", type=float, default=CONFIG["risk_per_trade_pct"])
    ap.add_argument("--min-dollar-vol", type=float, default=CONFIG["min_dollar_vol"])
    args = ap.parse_args()

    cfg = dict(CONFIG)
    cfg.update(top_n=args.top, account_size=args.account, risk_per_trade_pct=args.risk,
               min_dollar_vol=args.min_dollar_vol, sides=args.sides,
               earnings_mode=args.earnings, earnings_blackout_days=args.earnings_days,
               extended=(args.extended == "on"), gap_alert_pct=args.gap_alert)

    active = [(k, fn) for k, fn in SIGNALS
              if cfg["sides"] == "both" or SETUP_SIDE[k] == cfg["sides"]]

    print("═" * 62)
    print("  MARKET SIGNAL SCANNER — หุ้นสหรัฐ")
    print("═" * 62)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        print("\n[1/6] เตรียมรายชื่อหุ้น")
        tickers = get_universe(args.universe)
    if "SPY" not in tickers:
        tickers.append("SPY")
    print(f"  รวม {len(tickers)} ตัว")

    print("\n[2/6] ดึงราคาย้อนหลัง")
    data = download_prices(tickers, cfg["history_days"])
    print(f"  ได้ข้อมูลครบ {len(data)} ตัว")
    if len(data) < 5:
        print("\n! ดึงข้อมูลได้น้อยเกินไป — ตรวจการเชื่อมต่อเน็ต หรือลอง pip install -U yfinance")
        return 1

    print("\n[3/6] คำนวณสัญญาณ")
    rows, passed, hits = [], 0, []
    for t, df in data.items():
        m = compute_metrics(t, df)
        if m is None:
            continue
        rows.append(m)
        if not passes_liquidity(m, cfg):
            continue
        passed += 1
        for key, fn in active:
            got = fn(m)
            if got:
                _, score, why = got
                r = dict(m)
                r.update(setup=key, side=SETUP_SIDE[key], score=score, why=why,
                         plan=trade_plan(m, key, cfg))
                hits.append(r)
    print(f"  พบสัญญาณดิบ {len(hits)} รายการ จากหุ้นที่ผ่านตัวกรอง {passed} ตัว")

    # ── ตัวกรองวันประกาศงบ (ดึงเฉพาะหุ้นที่ติดสัญญาณ) ──
    print("\n[4/6] ตรวจวันประกาศงบ")
    excluded = 0
    if cfg["earnings_mode"] == "off" or not hits:
        for r in hits:
            r["earn_status"], r["earn_text"] = "unknown", "ไม่ได้เช็ก"
        print("  ข้ามการเช็ก")
    else:
        uniq = sorted({r["ticker"] for r in hits})[: cfg["earnings_max_lookup"]]
        print(f"  เช็ก {len(uniq)} ตัวที่ติดสัญญาณ")
        try:
            os.makedirs(args.outdir, exist_ok=True)
            einfo = get_earnings_info(
                uniq, cache_path=os.path.join(args.outdir, ".earnings_cache.json"))
        except Exception as e:  # noqa: BLE001
            print(f"  ! ดึงวันงบไม่สำเร็จ ({e.__class__.__name__}) — ปล่อยผ่านทั้งหมด")
            einfo = {}

        kept = []
        for r in hits:
            st, txt = earnings_verdict(einfo.get(r["ticker"]),
                                       cfg["earnings_blackout_days"], cfg["earnings_post_days"])
            r["earn_status"], r["earn_text"] = st, txt
            if cfg["earnings_mode"] == "exclude" and st == "blackout":
                excluded += 1
                continue
            kept.append(r)
        hits = kept
        if excluded:
            print(f"  ตัดออก {excluded} สัญญาณ (งบออกภายใน {cfg['earnings_blackout_days']} วัน)")

    # ── ราคานอกเวลาทำการ (pre-market / after-hours) ──
    print("\n[5/6] เทียบราคานอกเวลาทำการ")
    sess_now = current_session()
    alerts = 0
    if not cfg["extended"] or not hits:
        for r in hits:
            r["ext_status"], r["ext_text"] = "unknown", "ไม่ได้เช็ก"
            r["ext_chg"] = r["ext_price"] = r["ext_session"] = None
        print("  ข้ามการเช็ก")
    else:
        uniq = sorted({r["ticker"] for r in hits})
        print(f"  ตอนนี้ตลาดสหรัฐ: {SESSION_TH.get(sess_now, sess_now)} · เช็ก {len(uniq)} ตัว")
        try:
            quotes = get_extended_quotes(uniq)
        except Exception as e:  # noqa: BLE001
            print(f"  ! ดึงราคานอกเวลาไม่สำเร็จ ({e.__class__.__name__}) — ข้ามไป")
            quotes = {}
        alerts = attach_to_signals(hits, quotes, cfg["gap_alert_pct"])
        if alerts:
            print(f"  พบ {alerts} สัญญาณที่ราคาขยับเกิน {cfg['gap_alert_pct']:.0f}% นอกเวลา")

    results: dict[str, list[dict]] = {k: [] for k, _ in active}
    for r in hits:
        results[r["setup"]].append(r)
    for k in results:
        results[k].sort(key=lambda r: r["score"], reverse=True)
        results[k] = results[k][: cfg["top_n"]]

    reg = market_regime(data, rows)
    asof = max((m["date"] for m in rows), default=dt.date.today().isoformat())

    print("\n[6/6] บันทึกผล")
    os.makedirs(args.outdir, exist_ok=True)

    flat = []
    for k, rs in results.items():
        for r in rs:
            p = r["plan"]
            flat.append({
                "side": p["side"], "setup": k, "ticker": r["ticker"],
                "score": round(r["score"], 1), "close": round(r["close"], 2),
                "ret1d_%": round(r["ret1d"], 2), "ret5d_%": round(r["ret5d"], 2),
                "ret20d_%": round(r["ret20d"], 2), "rvol": round(r["rvol"], 2),
                "rsi14": round(r["rsi14"], 1), "adx14": round(r["adx14"], 1),
                "atr_%": round(r["atr_pct"], 2), "entry": round(p["entry"], 2),
                "stop": round(p["stop"], 2), "target": round(p["target"], 2),
                "rr": round(p["rr"], 2), "shares": p["shares"],
                "earnings": r.get("earn_text", ""),
                "ext_price": (round(r["ext_price"], 2) if r.get("ext_price") else ""),
                "ext_chg_%": (round(r["ext_chg"], 2) if r.get("ext_chg") is not None else ""),
                "ext_session": r.get("ext_session") or "",
                "ext_flag": r.get("ext_status", ""),
                "reason": r["why"],
            })
    csv_path = os.path.join(args.outdir, f"signals_{asof}.csv")
    pd.DataFrame(flat).to_csv(csv_path, index=False, encoding="utf-8-sig")

    html_path = os.path.join(args.outdir, f"report_{asof}.html")
    page = build_html(results, reg, len(rows), passed, cfg, asof, excluded, sess_now, alerts)
    for path in (html_path, os.path.join(args.outdir, "latest.html")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)

    print(f"  {csv_path}")
    print(f"  {html_path}")

    print("\n" + "─" * 62)
    print(f"สภาพตลาด: SPY {reg.get('spy_trend')} · หุ้นเหนือ EMA50 {_fmt(reg.get('breadth'), 0)}%"
          f" · ตลาดเอื้อ{reg.get('favors')}")
    print(f"  {reg['note']}")
    for side, title in (("long", "▲ ฝั่งซื้อ"), ("short", "▼ ฝั่งขายชอร์ต")):
        keys = [k for k, _ in active if SETUP_SIDE[k] == side]
        if not keys:
            continue
        print(f"\n{'═' * 62}\n{title}")
        for k in keys:
            rs = results[k]
            print(f"\n{SETUP_LABEL[k]}  ({len(rs)})")
            if not rs:
                print("   ไม่มีหุ้นเข้าเงื่อนไข")
            for r in rs[:10]:
                p = r["plan"]
                tag = "  [!งบ]" if r.get("earn_status") in ("blackout", "post") else ""
                print(f"   {r['ticker']:<6} score {r['score']:5.1f} | ${r['close']:>8,.2f} "
                      f"| stop ${p['stop']:>8,.2f} | tgt ${p['target']:>8,.2f} "
                      f"| {p['shares']:>4,} หุ้น{tag}")
                print(f"          {r['why']}")
    print("\n" + "─" * 62)
    print("เครื่องมือคัดกรองเท่านั้น ไม่ใช่คำแนะนำการลงทุน — ตรวจกราฟและปฏิทินข่าวก่อนเทรดเสมอ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

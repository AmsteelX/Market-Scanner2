"""
indicators.py - อินดิเคเตอร์พื้นฐานสำหรับ market scanner
เขียนด้วย pandas ล้วน ไม่ต้องพึ่ง TA-Lib (ติดตั้งยากบน Windows)
ทุกฟังก์ชันรับ pandas Series/DataFrame และคืน Series ที่ index ตรงกัน
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ค่าเฉลี่ย
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (RMA) แบบต้นฉบับ:
    ค่าแรก = ค่าเฉลี่ยธรรมดาของ n ค่าแรก จากนั้นค่อยเรียบด้วย alpha = 1/n
    (pandas ewm เฉย ๆ จะ seed ด้วยค่าแรกค่าเดียว ทำให้ค่าเพี้ยนจากตำรา)
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < n:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    seeded = pd.Series(np.nan, index=valid.index, dtype="float64")
    seeded.iloc[n - 1] = valid.iloc[:n].mean()
    seeded.iloc[n:] = valid.iloc[n:].to_numpy()

    out = seeded.ewm(alpha=1.0 / n, adjust=False).mean()
    return out.reindex(s.index)


# ---------------------------------------------------------------- โมเมนตัม
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI แบบ Wilder. คืนค่า 0-100"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder(gain, n)
    avg_loss = wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # ถ้าไม่มีวันลบเลย -> RSI = 100 ; ถ้าไม่มีวันบวกเลย -> RSI = 0
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, out.where(avg_loss == 0.0, 0.0))
    return out


# ---------------------------------------------------------------- ความผันผวน
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ADX แบบ Wilder — วัด 'ความแรงของเทรนด์' ไม่บอกทิศทาง (>20 = มีเทรนด์)"""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr_n = wilder(true_range(high, low, close), n)
    plus_di = 100.0 * wilder(plus_dm, n) / tr_n.replace(0.0, np.nan)
    minus_di = 100.0 * wilder(minus_dm, n) / tr_n.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder(dx, n)


def bollinger_pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """%B: 0 = แตะแบนด์ล่าง, 0.5 = กลาง, 1 = แตะแบนด์บน"""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def bollinger_bandwidth(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """ความกว้างแบนด์เทียบเส้นกลาง — ค่าต่ำ = บีบตัว (squeeze) มักนำหน้าการ breakout"""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return (2.0 * k * sd) / mid.replace(0.0, np.nan)


# ---------------------------------------------------------------- ตำแหน่งราคา
def rolling_high(high: pd.Series, n: int) -> pd.Series:
    return high.rolling(n, min_periods=n).max()


def rolling_low(low: pd.Series, n: int) -> pd.Series:
    return low.rolling(n, min_periods=n).min()


def pct_change_n(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n) * 100.0


def close_location(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """ปิดอยู่ตรงไหนของแท่ง: 1.0 = ปิดที่ high, 0.0 = ปิดที่ low"""
    rng = (high - low).replace(0.0, np.nan)
    return ((close - low) / rng).fillna(0.5)


def rvol(volume: pd.Series, n: int = 20) -> pd.Series:
    """Relative volume: วอลุ่มวันนี้ / ค่าเฉลี่ย n วันก่อนหน้า (ไม่รวมวันนี้)"""
    base = volume.shift(1).rolling(n, min_periods=max(5, n // 2)).mean()
    return volume / base.replace(0.0, np.nan)

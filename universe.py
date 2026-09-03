"""
universe.py - รายชื่อหุ้นที่จะสแกน
พยายามดึงสมาชิก S&P 500 / Nasdaq-100 สด ๆ จาก Wikipedia ก่อน
ถ้าดึงไม่ได้ (เน็ตมีปัญหา / โครงสร้างหน้าเปลี่ยน) จะถอยไปใช้รายชื่อสำรองที่ฝังไว้
"""
from __future__ import annotations

import pandas as pd

_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"

# รายชื่อสำรอง: หุ้นสภาพคล่องสูงที่เทรดสั้นได้จริง (สเปรดแคบ วอลุ่มหนา)
FALLBACK = """
AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO TSLA BRK-B LLY JPM V UNH XOM MA COST
JNJ PG HD ABBV NFLX BAC CRM CVX AMD KO PEP WFC MRK ADBE TMO LIN ACN CSCO MCD
ABT PM ORCL IBM GE TXN CAT QCOM DHR INTU VZ NOW AMGN PFE ISRG CMCSA SPGI RTX
UBER AMAT GS DIS NEE UNP LOW T PGR HON BKNG BLK SYK TJX AXP LRCX C BSX VRTX
MU ADI PANW ETN MDT ADP GILD SBUX MMC PLD CB REGN KLAC BA SO DE MO ELV ANET
CI ICE SHW DUK APH SNPS CME ZTS CDNS MCK EOG WM TT PYPL NKE MSI CVS MAR ITW
CTAS MCO PH ORLY EQIX APD GD CL FCX NOC EMR SLB ROP CSX PNC AON MMM USB TDG
COF AJG ECL HCA WELL DXCM AZO NSC AFL SPG PSA TFC CARR OKE MET F GM DAL AAL
UAL CCL NCLH RCL ABNB DASH SNAP PINS RBLX COIN HOOD SQ SHOP SOFI PLTR SMCI
MSTR ARM DDOG CRWD ZS NET SNOW MDB TEAM WDAY OKTA TWLO ROKU DKNG LYFT RIVN
LCID NIO XPEV LI PDD BABA JD BIDU TSM ASML SPOT MRNA BNTX ENPH FSLR RUN PLUG
CHPT AI IONQ RGTI SOUN BBAI TEM CELH LULU CMG DPZ WING SHAK YETI CROX DECK
ANF GPS URBN M JWN KSS DG DLTR FIVE BURL ROST TGT WMT SPY QQQ IWM DIA
XLF XLE XLK XLV XLI XLY XLP XLU XLB XLRE XBI SMH SOXL TQQQ ARKK GLD SLV USO
"""


def _from_wikipedia(url: str, col: str) -> list[str]:
    tables = pd.read_html(url)
    for t in tables:
        if col in t.columns:
            return [str(s).strip() for s in t[col].dropna().tolist()]
    raise ValueError(f"ไม่พบคอลัมน์ {col} ใน {url}")


def _clean(tickers) -> list[str]:
    out, seen = [], set()
    for t in tickers:
        t = str(t).strip().upper().replace(".", "-")  # BRK.B -> BRK-B (รูปแบบของ Yahoo)
        if not t or not all(ch.isalnum() or ch == "-" for ch in t):
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_universe(which: str = "both", verbose: bool = True) -> list[str]:
    """which: 'sp500' | 'nasdaq100' | 'both' | 'fallback'"""
    if which == "fallback":
        return _clean(FALLBACK.split())

    tickers: list[str] = []
    sources = []
    if which in ("sp500", "both"):
        sources.append((_WIKI_SP500, "Symbol", "S&P 500"))
    if which in ("nasdaq100", "both"):
        sources.append((_WIKI_NDX, "Ticker", "Nasdaq-100"))

    for url, col, label in sources:
        try:
            got = _from_wikipedia(url, col)
            tickers += got
            if verbose:
                print(f"  ดึง {label}: {len(got)} ตัว")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ! ดึง {label} ไม่สำเร็จ ({e.__class__.__name__}) — ใช้รายชื่อสำรองแทน")

    if not tickers:
        tickers = FALLBACK.split()
        if verbose:
            print(f"  ใช้รายชื่อสำรองทั้งหมด")

    return _clean(tickers)

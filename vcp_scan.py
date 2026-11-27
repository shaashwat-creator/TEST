"""
Minervini VCP scanner — NSE F&O stocks, Daily + 1-Hour.
Runs once after market close (GitHub Actions), sends the watchlist to WhatsApp via CallMeBot.

Daily  : full Minervini Trend Template + RS vs NIFTY + VCP (shrinking contractions, flat resistance)
1-Hour : VCP built inside the last ~2 weeks of hourly candles (stock must be in a daily uptrend)

Status:  BREAKOUT  = closed above pivot today on >= 1.5x avg volume
         FORMING   = tight final contraction, price within a few % under the pivot, volume drying up
"""
import os
import sys
import math
import time
import datetime as dt
import urllib.parse

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ───────────────────────── F&O universe (NSE, Nov 2026) ─────────────────────────
# Edit this list when NSE adds/removes stocks (twice a year).
FNO = """
360ONE ABB ABCAPITAL ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ADANIPOWER ALKEM AMBER AMBUJACEM
ANANDRATHI ANGELONE APLAPOLLO APOLLOHOSP ASHOKLEY ASIANPAINT ASTRAL ATHERENERG AUBANK AUROPHARMA
AXISBANK BAJAJ-AUTO BAJAJFINSV BAJFINANCE BANDHANBNK BANKBARODA BANKINDIA BDL BEL BHARATFORG
BHARTIARTL BHEL BIOCON BLUESTARCO BOSCHLTD BPCL BRITANNIA BSE CAMS CANBK CDSL CGPOWER CHOLAFIN
CIPLA COALINDIA COCHINSHIP COFORGE COLPAL CONCOR COROMANDEL CROMPTON CUB CUMMINSIND DABUR
DATAPATTNS DELHIVERY DIVISLAB DIXON DLF DMART DRREDDY EICHERMOT ENRIN ETERNAL FEDERALBNK
FORCEMOT FORTIS GAIL GLENMARK GMRAIRPORT GODFRYPHLP GODREJCP GODREJPROP GRASIM GVT&D HAL
HAVELLS HCLTECH HDFCAMC HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDPETRO HINDUNILVR HINDZINC
HYUNDAI ICICIBANK ICICIGI ICICIPRULI IDEA IDFCFIRSTB INDHOTEL INDIANB INDIGO INDUSINDBK
INDUSTOWER INFY INOXWIND IOC IRFC ITC JINDALSTEL JIOFIN JSWENERGY JSWSTEEL JUBLFOOD KALYANKJIL
KARURVYSYA KAYNES KEI KFINTECH KOTAKBANK KPITTECH LAURUSLABS LICHSGFIN LICI LODHA LT LTF LTM
LUPIN M&M MAHABANK MANAPPURAM MANKIND MARICO MARUTI MAXHEALTH MAZDOCK MCX MFSL MOTHERSON
MOTILALOFS MPHASIS MUTHOOTFIN NAM-INDIA NATIONALUM NAUKRI NAVINFLUOR NBCC NESTLEIND NHPC NMDC
NTPC NYKAA OBEROIRLTY OFSS OIL ONGC PAGEIND PATANJALI PAYTM PERSISTENT PETRONET PFC PGEL
PHOENIXLTD PIDILITIND PIIND PNB PNBHOUSING POLICYBZR POLYCAB POWERGRID POWERINDIA PREMIERENE
PRESTIGE RADICO RBLBANK RECLTD RELIANCE RVNL SAGILITY SAIL SBICARD SBILIFE SBIN SHREECEM
SHRIRAMFIN SIEMENS SOLARINDS SONACOMS SRF SUNPHARMA SUPREMEIND SUZLON SWIGGY SYRMA TATACONSUM
TATAELXSI TATAPOWER TATASTEEL TCS TECHM TIINDIA TITAN TMPV TORNTPHARM TRENT TVSMOTOR
UJJIVANSFB ULTRACEMCO UNIONBANK UNITDSPR UNOMINDA UPL VBL VEDL VMM VOLTAS WAAREEENER WIPRO
YESBANK ZYDUSLIFE
""".split()

BENCH = "^NSEI"  # NIFTY 50 on Yahoo

# ───────────────────────── Settings ─────────────────────────
DAILY = dict(piv=4, max_first=35.0, max_last=10.0, near=6.0, max_base=65, flat_tol=5.0)
HOURLY = dict(piv=3, max_first=12.0, max_last=4.0, near=2.5, max_base=70, flat_tol=2.5)  # 70 bars ≈ 2 weeks
MIN_T = 2          # at least 2 contractions
SHRINK = 0.85      # each contraction <= 85% of the previous one
VOL_DRY = 0.85     # 10-bar avg volume <= 85% of 50-bar avg  -> drying up
VOL_BO = 1.5       # breakout volume >= 1.5x 50-bar avg
MAX_MSG_ROWS = 15  # per section, to keep the WhatsApp message short


def yf_symbol(s: str) -> str:
    return s + ".NS"


# ───────────────────────── Data ─────────────────────────
def download(tickers, period, interval, tries=3):
    for i in range(tries):
        try:
            df = yf.download(tickers, period=period, interval=interval, group_by="ticker",
                             auto_adjust=True, threads=True, progress=False)
            if df is not None and len(df):
                return df
        except Exception as e:  # network hiccup
            print(f"download error ({interval}) try {i+1}: {e}")
        time.sleep(10)
    return None


def one(df, ticker):
    try:
        d = df[ticker].dropna(how="all")
    except Exception:
        return None
    if d is None or d.empty or "Close" not in d:
        return None
    d = d.dropna(subset=["High", "Low", "Close"])
    return d if len(d) else None


# ───────────────────────── Logic ─────────────────────────
def trend_template(d: pd.DataFrame) -> bool:
    if len(d) < 220:
        return False
    c = d["Close"]
    s50, s150, s200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    hi52, lo52 = d["High"].tail(252).max(), d["Low"].tail(252).min()
    x = c.iloc[-1]
    return bool(
        x > s150.iloc[-1] and x > s200.iloc[-1] and s150.iloc[-1] > s200.iloc[-1]
        and s200.iloc[-1] > s200.iloc[-21] and s50.iloc[-1] > s150.iloc[-1] and x > s50.iloc[-1]
        and x >= lo52 * 1.30 and x >= hi52 * 0.75
    )


def light_uptrend(d: pd.DataFrame) -> bool:
    """Gate for the hourly scan: stock must be in a daily uptrend."""
    if len(d) < 200:
        return False
    c = d["Close"]
    return bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1] > c.rolling(200).mean().iloc[-1])


def rs_ok(d: pd.DataFrame, bench: pd.DataFrame) -> bool:
    try:
        r = d["Close"].iloc[-1] / d["Close"].iloc[-64] - 1
        b = bench["Close"].iloc[-1] / bench["Close"].iloc[-64] - 1
        return bool(r > b)
    except Exception:
        return False


def swings(h: np.ndarray, l: np.ndarray, n: int):
    """Confirmed swing highs/lows (n bars each side). Returns list of (idx, kind, price) in time order."""
    out = []
    for i in range(n, len(h) - n):
        if h[i] == h[i - n:i + n + 1].max() and (h[i] > h[i - n:i].max() if n else True):
            out.append((i, "H", h[i]))
        if l[i] == l[i - n:i + n + 1].min() and (l[i] < l[i - n:i].min() if n else True):
            out.append((i, "L", l[i]))
    out.sort(key=lambda t: (t[0], 0 if t[1] == "H" else 1))
    return out


def vcp(d: pd.DataFrame, p: dict, window: int | None = None):
    """Detect a VCP. Returns dict or None."""
    if d is None or len(d) < 60:
        return None
    v_all = d["Volume"].astype(float).values
    d_w = d.tail(window) if window else d
    off = len(d) - len(d_w)
    h, l, c = d_w["High"].values, d_w["Low"].values, d_w["Close"].values

    base_hi = base_idx = None
    last_ph = last_ph_idx = None
    last_pl = None
    wait_low = False
    depths = []

    for idx, kind, price in swings(h, l, p["piv"]):
        if kind == "H":
            if base_hi is None or price > base_hi:
                base_hi, base_idx = price, idx
                depths, last_pl = [], None
            last_ph, last_ph_idx, wait_low = price, idx, True
        elif kind == "L" and wait_low and last_ph is not None and idx > last_ph_idx:
            depths.append((last_ph - price) / last_ph * 100)
            depths = depths[-6:]
            last_pl, wait_low = price, False

    # the current pullback may not be a confirmed swing low yet -> count it if it has formed
    if wait_low and last_ph is not None and len(l) - 1 - last_ph_idx >= 2:
        run_low = l[last_ph_idx + 1:].min()
        depth = (last_ph - run_low) / last_ph * 100
        if depth > 0.5:
            depths.append(depth)
            last_pl = run_low

    if not depths or last_ph is None or last_pl is None:
        return None

    # count consecutive shrinking contractions from newest backwards
    t, chain = 1, [depths[-1]]
    for i in range(len(depths) - 2, -1, -1):
        if depths[i + 1] <= depths[i] * SHRINK:
            t += 1
            chain.insert(0, depths[i])
        else:
            break

    n_bars = len(d_w)
    base_len = n_bars - 1 - base_idx
    close = c[-1]
    pivot, stop = last_ph, last_pl

    shape = (
        t >= MIN_T and chain[0] <= p["max_first"] and chain[-1] <= p["max_last"]
        and base_len <= p["max_base"] and close > stop
        and pivot >= base_hi * (1 - p["flat_tol"] / 100)      # flat resistance line
    )
    if not shape:
        return None

    vol = pd.Series(v_all)
    v50 = vol.rolling(50).mean().iloc[-1]
    v10_prev = vol.rolling(10).mean().iloc[-2]
    dist = (pivot - close) / pivot * 100
    prev_close = d["Close"].values[-2]

    breakout = close > pivot and prev_close <= pivot and v_all[-1] >= VOL_BO * v50
    forming = close <= pivot and dist <= p["near"] and v10_prev <= VOL_DRY * v50

    if not (breakout or forming):
        return None
    return dict(
        status="BREAKOUT" if breakout else "FORMING",
        t=t, chain=chain, pivot=pivot, stop=stop, close=close,
        dist=dist, risk=(pivot - stop) / pivot * 100,
        volx=v_all[-1] / v50 if v50 else float("nan"),
    )


# ───────────────────────── Message ─────────────────────────
def fmt(sym, r):
    ch = ">".join(f"{x:.0f}" if x >= 10 else f"{x:.1f}" for x in r["chain"])
    if r["status"] == "BREAKOUT":
        return f"🚀 {sym} BO {r['close']:.1f} > {r['pivot']:.1f} | vol {r['volx']:.1f}x | SL {r['stop']:.1f} | T{r['t']} {ch}%"
    return f"👀 {sym} {r['close']:.1f} | pivot {r['pivot']:.1f} ({r['dist']:.1f}% away) | SL {r['stop']:.1f} (risk {r['risk']:.1f}%) | T{r['t']} {ch}%"


def section(title, rows):
    if not rows:
        return f"*{title}*\nNone today\n"
    rows = sorted(rows, key=lambda x: (x[1]["status"] != "BREAKOUT", x[1]["dist"]))[:MAX_MSG_ROWS]
    return f"*{title}* ({len(rows)})\n" + "\n".join(fmt(s, r) for s, r in rows) + "\n"


def send_whatsapp(text: str):
    phone, key = os.getenv("CALLMEBOT_PHONE"), os.getenv("CALLMEBOT_APIKEY")
    if not phone or not key:
        print("No CallMeBot secrets set — printing only.")
        return
    # CallMeBot works best with shorter messages: split into chunks
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    chunks.append(cur)
    for ch in chunks:
        url = ("https://api.callmebot.com/whatsapp.php?phone=" + urllib.parse.quote(phone)
               + "&text=" + urllib.parse.quote(ch) + "&apikey=" + urllib.parse.quote(key))
        try:
            r = requests.get(url, timeout=30)
            print("CallMeBot:", r.status_code)
        except Exception as e:
            print("CallMeBot error:", e)
        time.sleep(4)


# ───────────────────────── Main ─────────────────────────
def main():
    ist_today = (dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).date()
    tickers = [yf_symbol(s) for s in FNO]

    daily = download(tickers + [BENCH], period="15mo", interval="1d")
    hourly = download(tickers, period="1mo", interval="60m")
    if daily is None:
        send_whatsapp(f"VCP scan {ist_today}: data download failed. Will retry tomorrow.")
        sys.exit(1)

    bench = one(daily, BENCH)
    last_bar = bench.index[-1].date() if bench is not None else None
    if last_bar != ist_today and os.getenv("FORCE_RUN") != "1":
        send_whatsapp(f"VCP scan {ist_today}: no new candle (market holiday?). No scan today.")
        return

    d_rows, h_rows, missing = [], [], []
    for s, tk in zip(FNO, tickers):
        d = one(daily, tk)
        if d is None:
            missing.append(s)
            continue
        # Daily: full Minervini
        if trend_template(d) and (bench is None or rs_ok(d, bench)):
            r = vcp(d, DAILY)
            if r:
                d_rows.append((s, r))
        # Hourly: VCP within last ~2 weeks, stock in daily uptrend
        if hourly is not None and light_uptrend(d):
            hd = one(hourly, tk)
            r = vcp(hd, HOURLY, window=HOURLY["max_base"]) if hd is not None else None
            if r:
                h_rows.append((s, r))

    msg = (f"📊 *VCP scan — {ist_today}* (F&O, after close)\n\n"
           + section("DAILY", d_rows) + "\n"
           + section("1-HOUR (last 2 wks)", h_rows)
           + "\n🚀=broke out today  👀=near pivot, watch\nSL=last contraction low. Check chart before trading.")
    if missing:
        msg += f"\n(no data: {', '.join(missing[:10])}{'…' if len(missing) > 10 else ''})"
    print(msg)
    send_whatsapp(msg)


if __name__ == "__main__":
    main()

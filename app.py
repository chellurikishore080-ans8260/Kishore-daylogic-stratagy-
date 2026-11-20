# Created by Kishore
# Kishore Signals - Liquidity Sweep + Body Fill scanner and backtest (SELL signals only)
# Data  : Binance Spot Vision (data.binance.vision) USDT pairs
# Coins : Top coins by market cap + liquidity (USDT pairs) + XAUTUSDT
# Rules : exactly the Pine "Kishore Signals" strategy (no fees, fixed size per trade)

import io
import time
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"
REST_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
COINGECKO = "https://api.coingecko.com/api/v3/coins/markets"
HEADERS = {"User-Agent": "Mozilla/5.0 (KishoreSignals)"}

IV_MIN = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
IV_MS = {k: v * 60_000 for k, v in IV_MIN.items()}
IST = "Asia/Kolkata"
GOLD = "XAUTUSDT"
COLS = ["t", "open", "high", "low", "close", "volume"]

STABLE = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "BUSD", "PYUSD", "USDD",
    "USDP", "USD1", "RLUSD", "GUSD", "FRAX", "LUSD", "SUSD", "USDG", "USDTB", "BFUSD",
    "USDX", "EURC", "EUR", "AEUR", "XUSD", "USDF", "USDT0", "SUSDE", "SUSDS", "USDY",
    "USDC.E", "USDL", "USDQ", "CUSD", "BUIDL", "USTB", "USYC", "OUSG", "FIGR_HELOC",
}
WRAPPED = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "WBETH", "CBBTC", "CBETH", "RETH", "BTCB",
    "EETH", "WBNB", "BNSOL", "JITOSOL", "MSOL", "LBTC", "SOLVBTC", "TBTC", "METH", "OSETH",
    "RSETH", "EZETH", "CMETH", "WTRX", "WSOL", "WAVAX", "SWETH", "ETHX", "FBTC", "XAUT",
}
STATIC_TOP = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK", "XLM", "BCH", "AVAX",
    "HBAR", "LTC", "SUI", "SHIB", "TON", "DOT", "UNI", "AAVE", "PEPE", "NEAR", "ETC", "ICP",
    "APT", "ONDO", "ARB", "ATOM", "FIL", "ALGO", "VET", "OP", "INJ", "RENDER", "TAO", "FET",
    "POL", "SEI", "TIA", "JUP", "STX", "CRV", "IMX", "LDO", "WLD", "ENA", "CAKE", "GRT",
    "SAND", "MANA", "AXS", "THETA", "FLOW", "EOS", "XTZ", "NEO", "KAVA", "RUNE", "GALA",
    "QNT", "PENDLE", "JASMY", "BONK", "FLOKI", "WIF", "ENS", "COMP", "SNX", "MKR", "1INCH",
    "CHZ", "DYDX", "ZEC", "DASH", "IOTA", "EGLD", "TRUMP", "PENGU", "VIRTUAL", "AR", "ZIL",
    "BAT", "ZRX", "ENJ", "ANKR", "KSM", "CELO", "ROSE", "MINA", "LRC", "SKL", "GMT", "APE",
    "BLUR", "PYTH", "STRK", "ORDI", "W", "NOT", "PEOPLE", "COW",
]


# ----------------------------------------------------------------------------
# Network helpers
# ----------------------------------------------------------------------------
_state = {"host": None}


def rest_get(path, params=None, timeout=15):
    hosts = list(REST_HOSTS)
    if _state["host"] in hosts:
        hosts.remove(_state["host"])
        hosts.insert(0, _state["host"])
    for h in hosts:
        try:
            r = requests.get(h + path, params=params, timeout=timeout, headers=HEADERS)
            if r.status_code == 200:
                _state["host"] = h
                return r.json()
        except Exception:
            continue
    return None


def empty_df():
    return pd.DataFrame(columns=COLS)


def norm(a):
    """Numeric clean-up. Binance spot files from 2025 use microsecond timestamps."""
    if a is None or len(a) == 0:
        return empty_df()
    a = a.apply(pd.to_numeric, errors="coerce").dropna()
    if a.empty:
        return empty_df()
    t = a["t"].astype("int64").to_numpy()
    t = np.where(t > 10 ** 14, t // 1000, t)
    a = a.copy()
    a["t"] = t
    return a.astype({"t": "int64"})[COLS]


def rest_klines(symbol, interval, start_ms, max_pages=300):
    rows = []
    cur = int(start_ms)
    ms = IV_MS[interval]
    for _ in range(max_pages):
        data = rest_get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000},
        )
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cur = int(data[-1][0]) + ms
    if not rows:
        return empty_df()
    a = pd.DataFrame([r[:6] for r in rows], columns=COLS)
    return norm(a)


def vision_month(symbol, interval, year, month):
    url = f"{VISION_BASE}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    r = None
    for _ in range(2):
        try:
            r = requests.get(url, timeout=25, headers=HEADERS)
            break
        except Exception:
            time.sleep(0.4)
    if r is None or r.status_code != 200:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                a = pd.read_csv(f, header=None, usecols=range(6), names=COLS)
        return norm(a)
    except Exception:
        return None


def finalize(df, tf, days):
    if df is None or len(df) == 0:
        return empty_df()
    now_ms = int(time.time() * 1000)
    ms = IV_MS[tf]
    df = df.drop_duplicates("t").sort_values("t")
    df = df[df["t"] + ms <= now_ms]                      # closed candles only
    df = df[df["t"] >= now_ms - int((days + 5) * 86_400_000)]
    df = df.reset_index(drop=True)
    if df.empty:
        return empty_df()
    df["time"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    )
    return df


def fetch_full(symbol, tf, days):
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=days + 5)
    start_ms = int(start.timestamp() * 1000)
    frames = []
    y, m = start.year, start.month
    while (y, m) < (now.year, now.month):
        d = vision_month(symbol, tf, y, m)
        if d is not None and len(d):
            frames.append(d)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    base = pd.concat(frames, ignore_index=True) if frames else empty_df()
    nxt = int(base["t"].max()) + IV_MS[tf] if len(base) else start_ms
    top = rest_klines(symbol, tf, nxt)                   # current month / latest candles
    allf = pd.concat([x for x in (base, top) if len(x)], ignore_index=True) if (len(base) or len(top)) else empty_df()
    return finalize(allf, tf, days)


def topup(symbol, tf, df, days):
    if df is None or len(df) == 0:
        return fetch_full(symbol, tf, days)
    base = df[COLS]
    nxt = int(base["t"].iloc[-1]) + IV_MS[tf]
    new = rest_klines(symbol, tf, nxt, max_pages=50)
    if len(new):
        base = pd.concat([base, new], ignore_index=True)
    return finalize(base, tf, days)


# ----------------------------------------------------------------------------
# Universe : top coins by market cap + liquidity, USDT pairs on Binance + XAUTUSDT
# ----------------------------------------------------------------------------
def binance_usdt_volume():
    data = rest_get("/api/v3/ticker/24hr", timeout=25)
    out = {}
    if not data:
        return out
    bad = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
    for x in data:
        s = x.get("symbol", "")
        if s.endswith("USDT") and not s.endswith(bad):
            base = s[:-4]
            if base in STABLE or base in WRAPPED:
                continue
            try:
                q = float(x.get("quoteVolume", 0))
            except Exception:
                q = 0.0
            if q > 0:
                out[s] = q
    return out


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_universe(n):
    vol = binance_usdt_volume()
    syms, src = [], ""
    try:
        r = requests.get(
            COINGECKO,
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1},
            timeout=25,
            headers=HEADERS,
        )
        if r.status_code == 200:
            for row in r.json():
                s = str(row.get("symbol", "")).upper()
                if not s or s in STABLE or s in WRAPPED:
                    continue
                pair = s + "USDT"
                if vol and pair not in vol:
                    continue
                if pair not in syms:
                    syms.append(pair)
            src = "CoinGecko market cap x Binance USDT pairs"
    except Exception:
        pass
    if len(syms) < 30 and vol:
        syms = [k for k, _ in sorted(vol.items(), key=lambda kv: kv[1], reverse=True)]
        src = "Binance 24h USDT volume (liquidity)"
    if len(syms) < 30:
        syms = [s + "USDT" for s in STATIC_TOP]
        src = "Built-in top list"
    syms = [s for s in syms if s != GOLD][:n]
    return syms + [GOLD], src


# ----------------------------------------------------------------------------
# Strategy engine (exact port of the Pine script)
# ----------------------------------------------------------------------------
def simulate(df, p, size, iv_min):
    """Returns (setups, trades). One position per coin, limit order valid for the next candle only."""
    n = len(df)
    setups, trades = [], []
    if n < p["trend"] + 3:
        return setups, trades

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    tm = df["time"].to_numpy()

    with np.errstate(invalid="ignore"):
        bull = np.ones(n, bool)
        for k in range(1, p["trend"] + 1):
            ck = df["close"].shift(k).to_numpy(float)
            ok = df["open"].shift(k).to_numpy(float)
            bull &= (ck > ok)
        c1 = df["close"].shift(1).to_numpy(float)
        o1 = df["open"].shift(1).to_numpy(float)
        l1 = df["low"].shift(1).to_numpy(float)
        h1 = df["high"].shift(1).to_numpy(float)
        pct = p["fill"] / 100.0
        sell = bull & (h > h1) & (c < o) & (c <= c1 - (c1 - o1) * pct)

    ep = p["entry"] / 100.0
    sb = p["slbuf"] / 100.0
    rr = p["rr"]
    iv = pd.Timedelta(minutes=iv_min)

    pending = None
    pos = None
    for i in range(n):
        just = False
        # 1) pending limit order from the previous candle: valid for this candle only
        if pending is not None:
            e = pending["entry"]
            sd = pending["side"]
            hit = (l[i] <= e) if sd == 1 else (h[i] >= e)
            if hit:
                fill = min(o[i], e) if sd == 1 else max(o[i], e)
                pos = dict(pending)
                pos["fill"] = fill
                pos["bar_in"] = i
                pending["setup"]["status"] = "OPEN"
                just = True
            else:
                pending["setup"]["status"] = "NOT FILLED"
            pending = None

        # 2) manage open position (SL has priority when SL and TP are both inside one candle)
        if pos is not None:
            sd = pos["side"]
            sl = pos["sl"]
            tp = pos["tp"]
            e = pos["entry"]
            if sd == 1:
                sl_hit = l[i] <= sl
                tp_ok = (not just) or c[i] >= o[i] or o[i] <= e
                tp_hit = tp_ok and h[i] >= tp
                sl_px = min(o[i], sl)
                tp_px = tp if just else max(o[i], tp)
            else:
                sl_hit = h[i] >= sl
                tp_ok = (not just) or c[i] <= o[i] or o[i] >= e
                tp_hit = tp_ok and l[i] <= tp
                sl_px = max(o[i], sl)
                tp_px = tp if just else min(o[i], tp)
            if sl_hit or tp_hit:
                ex = sl_px if sl_hit else tp_px
                fill = pos["fill"]
                risk = abs(e - sl)
                if sd == 1:
                    pnl = size * (ex - fill) / fill
                    r_mult = (ex - fill) / risk
                else:
                    pnl = size * (fill - ex) / fill
                    r_mult = (fill - ex) / risk
                res = "WIN" if pnl > 0 else "LOSS"
                trades.append({
                    "Side": "BUY" if sd == 1 else "SELL",
                    "Signal Time": pos["setup"]["signal_time"],
                    "Entry Candle": pd.Timestamp(tm[pos["bar_in"]]),
                    "Entry": fill, "SL": sl, "TP": tp,
                    "Exit Candle": pd.Timestamp(tm[i]),
                    "Exit": ex,
                    "Exit Type": "SL" if sl_hit else "TP",
                    "Result": res,
                    "PnL": pnl,
                    "Return %": pnl / size * 100.0,
                    "R": r_mult,
                    "ExitTS": pd.Timestamp(tm[i]),
                })
                pos["setup"]["status"] = res
                pos = None

        # 3) new setup at candle close (only when flat)
        if pos is None and sell[i]:
            side = -1
            e = h[i] - (h[i] - l[i]) * ep
            sl = h[i] * (1.0 + sb)
            risk = sl - e
            tp = e - risk * rr
            if risk > 0:
                setup = {
                    "bar": i,
                    "candle": pd.Timestamp(tm[i]),
                    "signal_time": pd.Timestamp(tm[i]) + iv,
                    "side": "BUY" if side == 1 else "SELL",
                    "entry": e, "sl": sl, "tp": tp,
                    "status": "PENDING",
                }
                setups.append(setup)
                pending = {"side": side, "entry": e, "sl": sl, "tp": tp, "setup": setup}

    if pending is not None:
        pending["setup"]["status"] = "ACTIVE"
    if pos is not None:                                  # still open at the last candle
        sd = pos["side"]
        fill = pos["fill"]
        ex = c[-1]
        risk = abs(pos["entry"] - pos["sl"])
        if sd == 1:
            pnl = size * (ex - fill) / fill
            r_mult = (ex - fill) / risk
        else:
            pnl = size * (fill - ex) / fill
            r_mult = (fill - ex) / risk
        trades.append({
            "Side": "BUY" if sd == 1 else "SELL",
            "Signal Time": pos["setup"]["signal_time"],
            "Entry Candle": pd.Timestamp(tm[pos["bar_in"]]),
            "Entry": fill, "SL": pos["sl"], "TP": pos["tp"],
            "Exit Candle": pd.NaT, "Exit": ex,
            "Exit Type": "-",
            "Result": "OPEN",
            "PnL": pnl,
            "Return %": pnl / size * 100.0,
            "R": r_mult,
            "ExitTS": pd.Timestamp(tm[-1]),
        })
    return setups, trades


@st.cache_resource
def store():
    return {"data": {}, "sim": {}}


def ensure_data(symbols, tf, days):
    S = store()["data"]
    key = (tf, days)
    if key not in S:
        while len(S) >= 3:
            S.pop(next(iter(S)))
        S[key] = {}
    cache = S[key]
    bucket = int(time.time() * 1000 // IV_MS[tf])
    todo = [s for s in symbols if cache.get(s) is None or cache[s][0] != bucket]

    if todo:
        prog = st.progress(0.0, text="Binance Vision data load avthundi...")

        def work(sym):
            item = cache.get(sym)
            if item is not None and len(item[1]) > 0:
                return sym, topup(sym, tf, item[1], days)
            return sym, fetch_full(sym, tf, days)

        done = 0
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = [ex.submit(work, s) for s in todo]
            for f in as_completed(futs):
                try:
                    sym, df = f.result()
                    cache[sym] = (bucket, df)
                except Exception:
                    pass
                done += 1
                prog.progress(done / len(todo), text=f"Data loading {done}/{len(todo)} coins")
        prog.empty()

    dfs = {s: cache[s][1] for s in symbols if s in cache and len(cache[s][1]) > 0}
    failed = [s for s in symbols if s not in dfs]
    return dfs, failed


def run_all(dfs, tf, days, p, size):
    sim = store()["sim"]
    if len(sim) > 3000:
        sim.clear()
    pkey = (p["trend"], p["fill"], p["entry"], p["slbuf"], p["rr"], size)
    all_setups, all_trades, nlen = [], [], {}
    for s, df in dfs.items():
        k = (s, tf, days, pkey, int(df["t"].iloc[-1]), len(df))
        if k not in sim:
            sim[k] = simulate(df, p, size, IV_MIN[tf])
        setups, trades = sim[k]
        nlen[s] = len(df)
        for x in setups:
            y = dict(x)
            y["Symbol"] = s
            y["age"] = len(df) - 1 - x["bar"]
            all_setups.append(y)
        for x in trades:
            y = dict(x)
            y["Symbol"] = s
            all_trades.append(y)
    return all_setups, pd.DataFrame(all_trades), nlen


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------
def fmt_price(x):
    try:
        x = float(x)
    except Exception:
        return ""
    ax = abs(x)
    if ax >= 100:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}"
    if ax >= 0.01:
        return f"{x:,.5f}"
    return f"{x:.8f}"


def fmt_time(x):
    if x is None or pd.isna(x):
        return "-"
    return pd.Timestamp(x).strftime("%d-%b-%Y %H:%M")


def color_pnl(v):
    try:
        if v > 0:
            return "color: #16a34a; font-weight: 600"
        if v < 0:
            return "color: #dc2626; font-weight: 600"
    except Exception:
        pass
    return ""


def show(df, color_cols=(), fmts=None):
    if df is None or len(df) == 0:
        return
    sty = df.style
    cols = [c for c in color_cols if c in df.columns]
    if cols:
        if hasattr(sty, "map"):
            sty = sty.map(color_pnl, subset=cols)
        else:
            sty = sty.applymap(color_pnl, subset=cols)
    if fmts:
        sty = sty.format({k: v for k, v in fmts.items() if k in df.columns})
    st.dataframe(sty, hide_index=True)


def summarize(tr):
    closed = tr[tr["Result"].isin(["WIN", "LOSS"])].copy()
    out = {"trades": len(closed)}
    if closed.empty:
        out.update(dict(wins=0, losses=0, wr=0.0, net=0.0, pf=0.0, dd=0.0, avg_r=0.0,
                        avg_win=0.0, avg_loss=0.0, best=0.0, worst=0.0))
        return out, closed
    wins = int((closed["Result"] == "WIN").sum())
    losses = int((closed["Result"] == "LOSS").sum())
    gw = closed.loc[closed["PnL"] > 0, "PnL"].sum()
    gl = -closed.loc[closed["PnL"] < 0, "PnL"].sum()
    closed = closed.sort_values("ExitTS")
    cum = closed["PnL"].cumsum().to_numpy()
    cum0 = np.r_[0.0, cum]
    dd = float((cum0 - np.maximum.accumulate(cum0)).min())
    out.update(
        wins=wins, losses=losses, wr=wins / len(closed) * 100.0,
        net=float(closed["PnL"].sum()),
        pf=(gw / gl) if gl > 0 else float("inf"),
        dd=dd, avg_r=float(closed["R"].mean()),
        avg_win=float(closed.loc[closed["PnL"] > 0, "PnL"].mean()) if wins else 0.0,
        avg_loss=float(closed.loc[closed["PnL"] < 0, "PnL"].mean()) if losses else 0.0,
        best=float(closed["PnL"].max()), worst=float(closed["PnL"].min()),
    )
    return out, closed


def status_text(s):
    return {
        "ACTIVE": "ACTIVE - limit order for next candle",
        "NOT FILLED": "Not filled (expired)",
        "OPEN": "Trade open",
        "WIN": "WIN",
        "LOSS": "LOSS",
    }.get(s, s)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def render(symbols, tf, days, p, size, recent, src):
    dfs, failed = ensure_data(symbols, tf, days)
    if not dfs:
        st.error("Binance Vision nundi data raaledu. Konchem sepu tarvata malli try cheyyandi.")
        return

    setups, tr, nlen = run_all(dfs, tf, days, p, size)
    last_candle = max(df["time"].iloc[-1] for df in dfs.values())

    st.caption(
        f"Coins scanned: {len(dfs)}/{len(symbols)}  |  Timeframe: {tf}  |  "
        f"Last closed candle (IST): {fmt_time(last_candle)}  |  Universe: {src}"
    )

    tab1, tab2 = st.tabs(["Live Signals", "Backtest"])

    # ---------------- Live signals ----------------
    with tab1:
        active = [x for x in setups if x["status"] == "ACTIVE"]
        iv = pd.Timedelta(minutes=IV_MIN[tf])
        st.subheader(f"Active signals ({len(active)})")
        st.caption("SELL rule match ayina coins. Entry = limit order, next candle lo matrame valid.")
        if active:
            rows = []
            for x in sorted(active, key=lambda z: z["Symbol"]):
                rows.append({
                    "Coin": x["Symbol"], "Side": x["side"],
                    "Signal Time (IST)": fmt_time(x["signal_time"]),
                    "Entry": fmt_price(x["entry"]),
                    "Stop Loss": fmt_price(x["sl"]),
                    "Take Profit": fmt_price(x["tp"]),
                    "Risk %": f"{abs(x['entry'] - x['sl']) / x['entry'] * 100:.2f}",
                    "Valid Till (IST)": fmt_time(x["signal_time"] + iv),
                })
            show(pd.DataFrame(rows))
        else:
            st.info("Ippudu ye coin lo rule match avvaledu. Kotha candle close ayyaka auto ga update avthundi.")

        # open trades
        if len(tr):
            op = tr[tr["Result"] == "OPEN"]
        else:
            op = pd.DataFrame()
        st.subheader(f"Open trades ({len(op)})")
        if len(op):
            rows = []
            for _, r in op.sort_values("Symbol").iterrows():
                rows.append({
                    "Coin": r["Symbol"], "Side": r["Side"],
                    "Entry Candle (IST)": fmt_time(r["Entry Candle"]),
                    "Entry": fmt_price(r["Entry"]),
                    "Stop Loss": fmt_price(r["SL"]),
                    "Take Profit": fmt_price(r["TP"]),
                    "Last Close": fmt_price(r["Exit"]),
                    "Running PnL": r["PnL"],
                })
            show(pd.DataFrame(rows), ["Running PnL"], {"Running PnL": "{:,.2f}"})
        else:
            st.caption("Open trade ledu.")

        st.subheader(f"Recent signals (last {recent} candles)")
        rec = [x for x in setups if x["age"] < recent]
        if rec:
            rows = []
            for x in sorted(rec, key=lambda z: (z["age"], z["Symbol"])):
                rows.append({
                    "Coin": x["Symbol"], "Side": x["side"],
                    "Signal Time (IST)": fmt_time(x["signal_time"]),
                    "Entry": fmt_price(x["entry"]),
                    "Stop Loss": fmt_price(x["sl"]),
                    "Take Profit": fmt_price(x["tp"]),
                    "Status": status_text(x["status"]),
                })
            show(pd.DataFrame(rows))
        else:
            st.caption("Ee range lo signals levu.")

    # ---------------- Backtest ----------------
    with tab2:
        st.subheader("Backtest result")
        st.caption(
            f"Period: last {days} days  |  Size per trade: {size:,.0f} USDT  |  No fees  |  "
            f"RR 1:{p['rr']:g}  |  Same candle lo SL and TP renda hit ayithe SL count avthundi (safe side)."
        )
        if tr is None or len(tr) == 0:
            st.info("Ee period lo trades levu.")
            return
        sm, closed = summarize(tr)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net PnL (USDT)", f"{sm['net']:,.2f}")
        c2.metric("Win Rate", f"{sm['wr']:.1f}%")
        c3.metric("Total Trades", f"{sm['trades']}")
        c4.metric("Wins / Losses", f"{sm['wins']} / {sm['losses']}")
        c5, c6, c7, c8 = st.columns(4)
        pf = "inf" if sm["pf"] == float("inf") else f"{sm['pf']:.2f}"
        c5.metric("Profit Factor", pf)
        c6.metric("Max Drawdown (USDT)", f"{sm['dd']:,.2f}")
        c7.metric("Avg R per trade", f"{sm['avg_r']:.2f}")
        c8.metric("Best / Worst trade", f"{sm['best']:,.0f} / {sm['worst']:,.0f}")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Avg Win", f"{sm['avg_win']:,.2f}")
        c10.metric("Avg Loss", f"{sm['avg_loss']:,.2f}")
        n_open = int((tr["Result"] == "OPEN").sum())
        open_pnl = float(tr.loc[tr["Result"] == "OPEN", "PnL"].sum()) if n_open else 0.0
        c11.metric("Open trades", f"{n_open}")
        c12.metric("Open PnL (not counted)", f"{open_pnl:,.2f}")

        if len(closed):
            st.subheader("Equity curve (USDT)")
            eq = closed[["ExitTS", "PnL"]].copy()
            eq["Equity"] = eq["PnL"].cumsum()
            st.line_chart(eq.set_index("ExitTS")["Equity"])

            st.subheader("Coin-wise result")
            g = closed.groupby("Symbol")
            by = pd.DataFrame({
                "Trades": g.size(),
                "Wins": g["Result"].apply(lambda s: int((s == "WIN").sum())),
                "Losses": g["Result"].apply(lambda s: int((s == "LOSS").sum())),
                "PnL": g["PnL"].sum(),
                "Avg R": g["R"].mean(),
            }).reset_index().rename(columns={"Symbol": "Coin"})
            by["Win Rate %"] = by["Wins"] / by["Trades"] * 100.0
            by = by[["Coin", "Trades", "Wins", "Losses", "Win Rate %", "PnL", "Avg R"]]
            by = by.sort_values("PnL", ascending=False)
            show(by, ["PnL", "Avg R"], {"Win Rate %": "{:.1f}", "PnL": "{:,.2f}", "Avg R": "{:.2f}"})

        st.subheader("All trades")
        coins = ["All"] + sorted(tr["Symbol"].unique().tolist())
        pick = st.selectbox("Coin filter", coins, key="coin_filter")
        v = tr if pick == "All" else tr[tr["Symbol"] == pick]
        v = v.sort_values("ExitTS", ascending=False)
        rows = []
        for _, r in v.iterrows():
            rows.append({
                "Coin": r["Symbol"], "Side": r["Side"],
                "Signal Time (IST)": fmt_time(r["Signal Time"]),
                "Entry Candle (IST)": fmt_time(r["Entry Candle"]),
                "Entry": fmt_price(r["Entry"]),
                "Stop Loss": fmt_price(r["SL"]),
                "Take Profit": fmt_price(r["TP"]),
                "Exit Candle (IST)": fmt_time(r["Exit Candle"]),
                "Exit": fmt_price(r["Exit"]),
                "Exit By": r["Exit Type"],
                "Result": r["Result"],
                "PnL": r["PnL"],
                "Return %": r["Return %"],
                "R": r["R"],
            })
        tdf = pd.DataFrame(rows)
        show(tdf, ["PnL", "Return %", "R"], {"PnL": "{:,.2f}", "Return %": "{:.2f}", "R": "{:.2f}"})
        st.download_button(
            "Download trades CSV",
            tdf.to_csv(index=False).encode("utf-8"),
            file_name=f"kishore_signals_{tf}_trades.csv",
            mime="text/csv",
        )

    if failed:
        with st.expander(f"Data raani coins ({len(failed)})"):
            st.write(", ".join(failed))


def main():
    st.set_page_config(page_title="Kishore Signals", page_icon="📈", layout="wide")
    st.title("Kishore Signals - Sell Scanner")

    with st.sidebar:
        st.header("Settings")
        tf = st.selectbox("Timeframe", ["1d", "4h", "1h", "30m", "15m"], index=0)
        days = st.slider("Backtest period (days)", 30, 1460, 365, step=5)
        top_n = st.slider("Top coins (market cap + liquidity)", 10, 100, 100, step=5)
        size = st.number_input("Trade size per trade (USDT)", min_value=1000, max_value=100_000_000,
                               value=100_000, step=10_000)
        recent = st.slider("Recent signals - last N candles", 1, 30, 3)
        st.subheader("Rules")
        trend = st.number_input("Minimum previous trend candles", 1, 10, 1, step=1)
        fill = st.number_input("Previous candle BODY fill (%)", 1.0, 100.0, 25.0, step=1.0)
        entry = st.number_input("Sweep candle retracement entry (%)", 1.0, 99.0, 50.0, step=1.0)
        slbuf = st.number_input("SL buffer beyond wick (%)", 0.0, 10.0, 0.1, step=0.05)
        rr = st.number_input("Risk to Reward (1:X)", 0.5, 20.0, 2.0, step=0.5)

    p = {"trend": int(trend), "fill": float(fill), "entry": float(entry),
         "slbuf": float(slbuf), "rr": float(rr)}

    symbols, src = get_universe(int(top_n))

    frag = getattr(st, "fragment", None)
    if frag is not None:
        runner = frag(run_every=60)(render)
    else:
        runner = render
    runner(symbols, tf, int(days), p, float(size), int(recent), src)


if __name__ == "__main__":
    main()

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ===== LOAD ENV =====
load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TG_TOKEN or not TG_CHAT_ID:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment variables")

# ===== CONFIG =====
# Data source: Binance Futures public API (for OHLC/volume)
BASE_URL = "https://fapi.binance.com"

# Top 10 high-liquidity, high-reputation futures pairs (adjust as you like)
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "TONUSDT",
    "AVAXUSDT",
]

SCAN_INTERVAL_SEC = 30  # scan every 60s

SL_PCT = 0.002     # 0.2% stop suggestion
TP1_PCT = 0.0035   # 0.35% TP1
TP2_PCT = 0.0065   # 0.65% TP2

IST = timezone(timedelta(hours=5, minutes=30))

INDIA_TARGET_SIGNALS = 4
US_LDN_TARGET_SIGNALS = 6

session_signal_count = 0
current_session_label = None

# Timeframes:
HIGHER_TF = "4h"    # trend
MID_TF_1 = "1h"     # zones
MID_TF_2 = "15m"    # zones
TRIGGER_TF_1 = "5m" # candlestick reaction / fakeouts
TRIGGER_TF_2 = "1m" # RSI filter

# Strict rating threshold
MIN_RATING = 70  # only send signals with rating >= 70/100


# ===== TELEGRAM =====
def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {
            "chat_id": TG_CHAT_ID,
            "text": msg[:4000],
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Telegram send failed:", e)


# ===== DATA / INDICATORS =====
def get_futures_klines(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BASE_URL + "/fapi/v1/klines", params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length=9):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["rsi9"] = rsi(df["close"], 9)
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


# ===== SWINGS / ZONES (EARLY S/R) =====
def find_swings(df: pd.DataFrame, lookback: int = 2):
    highs = []
    lows = []
    for i in range(lookback, len(df) - lookback):
        win_high = df["high"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == win_high.max():
            highs.append(i)

        win_low = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["low"].iloc[i] == win_low.min():
            lows.append(i)
    return highs, lows


def build_zones_from_swings(df: pd.DataFrame, highs, lows, tolerance_pct=0.15):
    zones = []
    prices = []

    for i in highs:
        price = df["high"].iloc[i]
        prices.append(("res", price))
    for i in lows:
        price = df["low"].iloc[i]
        prices.append(("sup", price))

    prices_sorted = sorted(prices, key=lambda x: x[1])

    for kind, price in prices_sorted:
        if not zones:
            width = price * tolerance_pct / 100.0
            zones.append({"type": kind, "low": price - width, "high": price + width})
            continue

        last = zones[-1]
        mid = (last["low"] + last["high"]) / 2
        if abs(price - mid) / mid < 0.003:  # within 0.3% → merge
            last["low"] = min(last["low"], price)
            last["high"] = max(last["high"], price)
        else:
            width = price * tolerance_pct / 100.0
            zones.append({"type": kind, "low": price - width, "high": price + width})

    return zones


def price_in_zone(price: float, zones, direction: str | None):
    candidates = []
    for z in zones:
        if z["low"] <= price <= z["high"]:
            if direction == "long" and z["type"] == "sup":
                candidates.append(z)
            elif direction == "short" and z["type"] == "res":
                candidates.append(z)
            elif direction is None:
                candidates.append(z)

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: x["high"] - x["low"])[0]


# ===== MULTI-TIMEFRAME LONG SIGNAL (HIGH RISK / HIGH REWARD) =====
def multi_tf_long_signal(sym: str):
    """
    Returns (ok, score, context) for a high-risk long setup using:
      - 4h / 1h trend
      - 1h & 15m support zones
      - 5m trigger candle (rejection) + 1m RSI + volume filter
    """

    # 4H / 1H: overall trend
    df_4h = compute_indicators(get_futures_klines(sym, HIGHER_TF, 200))
    df_1h = compute_indicators(get_futures_klines(sym, MID_TF_1, 200))

    last_4h = df_4h.iloc[-1]
    last_1h = df_1h.iloc[-1]

    ema_trend_up_4h = last_4h["ema9"] > last_4h["ema21"] > last_4h["ema50"]
    ema_trend_up_1h = last_1h["ema9"] > last_1h["ema21"] > last_1h["ema50"]

    if not (ema_trend_up_4h and ema_trend_up_1h):
        return False, None, None

    # Zones from 1h and 15m swings
    highs_1h, lows_1h = find_swings(df_1h)
    df_15m = compute_indicators(get_futures_klines(sym, MID_TF_2, 200))
    highs_15m, lows_15m = find_swings(df_15m)

    zones_1h = build_zones_from_swings(df_1h, highs_1h, lows_1h)
    zones_15m = build_zones_from_swings(df_15m, highs_15m, lows_15m)

    # Trigger: 5m candlestick and 1m RSI
    df_5m = compute_indicators(get_futures_klines(sym, TRIGGER_TF_1, 200))
    df_1m = compute_indicators(get_futures_klines(sym, TRIGGER_TF_2, 200))

    if len(df_5m) < 2 or len(df_1m) < 1:
        return False, None, None

    c5_prev = df_5m.iloc[-2]
    c5 = df_5m.iloc[-1]
    c1 = df_1m.iloc[-1]

    price = c1["close"]

    # Must be near a support zone (early S/R)
    all_zones = zones_1h + zones_15m
    zone = price_in_zone(price, all_zones, direction="long")
    if not zone:
        return False, None, None

    # 5m rejection candle near support (fake breakdown filter)
    wick_low = c5["low"]
    close_5m = c5["close"]
    open_5m = c5["open"]
    high_5m = c5["high"]

    body = abs(close_5m - open_5m)
    full_range = high_5m - wick_low
    long_lower_wick = (open_5m - wick_low > body * 1.2) or (close_5m - wick_low > body * 1.2)

    in_zone_5m = (zone["low"] <= wick_low <= zone["high"]) or (zone["low"] <= close_5m <= zone["high"])

    if not (in_zone_5m and long_lower_wick and close_5m > open_5m):
        return False, None, None

    # 1m RSI oversold to neutral
    if not (30 <= c1["rsi9"] <= 50):
        return False, None, None

    # Volume confirmation: 5m volume > 1.5x MA
    vol_score_5m = c5["volume"] / (c5["vol_ma"] + 1e-9)
    if vol_score_5m < 1.5:
        return False, None, None

    # Score: reward strong volume and price close to zone mid
    zone_mid = (zone["low"] + zone["high"]) / 2
    dist_to_zone_mid = abs(price - zone_mid) / price
    score = vol_score_5m / (1 + dist_to_zone_mid * 20)

    context = {
        "price": price,
        "zone": zone,
        "vol_score": vol_score_5m,
        "rsi": c1["rsi9"],
        "trend_desc": "4H & 1H uptrend near support zone",
    }
    return True, score, context


# ===== SESSION LOGIC =====
def get_session_label(now_ist: datetime):
    h = now_ist.hour
    m = now_ist.minute

    # India session: 09:15–15:30 IST
    if (h > 9 or (h == 9 and m >= 15)) and (h < 15 or (h == 15 and m <= 30)):
        return "INDIA"

    # US/London overlap: 18:30–23:30 IST
    if (h > 18 or (h == 18 and m >= 30)) and (h < 23 or (h == 23 and m <= 30)):
        return "US_LDN"

    return None


def get_session_target(label: str) -> int:
    if label == "INDIA":
        return INDIA_TARGET_SIGNALS
    if label == "US_LDN":
        return US_LDN_TARGET_SIGNALS
    return 0


# ===== MAIN LOOP =====
def main_loop():
    global session_signal_count, current_session_label

    print("Starting High-Risk Multi-TF Scalp Signal Bot (alerts only)…")
    send_telegram("📡 *High-Risk Multi-TF Scalp Signal Bot* started (alerts only, no auto-trading).")

    while True:
        try:
            now_ist = datetime.now(IST)
            session_label = get_session_label(now_ist)

            # Session change
            if session_label != current_session_label:
                session_signal_count = 0
                current_session_label = session_label
                if session_label:
                    msg = f"New session: {session_label} at {now_ist}."
                    print(msg)
                    send_telegram(f"🕒 {msg}")
                else:
                    msg = f"Outside trading sessions at {now_ist}."
                    print(msg)
                    send_telegram(f"🕒 {msg} No new signals.")

            if session_label is None:
                print(f"[{now_ist}] No active session. Sleeping {SCAN_INTERVAL_SEC}s.")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            target_signals = get_session_target(session_label)
            if session_signal_count >= target_signals:
                print(f"[{now_ist}] Session {session_label} reached target signals "
                      f"({session_signal_count}/{target_signals}). Waiting for next session.")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            print(f"[{now_ist}] Scanning {len(SYMBOLS)} symbols in session {session_label} "
                  f"(signals so far: {session_signal_count}/{target_signals})…")

            best_sym = None
            best_score = None
            best_ctx = None

            for sym in SYMBOLS:
                try:
                    ok, score, ctx = multi_tf_long_signal(sym)
                    if ok:
                        print(f"  ✓ Multi-TF long candidate on {sym} with raw score {score:.2f}")
                        if best_score is None or score > best_score:
                            best_score = score
                            best_sym = sym
                            best_ctx = ctx
                    else:
                        print(f"  - No valid multi-TF long on {sym}")
                except Exception as e_sym:
                    print(f"Error checking {sym}: {e_sym}")

            if best_sym and best_ctx:
                price = best_ctx["price"]
                sl_price = price * (1 - SL_PCT)
                tp1_price = price * (1 + TP1_PCT)
                tp2_price = price * (1 + TP2_PCT)

                # rating out of 100 from volume score
                raw_rating = min(best_ctx["vol_score"] * 20, 100)
                rating_int = int(round(raw_rating))

                if rating_int < MIN_RATING:
                    print(f"[{now_ist}] Best candidate {best_sym} rating {rating_int}/100 < {MIN_RATING}, no signal sent.")
                else:
                    trend_desc = best_ctx["trend_desc"]
                    leverage_hint = 10  # high-risk profile

                    reason = (
                        f"{trend_desc}, early support zone "
                        f"[{best_ctx['zone']['low']:.4f}–{best_ctx['zone']['high']:.4f}], "
                        f"1m RSI9={best_ctx['rsi']:.1f} from oversold, "
                        f"5m rejection candle with volume spike {best_ctx['vol_score']:.2f}x vs 20-bar MA."
                    )

                    msg = (
                        "📈 *HIGH-RISK SCALP SIGNAL*\n"
                        f"Session: *{session_label}*\n"
                        f"Coin: *{best_sym}*\n"
                        f"Trend: {trend_desc}\n"
                        f"Rating: *{rating_int}/100*\n"
                        f"Entry: `{price:.4f}`\n"
                        f"Book Profit (TP2): `{tp2_price:.4f}` (~{TP2_PCT*100:.2f}%)\n"
                        f"Partial Profit (TP1): `{tp1_price:.4f}` (~{TP1_PCT*100:.2f}%)\n"
                        f"Stop Loss: `{sl_price:.4f}` (~{SL_PCT*100:.2f}%)\n"
                        f"Suggested Leverage: *{leverage_hint}x* (high risk)\n"
                        f"Reason: {reason}\n\n"
                        "_Manual execution on Pi42 futures._"
                    )

                    print(
                        f"==> HIGH-RISK SIGNAL: {best_sym} | rating {rating_int}/100 | "
                        f"entry {price:.4f} | SL {sl_price:.4f} | TP1 {tp1_price:.4f} | TP2 {tp2_price:.4f}"
                    )
                    send_telegram(msg)
                    session_signal_count += 1
            else:
                print(f"[{now_ist}] No high-risk candidate above filters this cycle.")

        except Exception as e:
            print("Error in main loop:", e)
            send_telegram(f"⚠️ Error in signal loop: {e}")

        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()
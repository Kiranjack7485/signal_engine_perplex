import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ===== LOAD ENV =====
load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TG_TOKEN or not TG_CHAT_ID:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment variables")

# ===== CONFIG =====
BASE_URL = "https://api.binance.com"

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

SCAN_INTERVAL_SEC = 60

SL_PCT = 0.003      # 0.3% SL
TP1_PCT = 0.006     # 0.6% TP1
TP2_PCT = 0.012     # 1.2% TP2

IST = timezone(timedelta(hours=5, minutes=30))

TF_15M = "15m"
TF_1H = "1h"
TF_4H = "4h"
TF_1D = "1d"

MIN_RATING = 65           # trend rating threshold
MIN_15M_VOL_FACTOR = 1.3  # 15m volume must be >= 1.3x MA20
ATR_LEN_15M = 14          # ATR length for 15m extension check


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
def get_klines(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BASE_URL + "/api/v3/klines", params=params, timeout=8)
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


def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(length).mean()
    return df


# ===== TREND SCORING =====
def timeframe_trend_score(row) -> int:
    price = row["close"]
    ema20 = row["ema20"]
    ema50 = row["ema50"]

    if price > ema20 and ema20 > ema50:
        return 2
    elif price > ema20 and ema20 <= ema50:
        return 1
    elif price < ema20 and ema20 < ema50:
        return -2
    else:
        return -1


def build_trend_context(sym: str):
    df_1d = compute_basic_indicators(get_klines(sym, TF_1D, 200))
    df_4h = compute_basic_indicators(get_klines(sym, TF_4H, 200))
    df_1h = compute_basic_indicators(get_klines(sym, TF_1H, 200))
    df_15m = compute_basic_indicators(get_klines(sym, TF_15M, 200))
    df_15m = compute_atr(df_15m, ATR_LEN_15M)

    last_1d = df_1d.iloc[-1]
    last_4h = df_4h.iloc[-1]
    last_1h = df_1h.iloc[-1]
    last_15m = df_15m.iloc[-1]

    score_1d = timeframe_trend_score(last_1d)
    score_4h = timeframe_trend_score(last_4h)
    score_1h = timeframe_trend_score(last_1h)
    score_15m = timeframe_trend_score(last_15m)

    trend_score = score_1d + score_4h + score_1h + score_15m
    trend_rating = 100 * (trend_score + 8) / 16

    bonus = 0
    if 40 <= last_15m["rsi14"] <= 60:
        bonus += 5
    if trend_rating > 60 and last_15m["rsi14"] < 30:
        bonus += 5
    if last_1h["volume"] > 1.2 * (last_1h["vol_ma20"] + 1e-9):
        bonus += 3

    final_rating = max(0, min(100, trend_rating + bonus))

    context = {
        "trend_rating": final_rating,
        "trend_score_raw": trend_score,
        "rsi_1h": last_1h["rsi14"],
        "vol_1h": last_1h["volume"],
        "vol_ma_1h": last_1h["vol_ma20"],
        "score_1d": score_1d,
        "score_4h": score_4h,
        "score_1h": score_1h,
        "score_15m": score_15m,
    }
    return final_rating, context, df_15m


# ===== LONG SIGNAL LOGIC =====
def trend_long_signal(sym: str):
    """
    Long-only signal:
      - Multi-TF trend rating >= MIN_RATING
      - 15m price > EMA20
      - 15m RSI in [35, 70]
      - 15m volume >= MIN_15M_VOL_FACTOR * 15m vol MA20
      - Reversal risk low: not (15m extended AND 1h overbought strong up)
    """
    rating, ctx, df_15m = build_trend_context(sym)
    last_15m = df_15m.iloc[-1]

    price_15m = last_15m["close"]
    ema20_15m = last_15m["ema20"]
    rsi_15m = last_15m["rsi14"]
    vol_15m = last_15m["volume"]
    vol_ma20_15m = last_15m["vol_ma20"]
    atr_15m = last_15m["atr"]

    vol_factor_15m = vol_15m / (vol_ma20_15m + 1e-9)

    extended_15m = (price_15m > ema20_15m + 2 * atr_15m) or (rsi_15m > 70)
    extended_1h = (ctx["score_1h"] == 2 and ctx["rsi_1h"] > 70)

    ctx["price"] = price_15m
    ctx["rsi_15m"] = rsi_15m
    ctx["vol_15m"] = vol_15m
    ctx["vol_ma20_15m"] = vol_ma20_15m
    ctx["vol_factor_15m"] = vol_factor_15m
    ctx["atr14_15m"] = atr_15m
    ctx["extended_15m"] = extended_15m
    ctx["extended_1h"] = extended_1h

    basic_trend_ok = rating >= MIN_RATING
    basic_15m_ok = (price_15m > ema20_15m) and (35 <= rsi_15m <= 70)
    vol_ok = vol_factor_15m >= MIN_15M_VOL_FACTOR
    reversal_risk_low = not (extended_15m and extended_1h)

    ok = basic_trend_ok and basic_15m_ok and vol_ok and reversal_risk_low
    return ok, rating, ctx


# ===== SESSION LOGIC =====
def get_session_label(now_ist: datetime):
    h = now_ist.hour
    m = now_ist.minute

    if (h > 9 or (h == 9 and m >= 15)) and (h < 15 or (h == 15 and m <= 30)):
        return "INDIA"
    if (h > 18 or (h == 18 and m >= 30)) and (h < 23 or (h == 23 and m <= 30)):
        return "US_LDN"
    return None


def describe_tf_trend(score: int) -> str:
    if score == 2:
        return "strong up"
    if score == 1:
        return "mild up"
    if score == -1:
        return "mild down/choppy"
    if score == -2:
        return "strong down"
    return "neutral"


# ===== MAIN LOOP =====
def main_loop():
    current_session_label = None

    print("Bot started: Multi-TF Trend + Volume + Reversal-Risk (alerts only)")
    send_telegram("📡 *Bot started*: Multi-TF Trend + 15m Volume + Reversal-Risk (alerts only, no auto-trading).")

    while True:
        try:
            now_ist = datetime.now(IST)
            session_label = get_session_label(now_ist)

            if session_label != current_session_label:
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
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            best_sym = None
            best_rating = None
            best_ctx = None

            for sym in SYMBOLS:
                try:
                    ok, rating, ctx = trend_long_signal(sym)
                    if ok and (best_rating is None or rating > best_rating):
                        best_rating = rating
                        best_sym = sym
                        best_ctx = ctx
                except Exception as e_sym:
                    print(f"Error checking {sym}: {e_sym}")
                    send_telegram(f"⚠️ Error checking {sym}: {e_sym}")

            if best_sym and best_ctx:
                price = best_ctx["price"]
                sl_price = price * (1 - SL_PCT)
                tp1_price = price * (1 + TP1_PCT)
                tp2_price = price * (1 + TP2_PCT)

                rating_int = int(round(best_ctx["trend_rating"]))
                trend_1d = describe_tf_trend(best_ctx["score_1d"])
                trend_4h = describe_tf_trend(best_ctx["score_4h"])
                trend_1h = describe_tf_trend(best_ctx["score_1h"])
                trend_15m = describe_tf_trend(best_ctx["score_15m"])

                leverage_hint = 7
                reversal_note = "low"
                if best_ctx["extended_15m"] or best_ctx["extended_1h"]:
                    reversal_note = "elevated"

                reason = (
                    f"Trend: 1D={trend_1d}, 4H={trend_4h}, 1H={trend_1h}, 15m={trend_15m}; "
                    f"15m price above EMA20, 15m RSI14={best_ctx['rsi_15m']:.1f}; "
                    f"15m volume {best_ctx['vol_15m']:.0f} vs MA20 {best_ctx['vol_ma20_15m']:.0f} "
                    f"(factor {best_ctx['vol_factor_15m']:.2f}x); "
                    f"1H volume {best_ctx['vol_1h']:.0f} vs MA20 {best_ctx['vol_ma_1h']:.0f}; "
                    f"15m ATR14={best_ctx['atr14_15m']:.4f}, reversal risk={reversal_note}."
                )

                msg = (
                    "📈 *TREND + VOLUME + REVERSAL-RISK SCALP SIGNAL (LONG)*\n"
                    f"Session: *{session_label}*\n"
                    f"Coin: *{best_sym}*\n"
                    f"Trend Rating: *{rating_int}/100*\n"
                    f"Entry (15m close): `{price:.4f}`\n"
                    f"Book Profit (TP2): `{tp2_price:.4f}` (~{TP2_PCT*100:.2f}%)\n"
                    f"Partial Profit (TP1): `{tp1_price:.4f}` (~{TP1_PCT*100:.2f}%)\n"
                    f"Stop Loss: `{sl_price:.4f}` (~{SL_PCT*100:.2f}%)\n"
                    f"Suggested Leverage: *{leverage_hint}x* (trend-following)\n"
                    f"Reason: {reason}\n\n"
                    "_Manual execution on Pi42 futures._"
                )

                print(f"Signal sent: {best_sym} | rating {rating_int}/100 | entry {price:.4f}")
                send_telegram(msg)

        except Exception as e:
            print("Error in main loop:", e)
            send_telegram(f"⚠️ Error in signal loop: {e}")

        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()
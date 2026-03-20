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
    raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

# ===== CONFIG =====
BASE_URL = "https://fapi.binance.com"  # data source, adjust later if needed
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

SCAN_INTERVAL_SEC = 60

SL_PCT = 0.002     # 0.2% stop suggestion
TP1_PCT = 0.0035   # 0.35% TP1 suggestion
TP2_PCT = 0.0065   # 0.65% TP2 suggestion

IST = timezone(timedelta(hours=5, minutes=30))

INDIA_TARGET_SIGNALS = 4
US_LDN_TARGET_SIGNALS = 6

session_signal_count = 0
current_session_label = None


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
def get_futures_klines(symbol: str, interval: str = "1m", limit: int = 200):
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


def compute_indicators(df_1m, df_5m):
    for df in (df_1m, df_5m):
        df["ema9"] = ema(df["close"], 9)
        df["ema21"] = ema(df["close"], 21)
        df["ema50"] = ema(df["close"], 50)
        df["rsi9"] = rsi(df["close"], 9)
        df["vol_ma"] = df["volume"].rolling(20).mean()
    return df_1m, df_5m


def check_long_signal(df_1m, df_5m):
    c1 = df_1m.iloc[-1]
    c5 = df_5m.iloc[-1]

    trend_5m = c5["ema9"] > c5["ema21"] > c5["ema50"]
    trend_1m = c1["ema9"] > c1["ema21"] > c1["ema50"]
    if not (trend_5m and trend_1m):
        return False, None

    price = c1["close"]
    in_zone = c1["ema21"] * 0.999 <= price <= c1["ema21"] * 1.001
    if not in_zone:
        return False, None

    rsi_ok = 30 <= c1["rsi9"] <= 50
    if not rsi_ok:
        return False, None

    vol_ok = c1["volume"] > 1.5 * c1["vol_ma"]
    if not vol_ok:
        return False, None

    score = c1["volume"] / (c1["vol_ma"] + 1e-9)
    return True, score


# ===== SESSION LOGIC =====
def get_session_label(now_ist: datetime):
    h = now_ist.hour
    m = now_ist.minute

    if (h > 9 or (h == 9 and m >= 15)) and (h < 15 or (h == 15 and m <= 30)):
        return "INDIA"

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

    print("Starting Scalp Signal Bot (alerts only)…")
    send_telegram("📡 *Scalp Signal Bot* started (alerts only, no auto-trading).")

    while True:
        try:
            now_ist = datetime.now(IST)
            session_label = get_session_label(now_ist)

            # Detect and log session changes
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

            print(f"[{now_ist}] Scanning symbols in session {session_label} "
                  f"(signals so far: {session_signal_count}/{target_signals})…")

            best_sym = None
            best_score = None
            best_row = None

            for sym in SYMBOLS:
                df_1m = get_futures_klines(sym, "1m", 200)
                df_5m = get_futures_klines(sym, "5m", 200)
                df_1m, df_5m = compute_indicators(df_1m, df_5m)

                ok, score = check_long_signal(df_1m, df_5m)
                if ok:
                    print(f"  ✓ Candidate long on {sym} with score {score:.2f}")
                    if best_score is None or score > best_score:
                        best_score = score
                        best_sym = sym
                        best_row = df_1m.iloc[-1]
                else:
                    print(f"  - No valid long signal on {sym}")

            if best_sym:
                price = best_row["close"]
                sl_price = price * (1 - SL_PCT)
                tp1_price = price * (1 + TP1_PCT)
                tp2_price = price * (1 + TP2_PCT)

                raw_rating = min(best_score * 20, 100)
                rating_int = int(round(raw_rating))

                trend_desc = "Strong Uptrend (EMA9>21>50 on 1m & 5m)"
                leverage_hint = 7  # manual leverage hint

                reason = (
                    f"EMA trend up 1m & 5m, pullback near EMA21, "
                    f"RSI9={best_row['rsi9']:.1f} in 30–50, "
                    f"volume spike {best_score:.2f}x vs 20-bar MA."
                )

                msg = (
                    "📈 *SCALP SIGNAL*\n"
                    f"Session: *{session_label}*\n"
                    f"Coin: *{best_sym}*\n"
                    f"Trend: {trend_desc}\n"
                    f"Rating: *{rating_int}/100*\n"
                    f"Entry: `{price:.4f}`\n"
                    f"Book Profit (TP2): `{tp2_price:.4f}` (~{TP2_PCT*100:.2f}%)\n"
                    f"Partial Profit (TP1): `{tp1_price:.4f}` (~{TP1_PCT*100:.2f}%)\n"
                    f"Stop Loss: `{sl_price:.4f}` (~{SL_PCT*100:.2f}%)\n"
                    f"Suggested Leverage: *{leverage_hint}x*\n"
                    f"Reason: {reason}\n\n"
                    "_You execute manually on Pi42 futures._"
                )

                print(f"==> STRONG SIGNAL: {best_sym} | rating {rating_int}/100 | "
                      f"entry {price:.4f} | SL {sl_price:.4f} | TP1 {tp1_price:.4f} | TP2 {tp2_price:.4f}")
                send_telegram(msg)
                session_signal_count += 1
            else:
                print(f"[{now_ist}] No strong signal this cycle.")

        except Exception as e:
            print("Error in main loop:", e)
            send_telegram(f"⚠️ Error in signal loop: {e}")

        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()

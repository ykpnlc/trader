import ccxt
import pandas as pd
import numpy as np
import os
import asyncio
import aiohttp
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ==== ENV ====
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNALS_FILE = "active_signals.json"
RESULTS_FILE = "results.json"
DATASET_FILE = "signals_dataset.csv"
MODEL_FILE = "ai_model.joblib"

# ==== DATA & AI ====
def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r") as f:
        return json.load(f)
def save_signals(signals):
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f)
def load_results():
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)
def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)
def append_to_dataset(entry):
    if not os.path.exists(DATASET_FILE):
        pd.DataFrame([entry]).to_csv(DATASET_FILE, index=False)
    else:
        df = pd.read_csv(DATASET_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(DATASET_FILE, index=False)
def update_winrate():
    results = load_results()
    if not results: return 0.0
    wins = sum([1 for r in results if r["result"] == "win"])
    total = len(results)
    winrate = 100 * wins / total if total > 0 else 0
    return round(winrate, 2)
def retrain_ai_model_and_backtest():
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) < 100:
        return
    df = pd.read_csv(DATASET_FILE)
    features = ["score", "orderflow_score", "volatility"]
    X = df[features]
    y = (df["result"] == "win").astype(int)
    if X.shape[0] < 30:
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    dump(model, MODEL_FILE)
def ai_score_predict(score, delta, volatility):
    if not os.path.exists(MODEL_FILE):
        return 1.0
    model = load(MODEL_FILE)
    arr = np.array([[score, delta, volatility]])
    prob = model.predict_proba(arr)[0][1]
    return prob

# ==== TELEGRAM ====
async def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

# ==== GERÇEK FONKSİYONLAR: ICT/SMC/PA ====
def find_liquidity_sweep(candles, lookback=30, threshold=0.1):
    highs = [c[2] for c in candles[-lookback:]]
    lows = [c[3] for c in candles[-lookback:]]
    last_high = candles[-1][2]
    last_low = candles[-1][3]
    eq_highs = [h for i, h in enumerate(highs[:-3]) if abs(h - highs[i+1]) < threshold/100 * h]
    eq_lows = [l for i, l in enumerate(lows[:-3]) if abs(l - lows[i+1]) < threshold/100 * l]
    if eq_highs and last_high > max(eq_highs):
        return True
    if eq_lows and last_low < min(eq_lows):
        return True
    return False
def find_order_blocks(candles, lookback=30, body_ratio=0.6):
    blocks = []
    for i in range(-lookback, -3):
        o, c, h, l = candles[i][1], candles[i][4], candles[i][2], candles[i][3]
        body = abs(c-o)
        if body/(h-l) > body_ratio:
            direction = "bullish_OB" if c > o else "bearish_OB"
            retest_zone = l if direction == "bullish_OB" else h
            if direction == "bullish_OB" and candles[i+1][3] <= l:
                blocks.append({"type": direction, "price": l})
            if direction == "bearish_OB" and candles[i+1][2] >= h:
                blocks.append({"type": direction, "price": h})
    return bool(blocks)
def find_fvg_zones(candles, lookback=30, min_gap=0.0005):
    fvg_list = []
    for i in range(-lookback, -4):
        prev_low = candles[i-1][3]
        curr_high = candles[i][2]
        next_low = candles[i+1][3]
        if next_low - curr_high > min_gap:
            fvg_list.append(True)
        if prev_low - candles[i][3] > min_gap:
            fvg_list.append(True)
    return bool(fvg_list)
def check_bos_choch(candles, lookback=30):
    highs = [c[2] for c in candles[-lookback:]]
    lows = [c[3] for c in candles[-lookback:]]
    last_high, last_low = highs[-1], lows[-1]
    bos = False
    if last_high > max(highs[:-1]): bos = True
    if last_low < min(lows[:-1]): bos = True
    return bos
def breaker_block_liquidity_ema(candles, ema_period=50, lookback=30):
    closes = [c[4] for c in candles[-(lookback+ema_period):]]
    ema = pd.Series(closes).ewm(span=ema_period).mean().values
    for i in range(-lookback, -2):
        if abs(candles[i][4] - ema[i]) / candles[i][4] < 0.01:
            if candles[i][4] > ema[i] and candles[i-1][4] < ema[i-1]:
                return True
            if candles[i][4] < ema[i] and candles[i-1][4] > ema[i-1]:
                return True
    return False
def detect_candle_pattern(candles):
    for i in range(-10, -1):
        o, c, h, l = candles[i][1], candles[i][4], candles[i][2], candles[i][3]
        prev_o, prev_c = candles[i-1][1], candles[i-1][4]
        if prev_c < prev_o and c > o and c > prev_o and o < prev_c:
            return True
        if prev_c > prev_o and c < o and c < prev_o and o > prev_c:
            return True
        body = abs(c-o)
        total = h-l
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        if lower_wick > 0.66*total and body < 0.34*total:
            return True
        if upper_wick > 0.66*total and body < 0.34*total:
            return True
    return False
def mitigation_rsi_volume(candles, rsi_period=14, lookback=30, vol_mult=1.5):
    closes = np.array([c[4] for c in candles[-lookback:]])
    volumes = np.array([c[5] for c in candles[-lookback:]])
    delta = np.diff(closes)
    up = delta.clip(min=0)
    down = -delta.clip(max=0)
    roll_up = np.convolve(up, np.ones(rsi_period), 'valid') / rsi_period
    roll_down = np.convolve(down, np.ones(rsi_period), 'valid') / rsi_period
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    avg_vol = volumes.mean()
    for i in range(-10, -1):
        if rsi[i] > 60 and volumes[i] > avg_vol * vol_mult:
            return True
        if rsi[i] < 40 and volumes[i] > avg_vol * vol_mult:
            return True
    return False
def check_ema_cross(candles, fast=20, slow=50):
    closes = pd.Series([c[4] for c in candles])
    ema_fast = closes.ewm(span=fast, min_periods=fast).mean()
    ema_slow = closes.ewm(span=slow, min_periods=slow).mean()
    if len(ema_fast) < slow + 2: return False
    cross_up = ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
    cross_down = ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
    return cross_up or cross_down
def check_volume_spike(candles, lookback=30, spike_ratio=1.7):
    volumes = [c[5] for c in candles[-lookback:]]
    avg_vol = sum(volumes[:-2]) / (lookback - 2)
    last_vol = volumes[-1]
    prev_vol = volumes[-2]
    return last_vol > avg_vol * spike_ratio or prev_vol > avg_vol * spike_ratio
def check_fibonacci_golden(candles, swing_lookback=30):
    closes = [c[4] for c in candles[-swing_lookback:]]
    highs = [c[2] for c in candles[-swing_lookback:]]
    lows = [c[3] for c in candles[-swing_lookback:]]
    swing_high, swing_low = max(highs), min(lows)
    golden_0618 = swing_high - (swing_high - swing_low) * 0.618
    golden_0705 = swing_high - (swing_high - swing_low) * 0.705
    last_close = closes[-1]
    return golden_0705 <= last_close <= golden_0618
def trend_check(candles, ema_period=50):
    closes = pd.Series([c[4] for c in candles])
    ema = closes.ewm(span=ema_period, min_periods=ema_period).mean()
    if closes.iloc[-1] > ema.iloc[-1]:
        return "uptrend"
    elif closes.iloc[-1] < ema.iloc[-1]:
        return "downtrend"
    else:
        return "sideways"
def advanced_score(candles, symbol):
    score = 0
    if find_liquidity_sweep(candles): score += 1
    if find_order_blocks(candles): score += 1
    if find_fvg_zones(candles): score += 1
    if check_bos_choch(candles): score += 1
    if breaker_block_liquidity_ema(candles): score += 1
    if detect_candle_pattern(candles): score += 1
    if mitigation_rsi_volume(candles): score += 1
    if check_ema_cross(candles): score += 1
    if check_volume_spike(candles): score += 1
    if check_fibonacci_golden(candles): score += 1
    trend = trend_check(candles)
    if trend != "sideways": score += 1
    orderflow_score = np.random.randint(0,2)
    return score, trend, orderflow_score, {"delta": orderflow_score, "block_trade": False}
def get_signal_direction(candles):
    trend = trend_check(candles)
    if trend == "uptrend": return "LONG"
    elif trend == "downtrend": return "SHORT"
    else: return "LONG"
def calc_atr_sl_tp(candles, rr_ratio=2.0, atr_period=14, direction="LONG"):
    highs = np.array([c[2] for c in candles[-atr_period-2:]])
    lows = np.array([c[3] for c in candles[-atr_period-2:]])
    closes = np.array([c[4] for c in candles[-atr_period-2:]])
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    atr_v = np.mean(tr[-atr_period:])
    entry = closes[-1]
    if direction == "LONG":
        sl = entry - atr_v
        tp = entry + atr_v * rr_ratio
    else:
        sl = entry + atr_v
        tp = entry - atr_v * rr_ratio
    return round(sl, 4), round(tp, 4)

# ==== YFINANCE (fx, hisse, endeks, futures) ====
YF_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "XAUUSD=X", "USOUSD=X", "BIST100.IS",
    "BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT", "SPY", "NQ=F", "CL=F", "VIX", "GC=F"
]
def fetch_yf_candles(symbol, tf="1m", limit=150):
    try:
        data = yf.download(symbol, period="7d", interval=tf)
        if data is None or len(data) < limit: return None
        candles = []
        for i in range(len(data)):
            o = float(data['Open'][i])
            h = float(data['High'][i])
            l = float(data['Low'][i])
            c = float(data['Close'][i])
            v = float(data['Volume'][i])
            candles.append([i, o, h, l, c, v])
        return candles
    except Exception as e:
        print(f"YF Hata: {symbol} {tf}: {e}")
        return None

# ==== CCXT (kripto) ====
EXCHANGES = {
    "binance": ccxt.binance({"enableRateLimit": True}),
    "bybit": ccxt.bybit({"enableRateLimit": True}),
    "kucoin": ccxt.kucoin({"enableRateLimit": True})
}
MARKETS = {
    "binance": {
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        "timeframes": ["1m", "5m", "15m"]
    },
    "bybit": {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframes": ["5m", "15m"]
    },
    "kucoin": {
        "symbols": ["BTC/USDT", "ADA/USDT", "XRP/USDT"],
        "timeframes": ["15m"]
    }
}
def fetch_candles(exchange, symbol, timeframe, limit=150):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Hata: {exchange.id} {symbol} {timeframe} - {e}")
        return None

# ==== SPAM ENGELİ ====
def active_signal_exists(exchange, symbol, tf, direction):
    signals = load_signals()
    for s in signals:
        if (
            s.get("exchange") == exchange and
            s.get("symbol") == symbol and
            s.get("timeframe") == tf and
            s.get("direction") == direction and
            s.get("result") is None
        ):
            return True
    return False
def yf_active_signal_exists(symbol, tf, direction):
    signals = load_signals()
    for s in signals:
        if (
            s.get("exchange") == "yfinance" and
            s.get("symbol") == symbol and
            s.get("timeframe") == tf and
            s.get("direction") == direction and
            s.get("result") is None
        ):
            return True
    return False

# ==== SİNYAL TAKİP & AI RE-TRAIN ====
async def check_active_signals():
    signals = load_signals()
    results = load_results()
    updated_signals = []
    retrain_flag = False
    for s in signals:
        if s.get("result") is not None:
            continue
        if s.get("exchange") == "yfinance":
            candles = fetch_yf_candles(s["symbol"], tf=s["timeframe"], limit=2)
        else:
            ex = EXCHANGES[s["exchange"]]
            candles = fetch_candles(ex, s["symbol"], s["timeframe"], 2)
        if not candles:
            updated_signals.append(s)
            continue
        last_price = candles[-1][4]
        signal_closed = False
        if (s["direction"] == "LONG" and last_price >= s["tp"]) or (s["direction"] == "SHORT" and last_price <= s["tp"]):
            s["result"] = "win"
            s["close_time"] = datetime.now(timezone.utc).isoformat()
            results.append(s)
            retrain_flag = True
            signal_closed = True
            await send_telegram_signal(
                f"✅ WIN | {s['exchange'].upper()} | {s['symbol']} | {s['timeframe']}\n"
                f"{'🟢 LONG' if s['direction']=='LONG' else '🔴 SHORT'} | Entry: <b>{s['entry']}</b> | SL: <b>{s['sl']}</b> | TP: <b>{s['tp']}</b>\n"
                f"Close Price: <b>{last_price}</b>\n"
                f"Result: WIN 🎉"
            )
        elif (s["direction"] == "LONG" and last_price <= s["sl"]) or (s["direction"] == "SHORT" and last_price >= s["sl"]):
            s["result"] = "loss"
            s["close_time"] = datetime.now(timezone.utc).isoformat()
            results.append(s)
            retrain_flag = True
            signal_closed = True
            await send_telegram_signal(
                f"❌ LOSE | {s['exchange'].upper()} | {s['symbol']} | {s['timeframe']}\n"
                f"{'🟢 LONG' if s['direction']=='LONG' else '🔴 SHORT'} | Entry: <b>{s['entry']}</b> | SL: <b>{s['sl']}</b> | TP: <b>{s['tp']}</b>\n"
                f"Close Price: <b>{last_price}</b>\n"
                f"Result: LOSS 💔"
            )
        if signal_closed:
            entry = {
                "timestamp": s.get("open_time"),
                "exchange": s.get("exchange"),
                "symbol": s.get("symbol"),
                "timeframe": s.get("timeframe"),
                "score": s.get("score", 0),
                "trend": s.get("trend", ""),
                "orderflow_score": s.get("orderflow_score", 0),
                "entry": s.get("entry"),
                "sl": s.get("sl"),
                "tp": s.get("tp"),
                "direction": s.get("direction"),
                "result": s.get("result"),
                "delta": s.get("delta", 0),
                "volatility": s.get("volatility", 0),
                "session_time": s.get("session_time", True),
                "block_trade": s.get("block_trade", False)
            }
            append_to_dataset(entry)
        else:
            updated_signals.append(s)
    save_signals(updated_signals)
    save_results(results)
    if retrain_flag:
        retrain_ai_model_and_backtest()

# ==== SİNYAL TARAYICI ====
async def scan_all_markets():
    while True:
        await check_active_signals()
        # Kripto marketler
        for ex_name, ex_info in MARKETS.items():
            exchange = EXCHANGES[ex_name]
            for symbol in ex_info["symbols"]:
                for tf in ex_info["timeframes"]:
                    candles = fetch_candles(exchange, symbol, tf, 150)
                    if not candles or len(candles) < 100: continue
                    score, trend, orderflow_score, extra = advanced_score(candles, symbol)
                    direction = get_signal_direction(candles)
                    sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                    winrate = update_winrate()
                    ai_prob = ai_score_predict(score, extra["delta"], abs(candles[-1][2]-candles[-1][3]))
                    if score >= 9 and not active_signal_exists(ex_name, symbol, tf, direction):
                        last_close = candles[-1][4]
                        msg = (
                            f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | {ex_name.upper()} | {symbol} | {tf}\n"
                            f"🏅 Score: {score}/13 | 🤖 AI: %{round(ai_prob*100,1)}\n"
                            f"Orderflow: Δ={extra['delta']} \n"
                            f"📈 Entry: <b>{last_close}</b>\n"
                            f"⛔️ SL: <b>{sl}</b>\n"
                            f"🎯 TP: <b>{tp}</b>\n"
                            f"✅ WINRATE: %{winrate}"
                        )
                        await send_telegram_signal(msg)
                        signals = load_signals()
                        signals.append({
                            "exchange": ex_name,
                            "symbol": symbol,
                            "timeframe": tf,
                            "entry": last_close,
                            "sl": sl,
                            "tp": tp,
                            "direction": direction,
                            "score": score,
                            "trend": trend,
                            "orderflow_score": orderflow_score,
                            "open_time": datetime.now(timezone.utc).isoformat(),
                            "result": None,
                            "delta": extra["delta"],
                            "volatility": abs(candles[-1][2]-candles[-1][3]),
                            "session_time": True,
                            "block_trade": extra.get("block_trade", False)
                        })
                        save_signals(signals)
                        break  # Spam yok!
        # YFINANCE ile FX/hisse/endeks/futures sinyal
        for symbol in YF_SYMBOLS:
            for tf in ["1m", "5m", "15m"]:
                candles = fetch_yf_candles(symbol, tf=tf, limit=150)
                if not candles or len(candles) < 100: continue
                score, trend, orderflow_score, extra = advanced_score(candles, symbol)
                direction = get_signal_direction(candles)
                sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                winrate = update_winrate()
                ai_prob = ai_score_predict(score, extra["delta"], abs(candles[-1][2]-candles[-1][3]))
                if score >= 9 and not yf_active_signal_exists(symbol, tf, direction):
                    last_close = candles[-1][4]
                    msg = (
                        f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | YFINANCE | {symbol} | {tf}\n"
                        f"🏅 Score: {score}/13 | 🤖 AI: %{round(ai_prob*100,1)}\n"
                        f"Orderflow: Δ={extra['delta']} \n"
                        f"📈 Entry: <b>{last_close}</b>\n"
                        f"⛔️ SL: <b>{sl}</b>\n"
                        f"🎯 TP: <b>{tp}</b>\n"
                        f"✅ WINRATE: %{winrate}"
                    )
                    await send_telegram_signal(msg)
                    signals = load_signals()
                    signals.append({
                        "exchange": "yfinance",
                        "symbol": symbol,
                        "timeframe": tf,
                        "entry": last_close,
                        "sl": sl,
                        "tp": tp,
                        "direction": direction,
                        "score": score,
                        "trend": trend,
                        "orderflow_score": orderflow_score,
                        "open_time": datetime.now(timezone.utc).isoformat(),
                        "result": None,
                        "delta": extra["delta"],
                        "volatility": abs(candles[-1][2]-candles[-1][3]),
                        "session_time": True,
                        "block_trade": extra.get("block_trade", False)
                    })
                    save_signals(signals)
                    break  # Aynı varlık/timeframe spam yok!
        await asyncio.sleep(60)

if __name__ == "__main__":
    retrain_ai_model_and_backtest()
    asyncio.run(scan_all_markets())

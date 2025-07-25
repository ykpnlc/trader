import ccxt
import pandas as pd
import numpy as np
import os
import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# ENV
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

SIGNALS_FILE = "active_signals.json"
RESULTS_FILE = "results.json"
AI_MODEL_FILE = "ai_model.pkl"
AI_DATASET_FILE = "ai_dataset.csv"

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
def update_winrate():
    results = load_results()
    if not results:
        return 0.0
    wins = sum([1 for r in results if r["result"] == "win"])
    total = len(results)
    winrate = 100 * wins / total if total > 0 else 0
    return round(winrate, 2)

def append_to_dataset(signal_dict, dataset_file=AI_DATASET_FILE):
    df = pd.DataFrame([signal_dict])
    if not os.path.exists(dataset_file):
        df.to_csv(dataset_file, index=False)
    else:
        df.to_csv(dataset_file, index=False, mode="a", header=False)

def retrain_ai_model_and_backtest(dataset_file=AI_DATASET_FILE):
    if not os.path.exists(dataset_file):
        print("Dataset yok, AI retrain skip.")
        return None
    df = pd.read_csv(dataset_file)
    for col in ["score", "trend", "delta", "volatility", "session_time"]:
        if col not in df.columns:
            df[col] = 0
    df["trend"] = df["trend"].replace({"uptrend": 1, "downtrend": -1, "sideways": 0})
    X = df[["score", "trend", "delta", "volatility", "session_time"]]
    y = (df["result"] == "win").astype(int)
    if len(df) < 50:
        print("Veri az, AI re-train/backtest atlanıyor.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    joblib.dump(model, AI_MODEL_FILE)
    score = model.score(X_test, y_test)
    print(f"AI Model retrain: {len(X_train)} train, {len(X_test)} test, Backtest Winrate: %{round(score*100,2)}")
    return model

def ai_score_predict(score, trend, delta, volatility, session_time):
    if not os.path.exists(AI_MODEL_FILE):
        return 1.0
    model = joblib.load(AI_MODEL_FILE)
    trend_num = 1 if trend == "uptrend" else (-1 if trend == "downtrend" else 0)
    X = np.array([[score, trend_num, delta, volatility, 1 if session_time else 0]])
    prob = model.predict_proba(X)[0][1]
    return prob

async def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

# === Orderflow (Binance Spot) ===
def get_orderflow_features(symbol="BTCUSDT"):
    try:
        from binance.client import Client as BinanceClient
        client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
        trades = client.get_recent_trades(symbol=symbol, limit=500)
        buy_vol = sum(float(t['qty']) for t in trades if not t['isBuyerMaker'])
        sell_vol = sum(float(t['qty']) for t in trades if t['isBuyerMaker'])
        delta = buy_vol - sell_vol
        block = any(float(t['qty']) > 25 for t in trades)
        return {"delta": delta, "block_trade": block}
    except Exception as e:
        return {"delta": 0, "block_trade": False}

def orderflow_confluence(symbol):
    of = get_orderflow_features(symbol.replace("/", ""))
    score = 0
    if abs(of["delta"]) > 20: score += 1
    if of["block_trade"]: score += 1
    return score, of

# === Gelişmiş TA Modülleri ===
def find_liquidity_zones(candles, lookback=30, threshold=0.15):
    equal_highs, equal_lows, liquidity_zones = [], [], []
    for i in range(2, lookback):
        h1 = candles[-i][2]; h2 = candles[-i-1][2]
        l1 = candles[-i][3]; l2 = candles[-i-1][3]
        if abs(h1 - h2) / h1 < threshold/100:
            equal_highs.append((i, h1))
        if abs(l1 - l2) / l1 < threshold/100:
            equal_lows.append((i, l1))
    last_high = candles[-1][2]; last_low = candles[-1][3]
    for idx, high in equal_highs:
        if last_high > high and candles[-1][4] < last_high:
            liquidity_zones.append({"type": "eq_high_sweep", "level": high})
    for idx, low in equal_lows:
        if last_low < low and candles[-1][4] > last_low:
            liquidity_zones.append({"type": "eq_low_sweep", "level": low})
    return liquidity_zones if liquidity_zones else False

def find_order_blocks(candles, lookback=30, body_ratio=0.6):
    order_blocks = []
    for i in range(-lookback, -1):
        o, c, h, l = candles[i][1], candles[i][4], candles[i][2], candles[i][3]
        body = abs(c - o)
        if (c < o and body / (h - l) > body_ratio):
            try:
                next1 = candles[i+1][4]; next2 = candles[i+2][4]
                if next1 > c and next2 > next1:
                    order_blocks.append({"type": "bullish_OB", "index": i, "low": l, "high": h})
            except: pass
        if (c > o and body / (h - l) > body_ratio):
            try:
                next1 = candles[i+1][4]; next2 = candles[i+2][4]
                if next1 < c and next2 < next1:
                    order_blocks.append({"type": "bearish_OB", "index": i, "low": l, "high": h})
            except: pass
    return order_blocks if order_blocks else False

def find_fvg_zones(candles, lookback=30, min_gap=0.0005):
    fvg_list = []
    for i in range(-lookback, -3):
        prev_low = candles[i-1][3]
        curr_high = candles[i][2]
        next_low = candles[i+1][3]
        curr_low = candles[i][3]
        if curr_high < next_low and (next_low - curr_high) > min_gap:
            fvg_list.append({"type": "bullish_fvg", "index": i, "gap_low": curr_high, "gap_high": next_low})
        if prev_low > curr_low and (prev_low - curr_low) > min_gap:
            fvg_list.append({"type": "bearish_fvg", "index": i, "gap_low": curr_low, "gap_high": prev_low})
    return fvg_list if fvg_list else False

def check_bos_choch(candles, lookback=30):
    bos, choch = False, False
    highs = [c[2] for c in candles[-lookback:]]
    lows = [c[3] for c in candles[-lookback:]]
    closes = [c[4] for c in candles[-lookback:]]
    highest, lowest, last_close = max(highs[:-1]), min(lows[:-1]), closes[-1]
    if highs[-1] > highest:
        bos = {"type": "BoS_up", "level": highs[-1], "prev_high": highest}
    if lows[-1] < lowest:
        bos = {"type": "BoS_down", "level": lows[-1], "prev_low": lowest}
    for i in range(-6, -2):
        if closes[i] > closes[i-1] and closes[-1] < closes[i]:
            choch = {"type": "Choch_down", "level": closes[-1], "prev_level": closes[i]}
        if closes[i] < closes[i-1] and closes[-1] > closes[i]:
            choch = {"type": "Choch_up", "level": closes[-1], "prev_level": closes[i]}
    return {"bos": bos, "choch": choch} if bos or choch else False

def breaker_block_liquidity_ema(candles, ema_period=50, lookback=30, proximity=0.1):
    closes = [c[4] for c in candles]
    ema = pd.Series(closes).ewm(span=ema_period).mean().values
    result = []
    for i in range(-lookback, -5):
        prev_low = candles[i-1][3]
        curr_high = candles[i][2]
        if abs(closes[i] - closes[i-2]) / closes[i] < proximity:
            if closes[i+1] < closes[i] and closes[i+2] > closes[i]:
                if abs(ema[i] - closes[i]) / closes[i] < proximity:
                    wick = candles[i][2] - candles[i][4] if closes[i] < closes[i-2] else candles[i][4] - candles[i][3]
                    if wick / (candles[i][2] - candles[i][3]) > 0.5:
                        result.append({"breaker_block_idx": i, "breaker_level": closes[i], "ema_value": ema[i]})
    return result if result else False

def detect_candle_pattern(candles, threshold=0.6):
    patterns = []
    for i in range(-10, -1):
        prev_high, prev_low = candles[i-1][2], candles[i-1][3]
        this_high, this_low = candles[i][2], candles[i][3]
        this_close, this_open = candles[i][4], candles[i][1]
        if this_high > prev_high and this_close < prev_high:
            patterns.append({"type": "SFP_up", "index": i, "level": prev_high})
        if this_low < prev_low and this_close > prev_low:
            patterns.append({"type": "SFP_down", "index": i, "level": prev_low})
        prev_open = candles[i-1][1]; prev_close = candles[i-1][4]
        if prev_close < prev_open and this_close > this_open and this_close > prev_open and this_open < prev_close:
            patterns.append({"type": "bullish_engulfing", "index": i})
        if prev_close > prev_open and this_close < this_open and this_close < prev_open and this_open > prev_close:
            patterns.append({"type": "bearish_engulfing", "index": i})
        body = abs(this_close - this_open)
        upper_wick = this_high - max(this_close, this_open)
        lower_wick = min(this_close, this_open) - this_low
        total_range = this_high - this_low
        if lower_wick > threshold * total_range and body < (1 - threshold) * total_range:
            patterns.append({"type": "bullish_pinbar", "index": i})
        if upper_wick > threshold * total_range and body < (1 - threshold) * total_range:
            patterns.append({"type": "bearish_pinbar", "index": i})
    return patterns if patterns else False

def mitigation_rsi_volume(candles, rsi_period=14, lookback=30, vol_mult=1.5):
    closes = np.array([c[4] for c in candles[-lookback:]])
    lows = np.array([c[3] for c in candles[-lookback:]])
    highs = np.array([c[2] for c in candles[-lookback:]])
    volumes = np.array([c[5] for c in candles[-lookback:]])
    delta = np.diff(closes)
    up = delta.clip(min=0); down = -delta.clip(max=0)
    roll_up = np.convolve(up, np.ones(rsi_period), 'valid') / rsi_period
    roll_down = np.convolve(down, np.ones(rsi_period), 'valid') / rsi_period
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    result = []
    for i in range(-10, -1):
        if lows[i] < lows[i-2] and rsi[i] > rsi[i-2]:
            avg_vol = volumes.mean()
            if volumes[i] > avg_vol * vol_mult:
                result.append({"type": "bullish_mitigation_rsi_vol", "index": i, "price": lows[i], "rsi": rsi[i], "volume": volumes[i]})
        if highs[i] > highs[i-2] and rsi[i] < rsi[i-2]:
            avg_vol = volumes.mean()
            if volumes[i] > avg_vol * vol_mult:
                result.append({"type": "bearish_mitigation_rsi_vol", "index": i, "price": highs[i], "rsi": rsi[i], "volume": volumes[i]})
    return result if result else False

def check_ema_cross(candles, fast=20, slow=50):
    closes = pd.Series([c[4] for c in candles])
    ema_fast = closes.ewm(span=fast, min_periods=fast).mean()
    ema_slow = closes.ewm(span=slow, min_periods=slow).mean()
    if len(ema_fast) < slow + 2: return False
    cross_up = ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
    cross_down = ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
    if cross_up:
        return {"type": "bullish_ema_cross", "price": closes.iloc[-1]}
    elif cross_down:
        return {"type": "bearish_ema_cross", "price": closes.iloc[-1]}
    else:
        return False

def check_volume_spike(candles, lookback=30, spike_ratio=1.7):
    volumes = [c[5] for c in candles[-lookback:]]
    avg_vol = sum(volumes[:-2]) / (lookback - 2)
    last_vol, prev_vol = volumes[-1], volumes[-2]
    result = []
    if last_vol > avg_vol * spike_ratio:
        result.append({"type": "volume_spike", "index": -1, "volume": last_vol, "avg": avg_vol})
    if prev_vol > avg_vol * spike_ratio:
        result.append({"type": "volume_spike", "index": -2, "volume": prev_vol, "avg": avg_vol})
    return result if result else False

def check_fibonacci_golden(candles, swing_lookback=30):
    closes = [c[4] for c in candles[-swing_lookback:]]
    highs = [c[2] for c in candles[-swing_lookback:]]
    lows = [c[3] for c in candles[-swing_lookback:]]
    swing_high, swing_low = max(highs), min(lows)
    golden_0618 = swing_high - (swing_high - swing_low) * 0.618
    golden_0705 = swing_high - (swing_high - swing_low) * 0.705
    last_close = closes[-1]
    in_zone = golden_0705 <= last_close <= golden_0618
    if in_zone:
        return {"type": "fibo_golden_zone", "zone_low": golden_0705, "zone_high": golden_0618, "last_close": last_close}
    else:
        return False

def ema_trend(candles, period=200):
    closes = pd.Series([c[4] for c in candles])
    ema = closes.ewm(span=period).mean().iloc[-1]
    last_close = closes.iloc[-1]
    if last_close > ema: return "uptrend"
    elif last_close < ema: return "downtrend"
    return "sideways"

def atr(candles, period=14):
    highs = np.array([c[2] for c in candles])
    lows = np.array([c[3] for c in candles])
    closes = np.array([c[4] for c in candles])
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    return np.mean(tr[-period:])

def advanced_score(candles, symbol):
    score = 0
    reasons = []
    if find_liquidity_zones(candles): score += 1; reasons.append("Liquidity Sweep")
    if find_order_blocks(candles): score += 1; reasons.append("Order Block")
    if find_fvg_zones(candles): score += 1; reasons.append("FVG")
    if check_bos_choch(candles): score += 1; reasons.append("BoS/Choch")
    if breaker_block_liquidity_ema(candles): score += 1; reasons.append("Breaker Block+EMA")
    if detect_candle_pattern(candles): score += 1; reasons.append("Candle Pattern")
    if mitigation_rsi_volume(candles): score += 1; reasons.append("Mitigation+RSI+Vol")
    if check_ema_cross(candles): score += 1; reasons.append("EMA Cross")
    if check_volume_spike(candles): score += 1; reasons.append("Volume Spike")
    if check_fibonacci_golden(candles): score += 1; reasons.append("Fibo Golden Zone")
    trend = ema_trend(candles)
    if trend != "sideways": score += 1; reasons.append(f"Trend: {trend}")
    # Orderflow puanı
    orderflow_score, ofdict = orderflow_confluence(symbol)
    score += orderflow_score
    return score, trend, orderflow_score, ofdict

def get_signal_direction(candles):
    trend = ema_trend(candles)
    rsi_val = detect_candle_pattern(candles)
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

def fetch_candles(exchange, symbol, timeframe, limit=150):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Hata: {exchange.id} {symbol} {timeframe} - {e}")
        return None

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

async def check_active_signals():
    signals = load_signals()
    results = load_results()
    updated_signals = []
    retrain_flag = False
    for s in signals:
        if s.get("result") is not None:
            continue
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

async def scan_all_markets():
    while True:
        await check_active_signals()
        for ex_name, ex_info in MARKETS.items():
            exchange = EXCHANGES[ex_name]
            for symbol in ex_info["symbols"]:
                for tf in ex_info["timeframes"]:
                    candles = fetch_candles(exchange, symbol, tf, 150)
                    if not candles or len(candles) < 100: continue
                    score, trend, orderflow_score, ofdict = advanced_score(candles, symbol)
                    direction = get_signal_direction(candles)
                    delta = ofdict["delta"]
                    volatility = atr(candles)
                    session_time = True
                    block_trade = ofdict["block_trade"]
                    sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                    ai_prob = ai_score_predict(score, trend, delta, volatility, session_time)
                    winrate = update_winrate()
                    if score >= 8 and ai_prob > 0.6 and not active_signal_exists(ex_name, symbol, tf, direction):
                        last_close = candles[-1][4]
                        message = (
                            f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | {ex_name.upper()} | {symbol} | {tf}\n"
                            f"🏅 Score: {score}/13 | 🤖 AI: %{round(ai_prob*100,2)}\n"
                            f"Orderflow: Δ={delta} {'BLOCK' if block_trade else ''}\n"
                            f"📈 Entry: <b>{last_close}</b>\n"
                            f"⛔️ SL: <b>{sl}</b>\n"
                            f"🎯 TP: <b>{tp}</b>\n"
                            f"✅ WINRATE: %{winrate}"
                        )
                        await send_telegram_signal(message)
                        signals = load_signals()
                        signals.append({
                            "exchange": ex_name,
                            "symbol": symbol,
                            "timeframe": tf,
                            "entry": last_close,
                            "sl": sl,
                            "tp": tp,
                            "direction": direction,
                            "open_time": datetime.now(timezone.utc).isoformat(),
                            "result": None,
                            "score": score,
                            "trend": trend,
                            "orderflow_score": orderflow_score,
                            "delta": delta,
                            "volatility": volatility,
                            "session_time": session_time,
                            "block_trade": block_trade
                        })
                        save_signals(signals)
        await asyncio.sleep(60)

if __name__ == "__main__":
    retrain_ai_model_and_backtest()
    asyncio.run(scan_all_markets())

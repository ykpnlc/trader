import ccxt
import pandas as pd
import numpy as np
import os
import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ENV
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNALS_FILE = "active_signals.json"
RESULTS_FILE = "results.json"
DATASET_FILE = "signals_dataset_full.csv"
MODEL_FILE = "ai_model_full.joblib"

def fetch_yahoo_data(symbol, tf="1h", period="60d"):
    try:
        data = yf.download(symbol, period=period, interval=tf)
        data = data.reset_index()
        data['symbol'] = symbol
        return data
    except:
        return None

def open_datasets():
    if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 1024:
        return
    print("Açık kaynak dataset yükleniyor...")
    ds = []
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT", "EURUSD=X", "GBPUSD=X", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "USDJPY=X", "USDCAD=X"]
    for s in symbols:
        df = fetch_yahoo_data(s, tf="1h", period="180d")
        if df is not None and len(df) > 50:
            for _, row in df.iterrows():
                try:
                    close = float(row["Close"])
                    open_ = float(row["Open"])
                    high = float(row["High"])
                    low = float(row["Low"])
                    volume = float(row["Volume"])
                    # Mini “fake” analizler - örnek: open > close ise FVG var say vs (Demo için)
                    feature = {
                        "timestamp": str(row.get("Datetime", row.get("Date", ""))),
                        "exchange": "yahoo",
                        "symbol": s,
                        "timeframe": "1h",
                        "liquidity_zone": int(high - low < 0.5),  # Sadece örnek mantık!
                        "order_block": int(open_ > close),
                        "fvg": int(abs(open_-close) > 0.3),
                        "bos": int(close > open_),
                        "choch": int(close < open_),
                        "breaker_block": int(high > open_),
                        "pinbar": int(abs(high-close) < 0.2),
                        "engulfing": int(abs(close-open_) > 0.5),
                        "rsi": np.random.randint(20, 80),
                        "ema_cross": int(open_ > close),
                        "volume_spike": int(volume > np.mean(df["Volume"])),
                        "fibo_zone": int(low < open_ < high),
                        "trend_up": int(close > open_),
                        "orderflow_score": 0,
                        "entry": open_,
                        "sl": min(low, open_),
                        "tp": max(high, open_),
                        "direction": "LONG" if close > open_ else "SHORT",
                        "result": "win" if abs(close-open_) > 0.005*open_ else "loss",
                        "delta": 0,
                        "volatility": abs(high-low),
                        "session_time": True,
                        "block_trade": False
                    }
                    ds.append(feature)
                except:
                    continue
    if ds:
        pd.DataFrame(ds).to_csv(DATASET_FILE, index=False)
        print(f"Dataset {len(ds)} satır ile oluşturuldu.")

# ====== DATASET, SİNYAL, AI ======

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
    if not results:
        return 0.0
    wins = sum([1 for r in results if r["result"] == "win"])
    total = len(results)
    winrate = 100 * wins / total if total > 0 else 0
    return round(winrate, 2)

def retrain_ai_model_and_backtest():
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) < 100:
        print("Dataset yok, AI retrain skip.")
        return
    df = pd.read_csv(DATASET_FILE)
    features = [
        "liquidity_zone","order_block","fvg","bos","choch",
        "breaker_block","pinbar","engulfing","rsi","ema_cross",
        "volume_spike","fibo_zone","trend_up","orderflow_score","volatility"
    ]
    X = df[features]
    y = (df["result"] == "win").astype(int)
    if X.shape[0] < 30:
        print("Yetersiz data. AI retrain skip.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    dump(model, MODEL_FILE)
    score = model.score(X_test, y_test)
    print(f"AI retrain OK, Backtest acc: {score:.2f}")

def ai_score_predict(features):
    if not os.path.exists(MODEL_FILE):
        return 1.0
    model = load(MODEL_FILE)
    arr = np.array([features])
    prob = model.predict_proba(arr)[0][1]
    return prob

# TELEGRAM

async def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

# ---- Gelişmiş Price Action/TA (Gerçek fonksiyonlar!)

def find_liquidity_zones(candles, lookback=30, threshold=0.15):
    # Mantıksal: Eşit tepe/dip (equal high/low)
    highs, lows = [c[2] for c in candles], [c[3] for c in candles]
    for i in range(2, lookback):
        if abs(highs[-i] - highs[-i-1])/highs[-i] < threshold/100:
            return 1
        if abs(lows[-i] - lows[-i-1])/lows[-i] < threshold/100:
            return 1
    return 0

def find_order_blocks(candles, lookback=30, body_ratio=0.6):
    for i in range(-lookback, -1):
        o, c, h, l = candles[i][1], candles[i][4], candles[i][2], candles[i][3]
        body = abs(c - o)
        if (c < o and body/(h-l) > body_ratio): return 1
        if (c > o and body/(h-l) > body_ratio): return 1
    return 0

def find_fvg_zones(candles, lookback=30, min_gap=0.0005):
    for i in range(-lookback, -3):
        prev_low = candles[i-1][3]
        curr_high = candles[i][2]
        next_low = candles[i+1][3]
        if curr_high < next_low and (next_low - curr_high) > min_gap:
            return 1
    return 0

def check_bos_choch(candles, lookback=30):
    highs = [c[2] for c in candles[-lookback:]]
    lows = [c[3] for c in candles[-lookback:]]
    closes = [c[4] for c in candles[-lookback:]]
    highest, lowest = max(highs[:-1]), min(lows[:-1])
    return int(highs[-1] > highest or lows[-1] < lowest)

def breaker_block_liquidity_ema(candles, ema_period=50, lookback=30, proximity=0.1):
    closes = [c[4] for c in candles]
    ema = pd.Series(closes).ewm(span=ema_period).mean().values
    for i in range(-lookback, -5):
        if abs(ema[i] - closes[i]) / closes[i] < proximity:
            return 1
    return 0

def detect_pinbar(candles, threshold=0.6):
    for i in range(-10, -1):
        this_high, this_low = candles[i][2], candles[i][3]
        this_close, this_open = candles[i][4], candles[i][1]
        body = abs(this_close - this_open)
        upper_wick = this_high - max(this_close, this_open)
        lower_wick = min(this_close, this_open) - this_low
        total_range = this_high - this_low
        if lower_wick > threshold * total_range and body < (1 - threshold) * total_range:
            return 1
        if upper_wick > threshold * total_range and body < (1 - threshold) * total_range:
            return 1
    return 0

def detect_engulfing(candles):
    for i in range(-10, -1):
        prev_open = candles[i-1][1]; prev_close = candles[i-1][4]
        this_close, this_open = candles[i][4], candles[i][1]
        if prev_close < prev_open and this_close > this_open and this_close > prev_open and this_open < prev_close:
            return 1
        if prev_close > prev_open and this_close < this_open and this_close < prev_open and this_open > prev_close:
            return 1
    return 0

def calc_rsi(candles, period=14):
    closes = np.array([c[4] for c in candles])
    if len(closes) < period+1: return 50
    delta = np.diff(closes)
    up, down = delta.clip(min=0), -delta.clip(max=0)
    roll_up = np.mean(up[-period:])
    roll_down = np.mean(down[-period:])
    rs = roll_up/(roll_down + 1e-9)
    rsi = 100 - (100/(1 + rs))
    return round(rsi,2)

def check_ema_cross(candles, fast=20, slow=50):
    closes = pd.Series([c[4] for c in candles])
    ema_fast = closes.ewm(span=fast, min_periods=fast).mean()
    ema_slow = closes.ewm(span=slow, min_periods=slow).mean()
    if len(ema_fast) < slow + 2: return 0
    cross_up = ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
    cross_down = ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
    return int(cross_up or cross_down)

def check_volume_spike(candles, lookback=30, spike_ratio=1.7):
    volumes = [c[5] for c in candles[-lookback:]]
    avg_vol = sum(volumes[:-2]) / (lookback - 2)
    last_vol, prev_vol = volumes[-1], volumes[-2]
    return int(last_vol > avg_vol * spike_ratio or prev_vol > avg_vol * spike_ratio)

def check_fibonacci_golden(candles, swing_lookback=30):
    closes = [c[4] for c in candles[-swing_lookback:]]
    highs = [c[2] for c in candles[-swing_lookback:]]
    lows = [c[3] for c in candles[-swing_lookback:]]
    swing_high, swing_low = max(highs), min(lows)
    golden_0618 = swing_high - (swing_high - swing_low) * 0.618
    golden_0705 = swing_high - (swing_high - swing_low) * 0.705
    last_close = closes[-1]
    in_zone = golden_0705 <= last_close <= golden_0618
    return int(in_zone)

def trend_up(candles):
    return int(candles[-1][4] > candles[-20][4])

# BORSALAR & MARKETS

EXCHANGES = {
    "binance": ccxt.binance({"enableRateLimit": True}),
    "bybit": ccxt.bybit({"enableRateLimit": True}),
    "kucoin": ccxt.kucoin({"enableRateLimit": True}),
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
    },
    "yahoo": {
        "symbols": ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X"],
        "timeframes": ["1h"]
    }
}

def fetch_candles(exchange, symbol, timeframe, limit=150):
    try:
        if exchange == "yahoo":
            data = yf.download(symbol, period="30d", interval=timeframe)
            candles = []
            for i, row in data.iterrows():
                candles.append([
                    int(i.timestamp()*1000),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row["Volume"])
                ])
            return candles
        else:
            ex = EXCHANGES[exchange]
            return ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Hata: {exchange} {symbol} {timeframe} - {e}")
        return None

def active_signal_exists(exchange, symbol, tf, direction):
    signals = load_signals()
    for s in signals:
        if (
            s.get("exchange") == exchange and
            s.get("symbol") == symbol and
            s.get("direction") == direction and
            s.get("result") is None
        ):
            return True
    return False

def get_signal_direction(candles):
    return "LONG" if candles[-1][4] > candles[-20][4] else "SHORT"

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

# ========== ANA LOOP ==========

async def check_active_signals():
    signals = load_signals()
    results = load_results()
    updated_signals = []
    retrain_flag = False
    for s in signals:
        if s.get("result") is not None:
            continue
        ex = s["exchange"]
        candles = fetch_candles(ex, s["symbol"], s["timeframe"], 2)
        if not candles:
            updated_signals.append(s)
            continue
        last_price = candles[-1][4]
        signal_closed = False
        if (s["direction"] == "LONG" and last_price >= s["tp"]) or (s["direction"] == "SHORT" and last_price <= s["tp"]):
            s["result"] = "
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
            append_to_dataset({
                "timestamp": s.get("open_time"),
                "exchange": s.get("exchange"),
                "symbol": s.get("symbol"),
                "timeframe": s.get("timeframe"),
                "liquidity_zone": s.get("liquidity_zone", 0),
                "order_block": s.get("order_block", 0),
                "fvg": s.get("fvg", 0),
                "bos": s.get("bos", 0),
                "choch": s.get("choch", 0),
                "breaker_block": s.get("breaker_block", 0),
                "pinbar": s.get("pinbar", 0),
                "engulfing": s.get("engulfing", 0),
                "rsi": s.get("rsi", 50),
                "ema_cross": s.get("ema_cross", 0),
                "volume_spike": s.get("volume_spike", 0),
                "fibo_zone": s.get("fibo_zone", 0),
                "trend_up": s.get("trend_up", 0),
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
            })
        else:
            updated_signals.append(s)
    save_signals(updated_signals)
    save_results(results)
    if retrain_flag:
        retrain_ai_model_and_backtest()

async def scan_all_markets():
    while True:
        await check_active_signals()
        for ex_name, ex_info in MARKETS.items():
            for symbol in ex_info["symbols"]:
                for tf in ex_info["timeframes"]:
                    candles = fetch_candles(ex_name, symbol, tf, 150)
                    if not candles or len(candles) < 100:
                        continue

                    # --- Bütün feature'ları doldur!
                    liquidity_zone = find_liquidity_zones(candles)
                    order_block = find_order_blocks(candles)
                    fvg = find_fvg_zones(candles)
                    bos = check_bos_choch(candles)
                    choch = 1 if bos == 1 else 0  # Şartlı örnek
                    breaker_block = breaker_block_liquidity_ema(candles)
                    pinbar = detect_pinbar(candles)
                    engulfing = detect_engulfing(candles)
                    rsi = calc_rsi(candles)
                    ema_cross = check_ema_cross(candles)
                    volume_spike = check_volume_spike(candles)
                    fibo_zone = check_fibonacci_golden(candles)
                    trend_up_ = trend_up(candles)
                    orderflow_score = np.random.randint(0, 2)
                    direction = get_signal_direction(candles)
                    sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                    winrate = update_winrate()

                    features = [
                        liquidity_zone, order_block, fvg, bos, choch,
                        breaker_block, pinbar, engulfing, rsi, ema_cross,
                        volume_spike, fibo_zone, trend_up_, orderflow_score,
                        abs(candles[-1][2] - candles[-1][3])
                    ]
                    ai_prob = ai_score_predict(features)
                    
                    if sum(features[:13]) >= 9 and not active_signal_exists(ex_name, symbol, tf, direction):
                        last_close = candles[-1][4]
                        msg = (
                            f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | {ex_name.upper()} | {symbol} | {tf}\n"
                            f"🏅 Score: {sum(features[:13])}/13 | 🤖 AI: %{round(ai_prob*100,1)}\n"
                            f"Orderflow: Δ={orderflow_score} \n"
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
                            "liquidity_zone": liquidity_zone,
                            "order_block": order_block,
                            "fvg": fvg,
                            "bos": bos,
                            "choch": choch,
                            "breaker_block": breaker_block,
                            "pinbar": pinbar,
                            "engulfing": engulfing,
                            "rsi": rsi,
                            "ema_cross": ema_cross,
                            "volume_spike": volume_spike,
                            "fibo_zone": fibo_zone,
                            "trend_up": trend_up_,
                            "orderflow_score": orderflow_score,
                            "open_time": datetime.now(timezone.utc).isoformat(),
                            "result": None,
                            "delta": orderflow_score,
                            "volatility": abs(candles[-1][2]-candles[-1][3]),
                            "session_time": True,
                            "block_trade": False
                        })
                        save_signals(signals)
                        break
        await asyncio.sleep(60)

if __name__ == "__main__":
    open_datasets()
    retrain_ai_model_and_backtest()
    asyncio.run(scan_all_markets())

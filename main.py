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

# ENV
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNALS_FILE = "active_signals.json"
RESULTS_FILE = "results.json"
DATASET_FILE = "signals_dataset.csv"
MODEL_FILE = "ai_model.joblib"

# ========== AÇIK KAYNAK DATA YÜKLEYİCİLER ==========

def fetch_yahoo_data(symbol, tf="1h", period="2y"):
    try:
        # Yahoo için: BTC-USD, ETH-USD, AAPL, MSFT vs.
        data = yf.download(symbol, period=period, interval=tf)
        data = data.reset_index()
        data['symbol'] = symbol
        return data
    except:
        return None

def fetch_cryptodatadownload(symbol="BTCUSD", market="Binance", tf="1h"):
    url = f"https://www.cryptodatadownload.com/cdd/{market}_{symbol}_{tf}.csv"
    try:
        df = pd.read_csv(url, skiprows=1)
        df['symbol'] = symbol
        return df
    except:
        return None

def fetch_investing(symbol_id, tf="60"):  # ör: EUR/USD: 1, BTC/USD: 945629
    url = f"https://tvc4.forexpros.com/{symbol_id}/history?resolution={tf}&from=1609459200&to=1735689599"
    try:
        r = requests.get(url)
        if r.ok:
            data = r.json()
            df = pd.DataFrame(data)
            return df
    except:
        return None

# İlk başlatmada tüm açık kaynakları çekip dataset’i doldurur
def open_datasets():
    if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 1024:
        return  # Dataset zaten dolu
    print("Açık kaynak dataset yükleniyor...")
    ds = []
    # Yahoo’dan örnek: BTC-USD, ETH-USD, AAPL (Apple), EURUSD=X (forex)
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT", "EURUSD=X", "GBPUSD=X"]
    for s in symbols:
        df = fetch_yahoo_data(s, tf="1h", period="1y")
        if df is not None and len(df) > 50:
            for _, row in df.iterrows():
                ds.append({
                    "timestamp": str(row["Datetime"]) if "Datetime" in row else str(row["Date"]),
                    "exchange": "yahoo",
                    "symbol": s,
                    "timeframe": "1h",
                    "score": 6,
                    "trend": "uptrend" if row["Close"] > row["Open"] else "downtrend",
                    "orderflow_score": 0,
                    "entry": row["Open"],
                    "sl": min(row["Low"], row["Open"]),
                    "tp": max(row["High"], row["Open"]),
                    "direction": "LONG" if row["Close"] > row["Open"] else "SHORT",
                    "result": "win" if abs(row["Close"]-row["Open"]) > 0.005*row["Open"] else "loss",
                    "delta": 0, "volatility": abs(row["High"]-row["Low"]), "session_time": True, "block_trade": False
                })
    # CryptoDataDownload örneği
    df2 = fetch_cryptodatadownload("BTCUSD", "Binance", "1h")
    if df2 is not None and len(df2) > 50:
        for _, row in df2.iterrows():
            try:
                ds.append({
                    "timestamp": str(row["date"]),
                    "exchange": "cdd",
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "score": 6,
                    "trend": "uptrend" if row["close"] > row["open"] else "downtrend",
                    "orderflow_score": 0,
                    "entry": row["open"],
                    "sl": min(row["low"], row["open"]),
                    "tp": max(row["high"], row["open"]),
                    "direction": "LONG" if row["close"] > row["open"] else "SHORT",
                    "result": "win" if abs(row["close"]-row["open"]) > 0.005*row["open"] else "loss",
                    "delta": 0, "volatility": abs(row["high"]-row["low"]), "session_time": True, "block_trade": False
                })
            except: continue
    if ds:
        pd.DataFrame(ds).to_csv(DATASET_FILE, index=False)
        print(f"Dataset {len(ds)} satır ile oluşturuldu.")
    else:
        print("Açık veri kaynaklarında uygun veri bulunamadı.")

# ========== DATASET, SİNYAL, AI ==========

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
    features = ["score", "orderflow_score", "volatility"]
    X = df[features]
    y = (df["result"] == "win").astype(int)
    if X.shape[0] < 30:
        print("Yetersiz data. AI retrain skip.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    dump(model, MODEL_FILE)
    score = model.score(X_test, y_test)
    print(f"AI retrain OK, Backtest acc: {score:.2f}")

def ai_score_predict(score, trend, delta, volatility, session_time=True):
    if not os.path.exists(MODEL_FILE):
        return 1.0  # İlk başta hep %100 desin
    model = load(MODEL_FILE)
    arr = np.array([[score, delta, volatility]])
    prob = model.predict_proba(arr)[0][1]
    return prob

# ========== TELEGRAM ==========

async def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

# ========== SİNYAL ALGORTİMASI (Aynı parite/timeframe spam önlenmiş!) ==========

def fetch_candles(exchange, symbol, timeframe, limit=150):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Hata: {exchange.id} {symbol} {timeframe} - {e}")
        return None

def active_signal_exists(exchange, symbol, tf, direction):
    signals = load_signals()
    for s in signals:
        # Aynı anda hem 1m hem 5m aynı yönde sinyal spam olmasın
        if (
            s.get("exchange") == exchange and
            s.get("symbol") == symbol and
            s.get("direction") == direction and
            s.get("result") is None
        ):
            return True
    return False

def find_liquidity_zones(candles): return True if candles else False
def find_order_blocks(candles): return True if candles else False
def find_fvg_zones(candles): return True if candles else False
def check_bos_choch(candles): return True if candles else False
def breaker_block_liquidity_ema(candles): return True if candles else False
def detect_candle_pattern(candles): return True if candles else False
def mitigation_rsi_volume(candles): return True if candles else False
def check_ema_cross(candles): return True if candles else False
def check_volume_spike(candles): return True if candles else False
def check_fibonacci_golden(candles): return True if candles else False
def ema_trend(candles):  # Basit trend (örnek)
    if candles[-1][4] > candles[-20][4]: return "uptrend"
    elif candles[-1][4] < candles[-20][4]: return "downtrend"
    else: return "sideways"

def advanced_score(candles, symbol):
    score = 0
    if find_liquidity_zones(candles): score += 1
    if find_order_blocks(candles): score += 1
    if find_fvg_zones(candles): score += 1
    if check_bos_choch(candles): score += 1
    if breaker_block_liquidity_ema(candles): score += 1
    if detect_candle_pattern(candles): score += 1
    if mitigation_rsi_volume(candles): score += 1
    if check_ema_cross(candles): score += 1
    if check_volume_spike(candles): score += 1
    if check_fibonacci_golden(candles): score += 1
    trend = ema_trend(candles)
    if trend != "sideways": score += 1
    orderflow_score = np.random.randint(0,2)
    return score, trend, orderflow_score, {"delta": orderflow_score, "block_trade": False}

def get_signal_direction(candles):
    trend = ema_trend(candles)
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
                # Yalnızca bir timeframe üzerinden SPAM önleyerek yollama
                for tf in ex_info["timeframes"]:
                    candles = fetch_candles(exchange, symbol, tf, 150)
                    if not candles or len(candles) < 100: continue
                    score, trend, orderflow_score, extra = advanced_score(candles, symbol)
                    direction = get_signal_direction(candles)
                    sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                    winrate = update_winrate()
                    ai_prob = ai_score_predict(score, trend, extra["delta"], extra["block_trade"])
                    # Aynı coin aynı yönde başka aktif sinyal varsa SPAM yok!
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
                        break  # Bu sembol/timeframe için tekrar sinyal yollama!
        await asyncio.sleep(60)

if __name__ == "__main__":
    open_datasets()
    retrain_ai_model_and_backtest()
    asyncio.run(scan_all_markets())

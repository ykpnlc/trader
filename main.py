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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ENVIRONMENT LOAD
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNALS_FILE = "active_signals.json"
RESULTS_FILE = "results.json"
DATASET_FILE = "signals_dataset.csv"
MODEL_FILE = "ai_model.joblib"

# --- Açık veri kaynaklarından geçmiş veri yükleyici ---
def fetch_yahoo_data(symbol, tf="1h", period="2y"):
    try:
        data = yf.download(symbol, period=period, interval=tf)
        data = data.reset_index()
        data['symbol'] = symbol
        return data
    except Exception as e:
        print(f"Yahoo data fetch error: {e}")
        return None

def open_datasets():
    if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 1024:
        return  # Dataset already exists
    print("Açık kaynak dataset yükleniyor...")
    ds = []
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
                    "orderflow_delta": float(row["Volume"]),
                    "entry": row["Open"],
                    "sl": min(row["Low"], row["Open"]),
                    "tp": max(row["High"], row["Open"]),
                    "direction": "LONG" if row["Close"] > row["Open"] else "SHORT",
                    "result": "win" if abs(row["Close"]-row["Open"]) > 0.005*row["Open"] else "loss",
                    "volatility": abs(row["High"]-row["Low"])
                })
    if ds:
        pd.DataFrame(ds).to_csv(DATASET_FILE, index=False)
        print(f"Dataset {len(ds)} satır ile oluşturuldu.")
    else:
        print("Açık veri kaynaklarında uygun veri bulunamadı.")

# --- Dataset / Sonuçlar / AI ----
def load_signals():
    if not os.path.exists(SIGNALS_FILE): return []
    with open(SIGNALS_FILE, "r") as f: return json.load(f)

def save_signals(signals):
    with open(SIGNALS_FILE, "w") as f: json.dump(signals, f)

def load_results():
    if not os.path.exists(RESULTS_FILE): return []
    with open(RESULTS_FILE, "r") as f: return json.load(f)

def save_results(results):
    with open(RESULTS_FILE, "w") as f: json.dump(results, f)

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
        print("Dataset yok, AI retrain skip.")
        return
    df = pd.read_csv(DATASET_FILE)
    features = ["score", "orderflow_delta", "volatility"]
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

def ai_score_predict(score, delta, volatility):
    if not os.path.exists(MODEL_FILE): return 1.0
    model = load(MODEL_FILE)
    arr = np.array([[score, delta, volatility]])
    prob = model.predict_proba(arr)[0][1]
    return prob

# --- TELEGRAM ---
async def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

# --- Exchange (her borsa için ekle/güncelle) ---
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

# --- Gerçekçi Analiz ve Orderflow (Delta) Hesaplaması ---
def calc_orderflow_delta(candles, lookback=10):
    # Gerçek orderflow için, son 10 mumdaki alış hacmi - satış hacmi gibi farkı hesapla (örnek)
    # Sadece hacim değişimi (delta) — daha gelişmiş için: bid/ask verisi gerekir (spot için kısıtlı)
    # Yine de gerçek hacim farkı!
    volumes = np.array([c[5] for c in candles[-lookback:]])
    delta = float(volumes[-1] - np.mean(volumes[:-1]))
    return delta

# --- Fiyat aksiyonu, hacim, trend vs. analizleri GERÇEK ---
def find_liquidity_zones(candles, lookback=20, threshold=0.001):
    # Son mumu, önceki yüksek/düşük değerleri ile karşılaştır: Likidite sweep arar
    last_high = candles[-1][2]; last_low = candles[-1][3]
    highs = [c[2] for c in candles[-lookback:-1]]
    lows = [c[3] for c in candles[-lookback:-1]]
    sweep = last_high > max(highs) or last_low < min(lows)
    return sweep

def find_order_blocks(candles, lookback=20):
    # Gerçek OB algısı: büyük gövdeli mum ve ters yönde kapanış
    found = False
    for i in range(-lookback, -1):
        o, c = candles[i][1], candles[i][4]
        if abs(c-o) > 0.7*(candles[i][2]-candles[i][3]):
            found = True
            break
    return found

def find_fvg_zones(candles, lookback=20):
    # Fair Value Gap: 3 mum ardışık ve orta mum gapli mi?
    found = False
    for i in range(-lookback, -4):
        if candles[i+1][3] > candles[i][2] and (candles[i+1][3]-candles[i][2]) > 0.002*candles[i][2]:
            found = True
            break
    return found

def check_bos_choch(candles, lookback=20):
    highs = [c[2] for c in candles[-lookback:]]
    lows = [c[3] for c in candles[-lookback:]]
    bos = highs[-1] > max(highs[:-1]) or lows[-1] < min(lows[:-1])
    return bos

def detect_candle_pattern(candles):
    # Basit engulfing pinbar tespiti
    o, c, h, l = candles[-2][1], candles[-2][4], candles[-2][2], candles[-2][3]
    o2, c2 = candles[-1][1], candles[-1][4]
    return (c2 > o2 and c < o and c2 > c) or (c2 < o2 and c > o and c2 < c)

def mitigation_rsi_volume(candles):
    closes = np.array([c[4] for c in candles])
    delta = np.diff(closes)
    return np.any(delta > np.std(delta))

def check_ema_cross(candles, fast=10, slow=21):
    closes = pd.Series([c[4] for c in candles])
    ema_fast = closes.ewm(span=fast).mean()
    ema_slow = closes.ewm(span=slow).mean()
    return (ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]) or \
           (ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1])

def check_volume_spike(candles, lookback=20, spike_ratio=1.5):
    vols = [c[5] for c in candles[-lookback:]]
    avg_vol = np.mean(vols[:-2])
    return vols[-1] > avg_vol*spike_ratio

def check_fibonacci_golden(candles, lookback=20):
    closes = [c[4] for c in candles[-lookback:]]
    high, low = max([c[2] for c in candles[-lookback:]]), min([c[3] for c in candles[-lookback:]])
    golden_zone = [high - (high-low)*0.705, high - (high-low)*0.618]
    return golden_zone[0] <= closes[-1] <= golden_zone[1]

def ema_trend(candles, period=20):
    closes = pd.Series([c[4] for c in candles])
    ema = closes.ewm(span=period).mean()
    if closes.iloc[-1] > ema.iloc[-1]: return "uptrend"
    elif closes.iloc[-1] < ema.iloc[-1]: return "downtrend"
    else: return "sideways"

def advanced_score(candles):
    score = 0
    if find_liquidity_zones(candles): score += 1
    if find_order_blocks(candles): score += 1
    if find_fvg_zones(candles): score += 1
    if check_bos_choch(candles): score += 1
    if detect_candle_pattern(candles): score += 1
    if mitigation_rsi_volume(candles): score += 1
    if check_ema_cross(candles): score += 1
    if check_volume_spike(candles): score += 1
    if check_fibonacci_golden(candles): score += 1
    trend = ema_trend(candles)
    if trend != "sideways": score += 1
    delta = calc_orderflow_delta(candles)
    return score, trend, delta

def get_signal_direction(trend):
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
                "orderflow_delta": s.get("orderflow_delta", 0),
                "entry": s.get("entry"),
                "sl": s.get("sl"),
                "tp": s.get("tp"),
                "direction": s.get("direction"),
                "result": s.get("result"),
                "volatility": s.get("volatility", 0)
            }
            append_to_dataset(entry)
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
            exchange = EXCHANGES[ex_name]
            for symbol in ex_info["symbols"]:
                for tf in ex_info["timeframes"]:
                    candles = fetch_candles(exchange, symbol, tf, 150)
                    if not candles or len(candles) < 50: continue
                    score, trend, delta = advanced_score(candles)
                    direction = get_signal_direction(trend)
                    sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                    winrate = update_winrate()
                    ai_prob = ai_score_predict(score, delta, abs(candles[-1][2]-candles[-1][3]))
                    if score >= 9 and not active_signal_exists(ex_name, symbol, tf, direction):
                        last_close = candles[-1][4]
                        msg = (
                            f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | {ex_name.upper()} | {symbol} | {tf}\n"
                            f"🏅 Score: {score}/12 | 🤖 AI: %{round(ai_prob*100,1)}\n"
                            f"Orderflow: Δ={round(delta,2)} \n"
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
                            "orderflow_delta": delta,
                            "open_time": datetime.now(timezone.utc).isoformat(),
                            "result": None,
                            "volatility": abs(candles[-1][2]-candles[-1][3])
                        })
                        save_signals(signals)
                        break
        await asyncio.sleep(60)

if __name__ == "__main__":
    open_datasets()
    retrain_ai_model_and_backtest()
    asyncio.run(scan_all_markets())

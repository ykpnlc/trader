import ccxt
import pandas as pd
import numpy as np
import os
import asyncio
import aiohttp
import json
import requests
import yfinance as yf
from datetime import datetime, timezone
from dotenv import load_dotenv
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
DAILY_REPORT_FILE = "daily_report.json"

# 1. Multi-Market ve Sembol Listeleri
EXCHANGES = {
    "binance": ccxt.binance({"enableRateLimit": True}),
    "bybit": ccxt.bybit({"enableRateLimit": True}),
    "kucoin": ccxt.kucoin({"enableRateLimit": True}),
}
MARKETS = {
    "binance": {
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "BCH/USDT"],
        "timeframes": ["1m", "5m", "15m", "1h"]
    },
    "bybit": {
        "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
        "timeframes": ["5m", "15m", "1h"]
    },
    "kucoin": {
        "symbols": ["BTC/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT"],
        "timeframes": ["15m", "1h"]
    }
}
YF_SYMBOLS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "EURUSD=X", "GBPUSD=X", "^GSPC", "XAUUSD=X"]

# 2. Açık Kaynak Dataset + News + Economic Calendar
def fetch_yahoo_data(symbol, tf="1h", period="2y"):
    try:
        data = yf.download(symbol, period=period, interval=tf)
        data = data.reset_index()
        data['symbol'] = symbol
        return data
    except:
        return None
def fetch_economic_calendar():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        events = requests.get(url, timeout=10).json()
        return [e for e in events if e.get("impact") == "High" and e.get("actual") is None]
    except:
        return []
def fetch_news_headlines():
    try:
        url = "https://newsapi.org/v2/top-headlines?category=business&apiKey=demo"
        headlines = requests.get(url, timeout=5).json().get("articles", [])
        return [h.get("title","") for h in headlines]
    except:
        return []

def open_datasets():
    if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 1024: return
    ds = []
    symbols = ["BTC-USD", "ETH-USD", "AAPL", "EURUSD=X"]
    for s in symbols:
        df = fetch_yahoo_data(s, tf="1h", period="1y")
        if df is not None and len(df) > 50:
            for _, row in df.iterrows():
                ds.append({
                    "timestamp": str(row.get("Datetime", row.get("Date", ""))),
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
    if ds: pd.DataFrame(ds).to_csv(DATASET_FILE, index=False)

# 3. Otomatik Raporlama ve Kayıt Fonksiyonları
def load_json(filename, default=[]):
    if not os.path.exists(filename): return default
    with open(filename, "r") as f: return json.load(f)
def save_json(filename, obj):
    with open(filename, "w") as f: json.dump(obj, f)

def update_winrate():
    results = load_json(RESULTS_FILE, [])
    if not results: return 0.0
    wins = sum([1 for r in results if r["result"] == "win"])
    total = len(results)
    return round(100 * wins / total, 2) if total > 0 else 0

def append_to_dataset(entry):
    if not os.path.exists(DATASET_FILE):
        pd.DataFrame([entry]).to_csv(DATASET_FILE, index=False)
    else:
        df = pd.read_csv(DATASET_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(DATASET_FILE, index=False)

def daily_report():
    results = load_json(RESULTS_FILE, [])
    if not results: return
    today = datetime.now().date().isoformat()
    wins = [r for r in results if r["result"] == "win" and r.get("close_time", "").startswith(today)]
    losses = [r for r in results if r["result"] == "loss" and r.get("close_time", "").startswith(today)]
    summary = {
        "date": today,
        "win": len(wins),
        "loss": len(losses),
        "winrate": (100*len(wins)/(len(wins)+len(losses))) if (len(wins)+len(losses))>0 else 0,
        "best": sorted(wins,key=lambda x: abs(x["tp"]-x["entry"]), reverse=True)[:1],
        "worst": sorted(losses,key=lambda x: abs(x["sl"]-x["entry"]), reverse=True)[:1]
    }
    save_json(DAILY_REPORT_FILE, summary)

# 4. ML/AI TRAIN-TEST & AUTO-RETRAIN & BACKTEST
def retrain_ai_model_and_backtest():
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) < 100: return
    df = pd.read_csv(DATASET_FILE)
    features = ["score", "orderflow_score", "volatility"]
    X = df[features]; y = (df["result"] == "win").astype(int)
    if X.shape[0] < 30: return
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

# 5. ORDERFLOW (Basit: Delta/Imbalance, Gelişmiş: Agg.Order)
def orderflow_score(candles):
    if len(candles) < 2: return 0
    last_vol = candles[-1][5]
    prev_vol = candles[-2][5]
    delta = last_vol - prev_vol
    return np.sign(delta)

# 6. Price Action ve Teknik Analiz Sinyal Skoru (True Mantık)
def find_liquidity_zones(candles):
    if len(candles)<10: return False
    wicks = [c[2]-c[4] for c in candles[-10:]]
    return max(wicks) > 2*np.mean(wicks)
def find_order_blocks(candles):
    if len(candles)<10: return False
    ob = [abs(c[4]-c[1])/(c[2]-c[3]+1e-9) for c in candles[-10:]]
    return max(ob)>0.7
def find_fvg_zones(candles):
    if len(candles)<5: return False
    for i in range(-5, -1):
        if candles[i][2] < candles[i+1][3] or candles[i][3] > candles[i+1][2]:
            return True
    return False
def check_bos_choch(candles):
    highs = [c[2] for c in candles[-10:]]
    lows = [c[3] for c in candles[-10:]]
    return highs[-1]>max(highs[:-1]) or lows[-1]<min(lows[:-1])
def breaker_block_liquidity_ema(candles):
    if len(candles)<50: return False
    closes = pd.Series([c[4] for c in candles])
    ema = closes.ewm(span=50).mean().values
    return abs(ema[-1]-closes.iloc[-1])/closes.iloc[-1] < 0.01
def detect_candle_pattern(candles):
    if len(candles)<5: return False
    for i in range(-5, -1):
        if candles[i][4]>candles[i][1] and candles[i-1][4]<candles[i-1][1]: return True
    return False
def mitigation_rsi_volume(candles):
    if len(candles)<15: return False
    closes = [c[4] for c in candles[-15:]]
    delta = np.diff(closes)
    rsi = 100 - (100/(1+np.mean([x for x in delta if x>0])/abs(np.mean([x for x in delta if x<0])+1e-5)))
    return rsi>60 or rsi<40
def check_ema_cross(candles, fast=10, slow=20):
    closes = pd.Series([c[4] for c in candles])
    ema_fast = closes.ewm(span=fast).mean()
    ema_slow = closes.ewm(span=slow).mean()
    if len(ema_fast)<slow+2: return False
    return (ema_fast.iloc[-2]<ema_slow.iloc[-2] and ema_fast.iloc[-1]>ema_slow.iloc[-1]) or (ema_fast.iloc[-2]>ema_slow.iloc[-2] and ema_fast.iloc[-1]<ema_slow.iloc[-1])
def check_volume_spike(candles):
    vols = [c[5] for c in candles[-15:]]
    return vols[-1]>2*np.mean(vols[:-1])
def check_fibonacci_golden(candles):
    highs = [c[2] for c in candles[-30:]]
    lows = [c[3] for c in candles[-30:]]
    golden = max(highs)-(max(highs)-min(lows))*0.618
    return abs(candles[-1][4]-golden)/golden < 0.01
def ema_trend(candles):
    if len(candles)<30: return "sideways"
    closes = pd.Series([c[4] for c in candles])
    ema = closes.ewm(span=30).mean()
    if closes.iloc[-1] > ema.iloc[-1]: return "uptrend"
    elif closes.iloc[-1] < ema.iloc[-1]: return "downtrend"
    else: return "sideways"

def advanced_score(candles, symbol, news_impact, eco_events):
    score = 0
    reasons = []
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
    # Orderflow, news ve eco events ek puan
    of_score = orderflow_score(candles)
    score += int(of_score)
    if news_impact: score -= 1  # Yüksek haberde işlem azalt
    if eco_events: score -= 1
    return score, trend, of_score, {"delta": of_score, "block_trade": False}

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

# 7. Sinyal Spam/Korrelasyon/Fazla Pozisyon Kontrolü
def active_signal_exists(exchange, symbol, tf, direction):
    signals = load_json(SIGNALS_FILE, [])
    for s in signals:
        if (s.get("exchange")==exchange and s.get("symbol")==symbol and s.get("direction")==direction and s.get("result") is None):
            return True
    return False

def fetch_candles(exchange, symbol, timeframe, limit=150):
    try: return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except: return None

def fetch_yf_candles(symbol, tf="1m", limit=150):
    try:
        interval_map = {"1m":"1m", "5m":"5m", "15m":"15m", "1h":"1h"}
        df = yf.download(symbol, period="2d", interval=interval_map[tf])
        candles = []
        for idx,row in df.iterrows():
            candles.append([
                int(idx.timestamp()*1000), row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]
            ])
        return candles[-limit:]
    except: return None

# 8. Main Loop & Sinyal Üretici
async def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, data=payload)

async def check_active_signals():
    signals = load_json(SIGNALS_FILE, [])
    results = load_json(RESULTS_FILE, [])
    updated_signals = []
    retrain_flag = False
    for s in signals:
        if s.get("result") is not None: continue
        ex = EXCHANGES.get(s["exchange"], None)
        if ex is None:
            if s["exchange"]=="yfinance":
                candles = fetch_yf_candles(s["symbol"], s["timeframe"], 2)
            else:
                candles = None
        else:
            candles = fetch_candles(ex, s["symbol"], s["timeframe"], 2)
        if not candles: updated_signals.append(s); continue
        last_price = candles[-1][4]
        signal_closed = False
        if (s["direction"]=="LONG" and last_price>=s["tp"]) or (s["direction"]=="SHORT" and last_price<=s["tp"]):
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
        elif (s["direction"]=="LONG" and last_price<=s["sl"]) or (s["direction"]=="SHORT" and last_price>=s["sl"]):
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
    save_json(SIGNALS_FILE, updated_signals)
    save_json(RESULTS_FILE, results)
    if retrain_flag: retrain_ai_model_and_backtest(); daily_report()

async def scan_all_markets():
    news_headlines = fetch_news_headlines()
    economic_events = fetch_economic_calendar()
    for sym in YF_SYMBOLS:
        candles = fetch_yf_candles(sym, "1h", 150)
        if not candles or len(candles)<50: continue
        score, trend, orderflow, extra = advanced_score(candles, sym, any(news_headlines), any(economic_events))
        direction = get_signal_direction(candles)
        sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
        winrate = update_winrate()
        ai_prob = ai_score_predict(score, extra["delta"], abs(candles[-1][2]-candles[-1][3]))
        if score>=9 and not active_signal_exists("yfinance", sym, "1h", direction):
            last_close = candles[-1][4]
            msg = (
                f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | YF | {sym} | 1h\n"
                f"🏅 Score: {score}/13 | 🤖 AI: %{round(ai_prob*100,1)}\n"
                f"Orderflow: Δ={extra['delta']}\n"
                f"📈 Entry: <b>{last_close}</b>\n"
                f"⛔️ SL: <b>{sl}</b>\n"
                f"🎯 TP: <b>{tp}</b>\n"
                f"✅ WINRATE: %{winrate}"
            )
            await send_telegram_signal(msg)
            signals = load_json(SIGNALS_FILE, [])
            signals.append({
                "exchange": "yfinance",
                "symbol": sym,
                "timeframe": "1h",
                "entry": last_close,
                "sl": sl,
                "tp": tp,
                "direction": direction,
                "score": score,
                "trend": trend,
                "orderflow_score": orderflow,
                "open_time": datetime.now(timezone.utc).isoformat(),
                "result": None,
                "delta": extra["delta"],
                "volatility": abs(candles[-1][2]-candles[-1][3]),
                "session_time": True,
                "block_trade": extra.get("block_trade", False)
            })
            save_json(SIGNALS_FILE, signals)
    for ex_name, ex_info in MARKETS.items():
        exchange = EXCHANGES[ex_name]
        for symbol in ex_info["symbols"]:
            for tf in ex_info["timeframes"]:
                candles = fetch_candles(exchange, symbol, tf, 150)
                if not candles or len(candles)<50: continue
                score, trend, orderflow, extra = advanced_score(candles, symbol, any(news_headlines), any(economic_events))
                direction = get_signal_direction(candles)
                sl, tp = calc_atr_sl_tp(candles, rr_ratio=2.0, direction=direction)
                winrate = update_winrate()
                ai_prob = ai_score_predict(score, extra["delta"], abs(candles[-1][2]-candles[-1][3]))
                if score>=9 and not active_signal_exists(ex_name, symbol, tf, direction):
                    last_close = candles[-1][4]
                    msg = (
                        f"🚨 {'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} | {ex_name.upper()} | {symbol} | {tf}\n"
                        f"🏅 Score: {score}/13 | 🤖 AI: %{round(ai_prob*100,1)}\n"
                        f"Orderflow: Δ={extra['delta']}\n"
                        f"📈 Entry: <b>{last_close}</b>\n"
                        f"⛔️ SL: <b>{sl}</b>\n"
                        f"🎯 TP: <b>{tp}</b>\n"
                        f"✅ WINRATE: %{winrate}"
                    )
                    await send_telegram_signal(msg)
                    signals = load_json(SIGNALS_FILE, [])
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
                        "orderflow_score": orderflow,
                        "open_time": datetime.now(timezone.utc).isoformat(),
                        "result": None,
                        "delta": extra["delta"],
                        "volatility": abs(candles[-1][2]-candles[-1][3]),
                        "session_time": True,
                        "block_trade": extra.get("block_trade", False)
                    })
                    save_json(SIGNALS_FILE, signals)
    await asyncio.sleep(60)

async def main_loop():
    open_datasets()
    retrain_ai_model_and_backtest()
    while True:
        await check_active_signals()
        await scan_all_markets()

if __name__ == "__main__":
    asyncio.run(main_loop())

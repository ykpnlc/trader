import os
import time
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import ccxt
import yfinance as yf
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import requests

# ENV yükle
load_dotenv()
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

FOREX_SYMBOLS = ["EUR_USD", "GBP_USD"]
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
STOCK_SYMBOLS = ["AAPL", "MSFT"]

active_signals = []

# ----- TELEGRAM FONKSİYONU -----
def send_telegram(message, reply_to=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = requests.post(url, data=payload)
    if r.ok:
        try:
            return r.json()["result"]["message_id"]
        except:
            return None
    return None

# ----- OANDA VERİ ÇEKME -----
def fetch_oanda(symbol, count=100, granularity="M1"):
    try:
        client = API(access_token=OANDA_API_KEY)
        params = {"count": count, "granularity": granularity, "price": "M"}
        r = InstrumentsCandles(instrument=symbol, params=params)
        data = client.request(r)
        candles = data['candles']
        df = pd.DataFrame([{
            "time": x["time"],
            "open": float(x["mid"]["o"]),
            "high": float(x["mid"]["h"]),
            "low": float(x["mid"]["l"]),
            "close": float(x["mid"]["c"]),
            "volume": float(x["volume"]),
        } for x in candles if x['complete']])
        return df
    except Exception as e:
        print(f"OANDA veri hatası: {symbol} -> {e}")
        return pd.DataFrame()

# ----- CCXT (KRİPTO) VERİ ÇEKME -----
def fetch_ccxt(symbol, timeframe='1m', limit=100):
    try:
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Kripto veri hatası: {symbol} -> {e}")
        return pd.DataFrame()

# ----- YFINANCE (HİSSE/ENDEKS) VERİ ÇEKME -----
def fetch_yfinance(symbol, interval="1m", period="2d"):
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]  # Sütun isimlerini küçük harfe çevir
        if 'close' not in df.columns:
            raise Exception("YFinance veri çekimi hatalı: 'close' sütunu yok!")
        df.rename(columns={"datetime": "time"}, inplace=True)
        return df
    except Exception as e:
        print(f"Hisse veri hatası: {symbol} -> {e}")
        return pd.DataFrame()

# ----- GERÇEK TEKNİK/PA ANALİZ FONKSİYONLARI -----
def calc_ema(series, period=8):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def detect_bos(df, lookback=20):
    highs = df["high"].tail(lookback)
    prev_highs = highs[:-1]
    if df["close"].iloc[-1] > prev_highs.max():
        return 1
    else:
        return 0

def detect_order_block(df, window=20):
    for i in range(-window, -1):
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        wick = df["high"].iloc[i] - df["low"].iloc[i]
        if body / wick > 0.7 and df["volume"].iloc[i] > df["volume"].mean():
            return 1
    return 0

def pa_features(df):
    features = {
        "ema_8": calc_ema(df["close"], 8).iloc[-1],
        "ema_21": calc_ema(df["close"], 21).iloc[-1],
        "rsi_14": calc_rsi(df["close"], 14).iloc[-1],
        "bos": detect_bos(df),
        "order_block": detect_order_block(df),
        # Ek modüller buraya eklenebilir
    }
    return features

# ----- AI MODEL LOADER & TRAINER -----
def load_or_train_ai_model():
    model_file = "ai_model.json"
    scaler_file = "scaler.pkl"

    # Eğer model ve scaler dosyaları varsa yükle
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = XGBClassifier()
        model.load_model(model_file)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        print("AI modeli ve scaler yüklendi.")
        return model, scaler

    # Yoksa: Açık kaynak bir dataset ile fit et!
    print("Model yok, yeni eğitim başlatılıyor...")
    # ÖRNEK DUMMY DATASET: Bunu kendi gerçek datasetinle değiştir!
    # Price action ve teknik feature'larla zenginleştirilmiş örnek data:
    # Sütunlar: ema_8, ema_21, rsi_14, bos, order_block, label
    # label: 1 (win), 0 (loss)
    df = pd.read_csv("btc_usdt_1m_features.csv")  # <-- örnek dosya, senin datasetin!
    features = ["ema_8", "ema_21", "rsi_14", "bos", "order_block"]
    X = df[features].values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier()
    model.fit(X_scaled, y)
    # Kaydet
    model.save_model(model_file)
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print("AI modeli ve scaler eğitildi ve kaydedildi.")
    return model, scaler

# ----- SİNYAL OLUŞTURMA VE TAKİP -----
def ai_predict(features, model, scaler):
    X = np.array(list(features.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    score = model.predict_proba(X_scaled)[0][1]
    return score

def check_and_send_signal(symbol, df, market_type, model, scaler):
    if len(df) < 25: return  # yeterli veri yoksa atla
    features = pa_features(df)
    ai_score = ai_predict(features, model, scaler)
    if ai_score > 0.65:
        entry = df["close"].iloc[-1]
        sl = entry - 0.005 * entry
        tp = entry + 0.01 * entry
        message = (
            f"*Sinyal!*\n"
            f"Parite: `{symbol}`\n"
            f"Giriş: `{entry}`\n"
            f"TP: `{tp}` | SL: `{sl}`\n"
            f"AI Skoru: `{round(ai_score,2)}`\n"
            f"EMA8: `{round(features['ema_8'],2)}` / EMA21: `{round(features['ema_21'],2)}`\n"
            f"RSI: `{round(features['rsi_14'],2)}`\n"
            f"BoS: `{features['bos']}` OrderBlock: `{features['order_block']}`\n"
            f"Market: {market_type}\n"
            f"Zaman: {df['time'].iloc[-1]}\n"
            f"`#TradeBot`"
        )
        msg_id = send_telegram(message)
        signal = {
            "symbol": symbol, "entry": entry, "tp": tp, "sl": sl,
            "msg_id": msg_id, "market_type": market_type, "active": True
        }
        active_signals.append(signal)

def track_signals(df_dict):
    for sig in active_signals:
        if not sig["active"]:
            continue
        df = df_dict.get(sig["symbol"], None)
        if df is None or len(df) == 0: continue
        last_price = df["close"].iloc[-1]
        if last_price >= sig["tp"]:
            send_telegram(f"🎯 *WIN!* Parite: `{sig['symbol']}` TP’ye ulaştı! Giriş: `{sig['entry']}` ➡️ Çıkış: `{last_price}`", reply_to=sig["msg_id"])
            sig["active"] = False
        elif last_price <= sig["sl"]:
            send_telegram(f"🛑 *LOSS!* Parite: `{sig['symbol']}` SL’ye ulaştı! Giriş: `{sig['entry']}` ➡️ Çıkış: `{last_price}`", reply_to=sig["msg_id"])
            sig["active"] = False

# ----- ANA LOOP -----
def main_loop():
    model, scaler = load_or_train_ai_model()
    while True:
        df_dict = {}
        for symbol in FOREX_SYMBOLS:
            df = fetch_oanda(symbol)
            if len(df): df_dict[symbol] = df
            check_and_send_signal(symbol, df, "FOREX", model, scaler)
        for symbol in CRYPTO_SYMBOLS:
            df = fetch_ccxt(symbol)
            if len(df): df_dict[symbol] = df
            check_and_send_signal(symbol, df, "CRYPTO", model, scaler)
        for symbol in STOCK_SYMBOLS:
            df = fetch_yfinance(symbol)
            if len(df): df_dict[symbol] = df
            check_and_send_signal(symbol, df, "STOCK", model, scaler)
        track_signals(df_dict)
        time.sleep(60)

if __name__ == "__main__":
    main_loop()

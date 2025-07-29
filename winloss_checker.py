
import ccxt
import pandas as pd
import time
from utils import load_active_signals, clear_active_signal, log_winloss
from telegram_logger import send_result_message

exchange = ccxt.binance()

def fetch_latest_price(pair):
    try:
        ticker = exchange.fetch_ticker(pair)
        return ticker['last']
    except:
        return None

def monitor_active_signals():
    print("📡 SL/TP kontrol döngüsü başlatıldı.")
    while True:
        active = load_active_signals()
        if not active:
            print("🔕 Aktif sinyal yok.")
            time.sleep(60)
            continue

        for key, signal in list(active.items()):
            pair = signal["symbol"]
            entry = signal["entry"]
            sl = signal["sl"]
            tp = signal["tp"]
            direction = signal["status"]
            latest = fetch_latest_price(pair)
            if latest is None:
                continue

            result = None
            if direction == "buy":
                if latest >= tp:
                    result = "WIN ✅ (TP)"
                elif latest <= sl:
                    result = "LOSS ❌ (SL)"
            elif direction == "sell":
                if latest <= tp:
                    result = "WIN ✅ (TP)"
                elif latest >= sl:
                    result = "LOSS ❌ (SL)"

            if result:
                print(f"📈 {pair} sonucu: {result}")
                send_result_message(pair, result)
                log_winloss(pair, result)
                clear_active_signal(pair, signal["timeframe"])

        time.sleep(30)  # her 30 saniyede kontrol

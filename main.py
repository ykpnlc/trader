
import time
import ccxt
import pandas as pd
from strategy import analyze_market
from telegram_logger import send_signal
from utils import has_active_signal, set_active_signal
from config import COIN_LIST, TIMEFRAMES, LOOP_INTERVAL

exchange = ccxt.binance()

def fetch_ohlcv(symbol, timeframe):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"Veri çekme hatası ({symbol} - {timeframe}):", str(e))
        return None

def main_loop():
    while True:
        print("⏳ Tarama başlıyor...")
        for pair in COIN_LIST:
            for tf_label, tf in TIMEFRAMES.items():
                symbol = pair
                if has_active_signal(pair, tf):
                    continue  # aynı coin ve timeframe için tekrar sinyal gönderme

                df = fetch_ohlcv(symbol, tf)
                if df is None or len(df) < 50:
                    continue

                result = analyze_market(df, symbol, tf)
                if result["entry"]:
                    sent = send_signal(
                        pair=result["symbol"],
                        direction=result["status"],
                        score=result["score"],
                        reasons=result["reasons"]
                    )
                    if sent:
                        set_active_signal(pair, tf, result)
                        print(f"✅ Sinyal gönderildi: {pair} ({tf})")
        print(f"🔁 Bekleniyor ({LOOP_INTERVAL} sn)...\n")
        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    main_loop()

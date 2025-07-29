
import time
from config import COIN_LIST, TIMEFRAMES, LOOP_INTERVAL
from strategy import analyze_market
from telegram_logger import send_signal
from data import fetch_ohlcv

def main_loop():
    print("🔄 Bot başlatıldı...")
    while True:
        print("⏳ Tarama başlıyor...")
        for symbol in COIN_LIST:
            for tf_label, tf in TIMEFRAMES.items():
                print(f"📊 {symbol} | Zaman dilimi: {tf_label} | Veri çekiliyor...")
                ohlcv = fetch_ohlcv(symbol, tf)
                if ohlcv is None or len(ohlcv) == 0:
                    print(f"❌ Veri çekme başarısız: {symbol} ({tf_label})")
                    continue
                print(f"✅ Veri alındı: {symbol} ({tf_label}) - {len(ohlcv)} mum")
                signal, details = analyze_market(ohlcv, symbol, tf_label)
                if signal:
                    print(f"📨 Sinyal bulundu: {symbol} ({tf_label}) - {signal}")
                    send_signal(symbol, tf_label, signal, details)
                else:
                    print(f"🚫 Sinyal yok: {symbol} ({tf_label})")
        print(f"⏱️ Bekleniyor ({LOOP_INTERVAL} sn)...")
        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    main_loop()

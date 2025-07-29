import time
import requests
import ccxt  # Binance için API
from datetime import datetime

# === TELEGRAM BİLGİLERİN ===
TELEGRAM_BOT_TOKEN = 'TELEGRAM_BOT_TOKENINIZI_GİRİN'
TELEGRAM_CHAT_ID = 'CHAT_ID_NIZI_GİRİN'

# === SEVİYELER ===
RESISTANCE = 3.1450
SUPPORT = 3.1050
BREAKOUT = 3.1500

# === EXCHANGE ===
exchange = ccxt.binance()

def get_price():
    ticker = exchange.fetch_ticker('XRP/USDT')
    return float(ticker['last'])

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    requests.post(url, data=data)

def check_levels():
    price = get_price()
    now = datetime.now().strftime("%H:%M:%S")
    
    if price >= BREAKOUT:
        send_telegram_message(f"[{now}] 🚀 XRP fiyatı {price:.4f} ile direnci kırdı! LONG için uygun olabilir.")
    elif price >= RESISTANCE:
        send_telegram_message(f"[{now}] ⚠️ XRP fiyatı {price:.4f} - direnç bölgesine yaklaştı.")
    elif price <= SUPPORT:
        send_telegram_message(f"[{now}] 🔻 XRP fiyatı {price:.4f} - destek bölgesine düştü. LONG fırsatı kollanabilir.")
    else:
        print(f"[{now}] XRP Fiyatı: {price:.4f} - İzleniyor...")

def main():
    while True:
        try:
            check_levels()
            time.sleep(60)  # 60 saniyede bir kontrol
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

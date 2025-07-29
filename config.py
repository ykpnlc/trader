import os

COIN_LIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "MATIC/USDT",
    "DOT/USDT", "TRX/USDT", "SHIB/USDT", "LTC/USDT", "ATOM/USDT",
    "UNI/USDT", "NEAR/USDT", "XLM/USDT", "INJ/USDT", "FIL/USDT"
]

TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

LOOP_INTERVAL = 60  # saniye

# Güvenli şekilde environment değişkenlerinden alınır
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
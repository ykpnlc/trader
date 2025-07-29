# config.py

# -------------------------------
# COIN ve ZAMAN AYARLARI
# -------------------------------

COIN_LIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
    "DOT/USDT", "TRX/USDT", "NEAR/USDT", "ATOM/USDT", "OP/USDT",
    "ARB/USDT", "LTC/USDT", "FTM/USDT", "ETC/USDT", "RNDR/USDT",
    "INJ/USDT", "APE/USDT", "GALA/USDT", "SAND/USDT", "MANA/USDT",
    "CRO/USDT", "AAVE/USDT", "FLOW/USDT", "CHZ/USDT", "EGLD/USDT",
    "DYDX/USDT", "ALGO/USDT", "HBAR/USDT", "1000PEPE/USDT", "UNI/USDT",
    "FIL/USDT", "RUNE/USDT", "LDO/USDT", "GMT/USDT", "XLM/USDT",
    "TWT/USDT", "ENS/USDT", "ZIL/USDT", "KAVA/USDT", "COMP/USDT",
    "CRV/USDT", "SKL/USDT", "1INCH/USDT", "WOO/USDT", "SUSHI/USDT"
]

TIMEFRAMES = {
    "scalp": "1m",
    "short": "15m",
    "intraday": "1h",
    "swing": "4h",
    "macro": "1d"
}

# -------------------------------
# LOOP AYARI (Kaç saniyede bir kontrol?)
# -------------------------------
LOOP_INTERVAL = 60  # 60 saniyede bir çalışır (1 dakika)

# -------------------------------
# TELEGRAM AYARLARI
# -------------------------------
TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
TELEGRAM_CHAT_ID = "your-chat-id"  # örn: -1001234567890

# -------------------------------
# STRATEJİ SKORLAMA EŞİĞİ
# -------------------------------
SIGNAL_SCORE_THRESHOLD = 7  # sinyal skoru en az 7 ise gönder

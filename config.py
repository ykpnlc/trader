import os

# === TELEGRAM AYARLARI ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === KULLANILACAK COIN LİSTESİ (ilk 50'den örnek) ===
COIN_LIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT",
    "LTC/USDT", "LINK/USDT", "TRX/USDT", "BCH/USDT", "NEAR/USDT",
    "ATOM/USDT", "UNI/USDT", "ETC/USDT", "XLM/USDT", "FIL/USDT"
]

# === PERFORMANS / SİNYAL KAYIT DOSYALARI ===
PERFORMANCE_PATH = "performance.json"
ACTIVE_SIGNALS_PATH = "active_signals.json"
SIGNAL_REGISTRY_PATH = "signal_registry.json"

# === ANALİZ SIKLIĞI (saniye) ===
LOOP_INTERVAL = 60  # her 60 saniyede bir tüm coinler kontrol edilir

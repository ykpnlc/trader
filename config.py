
from fetch_pairs import fetch_all_usdt_pairs

# Coin listesi otomatik çekilir
COIN_LIST = fetch_all_usdt_pairs()

# Zaman dilimleri (scalping için kısa zaman)
TIMEFRAMES = ["1m", "5m"]

# Sinyal puan eşiği (test modu için düşürüldü)
SIGNAL_SCORE_THRESHOLD = 5

# Telegram ayarları (Railway'de env olarak tanımlanmalı)
import os
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Test modu
TEST_MODE = True

import requests
import os
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_signal_message(symbol, signal):
    direction = signal['direction']
    score = signal['score']
    price = signal['price']
    entry = signal['entry']
    sl = signal['sl']
    tp = signal['tp']
    details = "\n".join(signal['details'])

    message = f"""
📡 <b>{direction} SİNYAL</b> – <code>{symbol}</code>
🎯 Skor: <b>{score}/10</b>
💰 Fiyat: <b>{price}</b>
📈 Entry: {entry}
🛑 SL: {sl}
🎯 TP: {tp}

🧠 Detaylar:
{details}
"""

    response = requests.post(
        f"{BASE_URL}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
    )

    if response.status_code == 200:
        return response.json()['result']['message_id']
    else:
        print("Telegram gönderim hatası:", response.text)
        return None

def send_result_message(symbol, result, original_message_id):
    emoji = "✅" if result == "win" else "❌"
    message = f"""
{emoji} <b>{symbol}</b> sinyali <b>{result.upper()}</b> olarak sonuçlandı.
"""
    requests.post(
        f"{BASE_URL}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "reply_to_message_id": original_message_id
        }
    )

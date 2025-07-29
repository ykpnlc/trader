
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_signal(pair, direction, score, reasons, chart_path=None):
    message = f"🔔 *ALERT* 🔔\n"               f"*Pair:* `{pair}`\n"               f"*Direction:* `{direction.upper()}`\n"               f"*Score:* `{score}`\n"               f"*Reasons:*\n" + "\n".join([f"- {r}" for r in reasons])

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram mesaj hatası:", e)
        return False

    if chart_path:
        send_photo(chart_path)

    return True

def send_result_message(pair, result):
    message = f"📊 *Signal Result:* `{pair}` → *{result}*"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram sonuç mesaj hatası:", e)

def send_photo(file_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(file_path, 'rb') as photo:
            data = {
                "chat_id": TELEGRAM_CHAT_ID
            }
            files = {
                "photo": photo
            }
            requests.post(url, data=data, files=files)
    except Exception as e:
        print("Telegram görsel gönderim hatası:", e)

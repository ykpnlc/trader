
import requests

def send_signal(pair, direction, exchange, timeframe, score, ai_score, orderflow, entry, sl, tp, winrate, bot_token, chat_id, reply_to_message_id=None):
    message = f"""
{'🔴 SHORT' if direction == 'SHORT' else '🟢 LONG'} | {exchange.upper()} | {pair.upper()} | {timeframe}
🏅 Score: {score}   🤖 AI: %{ai_score}
Orderflow: Δ={orderflow}
📈 Entry: {entry}
🛑 SL: {sl}
🎯 TP: {tp}
✅ WINRATE: %{winrate}
"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Sinyal gönderildi:", pair)
        return response.json().get("result", {}).get("message_id")
    else:
        print("❌ Telegram mesajı gönderilemedi.")
        return None

def send_result(bot_token, chat_id, result_type, reply_to_message_id):
    emoji = "✅ WIN" if result_type == "WIN" else "❌ LOSS"
    message = f"{emoji} - Pozisyon sonucu"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "reply_to_message_id": reply_to_message_id
    }

    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print(f"📨 {result_type} mesajı gönderildi.")
    else:
        print("❌ Sonuç mesajı gönderilemedi.")

import time
import json
from telegram import send_result_message
from config import PERFORMANCE_PATH, ACTIVE_SIGNALS_PATH
from market_data import fetch_ohlcv

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def check_signal_results(active_signals):
    updated = False
    new_active_signals = {}

    for symbol, info in active_signals.items():
        if info["status"] != "open":
            continue

        ohlcv = fetch_ohlcv(symbol, '1m', limit=5)
        closes = [c[4] for c in ohlcv[-3:]]

        hit_tp = any(c >= info["tp"] for c in closes)
        hit_sl = any(c <= info["sl"] for c in closes)

        if hit_tp:
            update_performance("win")
            send_result_message(symbol, "win", info["message_id"])
            updated = True

        elif hit_sl:
            update_performance("loss")
            send_result_message(symbol, "loss", info["message_id"])
            updated = True

        else:
            new_active_signals[symbol] = info  # halen açık işlem

    if updated:
        save_json(ACTIVE_SIGNALS_PATH, new_active_signals)
        return new_active_signals

    return None

def update_performance(result):
    data = load_json(PERFORMANCE_PATH)

    if "total_signals" not in data:
        data["total_signals"] = 0
        data["wins"] = 0
        data["losses"] = 0

    data["total_signals"] += 1
    if result == "win":
        data["wins"] += 1
    elif result == "loss":
        data["losses"] += 1

    wins = data["wins"]
    total = data["total_signals"]
    data["winrate"] = round((wins / total) * 100, 2) if total > 0 else 0

    save_json(PERFORMANCE_PATH, data)

from strategy import analyze_market
from telegram import send_signal_message, send_result_message
from market_data import fetch_price_data
from winrate_tracker import check_signal_results
from config import COIN_LIST, TIMEFRAMES, LOOP_INTERVAL
import json
import time
import os

SIGNAL_REGISTRY_PATH = "signal_registry.json"
ACTIVE_SIGNALS_PATH = "active_signals.json"

# Yükleme veya boş başlatma
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main_loop():
    signal_registry = load_json(SIGNAL_REGISTRY_PATH)
    active_signals = load_json(ACTIVE_SIGNALS_PATH)

    while True:
        for coin in COIN_LIST:
            try:
                data = fetch_price_data(coin)
                signal = analyze_market(coin, data)

                if signal:
                    coin_key = f"{coin}-{signal['direction']}"
                    if coin_key not in signal_registry:
                        msg_id = send_signal_message(coin, signal)
                        signal_registry[coin_key] = {
                            "timestamp": time.time()
                        }
                        active_signals[coin] = {
                            **signal,
                            "status": "open",
                            "message_id": msg_id
                        }
                        save_json(SIGNAL_REGISTRY_PATH, signal_registry)
                        save_json(ACTIVE_SIGNALS_PATH, active_signals)

            except Exception as e:
                print(f"{coin} analiz hatası: {e}")

        # TP veya SL kontrolü
        updated_signals = check_signal_results(active_signals)
        if updated_signals is not None:
            save_json(ACTIVE_SIGNALS_PATH, updated_signals)

        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    main_loop()

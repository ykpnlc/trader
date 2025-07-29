
import json
import os

# Aktif sinyallerin saklandığı dosya
ACTIVE_SIGNAL_FILE = "active_signals.json"

def score_signal(signals):
    return sum(1 for s in signals if s == 1)

def load_active_signals():
    if not os.path.exists(ACTIVE_SIGNAL_FILE):
        return {}
    with open(ACTIVE_SIGNAL_FILE, "r") as f:
        return json.load(f)

def save_active_signals(data):
    with open(ACTIVE_SIGNAL_FILE, "w") as f:
        json.dump(data, f, indent=2)

def has_active_signal(pair, tf):
    active = load_active_signals()
    key = f"{pair}-{tf}"
    return key in active

def set_active_signal(pair, tf, data):
    active = load_active_signals()
    key = f"{pair}-{tf}"
    active[key] = data
    save_active_signals(active)

def clear_active_signal(pair, tf):
    active = load_active_signals()
    key = f"{pair}-{tf}"
    if key in active:
        del active[key]
        save_active_signals(active)

def log_winloss(pair, result):
    log_file = "trade_log.json"
    if not os.path.exists(log_file):
        logs = []
    else:
        with open(log_file, "r") as f:
            logs = json.load(f)
    logs.append({"pair": pair, "result": result})
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

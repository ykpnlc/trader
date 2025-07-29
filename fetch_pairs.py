
import ccxt

def fetch_all_usdt_pairs():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    return sorted([symbol for symbol in markets if symbol.endswith("/USDT") and ":" not in symbol])

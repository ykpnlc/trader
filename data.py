
import ccxt
import pandas as pd
import time

exchange = ccxt.binance({
    'enableRateLimit': True
})

def fetch_ohlcv(symbol, timeframe, limit=500):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Veri çekme hatası ({symbol} - {timeframe}): {e}")
        return None

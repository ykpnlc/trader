import ccxt
import pandas as pd
import time

# Binance borsasını başlat
exchange = ccxt.binance()

# Zaman dilimlerini eşle
BINANCE_TIMEFRAMES = {
    '1d': '1d',
    '4h': '4h',
    '15m': '15m',
    '1m': '1m'
}

# Tek bir zaman dilimi için veri çek
def fetch_ohlcv(symbol, timeframe, limit=100):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Tüm zaman dilimlerini birlikte çek
def fetch_price_data(symbol):
    full_data = {}
    for tf_key, binance_tf in BINANCE_TIMEFRAMES.items():
        try:
            df = fetch_ohlcv(symbol, binance_tf)
            full_data[tf_key] = df
            time.sleep(0.2)  # Binance rate limit
        except Exception as e:
            print(f"{symbol} için {tf_key} veri alınamadı: {e}")
    return full_data


import pandas as pd
from strategy import analyze_market
import ccxt

exchange = ccxt.binance()

def fetch_historical_data(symbol, timeframe, limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def backtest(pair, timeframe):
    df = fetch_historical_data(pair, timeframe)
    if df is None or len(df) < 100:
        print(f"Veri yetersiz: {pair}")
        return

    signals = []
    for i in range(50, len(df)):
        slice_df = df.iloc[i-50:i]
        result = analyze_market(slice_df, pair, timeframe)
        if result["entry"]:
            signals.append({
                "entry_price": result["entry"],
                "sl": result["sl"],
                "tp": result["tp"],
                "close_price": df['close'].iloc[i],
                "timestamp": df['timestamp'].iloc[i],
                "status": result["status"]
            })

    wins, losses = 0, 0
    for s in signals:
        if s["status"] == "buy":
            if s["close_price"] >= s["tp"]:
                wins += 1
            elif s["close_price"] <= s["sl"]:
                losses += 1
        elif s["status"] == "sell":
            if s["close_price"] <= s["tp"]:
                wins += 1
            elif s["close_price"] >= s["sl"]:
                losses += 1

    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0
    print(f"Pair: {pair} | Timeframe: {timeframe}")
    print(f"Toplam Sinyal: {total} | Win: {wins} | Loss: {losses} | Winrate: %{winrate:.2f}")

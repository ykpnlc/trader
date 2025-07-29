
import pandas as pd

def detect_trend_structure(df):
    try:
        closes = df['close'].iloc[-10:]
        if all(closes.diff().dropna() > 0):
            return 1, "Uptrend (Higher Highs)"
        elif all(closes.diff().dropna() < 0):
            return 1, "Downtrend (Lower Lows)"
    except:
        return 0, ""
    return 0, ""

def detect_support_resistance(df):
    try:
        support = df['low'].rolling(window=20).min().iloc[-1]
        resistance = df['high'].rolling(window=20).max().iloc[-1]
        price = df['close'].iloc[-1]

        if abs(price - support) / price < 0.01:
            return 1, "Near Support Zone"
        elif abs(price - resistance) / price < 0.01:
            return 1, "Near Resistance Zone"
    except:
        return 0, ""
    return 0, ""

def detect_supply_demand(df):
    try:
        recent = df.iloc[-3:]
        if recent['close'].mean() > df['open'].iloc[-4]:
            return 1, "Demand Zone Formation"
        elif recent['close'].mean() < df['open'].iloc[-4]:
            return 1, "Supply Zone Formation"
    except:
        return 0, ""
    return 0, ""

def detect_wyckoff_phase(df):
    try:
        lows = df['low'].rolling(window=15).min()
        highs = df['high'].rolling(window=15).max()
        last = df.iloc[-1]

        if last['low'] < lows.iloc[-2] and last['close'] > df['close'].iloc[-2]:
            return 1, "Spring Phase (Wyckoff Accumulation)"
        elif last['high'] > highs.iloc[-2] and last['close'] < df['close'].iloc[-2]:
            return 1, "Upthrust Phase (Wyckoff Distribution)"
    except:
        return 0, ""
    return 0, ""

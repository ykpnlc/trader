
import pandas as pd

def detect_bos_choch(df):
    try:
        # Break of Structure: yeni HH veya LL
        recent_high = df['high'].iloc[-2]
        curr_high = df['high'].iloc[-1]
        recent_low = df['low'].iloc[-2]
        curr_low = df['low'].iloc[-1]

        if curr_high > recent_high:
            return 1, "BoS (Break of Structure) Up"
        elif curr_low < recent_low:
            return 1, "BoS (Break of Structure) Down"
    except:
        return 0, ""
    return 0, ""

def detect_order_blocks(df):
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if prev['close'] > prev['open'] and last['close'] < last['open']:
            return 1, "Bearish Order Block"
        elif prev['close'] < prev['open'] and last['close'] > last['open']:
            return 1, "Bullish Order Block"
    except:
        return 0, ""
    return 0, ""

def detect_fvg(df):
    try:
        prev = df.iloc[-3]
        curr = df.iloc[-1]
        if curr['low'] > prev['high']:
            return 1, "FVG Up (Fair Value Gap)"
        elif curr['high'] < prev['low']:
            return 1, "FVG Down (Fair Value Gap)"
    except:
        return 0, ""
    return 0, ""

def detect_liquidity_sweep(df):
    try:
        highs = df['high'].iloc[-5:-1]
        lows = df['low'].iloc[-5:-1]
        if df['high'].iloc[-1] > highs.max():
            return 1, "Liquidity Sweep Up"
        elif df['low'].iloc[-1] < lows.min():
            return 1, "Liquidity Sweep Down"
    except:
        return 0, ""
    return 0, ""

def detect_displacement(df):
    try:
        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        range_candle = df['high'].iloc[-1] - df['low'].iloc[-1]
        if body > range_candle * 0.7:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                return 1, "Displacement Up"
            else:
                return 1, "Displacement Down"
    except:
        return 0, ""
    return 0, ""

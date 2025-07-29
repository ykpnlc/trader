
import pandas as pd

def detect_engulfing(df):
    try:
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Bullish engulfing
        if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
            return 1, "Bullish Engulfing Pattern"

        # Bearish engulfing
        elif prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
            return 1, "Bearish Engulfing Pattern"
    except:
        return 0, ""
    return 0, ""

def detect_pinbar(df):
    try:
        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        upper_shadow = candle['high'] - max(candle['close'], candle['open'])
        lower_shadow = min(candle['close'], candle['open']) - candle['low']

        if lower_shadow > 2 * body and upper_shadow < body:
            return 1, "Bullish Pin Bar"
        elif upper_shadow > 2 * body and lower_shadow < body:
            return 1, "Bearish Pin Bar"
    except:
        return 0, ""
    return 0, ""

def detect_double_top_bottom(df):
    try:
        last = df['close'].iloc[-1]
        last20 = df['close'].iloc[-20:]
        high_count = (last20 > last20.mean() * 1.01).sum()
        low_count = (last20 < last20.mean() * 0.99).sum()

        if high_count >= 2 and last < last20.mean():
            return 1, "Double Top Likely"
        elif low_count >= 2 and last > last20.mean():
            return 1, "Double Bottom Likely"
    except:
        return 0, ""
    return 0, ""

def detect_eqh_eql(df):
    try:
        highs = df['high'].rolling(window=20).max()
        lows = df['low'].rolling(window=20).min()
        if abs(df['high'].iloc[-1] - highs.iloc[-1]) < 0.1:
            return 1, "EQH - Equal High Detected"
        if abs(df['low'].iloc[-1] - lows.iloc[-1]) < 0.1:
            return 1, "EQL - Equal Low Detected"
    except:
        return 0, ""
    return 0, ""

def detect_fractal_breakout(df):
    try:
        if df['high'].iloc[-1] > max(df['high'].iloc[-6:-1]):
            return 1, "Fractal Breakout Up"
        elif df['low'].iloc[-1] < min(df['low'].iloc[-6:-1]):
            return 1, "Fractal Breakout Down"
    except:
        return 0, ""
    return 0, ""

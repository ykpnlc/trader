import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# === DESTEK FONKSIYONLARI ===

def is_engulfing(df):
    prev = df.iloc[-2]
    last = df.iloc[-1]
    if (
        prev['close'] < prev['open'] and
        last['close'] > last['open'] and
        last['close'] > prev['open'] and
        last['open'] < prev['close']
    ):
        return True
    return False

def is_bos(df):
    highs = df['high'].rolling(window=3).max()
    last_high = df.iloc[-1]['high']
    return last_high > highs.iloc[-2]

def is_choch(df):
    lows = df['low'].rolling(window=3).min()
    last_low = df.iloc[-1]['low']
    return last_low < lows.iloc[-2]

def detect_fvg(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    gap = abs(prev['close'] - last['open'])
    return gap > (last['high'] - last['low']) * 0.4

def detect_liquidity_sweep(df):
    last_low = df.iloc[-1]['low']
    recent_lows = df['low'][-6:-1]
    return any(last_low < l for l in recent_lows)

def trend_score(df):
    ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
    ema50 = EMAIndicator(df['close'], window=50).ema_indicator()
    return int(df['close'].iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1])

def volume_spike(df):
    volumes = df['volume'][-10:]
    return df['volume'].iloc[-1] > volumes.mean() * 1.5

def bollinger_breakout(df):
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    return df['close'].iloc[-1] > bb.bollinger_hband().iloc[-1]

def get_timeframe_score(tf_signals):
    return sum(tf_signals.values())

# === ANA ANALİZ FONKSIYONU ===

def analyze_market(symbol, multi_tf_data):
    """
    multi_tf_data = {
        '1d': df,
        '4h': df,
        '15m': df,
        '1m': df
    }
    """
    total_score = 0
    details = []
    direction = None  # 'LONG' or 'SHORT'
    tf_signals = {}

    for tf, df in multi_tf_data.items():
        score = 0
        this_tf = []

        if is_engulfing(df):
            score += 2
            this_tf.append("Engulfing")
            direction = 'LONG'

        if is_bos(df):
            score += 2
            this_tf.append("BoS")

        if is_choch(df):
            score += 2
            this_tf.append("ChoCH")
            direction = 'SHORT'

        if detect_liquidity_sweep(df):
            score += 1
            this_tf.append("Liquidity Sweep")

        if detect_fvg(df):
            score += 1
            this_tf.append("FVG")

        if trend_score(df):
            score += 1
            this_tf.append("Trend Align")

        if volume_spike(df):
            score += 1
            this_tf.append("Volume Spike")

        if bollinger_breakout(df):
            score += 1
            this_tf.append("BB Break")

        tf_signals[tf] = score
        details.append(f"{tf.upper()}: {' | '.join(this_tf)}")

    total_score = get_timeframe_score(tf_signals)

    if total_score >= 7 and direction:
        last_close = multi_tf_data['1m'].iloc[-1]['close']
        signal = {
            "direction": direction,
            "score": total_score,
            "price": round(float(last_close), 4),
            "details": details,
            "entry": round(float(last_close), 4),
            "sl": round(float(last_close) * 0.98, 4),
            "tp": round(float(last_close) * 1.02, 4)
        }
        return signal
    return None

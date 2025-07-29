
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import numpy as np

def ema_analysis(df):
    try:
        ema8 = EMAIndicator(df['close'], window=8).ema_indicator()
        ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
        ema50 = EMAIndicator(df['close'], window=50).ema_indicator()
        ema200 = EMAIndicator(df['close'], window=200).ema_indicator()
        if df['close'].iloc[-1] > ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
            return 1, "Bullish EMA Stack"
        elif df['close'].iloc[-1] < ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
            return 1, "Bearish EMA Stack"
    except:
        return 0, ""
    return 0, ""

def rsi_analysis(df):
    try:
        rsi = RSIIndicator(df['close'], window=14).rsi()
        if rsi.iloc[-1] < 30:
            return 1, "RSI Oversold"
        elif rsi.iloc[-1] > 70:
            return 1, "RSI Overbought"
    except:
        return 0, ""
    return 0, ""

def macd_analysis(df):
    try:
        macd = MACD(df['close'])
        if macd.macd_diff().iloc[-1] > 0:
            return 1, "MACD Bullish Momentum"
        elif macd.macd_diff().iloc[-1] < 0:
            return 1, "MACD Bearish Momentum"
    except:
        return 0, ""
    return 0, ""

def atr_analysis(df):
    try:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        return 1, f"ATR: {atr.iloc[-1]:.2f}"
    except:
        return 0, ""

def bollinger_analysis(df):
    try:
        bb = BollingerBands(df['close'])
        if df['close'].iloc[-1] > bb.bollinger_hband().iloc[-1]:
            return 1, "Bollinger Breakout Up"
        elif df['close'].iloc[-1] < bb.bollinger_lband().iloc[-1]:
            return 1, "Bollinger Breakout Down"
    except:
        return 0, ""
    return 0, ""

def ichimoku_analysis(df):
    try:
        ichi = IchimokuIndicator(df['high'], df['low'])
        if df['close'].iloc[-1] > ichi.ichimoku_a().iloc[-1] > ichi.ichimoku_b().iloc[-1]:
            return 1, "Above Ichimoku Cloud"
        elif df['close'].iloc[-1] < ichi.ichimoku_a().iloc[-1] < ichi.ichimoku_b().iloc[-1]:
            return 1, "Below Ichimoku Cloud"
    except:
        return 0, ""
    return 0, ""

def obv_analysis(df):
    try:
        obv = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        if obv.iloc[-1] > obv.iloc[-2]:
            return 1, "Increasing OBV"
    except:
        return 0, ""
    return 0, ""

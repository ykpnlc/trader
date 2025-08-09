import pandas as pd
import numpy as np

from indicators import ema_analysis, rsi_analysis, macd_analysis
from patterns import detect_engulfing, detect_pinbar, detect_double_top_bottom
from smc import detect_bos_choch, detect_order_blocks, detect_fvg

# ==== Yardımcı hesaplar ====

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"].replace(0, np.nan)
    pv = df["close"] * vol
    return (pv.cumsum() / vol.cumsum()).fillna(method="ffill")

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def _last_swing_low(df: pd.DataFrame, lookback: int = 10) -> float:
    return float(df["low"].tail(lookback).min())

def _last_swing_high(df: pd.DataFrame, lookback: int = 10) -> float:
    return float(df["high"].tail(lookback).max())

def _clip_reasons(reasons, k=6):
    out = [str(r) for r in reasons if r]
    uniq = []
    for r in out:
        if r not in uniq:
            uniq.append(r)
    return uniq[:k]

# ==== Skor ağırlıkları ====
WEIGHTS_LONG = {
    "ema": 18, "rsi": 6, "macd": 8,
    "engulf": 8, "pinbar": 6, "dtb": 6,
    "bos_choch_up": 22, "ob_bull": 12, "fvg_bull": 14
}
WEIGHTS_SHORT = {
    "ema": 18, "rsi": 6, "macd": 8,
    "engulf": 8, "pinbar": 6, "dtb": 6,
    "bos_choch_dn": 22, "ob_bear": 12, "fvg_bear": 14
}

SCORE_CUT_LONG = 70
SCORE_CUT_SHORT = 70
ATR_MULT = 1.5
RR_MIN = 2.0

def _weighted_score(flags: dict, weights: dict) -> int:
    score = 0
    for k, w in weights.items():
        score += w * int(bool(flags.get(k, False)))
    return int(min(100, score))

# ==== Ana analiz fonksiyonu ====
def analyze_market(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": None,
        "score": 0,
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": None,
        "reasons": [],
        "status": "neutral"
    }

    try:
        if df is None or df.empty:
            result["status"] = "error"
            result["reasons"] = ["Boş veri"]
            return result

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = df.copy().dropna(subset=["open", "high", "low", "close", "volume"])
        if len(df) < 100:
            result["status"] = "error"
            result["reasons"] = ["Yetersiz bar sayısı (<100)"]
            return result

        # Ek göstergeler
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["ema200"] = df["close"].ewm(span=200).mean()
        df["vwap"] = _compute_vwap(df)
        df["atr14"] = _compute_atr(df, 14)

        last = df.iloc[-1]
        last_close = float(last["close"])
        atr = float(last["atr14"]) if np.isfinite(last["atr14"]) else None

        # Analiz modülleri
        ema_signal, ema_reason = ema_analysis(df)
        rsi_signal, rsi_reason = rsi_analysis(df)
        macd_signal, macd_reason = macd_analysis(df)

        engulfing_signal, engulfing_reason = detect_engulfing(df)
        pinbar_signal, pinbar_reason = detect_pinbar(df)
        dtb_signal, dtb_reason = detect_double_top_bottom(df)

        bos_choch_signal, bos_choch_reason = detect_bos_choch(df)
        ob_signal, ob_reason = detect_order_blocks(df)
        fvg_signal, fvg_reason = detect_fvg(df)

        # Trend/VWAP filtresi
        ema_trend_up = bool(df["ema50"].iloc[-1] > df["ema200"].iloc[-1])
        ema_trend_dn = not ema_trend_up
        vwap_up = bool(last_close > float(df["vwap"].iloc[-1]))
        vwap_dn = not vwap_up

        bos_up = bool(bos_choch_signal.get("up", False)) if isinstance(bos_choch_signal, dict) else bool(bos_choch_signal)
        bos_dn = bool(bos_choch_signal.get("down", False)) if isinstance(bos_choch_signal, dict) else False
        ob_bull = bool(ob_signal.get("bull", False)) if isinstance(ob_signal, dict) else bool(ob_signal)
        ob_bear = bool(ob_signal.get("bear", False)) if isinstance(ob_signal, dict) else False
        fvg_bull = bool(fvg_signal.get("bull", False)) if isinstance(fvg_signal, dict) else bool(fvg_signal)
        fvg_bear = bool(fvg_signal.get("bear", False)) if isinstance(fvg_signal, dict) else False

        engulf = bool(engulfing_signal)
        pinbar = bool(pinbar_signal)
        dtb = bool(dtb_signal)

        # LONG bayraklar
        flags_long = {
            "ema": (ema_signal is True) or (ema_trend_up and vwap_up),
            "rsi": (rsi_signal is True),
            "macd": (macd_signal is True),
            "engulf": engulf,
            "pinbar": pinbar,
            "dtb": dtb and bos_up,
            "bos_choch_up": bos_up,
            "ob_bull": ob_bull,
            "fvg_bull": fvg_bull
        }

        # SHORT bayraklar
        flags_short = {
            "ema": (ema_signal is False) or (ema_trend_dn and vwap_dn),
            "rsi": (rsi_signal is False),
            "macd": (macd_signal is False),
            "engulf": engulf,
            "pinbar": pinbar,
            "dtb": dtb and bos_dn,
            "bos_choch_dn": bos_dn,
            "ob_bear": ob_bear,
            "fvg_bear": fvg_bear
        }

        score_long = _weighted_score(flags_long, WEIGHTS_LONG)
        score_short = _weighted_score(flags_short, WEIGHTS_SHORT)

        reasons = _clip_reasons([
            ema_reason, rsi_reason, macd_reason,
            engulfing_reason, pinbar_reason, dtb_reason,
            bos_choch_reason, ob_reason, fvg_reason
        ], k=8)

        chosen_side = None
        chosen_score = 0

        if score_long >= SCORE_CUT_LONG and score_long >= score_short:
            chosen_side = "buy"
            chosen_score = score_long
        elif score_short >= SCORE_CUT_SHORT and score_short > score_long:
            chosen_side = "sell"
            chosen_score = score_short

        if chosen_side:
            entry = last_close
            if atr and np.isfinite(atr):
                sl = entry - ATR_MULT * atr if chosen_side == "buy" else entry + ATR_MULT * atr
            else:
                sl = _last_swing_low(df) if chosen_side == "buy" else _last_swing_high(df)

            dist = abs(entry - sl)
            tp = entry + RR_MIN * dist if chosen_side == "buy" else entry - RR_MIN * dist
            rr = round(abs(tp - entry) / max(1e-9, dist), 2)

            if dist / entry < 0.0005:
                chosen_side = None
                reasons.append("SL mesafesi çok küçük")

        if chosen_side:
            result.update({
                "side": chosen_side,
                "score": int(chosen_score),
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "rr": rr,
                "reasons": reasons,
                "status": "signal"
            })
        else:
            result.update({
                "score": int(max(score_long, score_short)),
                "reasons": reasons,
                "status": "neutral"
            })

    except Exception as e:
        result["status"] = "error"
        result["reasons"] = _clip_reasons(result.get("reasons", []) + [f"Hata: {e}"], k=10)

    return result

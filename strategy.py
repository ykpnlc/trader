import pandas as pd
import numpy as np

from indicators import ema_analysis, rsi_analysis, macd_analysis   # mevcut
from patterns import detect_engulfing, detect_pinbar, detect_double_top_bottom
from smc import detect_bos_choch, detect_order_blocks, detect_fvg
from utils import score_signal  # ister kullan, ama burada ağırlıklı skor da var

# ==== Basit yardımcılar ====

def _safe_series(s, fill=None):
    try:
        return s.fillna(fill) if fill is not None else s
    except Exception:
        return s

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"].replace(0, np.nan)
    pv = (df["close"] * vol)
    # Kümülatif vwap - bar içi doğru değil ama “proxy” olarak yeterli
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
    sl = df["low"].tail(lookback).min()
    return float(sl) if np.isfinite(sl) else float(df["low"].iloc[-1])

def _last_swing_high(df: pd.DataFrame, lookback: int = 10) -> float:
    sh = df["high"].tail(lookback).max()
    return float(sh) if np.isfinite(sh) else float(df["high"].iloc[-1])

def _clip_reasons(reasons, k=6):
    # kısa ve net tut
    out = [str(r) for r in reasons if r]
    # tekrarları at
    uniq = []
    for r in out:
        if r not in uniq:
            uniq.append(r)
    return uniq[:k]

# ==== Ağırlıklar (isteğe göre ayarla / hyperopt) ====
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

SCORE_CUT_LONG = 70  # 0-100
SCORE_CUT_SHORT = 70
ATR_MULT = 1.5       # SL için
RR_MIN = 2.0         # en az 2R

def _weighted_score_long(flags: dict) -> int:
    score = 0
    for k, w in WEIGHTS_LONG.items():
        score += w * int(bool(flags.get(k, False)))
    return int(min(100, score))

def _weighted_score_short(flags: dict) -> int:
    score = 0
    for k, w in WEIGHTS_SHORT.items():
        score += w * int(bool(flags.get(k, False)))
    return int(min(100, score))

# ==== Ana fonksiyon ====
def analyze_market(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """
    Girdi: OHLCV DataFrame (index: datetime), kolonlar: open, high, low, close, volume
    Çıktı: {symbol, timeframe, side, score, entry, sl, tp, reasons[], status}
    """
    signal_result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": None,           # "buy" | "sell"
        "score": 0,
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": None,
        "reasons": [],
        "status": "neutral"
    }

    try:
        # ---- Veri güvenlik kontrolleri ----
        if df is None or df.empty:
            signal_result["status"] = "error"
            signal_result["reasons"] = ["Boş DataFrame"]
            return signal_result

        # Datetime index ve NaN temizliği
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = df.copy()
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        if len(df) < 100:
            signal_result["status"] = "error"
            signal_result["reasons"] = ["Yetersiz bar sayısı (<100)"]
            return signal_result

        # ---- Göstergeler/proxyler (yerel hesap) ----
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["ema200"] = df["close"].ewm(span=200).mean()
        df["vwap"] = _compute_vwap(df)
        df["atr14"] = _compute_atr(df, 14)

        last = df.iloc[-1]
        last_close = float(last["close"])
        atr = float(last["atr14"]) if np.isfinite(last["atr14"]) else None

        # === Teknik Göstergeler (mevcut fonksiyonlar) ===
        ema_signal, ema_reason = ema_analysis(df)      # beklenen: bool veya -1/0/1 vs.
        rsi_signal, rsi_reason = rsi_analysis(df)
        macd_signal, macd_reason = macd_analysis(df)

        # === Price Action Patternleri ===
        engulfing_signal, engulfing_reason = detect_engulfing(df)
        pinbar_signal, pinbar_reason = detect_pinbar(df)
        dtb_signal, dtb_reason = detect_double_top_bottom(df)

        # === SMC ===
        bos_choch_signal, bos_choch_reason = detect_bos_choch(df)  # beklenen: {"up":bool,"down":bool} olabilir
        ob_signal, ob_reason = detect_order_blocks(df)             # {"bull":bool,"bear":bool} veya bool
        fvg_signal, fvg_reason = detect_fvg(df)                    # {"bull":bool,"bear":bool} veya bool

        # ---- Sinyal bayraklarını normalize et ----
        # EMA trend filtresi + VWAP üstü/altı (proxy)
        ema_trend_up = bool(df["ema50"].iloc[-1] > df["ema200"].iloc[-1])
        ema_trend_dn = bool(df["ema50"].iloc[-1] < df["ema200"].iloc[-1])
        vwap_up = bool(last_close > float(df["vwap"].iloc[-1]))
        vwap_dn = not vwap_up

        # bos/ob/fvg dökümü olası çıktı formatlarına karşı esnek
        bos_up = bool(bos_choch_signal.get("up", False)) if isinstance(bos_choch_signal, dict) else bool(bos_choch_signal)
        bos_dn = bool(bos_choch_signal.get("down", False)) if isinstance(bos_choch_signal, dict) else False

        ob_bull = bool(ob_signal.get("bull", False)) if isinstance(ob_signal, dict) else bool(ob_signal)
        ob_bear = bool(ob_signal.get("bear", False)) if isinstance(ob_signal, dict) else False

        fvg_bull = bool(fvg_signal.get("bull", False)) if isinstance(fvg_signal, dict) else bool(fvg_signal)
        fvg_bear = bool(fvg_signal.get("bear", False)) if isinstance(fvg_signal, dict) else False

        # Patternler (bool bekleniyor)
        engulf = bool(engulfing_signal)
        pinbar = bool(pinbar_signal)
        dtb = bool(dtb_signal)

        # ---- LONG skor ----
        flags_long = {
            "ema": (ema_signal is True) or (ema_trend_up and vwap_up),
            "rsi": (rsi_signal is True),
            "macd": (macd_signal is True),
            "engulf": engulf,
            "pinbar": pinbar,
            "dtb": dtb and bos_up,  # double bottom + yapısal teyit
            "bos_choch_up": bos_up,
            "ob_bull": ob_bull,
            "fvg_bull": fvg_b

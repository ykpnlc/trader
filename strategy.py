
import pandas as pd
from indicators import ema_analysis, rsi_analysis, macd_analysis
from patterns import detect_engulfing, detect_pinbar, detect_double_top_bottom
from smc import detect_bos_choch, detect_order_blocks, detect_fvg
from utils import score_signal

def analyze_market(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    signal_result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "score": 0,
        "entry": None,
        "sl": None,
        "tp": None,
        "reasons": [],
        "status": "neutral"
    }

    try:
        # === Teknik Göstergeler ===
        ema_signal, ema_reason = ema_analysis(df)
        rsi_signal, rsi_reason = rsi_analysis(df)
        macd_signal, macd_reason = macd_analysis(df)

        # === Price Action Patternleri ===
        engulfing_signal, engulfing_reason = detect_engulfing(df)
        pinbar_signal, pinbar_reason = detect_pinbar(df)
        dtb_signal, dtb_reason = detect_double_top_bottom(df)

        # === Smart Money Concepts ===
        bos_choch_signal, bos_choch_reason = detect_bos_choch(df)
        ob_signal, ob_reason = detect_order_blocks(df)
        fvg_signal, fvg_reason = detect_fvg(df)

        # === Puanlama ve sinyal hesaplama ===
        strategies = [
            ema_signal, rsi_signal, macd_signal,
            engulfing_signal, pinbar_signal, dtb_signal,
            bos_choch_signal, ob_signal, fvg_signal
        ]
        reasons = [
            ema_reason, rsi_reason, macd_reason,
            engulfing_reason, pinbar_reason, dtb_reason,
            bos_choch_reason, ob_reason, fvg_reason
        ]

        score = score_signal(strategies)
        signal_result["score"] = score
        signal_result["reasons"] = [r for r in reasons if r]

        # === Sinyal oluşturma şartı ===
        if score >= 7:  # örnek eşik puan
            last_close = df['close'].iloc[-1]
            signal_result["entry"] = last_close
            signal_result["sl"] = last_close * 0.99
            signal_result["tp"] = last_close * 1.02
            signal_result["status"] = "buy"

    except Exception as e:
        signal_result["status"] = "error"
        signal_result["reasons"].append(str(e))

    return signal_result

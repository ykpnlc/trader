
import matplotlib.pyplot as plt
import pandas as pd

def plot_signal_chart(df, pair, signal):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Mum çubuğu (candlestick) stili
    for i in range(len(df)):
        color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
        ax.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], color=color)
        ax.plot([i, i], [df['open'].iloc[i], df['close'].iloc[i]], color=color, linewidth=6)

    ax.set_title(f"Signal: {signal['status'].upper()} | {pair}")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Price")

    # SL, TP, Entry çizgileri
    ax.axhline(signal['entry'], color='blue', linestyle='--', label='Entry')
    ax.axhline(signal['tp'], color='green', linestyle='--', label='Take Profit')
    ax.axhline(signal['sl'], color='red', linestyle='--', label='Stop Loss')

    # Açıklamaları yaz
    for i, reason in enumerate(signal['reasons']):
        ax.text(0.01, 0.95 - i*0.05, reason, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    ax.legend()
    filename = f"{pair.replace('/', '_')}_signal.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

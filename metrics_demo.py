"""
metrics_demo.py — Goosos tutorial #5 companion code.

Takes Part 1's MA(20,50) crossover on SPY and runs the full
metrics suite on it: Sharpe vs Sortino vs Calmar, VaR/CVaR,
and drawdown duration. The point: one number never tells
the whole story.

Run:  python metrics_demo.py
Needs: vectorbt, yfinance, pandas, numpy, matplotlib
"""
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")
from metrics import metrics_table, _drawdown_series
try:
    from quant_toolkit.backtest import run_backtest, ma_crossover_signals
except ImportError:  # standalone tutorial repo
    from backtest import run_backtest, ma_crossover_signals

SYMBOL = "SPY"
FAST, SLOW = 20, 50


def load_data():
    """~3 years of daily SPY closes. Falls back to synthetic if offline."""
    try:
        import yfinance as yf
        df = yf.download(SYMBOL, period="3y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError("empty download")
        close = df["Close"].iloc[:, 0] if df["Close"].ndim > 1 else df["Close"]
        close = close.dropna()
        print(f"Data: {len(close)} daily bars of {SYMBOL} "
              f"({close.index[0].date()} -> {close.index[-1].date()})")
        return close
    except Exception as e:
        print(f"Download failed ({e}); using synthetic data")
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-01", periods=750, freq="B")
        px = 400 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 750)))
        return pd.Series(px, index=idx, name="Close")


def strategy_returns(price):
    """MA crossover backtest -> per-bar strategy returns."""
    entries, exits = ma_crossover_signals(price, fast=FAST, slow=SLOW)
    entries = entries.vbt.signals.fshift(1)
    exits = exits.vbt.signals.fshift(1)
    pf = run_backtest(price, entries, exits)
    # Per-bar returns of the strategy equity curve
    value = pf.value()
    rets = value.pct_change().dropna()
    return rets, pf


def print_table(m):
    print("\n=== MA(20,50) on SPY — full metrics ===")
    print(f"  Sharpe (total vol)      : {m['sharpe']:+.2f}")
    print(f"  Sortino (downside only) : {m['sortino']:+.2f}")
    print(f"  Calmar (ret / maxDD)    : {m['calmar']:+.2f}")
    print(f"  Max drawdown            : {m['max_drawdown']:.2%}")
    print(f"  Max DD duration        : {m['max_drawdown_duration_bars']} bars")
    print(f"  Daily VaR 95%           : {m['var_95']:.2%}")
    print(f"  Daily CVaR 95%          : {m['cvar_95']:.2%}")
    print(f"  Bars                    : {m['n_bars']}")


def plot_underwater(price, path="underwater.png"):
    """Drawdown underwater chart: depth AND duration visible."""
    rets, _ = strategy_returns(price)
    dd = _drawdown_series(rets)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd * 100, 0, color="#c0392b", alpha=0.35)
    ax.plot(dd.index, dd * 100, color="#c0392b", lw=1)
    ax.set_ylabel("Drawdown %")
    ax.set_title(f"MA({FAST},{SLOW}) on {SYMBOL} — underwater chart")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


def plot_metrics_compare(price, path="metrics_compare.png"):
    """Bar chart: Sharpe vs Sortino vs Calmar on the same strategy."""
    rets, _ = strategy_returns(price)
    m = metrics_table(rets)
    names = ["Sharpe", "Sortino", "Calmar"]
    vals = [m["sharpe"], m["sortino"], m["calmar"]]
    colors = ["#7f8c8d", "#2980b9", "#27ae60"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Ratio")
    ax.set_title(f"MA({FAST},{SLOW}) on {SYMBOL} — one strategy, three stories")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{v:+.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    price = load_data()
    rets, pf = strategy_returns(price)
    m = metrics_table(rets)
    print_table(m)
    plot_underwater(price)
    plot_metrics_compare(price)

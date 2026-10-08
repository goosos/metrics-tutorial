"""
metrics.py — Part 5 of "Build Your Own Quant Research System".

Beyond Sharpe: Sortino, Calmar, CVaR, and drawdown duration.
Every function takes a returns series — the lingua franca of
performance measurement.

Tutorial: https://goosos.com/performance-metrics-beyond-sharpe
Code: https://github.com/goosos/metrics-tutorial
"""
import numpy as np
import pandas as pd


def _as_series(returns):
    """Coerce to float pd.Series, drop NaNs."""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        raise ValueError("need at least 2 returns")
    return r


def sharpe(returns, risk_free=0.0, periods_per_year=252):
    """
    Sharpe ratio: mean excess return / total volatility.

    The baseline everyone quotes — and the one Part 5 argues
    is insufficient on its own. Included here so every metric
    in the comparison table comes from the same returns series.

    Parameters
    ----------
    returns : pd.Series
        Per-period simple returns.
    risk_free : float
        Per-period risk-free rate. Default 0 (fine for comparisons).
    periods_per_year : int
        Annualization factor. 252 = daily, 12 = monthly.
        WARNING: annualization assumes i.i.d. returns — see article §1.
    """
    r = _as_series(returns) - risk_free
    vol = r.std(ddof=1)
    if vol == 0:
        return float("nan")
    return float(r.mean() / vol * np.sqrt(periods_per_year))


def sortino(returns, target=0.0, periods_per_year=252):
    """
    Sortino ratio: mean excess return / downside deviation.

    Unlike Sharpe, only *downside* volatility counts. A strategy
    that spikes upward looks the same as a flat one — as it should,
    because nobody complains about upside volatility.

    Parameters
    ----------
    returns : pd.Series
        Per-period simple returns.
    target : float
        Minimum acceptable return (MAR) per period. Deviations are
        measured against this, not against the mean. Default 0.
    periods_per_year : int
        Annualization factor (same caveat as sharpe()).
    """
    r = _as_series(returns)
    downside = r[r < target] - target
    if len(downside) == 0:
        return float("inf")  # never fell below target: perfect
    dd = downside.std(ddof=1)
    if dd == 0:
        return float("nan")
    return float((r.mean() - target) / dd * np.sqrt(periods_per_year))


def _drawdown_series(returns):
    """Return (drawdown series, underwater bool series)."""
    r = _as_series(returns)
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd


def max_drawdown(returns):
    """Worst peak-to-trough loss, as a positive fraction (e.g. 0.115)."""
    dd = _drawdown_series(returns)
    return float(-dd.min())


def max_drawdown_duration(returns):
    """
    Longest underwater stretch, in bars.

    Depth tells you how much you lost; duration tells you how long
    you waited to get it back. A 10% drawdown that recovers in a
    week is noise. A 10% drawdown that lasts two years is a regime
    you didn't understand.

    Returns
    -------
    int
        Number of bars in the longest drawdown episode.
    """
    dd = _drawdown_series(returns)
    underwater = dd < 0
    if not underwater.any():
        return 0
    # Lengths of consecutive True runs
    groups = (~underwater).cumsum()[underwater]
    longest = int(groups.value_counts().max())
    return longest


def calmar(returns, periods_per_year=252):
    """
    Calmar ratio: annualized return / max drawdown.

    Answers "how much return per unit of worst pain?" A strategy
    with Calmar < 1 earned less than its worst drawdown — you'd
    need strong nerves to hold it.

    Parameters
    ----------
    returns : pd.Series
        Per-period simple returns.
    periods_per_year : int
        Used to annualize the geometric mean return.
    """
    r = _as_series(returns)
    mdd = max_drawdown(r)
    if mdd == 0:
        return float("inf")
    total = float((1.0 + r).prod() - 1.0)
    years = len(r) / periods_per_year
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return float(ann / mdd)


def var(returns, alpha=0.05):
    """
    Value at Risk: the loss you exceed only `alpha` of the time.

    var(returns, 0.05) = -0.02 means "on 95% of days you lose
    less than 2%." Historical VaR — no distribution assumed.

    Returns a negative number (it's a loss).
    """
    r = _as_series(returns)
    return float(r.quantile(alpha))


def cvar(returns, alpha=0.05):
    """
    Conditional VaR (expected shortfall): average loss *given*
    you're in the worst `alpha` tail.

    VaR tells you the threshold; CVaR tells you how bad it gets
    past the threshold. For fat-tailed return distributions CVaR
    is the honest number — VaR alone hides the tail shape.

    Returns a negative number (it's a loss).
    """
    r = _as_series(returns)
    threshold = r.quantile(alpha)
    tail = r[r <= threshold]
    return float(tail.mean())


def metrics_table(returns, target=0.0, alpha=0.05, periods_per_year=252):
    """
    One dict with every metric in this module, computed on the
    same returns series so the comparison table is apples-to-apples.

    Returns
    -------
    dict with keys: sharpe, sortino, calmar, max_drawdown,
    max_drawdown_duration_bars, var_95, cvar_95, n_bars.
    """
    r = _as_series(returns)
    return {
        "sharpe": sharpe(r, periods_per_year=periods_per_year),
        "sortino": sortino(r, target=target,
                           periods_per_year=periods_per_year),
        "calmar": calmar(r, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(r),
        "max_drawdown_duration_bars": max_drawdown_duration(r),
        "var_95": var(r, alpha=alpha),
        "cvar_95": cvar(r, alpha=alpha),
        "n_bars": len(r),
    }


if __name__ == "__main__":
    # Smoke test on synthetic data: no crash, sane values.
    rng = np.random.default_rng(7)
    fake = pd.Series(rng.normal(0.0005, 0.01, 500))
    t = metrics_table(fake)
    assert t["n_bars"] == 500
    assert 0 < t["max_drawdown"] < 1
    assert t["max_drawdown_duration_bars"] > 0
    assert t["var_95"] < 0 and t["cvar_95"] <= t["var_95"]
    print("metrics.py smoke test passed")
    for k, v in t.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

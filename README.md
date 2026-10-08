# Performance Metrics Tutorial (Part 5)

**Part 5 of [Build Your Own Quant Research System](https://goosos.com/tutorials/)** — beyond Sharpe: Sortino, Calmar, CVaR, and drawdown duration.

Sharpe 0.87 sounds tradeable. Calmar 0.65 with a 9-month underwater stretch sounds like a strategy you'd abandon in month four. Same strategy — that's why you need the full metrics panel, not one number.

## What you get

| File | Purpose |
|---|---|
| `metrics.py` | `sharpe()` + `sortino()` + `calmar()` + `var()`/`cvar()` + `max_drawdown_duration()` + `metrics_table()` — the Part 5 toolkit module |
| `metrics_demo.py` | End-to-end: MA(20,50) on SPY → full metrics table → underwater + comparison charts |
| `article.md` | Full tutorial text |

## Quick start

```bash
pip install -r requirements.txt
python metrics_demo.py
```

Expected output (real SPY data, Oct 2023–Oct 2026):

```
=== MA(20,50) on SPY — full metrics ===
  Sharpe (total vol)      : +0.87
  Sortino (downside only) : +0.93
  Calmar (ret / maxDD)    : +0.65
  Max drawdown            : 13.25%
  Max DD duration        : 195 bars
  Daily VaR 95%           : -1.07%
  Daily CVaR 95%          : -1.65%
  Bars                    : 751
```

## The toolkit

This tutorial contributes `metrics.py` to [goosos/quant-toolkit](https://github.com/goosos/quant-toolkit):

```python
from quant_toolkit.metrics import metrics_table

rets = pf.value().pct_change().dropna()
t = metrics_table(rets)  # sharpe, sortino, calmar, drawdowns, VaR/CVaR
```

## References

- Sharpe, W. F. (1994). *The Sharpe Ratio.*
- Sortino, F. A., & van der Meer, R. (1991). *Downside Risk.*
- Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.*

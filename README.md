# Hurricane SPY Automated Pipeline

This repository implements the **Hurricane SPY** multi-timeframe market prediction
framework described in the accompanying research note. The code provides an
end-to-end automation layer that transforms market microstructure inputs into
probabilistic support/resistance maps, directional calls with abstention logic,
and speed forecasts.

## Project Layout

```
hurricane_spy/
├── __init__.py
├── aggregation.py
├── config.py
├── data_structures.py
├── execution.py
├── diagnostics.py
├── features.py
├── gating.py
├── pipeline.py
└── scripts/
    ├── run_pipeline.py
    └── run_trading.py
```

## Getting Started

1. Install dependencies (requires Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```
2. Populate the data bundle expected by the pipeline (see
   `hurricane_spy/data_structures.py` for schema details).
3. Run the example pipeline:
   ```bash
   python -m hurricane_spy.scripts.run_pipeline
   ```

## Algorithm Overview

The implementation follows the structure of the Hurricane SPY specification:

- **Feature Construction** (`features.py`)
  - Hurricane intensity scale derived from realized volatility, drift, and gamma
    pressure.
  - Kernel-based support/resistance potential maps.
  - Directional drift estimator combining GEX sign, dark-pool index changes,
    technical alignment, and order-flow imbalance.
  - Speed forecasts that scale the local volatility surface with variance
    amplifiers.
  - Barrier hitting and conformal modules for probability calibration.

- **Stability Gating** (`gating.py`)
  - Event-aware abstention with configurable cooling windows.
  - Regime flip detection using a rolling CUSUM statistic and volatility regime
    classifier.
  - Exogenous flow dominance and hedging-pressure gating to suppress unreliable
    signals.

- **Aggregation** (`aggregation.py`)
  - Stress-weighted global minimum-variance (GMV) blending across timeframes.
  - Conformalized reliability adjustment and Brier score tracking per regime.

- **Pipeline Orchestration** (`pipeline.py`)
  - Validates inputs, computes per-timeframe forecasts, applies stability
    controls, and aggregates the final predictions.
  - Provides detailed diagnostics of calibration, gating decisions, and
    abstention rationale.

- **Trading Integration** (`execution.py`)
  - Lightweight Alpaca REST client with environment-loaded credentials.
  - Risk controls for mapping aggregate forecasts into target positions.
  - Trader orchestration that queries current positions, derives the
    rebalance required, and routes orders (including optional flatten-on-abstain).

## Sending Orders to Alpaca

The repository ships with a reference adapter that transforms Hurricane SPY
forecasts into Alpaca orders. To use it:

1. Export your Alpaca credentials (paper or live) as environment variables:
   ```bash
   export ALPACA_API_KEY="<your key>"
   export ALPACA_SECRET_KEY="<your secret>"
   # Optional override if not using the default paper endpoint
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
   ```
2. Adjust the trading configuration (symbol, sizing, risk thresholds) either by
   editing `hurricane_spy/scripts/run_trading.py` or by instantiating
   `TradingConfig` in your own harness.
3. Run the demonstration script to see the end-to-end automation, including the
   trade decision and any submitted order payloads:
   ```bash
   python -m hurricane_spy.scripts.run_trading
   ```

The `HurricaneTrader` class can also be embedded into schedulers or other
automation frameworks once you supply real market data bundles instead of the
dummy generator used in the example script.

See the inline documentation throughout the codebase for the precise formulas
and configuration options.

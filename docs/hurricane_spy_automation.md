# Hurricane SPY Automated Pipeline

This repository implements the **Hurricane SPY** multi-timeframe market prediction framework described in the accompanying research note. The code provides an end-to-end automation layer that transforms market microstructure inputs into probabilistic support/resistance maps, directional calls with abstention logic, and speed forecasts.

## Prerequisites

- Python **3.9 or later** with a virtual environment recommended.
- Access to historical SPY OHLCV data and, optionally, Unusual Whales order-flow
  feeds for live execution.
- Alpaca paper-trading or live credentials if you intend to route orders.
- `pip install -r requirements.txt`
- Optional: `UNUSUAL_WHALES_API_TOKEN` environment variable for live order-flow
  ingestion.

## Configuration Quick Start

1. Create an environment file (for example `.env`) and populate the following
   variables if you plan to exercise live integrations:

   ```bash
   export ALPACA_API_KEY="your-key"
   export ALPACA_SECRET_KEY="your-secret"
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # default
   export UNUSUAL_WHALES_API_TOKEN="optional-token"
   ```

2. Review `hurricane_spy/config.py` to adjust timeframe definitions,
   abstention thresholds, and aggregation weights. The defaults mirror the
   specification in the research note.

3. Construct or fetch a `MarketDataBundle` containing the inputs listed in
   `hurricane_spy/data_structures.py`. The helper `assemble_bundle` in
   `hurricane_spy/data_sources.py` can assist when ingesting data programmatically.

4. Dry-run the synthetic pipeline to confirm the environment is working:

   ```bash
   python -m hurricane_spy.scripts.run_pipeline --seed 42
   ```

5. Optional – validate the trading harness against Alpaca paper accounts:

   ```bash
   python -m hurricane_spy.scripts.run_trading --dry-run
   ```

   The `--dry-run` flag keeps the example from submitting real orders while still
   exercising authentication and sizing logic.

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
   `hurricane_spy/data_structures.py` for schema details). The regression tests
   demonstrate the minimum viable inputs for synthetic bundles.
3. Run the example pipeline with deterministic data if desired:
   ```bash
   python -m hurricane_spy.scripts.run_pipeline --seed 42
   ```
4. Inspect the console output which includes aggregate predictions,
   per-timeframe diagnostics, and gating decisions.

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

- **Data Source Bridges** (`data_sources.py`)
  - Convenience client for the Unusual Whales API with helpers to align order
    flow, dark pool prints, and gamma exposure streams.
  - Utilities for merging those feeds into the order-flow imbalance frame and
    assembling validated `MarketDataBundle` objects for the pipeline.

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
3. (Optional) Export your Unusual Whales API token to drive real-time order
   flow ingestion:
   ```bash
   export UNUSUAL_WHALES_API_TOKEN="<your token>"
   ```
4. Run the demonstration script to see the end-to-end automation, including the
   trade decision, any submitted order payloads, and the data ingestion mode
   (synthetic vs. live Unusual Whales feeds). Append `--dry-run` to the command
   to log the intended orders without submitting them:
   ```bash
   python -m hurricane_spy.scripts.run_trading --dry-run
   ```

The `HurricaneTrader` class can also be embedded into schedulers or other
automation frameworks. When the `UNUSUAL_WHALES_API_TOKEN` environment variable
is provided, the reference trading harness will automatically pull order-flow
and dark-pool data from the Unusual Whales API; otherwise it falls back to the
synthetic bundle generator for offline testing.

## Scheduling and Monitoring

- **Cron** example (run every weekday at 15:55 New York time):

  ```cron
  55 20 * * 1-5 /usr/bin/env bash -lc 'cd /path/to/Hurricane_V2 && \ 
      source .venv/bin/activate && \ 
      python -m hurricane_spy.scripts.run_trading >> logs/hurricane.log 2>&1'
  ```

- **systemd** service units can provide more control for long-running daemons.
  Ensure the unit exports the necessary environment variables or sources a file
  that does.

- The trading script emits structured logs describing the chosen gates and
  final trade decision. Pipe these logs into your observability stack or cloud
  logging provider for alerting.

## Regression Testing

- Run `pytest` locally before committing changes. The suite builds deterministic
  synthetic bundles and exercises:
  - Per-timeframe payload structure and gating behaviour
  - Stress-weighted GMV aggregation under multiple stress regimes
  - Validation failures for malformed bundles

- The GitHub Actions workflow `.github/workflows/test.yml` automatically runs
  the regression suite and uploads coverage artefacts on each push or pull
  request.

## Troubleshooting

- **Missing optional dependencies** – install `requests` if you plan to fetch
  live Unusual Whales data. The client raises a descriptive runtime error when
  the package is absent.
- **Alpaca authentication failures** – verify the API keys and ensure the base
  URL matches the environment (paper or live).
- **Data validation errors** – use the exceptions raised by
  `MarketDataBundle.validate` to identify missing columns or negative
  volatility entries. The regression tests include minimal reproductions for
  common mistakes.
- **Merge conflicts** – the README intentionally remains concise. Place detailed
  release notes or operational runbooks inside `docs/` to reduce churn.

See the inline documentation throughout the codebase for the precise formulas
and configuration options.

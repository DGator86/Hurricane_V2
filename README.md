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
├── diagnostics.py
├── features.py
├── gating.py
├── pipeline.py
└── scripts/
    └── run_pipeline.py
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

See the inline documentation throughout the codebase for the precise formulas
and configuration options.

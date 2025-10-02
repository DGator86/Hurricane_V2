# Hurricane_V2

This repository hosts the Hurricane SPY automation codebase. The Python
packages under `hurricane_spy/` expose the feature engineering, gating, and
aggregation logic for producing multi-timeframe support/resistance, direction,
and speed forecasts. The top-level scripts provide reproducible demos for both
signal generation and Alpaca paper trading integration.

## Documentation

- [Automation guide](docs/hurricane_spy_automation.md) – project structure,
  configuration, data requirements, and scheduler examples.
- [Regression tests](tests/test_pipeline.py) – deterministic coverage for the
  pipeline and stress-weighted GMV aggregator.
- [Example scripts](hurricane_spy/scripts/) – synthetic end-to-end pipeline
  runner and reference trading harness.

## Local development

1. Install dependencies: `pip install -r requirements.txt`
2. Run the regression suite: `pytest`
3. Exercise the synthetic demo: `python -m hurricane_spy.scripts.run_pipeline --seed 42`
4. (Optional) Dry-run the trading harness against Alpaca paper accounts:
   `python -m hurricane_spy.scripts.run_trading`

## Resolving README conflicts

If merge conflicts arise in this README, keep the project title above intact
and prefer appending release notes or additional links below the existing
sections.

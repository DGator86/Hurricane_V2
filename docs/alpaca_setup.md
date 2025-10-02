# Alpaca Integration Guide

This document explains how to connect the Hurricane SPY pipeline to the Alpaca
brokerage API.

## Prerequisites

1. Create an account at [Alpaca Markets](https://alpaca.markets/).
2. Install the Alpaca SDK:
   ```bash
   pip install alpaca-trade-api
   ```
3. Generate API credentials from the Alpaca dashboard. For safety the secret is
   visible only once when you create the key pair.

## Environment Variables

Export the credentials before running the pipeline:

```bash
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
# Optional: choose between paper (default) or live trading
export ALPACA_MODE=paper
```

It is recommended to store these values in a local `.env` file that is ignored
by Git (see the repository `.gitignore`). Never commit real credentials to the
repository.

## Running the Pipeline

The `run_pipeline` script provides a dry-run mode that does not require
credentials and instead uses an in-memory mock client:

```bash
python -m hurricane_spy.scripts.run_pipeline --dry-run
```

To execute real orders against the Alpaca paper trading endpoint, omit the
`--dry-run` flag after setting the environment variables above:

```bash
python -m hurricane_spy.scripts.run_pipeline --alpaca-mode paper
```

Pass `--alpaca-mode live` only after you are confident in your strategy and
understand the associated risks.

## Tuning Execution Behaviour

You can customise basic execution parameters using CLI flags:

```bash
python -m hurricane_spy.scripts.run_pipeline \
  --base-quantity 25 \
  --max-position 200 \
  --buy-threshold 0.62 \
  --sell-threshold 0.38 \
  --intensity-multiplier 0.4
```

These arguments are translated into an `ExecutionConfig` instance that the
`TradingExecutor` uses to convert aggregated forecasts into orders. Position
sizes are scaled by the hurricane intensity metric which captures the strength
of the current market regime.

## Observability

Execution logs are emitted to stdout. When using the synthetic client the
submitted orders are echoed to the console, providing a safe environment for
iterating on trading logic without touching live markets.

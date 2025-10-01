"""Example script wiring Hurricane SPY predictions to Alpaca execution."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from hurricane_spy import (
    AlpacaClient,
    AlpacaCredentials,
    HurricaneConfig,
    HurricaneSPY,
    HurricaneTrader,
    TimeframeConfig,
    TradingConfig,
)
from hurricane_spy.scripts.run_pipeline import generate_dummy_data


def main() -> None:
    index = pd.date_range(end=datetime.now(UTC), periods=500, freq="min")
    bundle = generate_dummy_data(index)

    config = HurricaneConfig(
        timeframes=[
            TimeframeConfig(name="1m", horizon_minutes=1, abstention_threshold=0.05, lambda_level=0.75),
            TimeframeConfig(name="5m", horizon_minutes=5, abstention_threshold=0.06, lambda_level=1.0),
            TimeframeConfig(name="30m", horizon_minutes=30, abstention_threshold=0.08, lambda_level=1.5),
        ]
    )
    pipeline = HurricaneSPY(config)

    credentials = AlpacaCredentials.from_env()
    alpaca = AlpacaClient(credentials)

    trading = TradingConfig(
        symbol="SPY",
        base_position_size=10,
        max_position=100,
        min_order_size=1,
        confidence_threshold=0.25,
        speed_position_scale=2.0,
        direction_score_threshold=0.05,
    )

    trader = HurricaneTrader(pipeline=pipeline, alpaca=alpaca, trading_config=trading)
    result, decision = trader.execute(bundle)

    aggregate = result["aggregate"]
    print("Aggregate forecasts:")
    for key, value in aggregate.items():
        if key == "weights":
            continue
        print(f"  {key}: {value}")

    print("\nTrade decision:")
    print(f"  action: {decision.action}")
    print(f"  target_position: {decision.target_position:.2f}")
    print(f"  order_quantity: {decision.order_quantity:.2f}")
    print(f"  confidence: {decision.confidence:.2f}")
    print(f"  reason: {decision.reason}")
    if decision.order_response:
        print("  order_response:", decision.order_response)


if __name__ == "__main__":
    main()

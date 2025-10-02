from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hurricane_spy import HurricaneConfig, HurricaneSPY, TimeframeConfig
from hurricane_spy.aggregation import StressWeightedGMV
from hurricane_spy.scripts.run_pipeline import generate_dummy_data


def _default_config() -> HurricaneConfig:
    return HurricaneConfig(
        timeframes=[
            TimeframeConfig(name="1m", horizon_minutes=1, abstention_threshold=0.05, lambda_level=0.75),
            TimeframeConfig(name="5m", horizon_minutes=5, abstention_threshold=0.06, lambda_level=1.0),
            TimeframeConfig(name="30m", horizon_minutes=30, abstention_threshold=0.08, lambda_level=1.5),
        ],
        cooling_window=3,
        gmvtikhonov=1e-2,
    )


def test_pipeline_produces_multi_timeframe_output():
    np.random.seed(0)
    index = pd.date_range("2025-01-01", periods=512, freq="min", tz="UTC")
    bundle = generate_dummy_data(index)
    pipeline = HurricaneSPY(_default_config())

    result = pipeline.run(bundle)

    assert set(result.keys()) == {"timeframes", "aggregate", "diagnostics"}
    assert set(result["timeframes"].keys()) == {"1m", "5m", "30m"}

    for name, payload in result["timeframes"].items():
        assert payload["direction_signal"] in {"up", "down", "abstain"}
        assert payload["speed"] >= 0
        assert "gates" in payload
        # Ensure per-timeframe diagnostics carry the configuration for traceability
        assert payload["config"]["name"] == name

    weights = result["aggregate"]["weights"]
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0

    diagnostics = result["diagnostics"]["gating"]
    assert not diagnostics.empty
    expected_cols = {"event", "regime_flip", "exogenous_flow", "hedging_pressure"}
    assert expected_cols.issubset(set(diagnostics.columns))


@pytest.mark.parametrize("stress_level", [0.0, 2.5, 5.0])
def test_stress_weighted_gmv_shifts_weight_towards_slower_horizons(stress_level: float):
    forecasts = {
        "1m": {"support_resistance": 0.1, "direction_score": 0.05, "speed": 0.3, "probability": 0.55, "hurricane_intensity": 0.2},
        "5m": {"support_resistance": 0.2, "direction_score": 0.04, "speed": 0.2, "probability": 0.52, "hurricane_intensity": 0.25},
        "30m": {"support_resistance": 0.3, "direction_score": 0.03, "speed": 0.1, "probability": 0.5, "hurricane_intensity": 0.3},
    }
    covariance = pd.DataFrame(
        np.diag([0.8, 0.5, 0.3]), index=["1m", "5m", "30m"], columns=["1m", "5m", "30m"]
    )
    aggregator = StressWeightedGMV(tikhonov=1e-3, stress_weight=0.4)

    weights = aggregator(forecasts, covariance, stress_level)["weights"]

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    # Higher stress should always allocate at least as much weight to the slowest horizon
    if stress_level > 0:
        high_stress_weights = aggregator(forecasts, covariance, stress_level + 1.0)["weights"]
        assert high_stress_weights["30m"] >= weights["30m"]
    else:
        assert weights["30m"] >= weights["1m"]

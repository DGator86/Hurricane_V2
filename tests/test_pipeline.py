"""Regression coverage for the Hurricane SPY pipeline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hurricane_spy.aggregation import StressWeightedGMV
from hurricane_spy.data_structures import MarketDataBundle


@pytest.fixture
def pipeline_result(pipeline, dummy_bundle):
    """Run the pipeline once per test invocation for deterministic assertions."""

    return pipeline.run(dummy_bundle)


def test_pipeline_topology_and_keys(pipeline_result):
    """The pipeline should return aggregate, per-timeframe, and diagnostics payloads."""

    assert set(pipeline_result.keys()) == {"timeframes", "aggregate", "diagnostics"}
    assert {"gating", "reliability"}.issubset(pipeline_result["diagnostics"].keys())
    assert not pipeline_result["diagnostics"]["gating"].empty


@pytest.mark.parametrize("timeframe", ["1m", "5m", "30m"])
def test_timeframe_payload_structure(pipeline_result, timeframe):
    """Each timeframe output should carry core metrics and traceability metadata."""

    payload = pipeline_result["timeframes"][timeframe]
    assert payload["direction_signal"] in {"up", "down", "abstain"}
    assert payload["speed"] >= 0
    assert 0.0 <= payload["probability"] <= 1.0
    assert "gates" in payload and isinstance(payload["gates"], dict)
    assert payload["config"]["name"] == timeframe


def test_pipeline_abstains_during_events(pipeline, dummy_bundle):
    """Event abstention should force signals to abstain when the latest bar is flagged."""

    assert dummy_bundle.events is not None
    events = dummy_bundle.events.copy()
    events.iloc[-1] = True
    bundle = replace(dummy_bundle, events=events)

    result = pipeline.run(bundle)
    for payload in result["timeframes"].values():
        assert payload["direction_signal"] == "abstain"
        assert payload["gates"]["event"] is True


@pytest.mark.parametrize("stress_level", [0.0, 2.5, 5.0])
def test_stress_weighted_gmv_allocates_to_slower_horizons(stress_level: float):
    """Weights should normalise and emphasise slower horizons as stress increases."""

    forecasts = {
        "1m": {
            "support_resistance": 0.1,
            "direction_score": 0.05,
            "speed": 0.3,
            "probability": 0.55,
            "hurricane_intensity": 0.2,
        },
        "5m": {
            "support_resistance": 0.2,
            "direction_score": 0.04,
            "speed": 0.2,
            "probability": 0.52,
            "hurricane_intensity": 0.25,
        },
        "30m": {
            "support_resistance": 0.3,
            "direction_score": 0.03,
            "speed": 0.1,
            "probability": 0.5,
            "hurricane_intensity": 0.3,
        },
    }
    covariance = pd.DataFrame(
        np.diag([0.8, 0.5, 0.3]),
        index=["1m", "5m", "30m"],
        columns=["1m", "5m", "30m"],
    )
    aggregator = StressWeightedGMV(tikhonov=1e-3, stress_weight=0.4)

    weights = aggregator(forecasts, covariance, stress_level)["weights"]

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    if stress_level > 0:
        increased = aggregator(forecasts, covariance, stress_level + 1.0)["weights"]
        assert increased["30m"] >= weights["30m"]
    else:
        assert weights["30m"] >= weights["1m"]


def test_gmv_raises_on_empty_covariance():
    """Stress-weighted GMV should guard against empty covariance matrices."""

    aggregator = StressWeightedGMV(tikhonov=1e-3, stress_weight=0.4)
    with pytest.raises(ValueError, match="Covariance matrix must not be empty"):
        aggregator({}, pd.DataFrame(), stress_level=1.0)


def test_bundle_validation_missing_close_column(
    bundle_without_close: MarketDataBundle, hurricane_config
) -> None:
    """Bundles lacking required price columns should fail validation."""

    with pytest.raises(ValueError, match="Missing price columns"):
        bundle_without_close.validate([tf.name for tf in hurricane_config.timeframes])


def test_bundle_validation_negative_realised_vol(
    bundle_with_negative_vol: MarketDataBundle, hurricane_config
) -> None:
    """Negative realised volatility entries must be rejected."""

    with pytest.raises(ValueError, match="Realised volatility contains negatives"):
        bundle_with_negative_vol.validate([tf.name for tf in hurricane_config.timeframes])


def test_end_to_end_pipeline_output_matches_index(pipeline, dummy_bundle):
    """Integration check verifying time-aligned diagnostic output."""

    result = pipeline.run(dummy_bundle)
    gating = result["diagnostics"]["gating"]
    assert len(gating) == len(pipeline.config.timeframes)
    weights = result["aggregate"]["weights"]
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0

"""Diagnostics and logging utilities for Hurricane SPY."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

from .features import brier_score


@dataclass
class ReliabilityTracker:
    """Track calibration and reliability statistics by regime."""

    regimes: List[str] = field(default_factory=lambda: ["calm", "trend", "storm", "pin"])
    records: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.records = {regime: [] for regime in self.regimes}

    def update(self, regime: str, probability: float, outcome: float) -> None:
        if regime not in self.records:
            self.records[regime] = []
        self.records[regime].append(brier_score(probability, outcome))

    def summary(self) -> pd.Series:
        return pd.Series({regime: np.mean(values) if values else np.nan for regime, values in self.records.items()})


@dataclass
class GatingDiagnostics:
    entries: List[Dict[str, float]] = field(default_factory=list)

    def log(self, timestamp, gates: Mapping[str, float]) -> None:
        record = {"timestamp": timestamp}
        record.update(gates)
        self.entries.append(record)

    def to_frame(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["timestamp"])
        return pd.DataFrame(self.entries).set_index("timestamp")

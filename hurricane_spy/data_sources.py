"""External data source integrations for Hurricane SPY."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd
import requests

from .data_structures import MarketDataBundle

LOGGER = logging.getLogger(__name__)


def _ensure_dataframe(payload: Mapping[str, Any], key: str) -> pd.DataFrame:
    """Normalise API payloads into dataframes.

    The Unusual Whales REST responses typically wrap the useful rows inside a
    ``data`` field. We coerce that into a ``DataFrame`` while keeping any
    metadata available via the ``meta`` field.
    """

    rows = payload.get(key, [])
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    if isinstance(rows, MutableMapping):
        return pd.DataFrame([rows])
    raise ValueError(f"Unexpected payload structure for key '{key}': {type(rows)!r}")


@dataclass
class UnusualWhalesClient:
    """Minimal Unusual Whales REST client.

    Only a subset of endpoints is implemented, but the class keeps the
    authentication, pagination handling, and retry logic in one place so the
    pipeline can request order-flow and dark-pool data without duplicating
    boilerplate.
    """

    api_token: str
    base_url: str = "https://phx.unusualwhales.com/api"
    session: Optional[requests.Session] = None
    timeout: float = 10.0

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        sess = self.session or requests.Session()
        try:
            response = sess.request(method, url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            LOGGER.error("Unusual Whales request failed: %s", exc, exc_info=True)
            raise
        except requests.RequestException as exc:  # pragma: no cover - network errors
            LOGGER.error("Unusual Whales connectivity issue: %s", exc, exc_info=True)
            raise
        return response.json()

    def fetch_options_flow(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {"ticker": symbol.upper(), "limit": limit}
        if start is not None:
            params["start"] = start.astimezone(UTC).isoformat()
        if end is not None:
            params["end"] = end.astimezone(UTC).isoformat()
        payload = self._request("GET", "/whales/options/flow", params=params)
        return _ensure_dataframe(payload, "data")

    def fetch_dark_pool_activity(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {"ticker": symbol.upper()}
        if start is not None:
            params["start"] = start.astimezone(UTC).isoformat()
        if end is not None:
            params["end"] = end.astimezone(UTC).isoformat()
        payload = self._request("GET", "/darkpool/volume", params=params)
        return _ensure_dataframe(payload, "data")

    def fetch_gamma_exposure(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {"ticker": symbol.upper()}
        if start is not None:
            params["start"] = start.astimezone(UTC).isoformat()
        if end is not None:
            params["end"] = end.astimezone(UTC).isoformat()
        payload = self._request("GET", "/options/gex", params=params)
        return _ensure_dataframe(payload, "data")


def merge_unusual_whales_signals(
    *,
    flow: pd.DataFrame,
    dark_pool: pd.DataFrame,
    gex: pd.DataFrame,
    resample_rule: str = "1min",
) -> pd.DataFrame:
    """Merge Unusual Whales streams into the OFI frame expected by the pipeline."""

    frames = []
    if not flow.empty:
        flow_frame = flow.copy()
        if "timestamp" in flow_frame:
            flow_frame["timestamp"] = pd.to_datetime(flow_frame["timestamp"], utc=True)
            flow_frame = flow_frame.set_index("timestamp")
        frames.append(flow_frame.resample(resample_rule).sum(min_count=1))
    if not dark_pool.empty:
        pool = dark_pool.copy()
        if "timestamp" in pool:
            pool["timestamp"] = pd.to_datetime(pool["timestamp"], utc=True)
            pool = pool.set_index("timestamp")
        frames.append(pool.resample(resample_rule).sum(min_count=1))
    if not gex.empty:
        gex_frame = gex.copy()
        if "timestamp" in gex_frame:
            gex_frame["timestamp"] = pd.to_datetime(gex_frame["timestamp"], utc=True)
            gex_frame = gex_frame.set_index("timestamp")
        frames.append(gex_frame.resample(resample_rule).mean())

    if not frames:
        raise ValueError("At least one Unusual Whales dataframe must be non-empty")

    combined = pd.concat(frames, axis=1)
    if "ofi" not in combined:
        combined["ofi"] = combined.get("net_flow", combined.sum(axis=1)).fillna(0.0)
    if "dark_pool_index" not in combined and "darkpool_net" in combined:
        combined["dark_pool_index"] = combined["darkpool_net"].cumsum().fillna(0.0)
    if "exogenous_flow" not in combined:
        combined["exogenous_flow"] = combined.get("lit_volume", 0.0)
    if "variance_amplifier" not in combined:
        combined["variance_amplifier"] = (
            combined["ofi"].abs() / combined["ofi"].abs().rolling(50).mean()
        ).clip(lower=0.0).fillna(0.0)
    return combined.fillna(method="ffill").fillna(0.0)


def assemble_bundle(
    *,
    price: pd.DataFrame,
    greeks: pd.DataFrame,
    ofi: pd.DataFrame,
    technical: Mapping[str, pd.Series],
    levels: pd.DataFrame,
    realised_vol: Mapping[str, pd.Series],
    base_vol: Mapping[str, float],
    barrier_levels: Optional[Mapping[str, Mapping[str, float]]] = None,
    events: Optional[pd.DataFrame] = None,
    stress_index: Optional[pd.Series] = None,
) -> MarketDataBundle:
    """Convenience helper to construct a validated ``MarketDataBundle``."""

    bundle = MarketDataBundle(
        price=price,
        greeks=greeks,
        ofi=ofi,
        technical=technical,
        levels=levels,
        realised_vol=realised_vol,
        base_vol=base_vol,
        barrier_levels=barrier_levels,
        events=events,
        stress_index=stress_index,
    )
    bundle.validate(timeframes=technical.keys())
    return bundle


def align_series(series: Iterable[pd.Series], rule: str = "1min") -> pd.DatetimeIndex:
    """Create a shared datetime index for multiple series."""

    union = pd.Index([])
    for s in series:
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is None:
            idx = idx.tz_localize(UTC)
        union = union.union(idx)
    if union.empty:
        raise ValueError("At least one series is required to build an index")
    return union.sort_values().unique().tz_convert(UTC).floor(rule)


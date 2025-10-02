"""Utilities for retrieving market data from the Unusual Whales API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import requests

from hurricane_spy.data_structures import MarketDataBundle

LOGGER = logging.getLogger(__name__)


class UnusualWhalesError(RuntimeError):
    """Raised when the Unusual Whales API returns an unrecoverable error."""


@dataclass
class UnusualWhalesClient:
    """Minimal HTTP client for the Unusual Whales REST API."""

    api_key: str
    base_url: str = "https://api.unusualwhales.com"
    timeout: float = 30.0
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        self._session = self.session or requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )

    def _request(self, method: str, path: str, **kwargs: object) -> Mapping[str, object]:
        url = f"{self.base_url.rstrip('/')}{path}"
        response = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if response.status_code == 401:
            raise UnusualWhalesError("Unusual Whales API rejected the supplied credentials (401).")
        if response.status_code == 403:
            raise UnusualWhalesError("Unusual Whales API access forbidden (403). Check plan permissions.")
        if response.status_code == 404:
            raise UnusualWhalesError("Requested resource not found on Unusual Whales (404).")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - defensive branch
            raise UnusualWhalesError(f"Unusual Whales request failed: {exc}") from exc
        payload: Mapping[str, object] = response.json()
        return payload

    def get_historic_chain_timeseries(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "minute",
        limit: Optional[int] = None,
    ) -> Iterable[Mapping[str, object]]:
        """Fetch a historic options-chain time-series for the underlying symbol."""

        params = {
            "symbol": symbol,
            "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "interval": interval,
        }
        if limit is not None:
            params["limit"] = limit
        payload = self._request("GET", f"/api/historic_chains/{symbol}", params=params)
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if data is None:
            raise UnusualWhalesError("Unexpected response structure from Unusual Whales API.")
        return data  # type: ignore[return-value]


def _select_first_available(frame: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    for candidate in candidates:
        if candidate in frame.columns:
            series = pd.to_numeric(frame[candidate], errors="coerce")
            return series
    return None


def _prepare_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    close = _select_first_available(frame, "close", "last", "underlying_price", "price")
    if close is None:
        raise UnusualWhalesError("Unusual Whales payload did not include a close/price field.")
    close.name = "close"

    open_ = _select_first_available(frame, "open", "open_price")
    if open_ is None:
        open_ = close.shift().fillna(close.iloc[0])
    open_.name = "open"

    high = _select_first_available(frame, "high", "high_price")
    if high is None:
        high = pd.concat([open_, close], axis=1).max(axis=1)
    high.name = "high"

    low = _select_first_available(frame, "low", "low_price")
    if low is None:
        low = pd.concat([open_, close], axis=1).min(axis=1)
    low.name = "low"

    volume = _select_first_available(frame, "volume", "share_volume", "total_volume", "sum_volume")
    if volume is None:
        volume = close.diff().abs().fillna(0.0) * 1_000
    volume = volume.clip(lower=0.0)
    volume.name = "volume"

    price = pd.concat([open_, high, low, close, volume], axis=1)
    price = price.fillna(method="ffill").dropna()
    return price


def _prepare_greek_frame(frame: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    gamma = _select_first_available(frame, "gamma")
    if gamma is None:
        gamma = returns.rolling(5, min_periods=1).mean()
    gamma = gamma.fillna(0.0)

    vanna = _select_first_available(frame, "vanna")
    if vanna is None:
        vanna = returns.rolling(15, min_periods=1).mean()
    vanna = vanna.fillna(0.0)

    charm = _select_first_available(frame, "charm")
    if charm is None:
        charm = returns.rolling(30, min_periods=1).mean()
    charm = charm.fillna(0.0)

    return pd.DataFrame({"gamma": gamma, "vanna": vanna, "charm": charm})


def _prepare_ofi_frame(frame: pd.DataFrame, returns: pd.Series, volume: pd.Series) -> pd.DataFrame:
    ofi = _select_first_available(frame, "ofi", "order_flow_imbalance", "net_premium")
    if ofi is None:
        ofi = (returns * volume).fillna(0.0)
    dark_pool = _select_first_available(frame, "dark_pool_index", "darkpool_index")
    if dark_pool is None:
        dark_pool = ofi.cumsum()
    exogenous = _select_first_available(frame, "exogenous_flow", "whale_score")
    if exogenous is None:
        exogenous = returns.rolling(20, min_periods=1).sum()
    variance_amplifier = _select_first_available(frame, "variance_amplifier", "iv_rank")
    if variance_amplifier is None:
        variance_amplifier = returns.rolling(30, min_periods=1).std().fillna(0.0)
        max_val = variance_amplifier.max()
        if max_val and not np.isnan(max_val):
            variance_amplifier = np.tanh(variance_amplifier / max_val)
    return pd.DataFrame(
        {
            "ofi": ofi.fillna(0.0),
            "dark_pool_index": dark_pool.fillna(method="ffill").fillna(0.0),
            "exogenous_flow": exogenous.fillna(0.0),
            "variance_amplifier": variance_amplifier.fillna(method="ffill").fillna(1.0),
        }
    )


def _compute_alignment(price: pd.DataFrame, window: int) -> pd.Series:
    close = price["close"]
    rolling_mean = close.rolling(window, min_periods=1).mean()
    rolling_std = close.rolling(window, min_periods=1).std().replace(0.0, np.nan)
    alignment = (close - rolling_mean) / rolling_std
    return np.tanh(alignment.fillna(0.0))


def _build_levels(price: pd.DataFrame, ofi: pd.DataFrame) -> pd.DataFrame:
    latest_ts = price.index[-1]
    recent_close = price["close"].iloc[-1]
    recent_vol = price["close"].pct_change().rolling(60, min_periods=1).std().iloc[-1]
    spread = max(recent_vol * recent_close * 4, recent_close * 0.01)
    level_grid = np.linspace(recent_close - spread / 2, recent_close + spread / 2, 5)
    dp_score = ofi["dark_pool_index"].diff().rolling(10, min_periods=1).mean().iloc[-1]
    volume_score = price["volume"].rolling(30, min_periods=1).apply(
        lambda values: (values[-1] - np.mean(values)) / (np.std(values) + 1e-6)
    ).iloc[-1]
    levels = pd.DataFrame(
        {
            "level": level_grid,
            "gamma_score": np.linspace(-1.0, 1.0, len(level_grid)),
            "dark_pool_score": np.tanh(dp_score),
            "volume_score": np.tanh(volume_score),
        },
        index=pd.Index([latest_ts] * len(level_grid)),
    )
    return levels


def _compute_realised_vol(price: pd.DataFrame, window: int) -> pd.Series:
    returns = price["close"].pct_change().fillna(0.0)
    realised = returns.rolling(window, min_periods=1).std().fillna(0.0) * np.sqrt(window)
    return realised


def load_market_data_from_unusual_whales(
    symbol: str,
    start: datetime,
    end: datetime,
    api_key: str,
    interval: str = "minute",
    client: Optional[UnusualWhalesClient] = None,
) -> MarketDataBundle:
    """Load a :class:`MarketDataBundle` using Unusual Whales data."""

    if start >= end:
        raise ValueError("start must be earlier than end when loading Unusual Whales data")

    whales_client = client or UnusualWhalesClient(api_key=api_key)
    raw = whales_client.get_historic_chain_timeseries(symbol=symbol, start=start, end=end, interval=interval)
    frame = pd.DataFrame(raw)
    if frame.empty:
        raise UnusualWhalesError("Unusual Whales returned no data for the requested window.")

    ts_col = next((c for c in ("timestamp", "time", "datetime") if c in frame.columns), None)
    if ts_col is None:
        raise UnusualWhalesError("Unusual Whales payload did not include a timestamp column.")

    frame[ts_col] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=[ts_col])
    frame = frame.set_index(ts_col).sort_index()
    frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
    if frame.empty:
        raise UnusualWhalesError("No Unusual Whales observations within the requested time window.")

    price = _prepare_price_frame(frame)
    returns = price["close"].pct_change().fillna(0.0)
    greeks = _prepare_greek_frame(frame, returns)
    ofi = _prepare_ofi_frame(frame, returns, price["volume"])

    technical = {
        "1m": _compute_alignment(price, 5),
        "5m": _compute_alignment(price, 15),
        "30m": _compute_alignment(price, 60),
    }

    levels = _build_levels(price, ofi)

    realised_vol = {
        "1m": _compute_realised_vol(price, 5),
        "5m": _compute_realised_vol(price, 15),
        "30m": _compute_realised_vol(price, 60),
    }
    base_vol = {name: float(series.rolling(60, min_periods=1).median().iloc[-1]) for name, series in realised_vol.items()}

    last_close = price["close"].iloc[-1]
    vol_scale = price["close"].pct_change().rolling(60, min_periods=1).std().iloc[-1]
    barrier_levels = {
        "1m": {"lower": last_close - 1.5 * vol_scale * last_close, "upper": last_close + 1.5 * vol_scale * last_close},
        "5m": {"lower": last_close - 2.0 * vol_scale * last_close, "upper": last_close + 2.0 * vol_scale * last_close},
        "30m": {"lower": last_close - 2.5 * vol_scale * last_close, "upper": last_close + 2.5 * vol_scale * last_close},
    }

    events = pd.DataFrame({"is_event": False}, index=price.index)

    stress_index = (
        price["close"].pct_change().rolling(120, min_periods=1).std()
        / price["close"].pct_change().rolling(360, min_periods=1).std()
    ).fillna(1.0).clip(lower=0.0)

    bundle = MarketDataBundle(
        price=price,
        greeks=greeks.reindex(price.index, method="ffill"),
        ofi=ofi.reindex(price.index, method="ffill").fillna(0.0),
        technical={k: v.reindex(price.index, method="ffill").fillna(0.0) for k, v in technical.items()},
        levels=levels,
        realised_vol={
            k: v.reindex(price.index, method="ffill").fillna(method="bfill").fillna(0.0)
            for k, v in realised_vol.items()
        },
        base_vol=base_vol,
        barrier_levels=barrier_levels,
        events=events,
        stress_index=stress_index.reindex(price.index, method="ffill").fillna(1.0),
    )
    bundle.validate(technical.keys())
    return bundle

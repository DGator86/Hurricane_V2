"""Trading execution utilities for Hurricane SPY using the Alpaca API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import json
import math
import os
from urllib import error, request


@dataclass
class AlpacaCredentials:
    """Credentials for authenticating with Alpaca."""

    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        """Load credentials from the standard environment variables."""

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set"
            )
        base_url = os.getenv("ALPACA_BASE_URL", cls.base_url)
        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url)


class AlpacaClient:
    """Light-weight REST client for interacting with Alpaca trading endpoints."""

    def __init__(self, credentials: AlpacaCredentials) -> None:
        self.credentials = credentials

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.secret_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.credentials.base_url.rstrip('/')}{path}"
        data = kwargs.pop("json", None)
        headers = self._headers()
        if data is not None:
            body = json.dumps(data).encode("utf-8")
        else:
            body = None
        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=10) as response:  # type: ignore[arg-type]
                text = response.read().decode("utf-8")
                if not text:
                    return {}
                return json.loads(text)
        except error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(path) from exc
            message = exc.read().decode("utf-8") if exc.fp else exc.reason
            raise RuntimeError(f"Alpaca request failed ({exc.code}): {message}") from exc
        except error.URLError as exc:
            raise ConnectionError(f"Failed to reach Alpaca endpoint: {exc.reason}") from exc

    def get_account(self) -> Dict[str, Any]:
        """Return the Alpaca account details."""

        return self._request("GET", "/v2/account")

    def get_position(self, symbol: str) -> float:
        """Return the current position quantity for the given symbol."""

        try:
            payload = self._request("GET", f"/v2/positions/{symbol}")
        except FileNotFoundError:
            return 0.0
        qty = payload.get("qty") or payload.get("quantity") or 0
        try:
            return float(qty)
        except (TypeError, ValueError):
            return 0.0

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Submit an order and return the API response."""

        payload = {
            "symbol": symbol,
            "qty": f"{abs(qty):.4f}",
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        return self._request("POST", "/v2/orders", json=payload)

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close the entire open position for the given symbol."""

        return self._request("DELETE", f"/v2/positions/{symbol}")


@dataclass
class TradingConfig:
    """Risk and execution settings for converting pipeline signals into trades."""

    symbol: str
    base_position_size: float
    max_position: float
    min_order_size: float = 1.0
    confidence_threshold: float = 0.2
    flatten_on_abstain: bool = True
    speed_position_scale: float = 1.0
    direction_score_threshold: float = 0.1
    max_hurricane_intensity: float = 4.5
    position_tolerance: float = 0.1
    time_in_force: str = "day"
    order_type: str = "market"


@dataclass
class TradeDecision:
    """Structured description of the chosen trading action."""

    action: str
    target_position: float
    order_quantity: float
    confidence: float
    reason: str
    side: Optional[str] = None
    order_response: Optional[Mapping[str, Any]] = None


class HurricaneTrader:
    """Bridge Hurricane SPY predictions with Alpaca order execution."""

    def __init__(
        self,
        pipeline: Any,
        alpaca: AlpacaClient,
        trading_config: TradingConfig,
    ) -> None:
        self.pipeline = pipeline
        self.alpaca = alpaca
        self.config = trading_config

    def _weights(self, aggregate: Mapping[str, Any], timeframes: Mapping[str, Mapping[str, Any]]) -> Mapping[str, float]:
        weights = aggregate.get("weights")
        if weights:
            return weights
        fallback_weight = 1.0 / max(len(timeframes), 1)
        return {name: fallback_weight for name in timeframes}

    def _aggregate_probability(
        self,
        aggregate: Mapping[str, Any],
        timeframes: Mapping[str, Mapping[str, Any]],
    ) -> float:
        if "probability" in aggregate:
            return float(aggregate["probability"])
        weights = self._weights(aggregate, timeframes)
        return float(
            sum(weights[name] * timeframes[name].get("probability", 0.5) for name in weights)
        )

    def _aggregate_intensity(
        self,
        aggregate: Mapping[str, Any],
        timeframes: Mapping[str, Mapping[str, Any]],
    ) -> float:
        if "hurricane_intensity" in aggregate:
            return float(aggregate["hurricane_intensity"])
        weights = self._weights(aggregate, timeframes)
        return float(
            sum(weights[name] * timeframes[name].get("hurricane_intensity", 0.0) for name in weights)
        )

    def _derive_signal(
        self,
        aggregate: Mapping[str, Any],
        probability_up: float,
    ) -> Tuple[str, float]:
        direction_score = float(aggregate.get("direction_score", 0.0))
        if direction_score > self.config.direction_score_threshold:
            direction = "up"
        elif direction_score < -self.config.direction_score_threshold:
            direction = "down"
        else:
            return "abstain", 0.0

        if direction == "up":
            directional_confidence = probability_up
        else:
            directional_confidence = 1.0 - probability_up
        confidence = max(0.0, min(1.0, 2.0 * (directional_confidence - 0.5)))
        return direction, confidence

    def _target_position(self, direction: str, confidence: float, speed: float) -> float:
        base_size = self.config.base_position_size
        speed_component = self.config.speed_position_scale * max(speed, 0.0)
        raw_target = (base_size + speed_component) * confidence
        raw_target = min(raw_target, self.config.max_position)
        if direction == "down":
            raw_target *= -1.0
        return raw_target

    def _build_decision(
        self,
        aggregate: Mapping[str, Any],
        timeframes: Mapping[str, Mapping[str, Any]],
        current_position: float,
    ) -> TradeDecision:
        probability_up = self._aggregate_probability(aggregate, timeframes)
        direction, confidence = self._derive_signal(aggregate, probability_up)

        intensity = self._aggregate_intensity(aggregate, timeframes)
        if intensity >= self.config.max_hurricane_intensity:
            return TradeDecision(
                action="hold",
                target_position=current_position,
                order_quantity=0.0,
                confidence=confidence,
                reason=f"intensity {intensity:.2f} exceeds limit",
            )

        if direction == "abstain":
            if self.config.flatten_on_abstain and abs(current_position) > self.config.position_tolerance:
                return TradeDecision(
                    action="flatten",
                    target_position=0.0,
                    order_quantity=abs(current_position),
                    confidence=confidence,
                    reason="abstention triggered flatten",
                    side="sell" if current_position > 0 else "buy",
                )
            return TradeDecision(
                action="hold",
                target_position=current_position,
                order_quantity=0.0,
                confidence=confidence,
                reason="abstain",
            )

        if confidence < self.config.confidence_threshold:
            return TradeDecision(
                action="hold",
                target_position=current_position,
                order_quantity=0.0,
                confidence=confidence,
                reason=f"confidence {confidence:.2f} below threshold",
            )

        speed = float(aggregate.get("speed", 0.0))
        target_position = self._target_position(direction, confidence, speed)
        delta = target_position - current_position
        if abs(delta) < self.config.min_order_size or math.isclose(delta, 0.0, abs_tol=self.config.position_tolerance):
            return TradeDecision(
                action="hold",
                target_position=current_position,
                order_quantity=0.0,
                confidence=confidence,
                reason="change below order size threshold",
            )

        side = "buy" if delta > 0 else "sell"
        return TradeDecision(
            action=side,
            target_position=target_position,
            order_quantity=abs(delta),
            confidence=confidence,
            reason="rebalance toward target",
            side=side,
        )

    def execute(self, data_bundle: Any) -> Tuple[Dict[str, Any], TradeDecision]:
        """Run the pipeline, derive an execution decision, and place orders if needed."""

        result = self.pipeline.run(data_bundle)
        aggregate = result["aggregate"]
        timeframes = result["timeframes"]
        current_position = self.alpaca.get_position(self.config.symbol)
        decision = self._build_decision(aggregate, timeframes, current_position)

        order_response: Optional[Mapping[str, Any]] = None
        if decision.action == "flatten":
            order_response = self.alpaca.close_position(self.config.symbol)
        elif decision.action in ("buy", "sell"):
            order_response = self.alpaca.submit_order(
                symbol=self.config.symbol,
                qty=decision.order_quantity,
                side=decision.side or decision.action,
                order_type=self.config.order_type,
                time_in_force=self.config.time_in_force,
            )

        if order_response is not None:
            decision.order_response = order_response

        return result, decision

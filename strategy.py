
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class Signal:
    side: str
    price: float
    score: float
    notional: float
    reason: str


class ConvergenceStrategy:
    """
    Bot B — Trader Behavior Replica

    Observable behavior reproduced from the trader research:

    Layer A:
        0.01 <= price <= 0.30
        Very frequent / very small entries.

    Layer B:
        0.90 <= price <= 0.995
        Less frequent / substantially larger entries.

    Other observed behavior:
        - rapid repeated entries are allowed
        - no artificial per-market trade-count limit
        - no forced exits or hedges
        - positions ride to resolution
        - final-minute entries are prohibited
        - dollar exposure remains the safety constraint

    IMPORTANT:
        The trader's true private entry trigger is unknown.
        Momentum/convergence is therefore only an observable-market
        proxy. It is NOT claimed to be the trader's exact model.
    """

    def __init__(
        self,
        bankroll=1000,
        max_market_exposure=50,
        max_order=10,

        layer_a_min_price=0.01,
        layer_a_max_price=0.30,

        layer_b_min_price=0.90,
        layer_b_max_price=0.995,

        layer_a_base_notional=0.10,
        layer_a_max_notional=0.50,

        layer_b_base_notional=5.00,
        layer_b_max_notional=10.00,

        start_sec=0,
        stop_sec=240,

        min_score=0.50,

        layer_a_min_score=0.50,
        layer_b_min_score=0.50,

        max_depth_participation=0.25,
        max_asset_exposure=50,
    ):
        self.bankroll = float(bankroll)

        self.max_market_exposure = float(max_market_exposure)
        self.max_order = float(max_order)

        self.layer_a_min_price = float(layer_a_min_price)
        self.layer_a_max_price = float(layer_a_max_price)

        self.layer_b_min_price = float(layer_b_min_price)
        self.layer_b_max_price = float(layer_b_max_price)

        self.layer_a_base_notional = float(layer_a_base_notional)
        self.layer_a_max_notional = float(layer_a_max_notional)

        self.layer_b_base_notional = float(layer_b_base_notional)
        self.layer_b_max_notional = float(layer_b_max_notional)

        self.start_sec = float(start_sec)
        self.stop_sec = float(stop_sec)

        self.min_score = float(min_score)

        self.layer_a_min_score = float(layer_a_min_score)
        self.layer_b_min_score = float(layer_b_min_score)

        self.max_depth_participation = float(max_depth_participation)
        self.max_asset_exposure = float(max_asset_exposure)

    @staticmethod
    def _features(ask, bid, hist, now):
        if ask is None:
            return None

        ask = float(ask)

        if not 0.0 < ask < 1.0:
            return None

        bid = float(bid) if bid is not None else None

        if bid is not None:
            spread = max(0.0, ask - bid)
        else:
            spread = 0.02

        history = []

        for item in hist or []:
            try:
                timestamp = float(item[0])
                price = float(item[1])

                if 0.0 < price < 1.0:
                    history.append((timestamp, price))

            except (TypeError, ValueError, IndexError):
                continue

        momentum = 0.0
        acceleration = 0.0

        if history:

            def nearest(seconds):
                return min(
                    history,
                    key=lambda x: abs((now - x[0]) - seconds)
                )[1]

            p30 = nearest(30.0)
            p10 = nearest(10.0)

            momentum = ask - p30

            acceleration = (
                (ask - p10)
                - (p10 - p30)
            )

        return spread, momentum, acceleration

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, value))

    def _layer(self, price):
        """
        Determine whether a price belongs to one of the two
        primary observed trader regimes.
        """

        price = float(price)

        if (
            self.layer_a_min_price
            <= price
            <= self.layer_a_max_price
        ):
            return "A"

        if (
            self.layer_b_min_price
            <= price
            <= self.layer_b_max_price
        ):
            return "B"

        return None

    def _score(
        self,
        ask,
        depth,
        spread,
        momentum,
        acceleration,
        elapsed,
    ):
        """
        Ranking score.

        This is deliberately NOT called probability.

        The historical trader dataset does not reveal the trader's
        internal probability estimate, so we must not pretend that
        this score is calibrated probability.
        """

        momentum_score = self._clamp(
            0.5 + momentum * 8.0
        )

        acceleration_score = self._clamp(
            0.5 + acceleration * 10.0
        )

        # Price is confirmation only.
        # Distance from 50c measures extremeness, not direction.
        price_score = self._clamp(
            abs(ask - 0.50) / 0.50
        )

        span = max(
            1.0,
            self.stop_sec - self.start_sec
        )

        time_score = self._clamp(
            (elapsed - self.start_sec) / span
        )

        spread_score = self._clamp(
            1.0
            - max(0.0, spread - 0.01) / 0.05
        )

        depth_reference = max(
            1.0,
            self.max_order / max(ask, 0.01)
        )

        depth_score = self._clamp(
            float(depth) / depth_reference
        )

        score = (
            0.30 * momentum_score
            + 0.18 * acceleration_score
            + 0.18 * price_score
            + 0.14 * time_score
            + 0.12 * depth_score
            + 0.08 * spread_score
        )

        return self._clamp(score)

    def _size_for_layer(self, layer, score):
        """
        Layer-specific sizing.

        Layer A:
            deliberately tiny, high-frequency positions.

        Layer B:
            substantially larger positions.

        Score changes size inside each regime but does not change
        the regime itself.
        """

        score = self._clamp(score)

        if layer == "A":

            size = (
                self.layer_a_base_notional
                + score
                * (
                    self.layer_a_max_notional
                    - self.layer_a_base_notional
                )
            )

            return min(
                self.layer_a_max_notional,
                size,
            )

        if layer == "B":

            size = (
                self.layer_b_base_notional
                + score
                * (
                    self.layer_b_max_notional
                    - self.layer_b_base_notional
                )
            )

            return min(
                self.layer_b_max_notional,
                size,
            )

        return 0.0

    def decide(
        self,
        elapsed,
        up_ask,
        down_ask,
        up_bid,
        down_bid,
        up_history,
        down_history,
        current_exposure,
        available_cash,
        up_depth=0.0,
        down_depth=0.0,
        now=None,
        asset_exposure=0.0,
    ) -> Optional[Signal]:

        if now is None:
            now = time.time()

        elapsed = float(elapsed)

        # Start immediately.
        if elapsed < self.start_sec:
            return None

        # HARD STOP.
        #
        # Five-minute markets end at 300s.
        # stop_sec=240 therefore blocks the final 60 seconds.
        if elapsed >= self.stop_sec:
            return None

        current_exposure = float(current_exposure)
        asset_exposure = float(asset_exposure)
        available_cash = float(available_cash)

        remaining_budget = min(
            self.max_market_exposure
            - current_exposure,

            self.max_asset_exposure
            - asset_exposure,
        )

        if remaining_budget <= 0.0:
            return None

        if available_cash <= 0.0:
            return None

        candidates = []

        observations = (
            (
                "Up",
                up_ask,
                up_bid,
                up_depth,
                up_history,
            ),
            (
                "Down",
                down_ask,
                down_bid,
                down_depth,
                down_history,
            ),
        )

        for (
            side,
            ask,
            bid,
            depth,
            history,
        ) in observations:

            features = self._features(
                ask=ask,
                bid=bid,
                hist=history,
                now=now,
            )

            if features is None:
                continue

            (
                spread,
                momentum,
                acceleration,
            ) = features

            ask = float(ask)
            depth = max(0.0, float(depth))

            layer = self._layer(ask)

            # Ignore the middle region for the primary replica.
            if layer is None:
                continue

            score = self._score(
                ask=ask,
                depth=depth,
                spread=spread,
                momentum=momentum,
                acceleration=acceleration,
                elapsed=elapsed,
            )

            if layer == "A":
                minimum_score = self.layer_a_min_score
            else:
                minimum_score = self.layer_b_min_score

            if score < minimum_score:
                continue

            candidates.append(
                {
                    "side": side,
                    "ask": ask,
                    "bid": bid,
                    "depth": depth,
                    "layer": layer,
                    "score": score,
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "spread": spread,
                }
            )

        if not candidates:
            return None

        # Pick the strongest observable signal.
        best = max(
            candidates,
            key=lambda item: item["score"],
        )

        ask = best["ask"]
        depth = best["depth"]
        layer = best["layer"]
        score = best["score"]

        target_notional = self._size_for_layer(
            layer,
            score,
        )

        # Absolute safety ceiling.
        target_notional = min(
            target_notional,
            self.max_order,
        )

        # Market/asset exposure.
        target_notional = min(
            target_notional,
            remaining_budget,
        )

        # Cash.
        target_notional = min(
            target_notional,
            available_cash,
        )

        # Visible depth protection.
        depth_cap = (
            depth
            * ask
            * self.max_depth_participation
        )

        target_notional = min(
            target_notional,
            depth_cap,
        )

        if target_notional <= 0.0:
            return None

        reason = (
            f"layer={layer} "
            f"score={score:.3f} "
            f"momentum={best['momentum']:+.4f} "
            f"accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} "
            f"depth={depth:.2f} "
            f"elapsed={elapsed:.1f}s "
            f"replica=trader-behavior"
        )

        return Signal(
            side=best["side"],
            price=ask,
            score=score,
            notional=round(
                target_notional,
                2,
            ),
            reason=reason,
        )


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
    BOT B V3.2

    Independent paper-trading strategy based on observable behavior
    found in the trader dataset.

    Price regimes:

        CHEAP : 0.01 <= price < 0.30
        MID   : 0.30 <= price < 0.70
        CORE  : 0.70 <= price < 0.90
        HIGH  : 0.90 <= price < 0.995

    The trader's private entry trigger is not observable in the dataset.
    Therefore this is an inferred strategy, not a guaranteed reconstruction.

    Hard rule:

        No new entries during the final 60 seconds.
    """

    VERSION = "V3.2"

    CHEAP_MIN = 0.01
    CHEAP_MAX = 0.30

    MID_MIN = 0.30
    MID_MAX = 0.70

    CORE_MIN = 0.70
    CORE_MAX = 0.90

    HIGH_MIN = 0.90
    HIGH_MAX = 0.995

    def __init__(
        self,
        bankroll=1000,
        max_market_exposure=25,
        max_order=10,
        layer_a_min_price=0.01,
        layer_a_max_price=0.30,
        layer_b_min_price=0.90,
        layer_b_max_price=0.995,
        layer_a_base_notional=0.15,
        layer_a_max_notional=1.00,
        layer_b_base_notional=2.00,
        layer_b_max_notional=3.00,
        start_sec=0,
        stop_sec=240,
        min_score=0.50,
        layer_a_min_score=0.45,
        layer_b_min_score=0.82,
        max_depth_participation=0.25,
        max_asset_exposure=35,
        max_total_exposure=60,
    ):
        self.bankroll = float(bankroll)

        self.max_market_exposure = min(
            float(max_market_exposure),
            25.0,
        )

        self.max_asset_exposure = min(
            float(max_asset_exposure),
            35.0,
        )

        self.max_order = min(
            float(max_order),
            10.0,
        )

        # V3.2 pacing: keep a meaningful cash reserve for subsequent
        # 5-minute markets instead of consuming the full bankroll early.
        self.max_total_exposure = min(
            float(max_total_exposure),
            100.0,
        )

        self.layer_a_min_price = float(
            layer_a_min_price
        )

        self.layer_a_max_price = float(
            layer_a_max_price
        )

        self.layer_b_min_price = float(
            layer_b_min_price
        )

        self.layer_b_max_price = float(
            layer_b_max_price
        )

        self.layer_a_base_notional = max(
            0.10,
            float(layer_a_base_notional),
        )

        self.layer_a_max_notional = min(
            self.max_order,
            float(layer_a_max_notional),
        )

        self.layer_b_base_notional = min(
            self.max_order,
            float(layer_b_base_notional),
        )

        self.layer_b_max_notional = min(
            self.max_order,
            float(layer_b_max_notional),
        )

        self.start_sec = float(
            start_sec
        )

        self.stop_sec = float(
            stop_sec
        )

        self.min_score = float(
            min_score
        )

        self.layer_a_min_score = float(
            layer_a_min_score
        )

        self.layer_b_min_score = max(
            0.82,
            float(layer_b_min_score),
        )

        self.max_depth_participation = min(
            0.25,
            max(
                0.01,
                float(max_depth_participation),
            ),
        )

    @staticmethod
    def _clamp(
        value,
        low=0.0,
        high=1.0,
    ):
        return max(
            low,
            min(
                high,
                float(value),
            ),
        )

    def _regime(self, price):
        price = float(price)

        if (
            self.layer_a_min_price
            <= price
            <
            self.layer_a_max_price
        ):
            return "CHEAP"

        if (
            self.layer_a_max_price
            <= price
            <
            self.layer_b_min_price
        ):
            if price < 0.70:
                return "MID"

            return "CORE"

        if (
            self.layer_b_min_price
            <= price
            <
            self.layer_b_max_price
        ):
            return "HIGH"

        return None

    @staticmethod
    def _features(
        ask,
        bid,
        history,
        now,
    ):
        if ask is None:
            return None

        try:
            ask = float(ask)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not (
            0.0
            < ask
            < 1.0
        ):
            return None

        bid_value = None

        if bid is not None:
            try:
                bid_value = float(bid)
            except (
                TypeError,
                ValueError,
            ):
                bid_value = None

        if bid_value is not None:
            spread = max(
                0.0,
                ask - bid_value,
            )
        else:
            spread = 0.02

        points = []

        for item in history or []:
            try:
                timestamp = float(
                    item[0]
                )

                price = float(
                    item[1]
                )

                if (
                    0.0
                    < price
                    < 1.0
                ):
                    points.append(
                        (
                            timestamp,
                            price,
                        )
                    )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

        points.sort()

        if not points:
            return (
                spread,
                0.0,
                0.0,
            )

        def nearest(seconds_ago):
            return min(
                points,
                key=lambda x:
                    abs(
                        (
                            now
                            - x[0]
                        )
                        -
                        seconds_ago
                    ),
            )[1]

        p30 = nearest(
            30.0
        )

        p10 = nearest(
            10.0
        )

        momentum = (
            ask
            - p30
        )

        acceleration = (
            ask
            - p10
        ) - (
            p10
            - p30
        )

        return (
            spread,
            momentum,
            acceleration,
        )

    def _score(
        self,
        ask,
        depth,
        spread,
        momentum,
        acceleration,
        elapsed,
    ):
        momentum_score = self._clamp(
            0.5
            +
            momentum
            * 8.0
        )

        acceleration_score = self._clamp(
            0.5
            +
            acceleration
            * 10.0
        )

        price_extremeness = self._clamp(
            abs(
                ask
                - 0.50
            )
            /
            0.50
        )

        time_score = self._clamp(
            (
                elapsed
                -
                self.start_sec
            )
            /
            max(
                1.0,
                self.stop_sec
                -
                self.start_sec,
            )
        )

        spread_score = self._clamp(
            1.0
            -
            max(
                0.0,
                spread
                -
                0.01,
            )
            /
            0.05
        )

        depth_reference = max(
            1.0,
            self.max_order
            /
            max(
                ask,
                0.01,
            ),
        )

        depth_score = self._clamp(
            float(depth)
            /
            depth_reference
        )

        score = (
            0.30
            * momentum_score
            +
            0.18
            * acceleration_score
            +
            0.18
            * price_extremeness
            +
            0.14
            * time_score
            +
            0.12
            * depth_score
            +
            0.08
            * spread_score
        )

        return self._clamp(
            score
        )

    def _allowed(
        self,
        regime,
        score,
        momentum,
        acceleration,
        spread,
        depth,
        elapsed,
    ):
        if depth <= 0:
            return False

        if spread > 0.05:
            return False

        # Global score floor.
        if score < self.min_score:
            return False

        if regime == "CHEAP":
            return (
                score
                >=
                self.layer_a_min_score
                and
                momentum
                >=
                -0.0025
                and
                acceleration
                >=
                -0.0040
            )

        if regime == "MID":
            return (
                score
                >=
                self.min_score
                and
                momentum
                >=
                -0.0015
                and
                acceleration
                >=
                -0.0020
            )

        if regime == "CORE":
            return (
                score
                >=
                self.min_score
                and
                momentum
                >=
                -0.0020
                and
                acceleration
                >=
                -0.0025
            )

        if regime == "HIGH":
            return (
                score
                >=
                self.layer_b_min_score
                and
                momentum
                >=
                0.0020
                and
                acceleration
                >=
                0.0
                and
                spread
                <=
                0.035
                and
                elapsed
                <=
                self.stop_sec
            )

        return False

    def _size(
        self,
        regime,
        price,
        score,
    ):
        """V3.2 empirical continuous sizing with tighter favorite sizing."""
        price = self._clamp(price)
        score = self._clamp(score)

        if regime == "CHEAP":
            progress = self._clamp(
                (price - self.layer_a_min_price)
                / max(0.0001, self.layer_a_max_price - self.layer_a_min_price)
            )
            size = (
                self.layer_a_base_notional
                + (self.layer_a_max_notional - self.layer_a_base_notional)
                * progress
                * (0.65 + 0.35 * score)
            )
            return min(
                self.layer_a_max_notional,
                self.max_order,
                max(self.layer_a_base_notional, size),
            )

        if regime == "MID":
            return min(
                self.max_order,
                1.75,
                max(0.35, 0.35 + 1.40 * score),
            )

        if regime == "CORE":
            return min(
                self.max_order,
                2.50,
                max(0.60, 0.60 + 1.90 * score),
            )

        if regime == "HIGH":
            strength = self._clamp(
                (score - self.layer_b_min_score)
                / max(0.0001, 1.0 - self.layer_b_min_score)
            )
            size = 0.60 + 1.40 * strength
            return min(
                self.max_order,
                2.00,
                max(0.60, size),
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
        total_exposure=0.0,
    ):
        if now is None:
            now = time.time()

        now = float(
            now
        )

        elapsed = float(
            elapsed
        )

        # ========================================================
        # HARD TRADING WINDOW
        # ========================================================

        if elapsed < self.start_sec:
            return None

        if elapsed >= self.stop_sec:
            return None

        # ========================================================
        # EXPOSURE
        # ========================================================

        current_exposure = max(
            0.0,
            float(
                current_exposure
            ),
        )

        asset_exposure = max(
            0.0,
            float(
                asset_exposure
            ),
        )

        total_exposure = max(
            0.0,
            float(
                total_exposure
            ),
        )

        available_cash = max(
            0.0,
            float(
                available_cash
            ),
        )

        remaining = min(
            self.max_market_exposure - current_exposure,
            self.max_asset_exposure - asset_exposure,
            self.max_total_exposure - total_exposure,
            available_cash,
        )

        if remaining < 0.10:
            return None

        # ========================================================
        # CANDIDATES
        # ========================================================

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

        candidates = []

        for (
            side,
            ask,
            bid,
            depth,
            history,
        ) in observations:

            features = self._features(
                ask,
                bid,
                history,
                now,
            )

            if features is None:
                continue

            (
                spread,
                momentum,
                acceleration,
            ) = features

            ask = float(
                ask
            )

            depth = max(
                0.0,
                float(
                    depth
                ),
            )

            regime = self._regime(
                ask
            )

            if regime is None:
                continue

            score = self._score(
                ask,
                depth,
                spread,
                momentum,
                acceleration,
                elapsed,
            )

            if not self._allowed(
                regime,
                score,
                momentum,
                acceleration,
                spread,
                depth,
                elapsed,
            ):
                continue

            ranking = score

            # The observed trader behavior is heavily concentrated
            # in the cheap region. Give cheap opportunities a modest
            # ranking preference rather than forcing them.

            if regime == "CHEAP":
                ranking += 0.04

            elif regime == "MID":
                ranking += 0.02

            elif regime == "HIGH":
                ranking -= 0.10

            candidates.append(
                {
                    "side": side,
                    "ask": ask,
                    "depth": depth,
                    "regime": regime,
                    "score": score,
                    "ranking": ranking,
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "spread": spread,
                }
            )

        if not candidates:
            return None

        # ========================================================
        # BEST CANDIDATE
        # ========================================================

        best = max(
            candidates,
            key=lambda item:
                item["ranking"],
        )

        # ========================================================
        # POSITION SIZE
        # ========================================================

        size = self._size(
            best["regime"],
            best["ask"],
            best["score"],
        )

        size = min(
            size,
            self.max_order,
            remaining,
        )

        # ========================================================
        # DEPTH PARTICIPATION CAP
        # ========================================================

        depth_cap = (
            best["depth"]
            *
            best["ask"]
            *
            self.max_depth_participation
        )

        size = min(
            size,
            depth_cap,
        )

        if size < 0.10:
            return None

        # ========================================================
        # REASON
        # ========================================================

        seconds_left = max(
            0.0,
            300.0
            -
            elapsed,
        )

        reason = (
            f"V3.2 "
            f"regime={best['regime']} "
            f"score={best['score']:.3f} "
            f"momentum={best['momentum']:+.4f} "
            f"accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} "
            f"depth={best['depth']:.2f} "
            f"elapsed={elapsed:.1f}s "
            f"seconds_left={seconds_left:.1f} "
            f"independent=true"
        )

        return Signal(
            side=best["side"],
            price=best["ask"],
            score=best["score"],
            notional=round(
                size,
                2,
            ),
            reason=reason,
        )

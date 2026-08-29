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
    BOT B V2

    Independent paper-trading strategy based on the observable behavioral
    characteristics found in the trader dataset.

    IMPORTANT:
    This is NOT a copy-trading strategy. It does not read or follow the
    trader's individual orders.

    V2 price regimes:

        CHEAP : 0.01 <= price < 0.30
        MID   : 0.30 <= price < 0.70
        CORE  : 0.70 <= price < 0.90
        HIGH  : 0.90 <= price < 0.995

    The trader's actual private entry trigger is unknown, so the signal
    calculation is an independently inferred market-microstructure
    hypothesis.

    Hard rule:

        No new entries during the final 60 seconds.
    """

    # ============================================================
    # V2 PRICE REGIMES
    # ============================================================

    CHEAP_MIN = 0.01
    CHEAP_MAX = 0.30

    MID_MIN = 0.30
    MID_MAX = 0.70

    CORE_MIN = 0.70
    CORE_MAX = 0.90

    HIGH_MIN = 0.90
    HIGH_MAX = 0.995

    # ============================================================
    # INITIALIZATION
    # ============================================================

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
    ):
        self.bankroll = float(bankroll)

        # --------------------------------------------------------
        # HARD V2 RISK CAPS
        # --------------------------------------------------------

        # Even if the environment contains the old V1 values of 50,
        # V2 will never allow these limits to exceed the V2 caps.

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

        # --------------------------------------------------------
        # V2 REGIME BOUNDARIES
        # --------------------------------------------------------

        # CHEAP regime
        self.layer_a_min_price = self.CHEAP_MIN
        self.layer_a_max_price = self.CHEAP_MAX

        # HIGH regime
        self.layer_b_min_price = self.HIGH_MIN
        self.layer_b_max_price = self.HIGH_MAX

        # --------------------------------------------------------
        # CHEAP SIZING
        # --------------------------------------------------------

        self.layer_a_base_notional = max(
            0.10,
            float(layer_a_base_notional),
        )

        self.layer_a_max_notional = min(
            1.00,
            float(layer_a_max_notional),
        )

        # --------------------------------------------------------
        # HIGH SIZING
        # --------------------------------------------------------

        self.layer_b_base_notional = min(
            2.00,
            float(layer_b_base_notional),
        )

        self.layer_b_max_notional = min(
            3.00,
            float(layer_b_max_notional),
        )

        # --------------------------------------------------------
        # TIMING
        # --------------------------------------------------------

        self.start_sec = float(
            start_sec
        )

        self.stop_sec = float(
            stop_sec
        )

        # --------------------------------------------------------
        # SIGNAL THRESHOLDS
        # --------------------------------------------------------

        self.min_score = float(
            min_score
        )

        self.layer_a_min_score = float(
            layer_a_min_score
        )

        # HIGH regime is deliberately much stricter than V1.
        #
        # V1 repeatedly bought 90-100c favorites and lost money.
        # Therefore a HIGH trade requires >= 0.82 score regardless
        # of a lower environment setting.

        self.layer_b_min_score = max(
            0.82,
            float(layer_b_min_score),
        )

        # --------------------------------------------------------
        # ORDER BOOK LIMIT
        # --------------------------------------------------------

        self.max_depth_participation = min(
            0.25,
            max(
                0.01,
                float(max_depth_participation),
            ),
        )

    # ============================================================
    # BASIC HELPERS
    # ============================================================

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

    # ============================================================
    # PRICE REGIME
    # ============================================================

    @staticmethod
    def _regime(price):
        """
        Determine which V2 price regime a contract belongs to.
        """

        price = float(price)

        if (
            0.01
            <= price
            < 0.30
        ):
            return "CHEAP"

        if (
            0.30
            <= price
            < 0.70
        ):
            return "MID"

        if (
            0.70
            <= price
            < 0.90
        ):
            return "CORE"

        if (
            0.90
            <= price
            < 0.995
        ):
            return "HIGH"

        return None

    # ============================================================
    # MARKET FEATURES
    # ============================================================

    @staticmethod
    def _features(
        ask,
        bid,
        history,
        now,
    ):
        """
        Extract observable short-term market features.

        Returns:

            spread
            momentum
            acceleration
        """

        if ask is None:
            return None

        ask = float(
            ask
        )

        if not (
            0.0
            < ask
            < 1.0
        ):
            return None

        if bid is not None:
            bid = float(
                bid
            )

        spread = (
            max(
                0.0,
                ask - bid,
            )
            if bid is not None
            else 0.02
        )

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

        # No historical observations yet.
        if not points:
            return (
                spread,
                0.0,
                0.0,
            )

        def nearest(
            seconds_ago
        ):
            return min(
                points,
                key=lambda x:
                    abs(
                        (
                            now
                            - x[0]
                        )
                        - seconds_ago
                    ),
            )[1]

        # --------------------------------------------------------
        # 30 SECOND MOMENTUM
        # --------------------------------------------------------

        p30 = nearest(
            30.0
        )

        # --------------------------------------------------------
        # 10 SECOND PRICE
        # --------------------------------------------------------

        p10 = nearest(
            10.0
        )

        # Positive = current ask is higher than historical ask.
        momentum = (
            ask
            - p30
        )

        # Measures whether recent movement is accelerating.
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

    # ============================================================
    # V2 SCORE
    # ============================================================

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
        Calculate the V2 observable-market ranking score.

        This is NOT a probability of winning.
        """

        # --------------------------------------------------------
        # MOMENTUM
        # --------------------------------------------------------

        momentum_score = self._clamp(
            0.5
            +
            momentum
            * 8.0
        )

        # --------------------------------------------------------
        # ACCELERATION
        # --------------------------------------------------------

        acceleration_score = self._clamp(
            0.5
            +
            acceleration
            * 10.0
        )

        # --------------------------------------------------------
        # PRICE EXTREMENESS
        # --------------------------------------------------------

        price_extremeness = self._clamp(
            abs(
                ask
                - 0.50
            )
            / 0.50
        )

        # --------------------------------------------------------
        # TIME
        # --------------------------------------------------------

        time_score = self._clamp(
            (
                elapsed
                - self.start_sec
            )
            /
            max(
                1.0,
                self.stop_sec
                - self.start_sec,
            )
        )

        # --------------------------------------------------------
        # SPREAD
        # --------------------------------------------------------

        spread_score = self._clamp(
            1.0
            -
            max(
                0.0,
                spread
                - 0.01,
            )
            / 0.05
        )

        # --------------------------------------------------------
        # DEPTH
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # FINAL SCORE
        # --------------------------------------------------------

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

    # ============================================================
    # REGIME ENTRY RULES
    # ============================================================

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
        """
        Determine whether a candidate is eligible for entry.
        """

        # Basic market-quality filter.
        if depth <= 0:
            return False

        if spread > 0.05:
            return False

        # --------------------------------------------------------
        # CHEAP
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # MID
        # --------------------------------------------------------

        if regime == "MID":

            return (
                score
                >=
                0.58

                and

                momentum
                >=
                -0.0015

                and

                acceleration
                >=
                -0.0020
            )

        # --------------------------------------------------------
        # CORE
        # --------------------------------------------------------

        if regime == "CORE":

            return (
                score
                >=
                0.58

                and

                momentum
                >=
                -0.0020

                and

                acceleration
                >=
                -0.0025
            )

        # --------------------------------------------------------
        # HIGH
        # --------------------------------------------------------

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
                >=
                90.0
            )

        return False

    # ============================================================
    # POSITION SIZING
    # ============================================================

    def _size(
        self,
        regime,
        price,
        score,
    ):
        """
        V2 sizing function.

        Cheap contracts receive small positions.

        Position size increases through the middle/core regimes.

        HIGH-price contracts are deliberately capped because the
        previous V1 90-100c behavior generated disproportionate losses.
        """

        price = self._clamp(
            price
        )

        score = self._clamp(
            score
        )

        # --------------------------------------------------------
        # CHEAP
        # --------------------------------------------------------

        if regime == "CHEAP":

            # Cheap prices get small notional.
            #
            # Typical range:
            # approximately $0.15-$0.50.
            #
            # Stronger conviction can increase the size modestly.

            base = (
                0.15
                +
                0.35
                *
                self._clamp(
                    price
                    /
                    0.30
                )
            )

            size = (
                base
                *
                (
                    0.65
                    +
                    0.35
                    * score
                )
            )

            return min(
                1.00,
                max(
                    0.15,
                    size,
                ),
            )

        # --------------------------------------------------------
        # MID
        # --------------------------------------------------------

        if regime == "MID":

            return min(
                2.50,
                max(
                    0.50,
                    0.50
                    +
                    2.00
                    * score,
                ),
            )

        # --------------------------------------------------------
        # CORE
        # --------------------------------------------------------

        if regime == "CORE":

            return min(
                4.00,
                max(
                    1.00,
                    1.00
                    +
                    3.00
                    * score,
                ),
            )

        # --------------------------------------------------------
        # HIGH
        # --------------------------------------------------------

        if regime == "HIGH":

            # Never restore V1's flat $5-$10 behavior.
            return min(
                2.50,
                max(
                    0.50,
                    0.50
                    +
                    2.00
                    *
                    max(
                        0.0,
                        score
                        -
                        0.82,
                    )
                    /
                    0.18,
                ),
            )

        return 0.0

    # ============================================================
    # MAIN DECISION ENGINE
    # ============================================================

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
    ):
        """
        Produce a paper-trade Signal or None.
        """

        if now is None:
            now = time.time()

        now = float(
            now
        )

        elapsed = float(
            elapsed
        )

        # ========================================================
        # HARD TIMING GATE
        # ========================================================

        # STOP_TRADING_SECOND=240 on a 300-second market means
        # the final 60 seconds are completely blocked.

        if (
            elapsed
            <
            self.start_sec
        ):
            return None

        if (
            elapsed
            >=
            self.stop_sec
        ):
            return None

        # ========================================================
        # RISK INPUTS
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

        available_cash = max(
            0.0,
            float(
                available_cash
            ),
        )

        # ========================================================
        # REMAINING RISK
        # ========================================================

        remaining = min(
            self.max_market_exposure
            -
            current_exposure,

            self.max_asset_exposure
            -
            asset_exposure,

            available_cash,
        )

        if remaining < 0.10:
            return None

        # ========================================================
        # CANDIDATES
        # ========================================================

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

            # ====================================================
            # REGIME
            # ====================================================

            regime = self._regime(
                ask
            )

            if regime is None:
                continue

            # ====================================================
            # SCORE
            # ====================================================

            score = self._score(
                ask,
                depth,
                spread,
                momentum,
                acceleration,
                elapsed,
            )

            # ====================================================
            # ENTRY FILTER
            # ====================================================

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

            # ====================================================
            # RANKING
            # ====================================================

            ranking = score

            # Cheap trades are intentionally given a modest
            # ranking preference so HIGH-price trades do not
            # automatically dominate the candidate list.

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

        # ========================================================
        # NO VALID CANDIDATE
        # ========================================================

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
        # SIZE
        # ========================================================

        size = self._size(
            best["regime"],
            best["ask"],
            best["score"],
        )

        # Global caps.
        size = min(
            size,
            self.max_order,
            remaining,
        )

        # ========================================================
        # ORDER BOOK DEPTH CAP
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

        # ========================================================
        # MINIMUM SIGNAL SIZE
        # ========================================================

        if size < 0.10:
            return None

        # ========================================================
        # RESEARCH REASON
        # ========================================================

        seconds_left = max(
            0.0,
            300.0
            -
            elapsed,
        )

        reason = (
            f"V2 "
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

        # ========================================================
        # SIGNAL
        # ========================================================

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

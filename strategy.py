from dataclasses import dataclass
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
    BOT B V2.2

    Paper-only behavioral proxy for the trader dataset.

    V2.2 keeps the original cheap/high structure, but makes risk and
    entry gating explicit:

      - CHEAP: 0.01 <= price < 0.30
      - MID:   0.30 <= price < 0.70
      - CORE:  0.70 <= price < 0.90
      - HIGH:  0.90 <= price < 0.995

    Safety caps are hard-capped in code:
      - $10 maximum order
      - $25 maximum exposure per market
      - $35 maximum exposure per underlying asset
      - global portfolio exposure is supplied as total_exposure
      - no new entries during the final 60 seconds

    This is an inferred strategy, not a guaranteed reconstruction of any
    private trader's trigger.
    """

    VERSION = "V2.2"

    CHEAP_MIN = 0.01
    CHEAP_MAX = 0.30
    MID_MIN = 0.30
    MID_MAX = 0.70
    CORE_MIN = 0.70
    CORE_MAX = 0.90
    HIGH_MIN = 0.90
    HIGH_MAX = 0.995

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 25.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0

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
        layer_a_min_score=0.50,
        layer_b_min_score=0.82,
        max_depth_participation=0.25,
        max_asset_exposure=35,
        max_total_exposure=25,
        hard_cutoff_seconds=60,
    ):
        self.bankroll = float(bankroll)

        self.max_market_exposure = min(
            max(0.0, float(max_market_exposure)),
            self.HARD_MAX_MARKET_EXPOSURE,
        )
        self.max_asset_exposure = min(
            max(0.0, float(max_asset_exposure)),
            self.HARD_MAX_ASSET_EXPOSURE,
        )
        self.max_total_exposure = min(
            max(0.0, float(max_total_exposure)),
            self.HARD_MAX_MARKET_EXPOSURE,
        )
        self.max_order = min(
            max(0.0, float(max_order)),
            self.HARD_MAX_ORDER,
        )

        self.layer_a_min_price = float(layer_a_min_price)
        self.layer_a_max_price = float(layer_a_max_price)
        self.layer_b_min_price = float(layer_b_min_price)
        self.layer_b_max_price = float(layer_b_max_price)

        self.layer_a_base_notional = max(0.10, float(layer_a_base_notional))
        self.layer_a_max_notional = min(
            self.max_order, float(layer_a_max_notional)
        )
        self.layer_b_base_notional = min(
            self.max_order, float(layer_b_base_notional)
        )
        self.layer_b_max_notional = min(
            self.max_order, float(layer_b_max_notional)
        )

        self.start_sec = float(start_sec)
        self.stop_sec = float(stop_sec)
        self.min_score = float(min_score)
        self.layer_a_min_score = max(0.50, float(layer_a_min_score))
        self.layer_b_min_score = max(0.82, float(layer_b_min_score))

        self.max_depth_participation = min(
            0.25, max(0.01, float(max_depth_participation))
        )
        self.hard_cutoff_seconds = max(
            self.HARD_CUTOFF_SECONDS,
            float(hard_cutoff_seconds),
        )

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    def _regime(self, price):
        price = float(price)
        if self.layer_a_min_price <= price < self.layer_a_max_price:
            return "CHEAP"
        if self.layer_a_max_price <= price < self.layer_b_min_price:
            return "MID" if price < 0.70 else "CORE"
        if self.layer_b_min_price <= price < self.layer_b_max_price:
            return "HIGH"
        return None

    @staticmethod
    def _features(ask, bid, history, now):
        if ask is None:
            return None
        try:
            ask = float(ask)
        except (TypeError, ValueError):
            return None
        if not 0.0 < ask < 1.0:
            return None

        try:
            bid_value = None if bid is None else float(bid)
        except (TypeError, ValueError):
            bid_value = None

        spread = (
            max(0.0, ask - bid_value)
            if bid_value is not None
            else 0.02
        )

        points = []
        for item in history or []:
            try:
                timestamp = float(item[0])
                price = float(item[1])
                if 0.0 < price < 1.0:
                    points.append((timestamp, price))
            except (TypeError, ValueError, IndexError):
                continue

        points.sort()
        if not points:
            return spread, 0.0, 0.0

        def nearest(seconds_ago):
            return min(
                points,
                key=lambda x: abs((now - x[0]) - seconds_ago),
            )[1]

        p30 = nearest(30.0)
        p10 = nearest(10.0)
        momentum = ask - p30
        acceleration = (ask - p10) - (p10 - p30)
        return spread, momentum, acceleration

    def _score(
        self,
        ask,
        depth,
        spread,
        momentum,
        acceleration,
        elapsed,
    ):
        momentum_score = self._clamp(0.5 + momentum * 8.0)
        acceleration_score = self._clamp(0.5 + acceleration * 10.0)
        price_extremeness = self._clamp(abs(ask - 0.50) / 0.50)
        time_score = self._clamp(
            (elapsed - self.start_sec)
            / max(1.0, self.stop_sec - self.start_sec)
        )
        spread_score = self._clamp(
            1.0 - max(0.0, spread - 0.01) / 0.05
        )
        depth_reference = max(1.0, self.max_order / max(ask, 0.01))
        depth_score = self._clamp(float(depth) / depth_reference)

        score = (
            0.30 * momentum_score
            + 0.18 * acceleration_score
            + 0.18 * price_extremeness
            + 0.14 * time_score
            + 0.12 * depth_score
            + 0.08 * spread_score
        )
        return self._clamp(score)

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
        if depth <= 0 or spread > 0.05:
            return False
        if score < self.min_score:
            return False

        # Cheap remains the primary small-size hunting zone, but flat/
        # deteriorating markets are rejected instead of being accepted
        # merely because price is low.
        if regime == "CHEAP":
            return (
                score >= self.layer_a_min_score
                and momentum >= -0.0010
                and acceleration >= -0.0015
            )

        # Middle prices require actual positive confirmation. This removes
        # the old behavior where a nearly flat MID/CORE setup could pass.
        if regime == "MID":
            return (
                score >= max(self.min_score, 0.62)
                and momentum >= 0.0010
                and acceleration >= -0.0005
                and spread <= 0.04
            )

        if regime == "CORE":
            return (
                score >= max(self.min_score, 0.66)
                and momentum >= 0.0015
                and acceleration >= 0.0
                and spread <= 0.04
            )

        if regime == "HIGH":
            return (
                score >= self.layer_b_min_score
                and momentum >= 0.0020
                and acceleration >= 0.0
                and spread <= 0.035
                and elapsed < self.stop_sec
            )

        return False

    def _size(self, regime, price, score):
        price = self._clamp(price)
        score = self._clamp(score)

        if regime == "CHEAP":
            progress = self._clamp(
                (price - self.layer_a_min_price)
                / max(
                    0.0001,
                    self.layer_a_max_price - self.layer_a_min_price,
                )
            )
            size = (
                self.layer_a_base_notional
                + (
                    self.layer_a_max_notional
                    - self.layer_a_base_notional
                )
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
                2.50,
                max(0.50, 0.50 + 2.00 * score),
            )

        if regime == "CORE":
            return min(
                self.max_order,
                4.00,
                max(1.00, 1.00 + 3.00 * score),
            )

        if regime == "HIGH":
            strength = self._clamp(
                (score - self.layer_b_min_score)
                / max(0.0001, 1.0 - self.layer_b_min_score)
            )
            size = (
                self.layer_b_base_notional
                + (
                    self.layer_b_max_notional
                    - self.layer_b_base_notional
                )
                * strength
            )
            return min(
                self.layer_b_max_notional,
                self.max_order,
                max(self.layer_b_base_notional, size),
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

        now = float(now)
        elapsed = float(elapsed)
        current_exposure = max(0.0, float(current_exposure))
        asset_exposure = max(0.0, float(asset_exposure))
        total_exposure = max(0.0, float(total_exposure))
        available_cash = max(0.0, float(available_cash))

        if elapsed < self.start_sec or elapsed >= self.stop_sec:
            return None

        # The final-minute rule is independent of stop_sec. If stop_sec is
        # ever configured incorrectly, this still prevents late entries.
        market_seconds_left = max(0.0, 300.0 - elapsed)
        if market_seconds_left <= self.hard_cutoff_seconds:
            return None

        remaining = min(
            self.max_market_exposure - current_exposure,
            self.max_asset_exposure - asset_exposure,
            self.max_total_exposure - total_exposure,
            available_cash,
            self.max_order,
        )
        if remaining < 0.10:
            return None

        observations = (
            ("Up", up_ask, up_bid, up_depth, up_history),
            ("Down", down_ask, down_bid, down_depth, down_history),
        )

        candidates = []
        for side, ask, bid, depth, history in observations:
            features = self._features(ask, bid, history, now)
            if features is None:
                continue

            spread, momentum, acceleration = features
            ask = float(ask)
            depth = max(0.0, float(depth))
            regime = self._regime(ask)
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
            if regime == "CHEAP":
                ranking += 0.06
            elif regime == "MID":
                ranking -= 0.02
            elif regime == "CORE":
                ranking -= 0.04
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

        best = max(candidates, key=lambda item: item["ranking"])
        size = self._size(
            best["regime"],
            best["ask"],
            best["score"],
        )

        size = min(size, self.max_order, remaining)

        depth_cap = (
            best["depth"]
            * best["ask"]
            * self.max_depth_participation
        )
        size = min(size, depth_cap)

        # Re-apply every hard cap after depth sizing.
        size = min(
            size,
            self.HARD_MAX_ORDER,
            self.HARD_MAX_MARKET_EXPOSURE - current_exposure,
            self.HARD_MAX_ASSET_EXPOSURE - asset_exposure,
            self.HARD_MAX_MARKET_EXPOSURE - total_exposure,
            available_cash,
        )

        if size < 0.10:
            return None

        seconds_left = market_seconds_left
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
            f"global_exposure={total_exposure:.2f} "
            f"independent=true"
        )

        return Signal(
            side=best["side"],
            price=best["ask"],
            score=best["score"],
            notional=round(size, 2),
            reason=reason,
        )

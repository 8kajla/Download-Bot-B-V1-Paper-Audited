from dataclasses import dataclass
import math
import time


@dataclass
class Signal:
    side: str
    price: float
    score: float
    notional: float
    reason: str


class ConvergenceStrategy:
    """BOT B V3.1 behavioral-replica strategy.

    The sizing model is based on the observed reference-trader relationship
    between execution price and order notional. It is deliberately scaled to
    the current $1,000 / $100 research account and remains BUY-only.

    This is an inferred behavioral proxy; the reference trader's private
    pre-trade trigger is not observable in the execution-only dataset.
    """

    VERSION = "V3.1"

    CHEAP_MIN, CHEAP_MAX = 0.01, 0.30
    MID_MIN, MID_MAX = 0.30, 0.70
    CORE_MIN, CORE_MAX = 0.70, 0.90
    HIGH_MIN, HIGH_MAX = 0.90, 0.995

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_MAX_TOTAL_EXPOSURE = 100.0
    HARD_CUTOFF_SECONDS = 60.0

    # Reference empirical average order sizes by price bucket. These are
    # scaled to the research account, not used as literal live-money sizes.
    PRICE_ANCHORS = (
        (0.01, 0.47),
        (0.10, 0.47),
        (0.20, 0.75),
        (0.30, 1.07),
        (0.40, 1.48),
        (0.50, 2.17),
        (0.60, 3.08),
        (0.70, 3.93),
        (0.80, 5.25),
        (0.90, 8.28),
        (0.95, 30.78),
        (0.995, 30.78),
    )
    REFERENCE_SCALE = HARD_MAX_ORDER / 30.78

    def __init__(
        self,
        bankroll=1000,
        max_market_exposure=100,
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
        max_total_exposure=100,
        hard_cutoff_seconds=60,
    ):
        self.bankroll = float(bankroll)
        self.max_market_exposure = min(max(0.0, float(max_market_exposure)), self.HARD_MAX_MARKET_EXPOSURE)
        self.max_asset_exposure = min(max(0.0, float(max_asset_exposure)), self.HARD_MAX_ASSET_EXPOSURE)
        self.max_total_exposure = min(max(0.0, float(max_total_exposure)), self.HARD_MAX_TOTAL_EXPOSURE)
        self.max_order = min(max(0.0, float(max_order)), self.HARD_MAX_ORDER)

        self.layer_a_min_price = float(layer_a_min_price)
        self.layer_a_max_price = float(layer_a_max_price)
        self.layer_b_min_price = float(layer_b_min_price)
        self.layer_b_max_price = float(layer_b_max_price)
        self.layer_a_base_notional = max(0.10, float(layer_a_base_notional))
        self.layer_a_max_notional = min(self.max_order, float(layer_a_max_notional))
        self.layer_b_base_notional = min(self.max_order, float(layer_b_base_notional))
        self.layer_b_max_notional = min(self.max_order, float(layer_b_max_notional))

        self.start_sec = float(start_sec)
        self.stop_sec = float(stop_sec)
        self.min_score = float(min_score)
        self.layer_a_min_score = max(0.0, float(layer_a_min_score))
        self.layer_b_min_score = max(0.0, float(layer_b_min_score))
        self.max_depth_participation = min(0.25, max(0.01, float(max_depth_participation)))
        self.hard_cutoff_seconds = max(self.HARD_CUTOFF_SECONDS, float(hard_cutoff_seconds))

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    def _regime(self, price):
        p = float(price)
        if self.layer_a_min_price <= p < self.layer_a_max_price:
            return "CHEAP"
        if self.layer_a_max_price <= p < self.layer_b_min_price:
            return "MID" if p < 0.70 else "CORE"
        if self.layer_b_min_price <= p < self.layer_b_max_price:
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
        spread = max(0.0, ask - bid_value) if bid_value is not None else 0.02

        points = []
        for item in history or []:
            try:
                ts, price = float(item[0]), float(item[1])
                if 0.0 < price < 1.0:
                    points.append((ts, price))
            except (TypeError, ValueError, IndexError):
                continue
        points.sort()
        if not points:
            return spread, 0.0, 0.0

        def nearest(seconds_ago):
            return min(points, key=lambda x: abs((now - x[0]) - seconds_ago))[1]

        p30 = nearest(30.0)
        p10 = nearest(10.0)
        momentum = ask - p30
        acceleration = (ask - p10) - (p10 - p30)
        return spread, momentum, acceleration

    def _score(self, ask, depth, spread, momentum, acceleration, elapsed):
        momentum_score = self._clamp(0.5 + momentum * 8.0)
        acceleration_score = self._clamp(0.5 + acceleration * 10.0)
        price_extremeness = self._clamp(abs(ask - 0.50) / 0.50)
        time_score = self._clamp((elapsed - self.start_sec) / max(1.0, self.stop_sec - self.start_sec))
        spread_score = self._clamp(1.0 - max(0.0, spread - 0.01) / 0.05)
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

    def _allowed(self, regime, score, momentum, acceleration, spread, depth, elapsed):
        if depth <= 0 or spread > 0.05 or score < self.min_score:
            return False
        if regime == "CHEAP":
            return score >= self.layer_a_min_score and momentum >= -0.0010 and acceleration >= -0.0015
        if regime == "MID":
            return score >= max(self.min_score, 0.55) and momentum >= 0.0010 and acceleration >= -0.0005 and spread <= 0.04
        if regime == "CORE":
            return score >= max(self.min_score, 0.58) and momentum >= 0.0015 and acceleration >= 0.0 and spread <= 0.04
        if regime == "HIGH":
            return score >= self.layer_b_min_score and momentum >= 0.0020 and acceleration >= 0.0 and spread <= 0.035 and elapsed < self.stop_sec
        return False

    @classmethod
    def _reference_size(cls, price):
        """Interpolate the empirical average order-size curve in log-space."""
        p = max(cls.PRICE_ANCHORS[0][0], min(cls.PRICE_ANCHORS[-1][0], float(price)))
        anchors = cls.PRICE_ANCHORS
        if p <= anchors[0][0]:
            raw = anchors[0][1]
        elif p >= anchors[-1][0]:
            raw = anchors[-1][1]
        else:
            raw = anchors[0][1]
            for (p0, s0), (p1, s1) in zip(anchors, anchors[1:]):
                if p0 <= p <= p1:
                    if p1 == p0:
                        raw = s1
                    else:
                        w = (p - p0) / (p1 - p0)
                        raw = math.exp(math.log(s0) + w * (math.log(s1) - math.log(s0)))
                    break
        return raw * cls.REFERENCE_SCALE

    def _size(self, regime, price, score):
        """Continuous empirical price-to-size curve, scaled to this account."""
        p = max(0.01, min(0.995, float(price)))
        score = self._clamp(score)
        base = self._reference_size(p)

        # Keep the observed price relationship dominant. Signal strength is a
        # modest multiplier rather than a separate tiered sizing system.
        signal_multiplier = 0.85 + 0.30 * score
        size = base * signal_multiplier

        # Preserve the existing small-trade floor for paper execution and the
        # configurable cheap/high compatibility limits only where they are
        # safety-oriented. The empirical curve remains the primary driver.
        if regime == "CHEAP":
            size = max(0.10, size)
        else:
            size = max(0.10, size)

        return min(self.max_order, size)

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
        seconds_left = max(0.0, 300.0 - elapsed)
        if seconds_left <= self.hard_cutoff_seconds:
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

        candidates = []
        for side, ask, bid, depth, history in (
            ("Up", up_ask, up_bid, up_depth, up_history),
            ("Down", down_ask, down_bid, down_depth, down_history),
        ):
            features = self._features(ask, bid, history, now)
            if features is None:
                continue
            spread, momentum, acceleration = features
            ask = float(ask)
            depth = max(0.0, float(depth))
            regime = self._regime(ask)
            if regime is None:
                continue
            score = self._score(ask, depth, spread, momentum, acceleration, elapsed)
            if not self._allowed(regime, score, momentum, acceleration, spread, depth, elapsed):
                continue

            ranking = score
            # Cheap executions are more frequent in the reference data; this
            # is only a small ranking preference, not a size multiplier.
            if regime == "CHEAP":
                ranking += 0.04
            elif regime == "HIGH":
                ranking -= 0.02

            candidates.append({
                "side": side,
                "ask": ask,
                "depth": depth,
                "regime": regime,
                "score": score,
                "ranking": ranking,
                "momentum": momentum,
                "acceleration": acceleration,
                "spread": spread,
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item["ranking"])
        size = min(self._size(best["regime"], best["ask"], best["score"]), self.max_order, remaining)

        depth_cap = best["depth"] * best["ask"] * self.max_depth_participation
        size = min(size, depth_cap)
        size = min(
            size,
            self.HARD_MAX_ORDER,
            self.HARD_MAX_MARKET_EXPOSURE - current_exposure,
            self.HARD_MAX_ASSET_EXPOSURE - asset_exposure,
            self.HARD_MAX_TOTAL_EXPOSURE - total_exposure,
            available_cash,
        )
        if size < 0.10:
            return None

        reason = (
            f"V3.1 regime={best['regime']} "
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

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
    Bot B V2 — independent paper strategy inspired by observed trader behavior.

    This is NOT a copy bot. It never reads trader activity.

    V2 behavioral targets:
      - 0.01-0.30 cheap regime: primary trade-count regime
      - 0.30-0.70 middle regime: genuine coverage
      - 0.70-0.90 core regime: preserved because V1 was profitable here
      - 0.90-0.995 high regime: sharply restricted because V1 lost here
      - dynamic sizing
      - rapid repeated entries
      - no forced exits/hedges
      - final 60 seconds blocked

    The trader's actual private trigger is unknown. The observable
    microstructure score below is therefore a research hypothesis, not
    a claim that we recovered his private algorithm.
    """

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
        layer_a_max_notional=3.00,
        layer_b_base_notional=2.00,
        layer_b_max_notional=7.50,
        start_sec=0,
        stop_sec=240,
        min_score=0.50,
        layer_a_min_score=0.45,
        layer_b_min_score=0.65,
        max_depth_participation=0.25,
        max_asset_exposure=35,
    ):
        self.bankroll = float(bankroll)

        # Keep compatibility with the existing bot.py constructor. V2 owns
        # the regime boundaries so stale V1 A/B environment variables cannot
        # silently restore the old two-layer strategy.
        self.max_market_exposure = min(float(max_market_exposure), 25.0)
        self.max_order = float(max_order)

        self.layer_a_min_price = self.CHEAP_MIN
        self.layer_a_max_price = self.CHEAP_MAX
        self.layer_b_min_price = self.HIGH_MIN
        self.layer_b_max_price = self.HIGH_MAX

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
        self.max_asset_exposure = min(float(max_asset_exposure), 35.0)

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    @staticmethod
    def _features(ask, bid, hist, now):
        if ask is None:
            return None

        ask = float(ask)
        if not 0.0 < ask < 1.0:
            return None

        bid = float(bid) if bid is not None else None
        spread = max(0.0, ask - bid) if bid is not None else 0.02

        history = []
        for item in hist or []:
            try:
                timestamp = float(item[0])
                price = float(item[1])
                if 0.0 < price < 1.0:
                    history.append((timestamp, price))
            except (TypeError, ValueError, IndexError):
                continue

        history.sort(key=lambda x: x[0])

        momentum = 0.0
        acceleration = 0.0

        if history:
            def nearest(seconds):
                return min(
                    history,
                    key=lambda x: abs((now - x[0]) - seconds),
                )[1]

            p30 = nearest(30.0)
            p10 = nearest(10.0)

            momentum = ask - p30
            acceleration = (ask - p10) - (p10 - p30)

        return spread, momentum, acceleration

    @staticmethod
    def _regime(price):
        price = float(price)

        if 0.01 <= price < 0.30:
            return "CHEAP"
        if 0.30 <= price < 0.70:
            return "MID"
        if 0.70 <= price < 0.90:
            return "CORE"
        if 0.90 <= price < 1.00:
            return "HIGH"

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
        Observable-market ranking score, NOT calibrated probability.
        """
        momentum_score = self._clamp(0.5 + momentum * 8.0)
        acceleration_score = self._clamp(0.5 + acceleration * 10.0)
        price_extremeness = self._clamp(abs(ask - 0.50) / 0.50)

        span = max(1.0, self.stop_sec - self.start_sec)
        time_score = self._clamp(
            (elapsed - self.start_sec) / span
        )

        spread_score = self._clamp(
            1.0 - max(0.0, spread - 0.01) / 0.05
        )

        depth_reference = max(
            1.0,
            self.max_order / max(ask, 0.01),
        )
        depth_score = self._clamp(
            float(depth) / depth_reference
        )

        return self._clamp(
            0.30 * momentum_score
            + 0.18 * acceleration_score
            + 0.18 * price_extremeness
            + 0.14 * time_score
            + 0.12 * depth_score
            + 0.08 * spread_score
        )

    def _cheap_signal(
        self,
        score,
        momentum,
        acceleration,
        spread,
        depth,
        ask,
    ):
        """
        Cheap-layer hypothesis.

        The old V1 price score mechanically suppressed prices below 0.50.
        V2 therefore uses the score only as a ranking threshold and requires
        basic observable book quality.
        """
        return (
            score >= self.layer_a_min_score
            and momentum >= -0.0025
            and acceleration >= -0.0025
            and spread <= 0.05
            and depth > 0.0
            and ask >= self.CHEAP_MIN
        )

    def _mid_signal(
        self,
        score,
        momentum,
        acceleration,
        spread,
        depth,
    ):
        return (
            score >= 0.58
            and momentum >= -0.0015
            and acceleration >= -0.0020
            and spread <= 0.05
            and depth > 0.0
        )

    def _core_signal(
        self,
        score,
        momentum,
        acceleration,
        spread,
        depth,
    ):
        return (
            score >= 0.58
            and momentum >= -0.0020
            and acceleration >= -0.0025
            and spread <= 0.05
            and depth > 0.0
        )

    def _high_signal(
        self,
        score,
        momentum,
        acceleration,
        spread,
        depth,
        elapsed,
    ):
        """
        V1's 90-100c late-convergence regime was the confirmed loss driver.
        Keep it only as a very selective, small-size hypothesis.
        """
        return (
            score >= max(0.82, self.layer_b_min_score + 0.17)
            and momentum >= 0.0020
            and acceleration >= 0.0
            and spread <= 0.035
            and depth > 0.0
            and elapsed >= 90.0
        )

    def _size_for_regime(
        self,
        regime,
        price,
        score,
        momentum,
        current_exposure,
    ):
        """
        Restrained empirical-shape sizing.

        Research shows notional rises strongly with price. The matched sample
        also showed size increasing during many bursts. V2 uses a conservative
        curve because trader profitability is unverified and V1 had dangerous
        concentration.
        """
        score = self._clamp(score)
        price = self._clamp(price)

        if regime == "CHEAP":
            # Cheap contracts dominate trade count but only ~12.9% of trader
            # dollar volume. Keep them small while allowing conviction scaling.
            base = 0.15 + 0.35 * self._clamp(price / 0.30)
            scale = 0.40 + 0.60 * score
            size = base * scale
            return min(3.00, max(0.15, size))

        if regime == "MID":
            size = 0.75 + 2.25 * score
            return min(3.50, max(0.75, size))

        if regime == "CORE":
            # Preserve the V1 70-90c regime while eliminating flat $5 sizing
            # at mediocre scores.
            size = 1.50 + 4.50 * score
            if score < 0.70:
                size = min(size, 3.00)
            return min(6.00, max(1.50, size))

        if regime == "HIGH":
            # Explicitly shrink the V1 loss-making 90-100c regime.
            size = 0.75 + 2.75 * max(0.0, score - 0.82) / 0.18
            if score < 0.92:
                size = min(size, 1.50)
            return min(3.00, max(0.75, size))

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

        if elapsed < self.start_sec:
            return None

        # Five-minute markets: final 60 seconds are blocked exactly.
        if elapsed >= self.stop_sec:
            return None

        current_exposure = float(current_exposure)
        asset_exposure = float(asset_exposure)
        available_cash = float(available_cash)

        remaining_budget = min(
            self.max_market_exposure - current_exposure,
            self.max_asset_exposure - asset_exposure,
        )

        if remaining_budget <= 0.0 or available_cash <= 0.0:
            return None

        candidates = []

        observations = (
            ("Up", up_ask, up_bid, up_depth, up_history),
            ("Down", down_ask, down_bid, down_depth, down_history),
        )

        for side, ask, bid, depth, history in observations:
            features = self._features(
                ask,
                bid,
                history,
                now,
            )

            if features is None:
                continue

            spread, momentum, acceleration = features
            ask = float(ask)
            depth = max(0.0, float(depth))

            regime = self._regime(ask)
            if regime is None:
                continue

            score = self._score(
                ask=ask,
                depth=depth,
                spread=spread,
                momentum=momentum,
                acceleration=acceleration,
                elapsed=elapsed,
            )

            if regime == "CHEAP":
                allowed = self._cheap_signal(
                    score,
                    momentum,
                    acceleration,
                    spread,
                    depth,
                    ask,
                )
            elif regime == "MID":
                allowed = self._mid_signal(
                    score,
                    momentum,
                    acceleration,
                    spread,
                    depth,
                )
            elif regime == "CORE":
                allowed = self._core_signal(
                    score,
                    momentum,
                    acceleration,
                    spread,
                    depth,
                )
            else:
                allowed = self._high_signal(
                    score,
                    momentum,
                    acceleration,
                    spread,
                    depth,
                    elapsed,
                )

            if not allowed:
                continue

            ranking_score = score

            # Prevent the previously loss-making high-price regime from
            # automatically dominating a comparable lower-price opportunity.
            if regime == "HIGH":
                ranking_score -= 0.10
            elif regime == "MID":
                ranking_score += 0.02

            candidates.append(
                {
                    "side": side,
                    "ask": ask,
                    "bid": bid,
                    "depth": depth,
                    "regime": regime,
                    "score": score,
                    "ranking_score": ranking_score,
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "spread": spread,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: item["ranking_score"],
        )

        ask = best["ask"]
        depth = best["depth"]
        regime = best["regime"]
        score = best["score"]

        target_notional = self._size_for_regime(
            regime=regime,
            price=ask,
            score=score,
            momentum=best["momentum"],
            current_exposure=current_exposure,
        )

        target_notional = min(
            target_notional,
            self.max_order,
        )
        target_notional = min(
            target_notional,
            remaining_budget,
        )
        target_notional = min(
            target_notional,
            available_cash,
        )

        depth_cap = (
            depth
            * ask
            * self.max_depth_participation
        )
        target_notional = min(
            target_notional,
            depth_cap,
        )

        if target_notional < 0.10:
            return None

        reason = (
            f"v2_regime={regime} "
            f"score={score:.3f} "
            f"momentum={best['momentum']:+.4f} "
            f"accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} "
            f"depth={depth:.2f} "
            f"elapsed={elapsed:.1f}s "
            f"remaining={300.0 - elapsed:.1f}s "
            f"replica=behavioral-hypothesis"
        )

        return Signal(
            side=best["side"],
            price=ask,
            score=score,
            notional=round(target_notional, 2),
            reason=reason,
        )

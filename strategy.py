
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
    """BOT B V6 — regime-specific behavioral model.

    Confirmed observations incorporated:
      * BUY-only behavior in the available trader history.
      * ~2s median intertrade cadence with rapid bursts.
      * ~89% same-side continuation with occasional resets.
      * CHEAP entries disproportionately occur while price is falling.
      * CORE/HIGH entries disproportionately occur while price is rising.
      * Entry size increases with entry count, with the strongest effect in
        CORE/HIGH; HIGH starts smaller and then scales sharply.
      * Sizing is smooth/continuous; clean discrete share tiers were not found.
      * No hard-coded asset-specific distribution is used because the latest
        asset x regime estimates were not stable enough to justify it.
      * Final 60 seconds remain a hard no-entry zone.

    Unknowns intentionally NOT hard-coded as facts:
      * the trader's true pre-trade trigger,
      * whether his fills come from a static/relative resting ladder,
      * his exact order-book inputs.
    """

    VERSION = "V6"

    CHEAP_MIN, CHEAP_MAX = 0.01, 0.30
    MID_MIN, MID_MAX = 0.30, 0.70
    CORE_MIN, CORE_MAX = 0.70, 0.90
    HIGH_MIN, HIGH_MAX = 0.90, 0.995

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_TOTAL_EXPOSURE = 300.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0

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
        max_total_exposure=300,
        hard_cutoff_seconds=60,
        min_trade_gap_seconds=2,
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
            self.HARD_MAX_TOTAL_EXPOSURE,
        )
        self.max_order = min(
            max(0.0, float(max_order)),
            self.HARD_MAX_ORDER,
        )

        self.layer_a_min_price = float(layer_a_min_price)
        self.layer_a_max_price = float(layer_a_max_price)
        self.layer_b_min_price = float(layer_b_min_price)
        self.layer_b_max_price = float(layer_b_max_price)

        self.start_sec = float(start_sec)
        self.stop_sec = min(240.0, float(stop_sec))

        self.min_score = float(min_score)
        self.layer_a_min_score = float(layer_a_min_score)
        self.layer_b_min_score = float(layer_b_min_score)

        self.max_depth_participation = min(
            0.25,
            max(0.01, float(max_depth_participation)),
        )

        self.hard_cutoff_seconds = max(
            self.HARD_CUTOFF_SECONDS,
            float(hard_cutoff_seconds),
        )
        self.min_trade_gap_seconds = max(
            0.0,
            float(min_trade_gap_seconds),
        )

        # Kept for Railway/backward compatibility.
        self.layer_a_base_notional = max(
            0.10,
            float(layer_a_base_notional),
        )
        self.layer_a_max_notional = min(
            self.max_order,
            max(
                self.layer_a_base_notional,
                float(layer_a_max_notional),
            ),
        )
        self.layer_b_base_notional = min(
            self.max_order,
            max(0.10, float(layer_b_base_notional)),
        )
        self.layer_b_max_notional = min(
            self.max_order,
            max(
                self.layer_b_base_notional,
                float(layer_b_max_notional),
            ),
        )

        self._last_trade_at = None

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    def _regime(self, price):
        p = float(price)
        if self.CHEAP_MIN <= p < self.CHEAP_MAX:
            return "CHEAP"
        if self.MID_MIN <= p < self.MID_MAX:
            return "MID"
        if self.CORE_MIN <= p < self.CORE_MAX:
            return "CORE"
        if self.HIGH_MIN <= p < self.HIGH_MAX:
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
            bid = None if bid is None else float(bid)
        except (TypeError, ValueError):
            bid = None

        spread = (
            max(0.0, ask - bid)
            if bid is not None
            else 0.02
        )

        points = []
        for item in history or []:
            try:
                t, p = float(item[0]), float(item[1])
                if 0.0 < p < 1.0:
                    points.append((t, p))
            except (TypeError, ValueError, IndexError):
                continue

        points.sort()

        if not points:
            return spread, 0.0, 0.0

        def nearest(seconds_ago):
            return min(
                points,
                key=lambda item: abs(
                    (now - item[0]) - seconds_ago
                ),
            )[1]

        p30 = nearest(30.0)
        p10 = nearest(10.0)

        momentum = ask - p30
        acceleration = (
            (ask - p10)
            - (p10 - p30)
        )

        return spread, momentum, acceleration

    def _base_score(
        self,
        ask,
        depth,
        spread,
        momentum,
        acceleration,
        elapsed,
    ):
        momentum_score = self._clamp(
            0.5 + momentum * 8.0
        )
        acceleration_score = self._clamp(
            0.5 + acceleration * 10.0
        )
        extremeness = self._clamp(
            abs(ask - 0.50) / 0.50
        )
        time_score = self._clamp(
            (elapsed - self.start_sec)
            / max(
                1.0,
                self.stop_sec - self.start_sec,
            )
        )
        spread_score = self._clamp(
            1.0
            - max(0.0, spread - 0.01)
            / 0.05
        )

        depth_dollars = (
            max(0.0, float(depth))
            * max(ask, 0.01)
        )
        depth_score = self._clamp(
            depth_dollars / 20.0
        )

        return self._clamp(
            0.25 * momentum_score
            + 0.16 * acceleration_score
            + 0.18 * extremeness
            + 0.13 * time_score
            + 0.14 * depth_score
            + 0.14 * spread_score
        )

    def _regime_trigger_score(
        self,
        regime,
        base_score,
        momentum,
        acceleration,
        elapsed,
    ):
        """Apply the observed weakness/strength gradient."""
        score = base_score

        if regime == "CHEAP":
            # Reward weak/falling price behavior, but permit flat prices.
            weakness = self._clamp(
                (-momentum + 0.002) / 0.04,
                0.0,
                1.0,
            )
            score += 0.10 * weakness

            persistence = self._clamp(
                (-acceleration + 0.002) / 0.04,
                0.0,
                1.0,
            )
            score += 0.05 * persistence

            if elapsed >= 120:
                score += 0.025

        elif regime == "MID":
            # MID is deliberately centered around a neutral-to-mildly
            # directional state.
            neutral = 1.0 - self._clamp(
                abs(momentum - 0.002) / 0.035
            )
            score += 0.05 * neutral

        elif regime == "CORE":
            strength = self._clamp(
                (momentum - 0.002) / 0.035,
                0.0,
                1.0,
            )
            score += 0.10 * strength

            accel = self._clamp(
                (acceleration - 0.001) / 0.025,
                0.0,
                1.0,
            )
            score += 0.06 * accel

        elif regime == "HIGH":
            strength = self._clamp(
                (momentum - 0.003) / 0.025,
                0.0,
                1.0,
            )
            score += 0.13 * strength

            accel = self._clamp(
                (acceleration + 0.001) / 0.02,
                0.0,
                1.0,
            )
            score += 0.06 * accel

            # HIGH executions are materially later than other regimes.
            if elapsed >= 150:
                score += 0.07

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
        if depth <= 0:
            return False

        if spread > 0.05:
            return False

        if regime == "CHEAP":
            return (
                score >= max(
                    self.layer_a_min_score,
                    0.49,
                )
                and momentum <= 0.012
                and acceleration <= 0.018
            )

        if regime == "MID":
            return (
                score >= max(
                    self.min_score,
                    0.63,
                )
                and momentum >= -0.010
                and momentum <= 0.050
                and acceleration >= -0.014
                and acceleration <= 0.018
                and spread <= 0.045
            )

        if regime == "CORE":
            return (
                score >= 0.72
                and momentum >= 0.004
                and spread <= 0.035
            )

        if regime == "HIGH":
            return (
                score >= max(
                    self.layer_b_min_score,
                    0.83,
                )
                and momentum >= 0.004
                and elapsed >= 120
                and spread <= 0.03
            )

        return False

    def _size(
        self,
        regime,
        price,
        score,
        entry_count=0,
        elapsed_since_first_entry=0.0,
    ):
        """Smooth, regime-specific size curve calibrated to observed medians.

        Entry tiers are continuous rather than hard share buckets.
        """
        p = self._clamp(price, 0.01, 0.995)
        s = self._clamp(score)
        n = max(0, int(entry_count))
        age = max(
            0.0,
            float(elapsed_since_first_entry),
        )

        # Smooth entry-count multiplier.
        if n <= 0:
            entry_factor = 1.00
        elif n <= 2:
            entry_factor = 1.12
        elif n <= 4:
            entry_factor = 1.24
        else:
            entry_factor = 1.30

        if regime == "CHEAP":
            x = self._clamp(
                (p - 0.01) / 0.29
            )
            # At 20c, starter ~0.40 and 4th+ ~0.61.
            base = (
                0.27
                + 0.24
                * (x ** 0.65)
                * (0.85 + 0.15 * s)
            )
            size = (
                base
                * (0.92 + 0.18 * entry_factor)
            )

        elif regime == "MID":
            x = self._clamp(
                (p - 0.30) / 0.40
            )
            # Around 50c, approx $1.9-$2.1 across entries.
            base = (
                1.35
                + 0.60
                * (x ** 0.70)
                * (0.85 + 0.15 * s)
            )
            size = base * (
                0.92
                + 0.08 * entry_factor
            )

        elif regime == "CORE":
            x = self._clamp(
                (p - 0.70) / 0.20
            )
            # Around 80c, approx $4.3 starter and ~$5 after several entries.
            base = (
                3.45
                + 1.05
                * (x ** 0.60)
                * (0.85 + 0.15 * s)
            )
            size = base * (
                0.88
                + 0.12 * entry_factor
            )

        elif regime == "HIGH":
            x = self._clamp(
                (p - 0.90) / 0.095
            )
            # Around 95c, starter ~5.7; later entries hit the $10 cap.
            starter = (
                4.20
                + 1.55
                * (x ** 0.50)
                * (0.85 + 0.15 * s)
            )
            if n == 0:
                size = starter
            else:
                size = (
                    starter
                    + 5.75
                    * self._clamp(
                        n / 4.0,
                        0.0,
                        1.0,
                    )
                    + (
                        0.40
                        if age >= 90
                        else 0.0
                    )
                )

        else:
            return 0.0

        return min(
            self.max_order,
            max(0.10, size),
        )

    @staticmethod
    def _reset_allowed(
        thesis_side,
        thesis_price,
        candidate_side,
        candidate_price,
        current_elapsed,
    ):
        """Conservative reset rule reflecting the observed ~10% flips.

        A side flip is allowed only after a large probability-state move.
        This is an explicit hypothesis, not a claimed reconstruction of the
        trader's private trigger.
        """
        if not thesis_side or thesis_side == candidate_side:
            return True

        if thesis_price is None:
            return False

        jump = abs(
            float(candidate_price)
            - float(thesis_price)
        )

        # Very large regime/state transitions can justify a directional reset.
        return (
            jump >= 0.45
            and current_elapsed >= 60.0
        )

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
        market_entry_count=0,
        seconds_since_first_entry=0.0,
        thesis_side=None,
        thesis_price=None,
    ):
        now = (
            time.time()
            if now is None
            else float(now)
        )
        elapsed = float(elapsed)

        if elapsed < self.start_sec:
            return None

        if elapsed >= self.stop_sec:
            return None

        if (
            300.0 - elapsed
            <= self.hard_cutoff_seconds
        ):
            return None

        if (
            self._last_trade_at is not None
            and now - self._last_trade_at
            < self.min_trade_gap_seconds
        ):
            return None

        remaining = min(
            self.max_market_exposure
            - float(current_exposure),
            self.max_asset_exposure
            - float(asset_exposure),
            self.max_total_exposure
            - float(total_exposure),
            float(available_cash),
            self.max_order,
        )

        if remaining < 0.10:
            return None

        candidates = []

        for side, ask, bid, depth, hist in (
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
        ):
            features = self._features(
                ask,
                bid,
                hist,
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

            base_score = self._base_score(
                ask,
                depth,
                spread,
                momentum,
                acceleration,
                elapsed,
            )
            score = self._regime_trigger_score(
                regime,
                base_score,
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

            if not self._reset_allowed(
                thesis_side,
                thesis_price,
                side,
                ask,
                elapsed,
            ):
                continue

            # Candidate ranking reflects the same observed monotonic
            # weakness→strength gradient without imposing a fixed quota.
            direction_bias = {
                "CHEAP": -momentum * 1.25,
                "MID": -abs(momentum - 0.002) * 0.25,
                "CORE": momentum * 1.25,
                "HIGH": momentum * 1.60,
            }[regime]

            reset_bonus = 0.0
            if (
                thesis_side
                and thesis_side != side
                and self._reset_allowed(
                    thesis_side,
                    thesis_price,
                    side,
                    ask,
                    elapsed,
                )
            ):
                reset_bonus = 0.05

            candidates.append(
                {
                    "side": side,
                    "ask": ask,
                    "depth": depth,
                    "regime": regime,
                    "score": score,
                    "ranking": (
                        score
                        + direction_bias
                        + reset_bonus
                    ),
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "spread": spread,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: item["ranking"],
        )

        desired = self._size(
            best["regime"],
            best["ask"],
            best["score"],
            market_entry_count,
            seconds_since_first_entry,
        )

        size = min(
            desired,
            remaining,
        )

        depth_cap = (
            best["depth"]
            * best["ask"]
            * self.max_depth_participation
        )

        size = min(
            size,
            depth_cap,
            self.HARD_MAX_ORDER,
            self.HARD_MAX_MARKET_EXPOSURE
            - float(current_exposure),
            self.HARD_MAX_ASSET_EXPOSURE
            - float(asset_exposure),
            self.HARD_MAX_TOTAL_EXPOSURE
            - float(total_exposure),
            float(available_cash),
        )

        if size < 0.10:
            return None

        self._last_trade_at = now

        mode = (
            "STARTER"
            if market_entry_count == 0
            else "ADD_ON"
        )

        if thesis_side and thesis_side != best["side"]:
            mode = "RESET"

        reason = (
            f"V6 regime={best['regime']} "
            f"mode={mode} "
            f"entry_count={int(market_entry_count)} "
            f"score={best['score']:.3f} "
            f"momentum={best['momentum']:+.4f} "
            f"accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} "
            f"depth={best['depth']:.2f} "
            f"elapsed={elapsed:.1f}s "
            f"since_first={float(seconds_since_first_entry):.1f}s "
            f"seconds_left={300.0 - elapsed:.1f} "
            f"global_exposure={float(total_exposure):.2f} "
            f"global_cap={self.max_total_exposure:.2f} "
            f"independent=false"
        )

        return Signal(
            best["side"],
            best["ask"],
            best["score"],
            round(size, 2),
            reason,
        )

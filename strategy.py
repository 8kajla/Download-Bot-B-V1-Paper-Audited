
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
    """BOT B V5 — four-regime, starter/add-on research model.

    Evidence incorporated from the synchronized trader dataset:
      * CHEAP is high-frequency, small-size, gap-tolerant and often later.
      * MID is moderate-frequency with comparatively stable ~$2 sizing.
      * CORE is less frequent and larger, with stronger confirmation.
      * HIGH is rare, late and very capital-intensive.
      * Repeated entries are a central behavior; first entries are generally
        smaller than later entries, especially in CORE/HIGH.

    Important:
      This is a behavioral research model, not a claim of the trader's
      private/hidden trigger.
    """

    VERSION = "V5"

    CHEAP_MIN, CHEAP_MAX = 0.01, 0.30
    MID_MIN, MID_MAX = 0.30, 0.70
    CORE_MIN, CORE_MAX = 0.70, 0.90
    HIGH_MIN, HIGH_MAX = 0.90, 0.995

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_TOTAL_EXPOSURE = 300.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0

    # Historical reference only; never a hard quota.
    TARGETS = {
        "CHEAP": 0.476,
        "MID": 0.347,
        "CORE": 0.105,
        "HIGH": 0.072,
    }

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
        self.max_asset_exposure = min(
            self.max_asset_exposure,
            float(max_asset_exposure),
        )

        self.hard_cutoff_seconds = max(
            60.0,
            float(hard_cutoff_seconds),
        )
        self.min_trade_gap_seconds = max(
            0.0,
            float(min_trade_gap_seconds),
        )

        # Keep these for compatibility with the existing environment.
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
            max(
                0.10,
                float(layer_b_base_notional),
            ),
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
    def _clamp(value, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(value)))

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

        pts = []
        for item in history or []:
            try:
                t, p = float(item[0]), float(item[1])
                if 0.0 < p < 1.0:
                    pts.append((t, p))
            except (TypeError, ValueError, IndexError):
                continue

        if not pts:
            return spread, 0.0, 0.0

        def nearest(seconds):
            return min(
                pts,
                key=lambda item: abs((now - item[0]) - seconds),
            )[1]

        p30 = nearest(30)
        p10 = nearest(10)

        momentum = ask - p30
        acceleration = (
            (ask - p10) - (p10 - p30)
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
        ms = self._clamp(0.5 + momentum * 8.0)
        acs = self._clamp(0.5 + acceleration * 10.0)

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
            0.29 * ms
            + 0.18 * acs
            + 0.16 * extremeness
            + 0.15 * time_score
            + 0.12 * depth_score
            + 0.10 * spread_score
        )

    def _regime_score(
        self,
        regime,
        base_score,
        momentum,
        acceleration,
        spread,
        elapsed,
    ):
        """Regime-specific score shaping.

        This is intentionally separate by regime rather than one universal
        scoring policy.
        """
        score = base_score

        if regime == "CHEAP":
            # Cheap is driven more by extremeness and persistence than by
            # requiring positive momentum.
            score += 0.04 * self._clamp(
                -momentum / 0.05
            )
            score += 0.03 * self._clamp(
                -acceleration / 0.05
            )

            if elapsed >= 120:
                score += 0.03

        elif regime == "MID":
            # MID retains a modest trend component.
            score += 0.04 * self._clamp(
                momentum / 0.05,
                -1,
                1,
            )

        elif regime == "CORE":
            # CORE requires much stronger directional confirmation.
            score += 0.08 * self._clamp(
                momentum / 0.04,
                -1,
                1,
            )
            score += 0.06 * self._clamp(
                acceleration / 0.03,
                -1,
                1,
            )

        elif regime == "HIGH":
            # HIGH behaves more like late confirmation.
            score += 0.10 * self._clamp(
                momentum / 0.03,
                -1,
                1,
            )
            if elapsed >= 165:
                score += 0.05

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
        entry_count,
    ):
        if depth <= 0:
            return False

        if spread > 0.05:
            return False

        if regime == "CHEAP":
            return (
                score >= 0.49
                and momentum >= -0.015
                and acceleration >= -0.018
                and spread <= 0.05
            )

        if regime == "MID":
            return (
                score >= 0.60
                and momentum >= -0.004
                and acceleration >= -0.008
                and spread <= 0.045
            )

        if regime == "CORE":
            return (
                score >= 0.77
                and momentum >= 0.008
                and acceleration >= 0.002
                and spread <= 0.03
            )

        if regime == "HIGH":
            return (
                score >= 0.83
                and momentum >= 0.003
                and acceleration >= -0.001
                and spread <= 0.03
                and elapsed >= 90
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
        """Piecewise sizing curve with smaller starters and larger add-ons."""
        p = self._clamp(price, 0.01, 0.995)
        s = self._clamp(score)
        entry_count = max(0, int(entry_count))
        age = max(
            0.0,
            float(elapsed_since_first_entry),
        )

        starter = entry_count == 0

        if regime == "CHEAP":
            x = self._clamp(
                (p - 0.01) / 0.29
            )
            starter_size = (
                0.20
                + 0.55
                * (x ** 0.65)
                * (0.80 + 0.20 * s)
            )

            addon_size = (
                0.30
                + 1.15
                * (x ** 0.55)
                * (0.80 + 0.20 * s)
            )

            return min(
                self.max_order,
                starter_size
                if starter
                else addon_size,
            )

        if regime == "MID":
            x = self._clamp(
                (p - 0.30) / 0.40
            )

            starter_size = (
                0.70
                + 1.40
                * (x ** 0.70)
                * (0.82 + 0.18 * s)
            )

            addon_size = (
                1.10
                + 2.30
                * (x ** 0.65)
                * (0.82 + 0.18 * s)
            )

            return min(
                self.max_order,
                starter_size
                if starter
                else addon_size,
            )

        if regime == "CORE":
            x = self._clamp(
                (p - 0.70) / 0.20
            )

            starter_size = (
                1.75
                + 2.00
                * (x ** 0.60)
                * (0.85 + 0.15 * s)
            )

            addon_size = (
                3.00
                + 3.60
                * (x ** 0.55)
                * (0.85 + 0.15 * s)
            )

            return min(
                self.max_order,
                6.50,
                starter_size
                if starter
                else addon_size,
            )

        if regime == "HIGH":
            x = self._clamp(
                (p - 0.90) / 0.095
            )

            # The observed trader has a very large late HIGH allocation.
            # We cannot exceed our $10 research order ceiling.
            starter_size = (
                2.50
                + 3.50
                * (x ** 0.55)
                * (0.85 + 0.15 * s)
            )

            addon_size = (
                5.00
                + 5.00
                * (x ** 0.45)
                * (0.90 + 0.10 * s)
            )

            # Later add-ons can be slightly larger, matching the observed
            # tendency for late HIGH entries to concentrate capital.
            if not starter and age >= 90:
                addon_size *= 1.05

            return min(
                self.max_order,
                starter_size
                if starter
                else addon_size,
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
        market_entry_count=0,
        seconds_since_first_entry=0.0,
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
            max(0.0, 300.0 - elapsed)
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

            score = self._regime_score(
                regime,
                base_score,
                momentum,
                acceleration,
                spread,
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
                market_entry_count,
            ):
                continue

            # Small soft prior to prevent CORE from dominating the selected
            # candidate when multiple regimes are simultaneously eligible.
            bias = {
                "CHEAP": 0.05,
                "MID": 0.03,
                "CORE": -0.06,
                "HIGH": -0.01,
            }[regime]

            candidates.append({
                "side": side,
                "ask": ask,
                "depth": depth,
                "regime": regime,
                "score": score,
                "ranking": score + bias,
                "momentum": momentum,
                "acceleration": acceleration,
                "spread": spread,
            })

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

        mode = "STARTER" if market_entry_count == 0 else "ADD_ON"

        reason = (
            f"V5 regime={best['regime']} "
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
            f"independent=true"
        )

        return Signal(
            best["side"],
            best["ask"],
            best["score"],
            round(size, 2),
            reason,
        )

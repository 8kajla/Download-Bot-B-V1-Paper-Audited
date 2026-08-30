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
    """V7 behavioral model.

    The model is deliberately built from confirmed trader observations:
    BUY-only behavior, ~2s bursts, strong side persistence, CHEAP weakness
    buying, CORE/HIGH strength buying, entry-count-aware smooth sizing, and a
    hard final-minute cutoff.

    V7 changes from V6:
      * MID is easier to enter so it cannot disappear from the sample.
      * CORE/HIGH require materially stronger confirmation.
      * Add-ons require signal persistence; weak signals stop scaling.
      * Opposite-side resets require a large state move plus stronger evidence,
        and a cooldown prevents ping-pong resets.
      * Sizing remains smooth/continuous; no artificial share buckets.
    """

    VERSION = "V7"

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
        layer_a_min_score=0.45,
        layer_b_base_notional=2.00,
        layer_b_max_notional=3.00,
        layer_b_min_score=0.82,
        start_sec=0,
        stop_sec=240,
        min_score=0.50,
        max_depth_participation=0.25,
        max_asset_exposure=35,
        max_total_exposure=300,
        hard_cutoff_seconds=60,
        min_trade_gap_seconds=2,
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
        self.layer_a_max_notional = min(self.max_order, max(self.layer_a_base_notional, float(layer_a_max_notional)))
        self.layer_a_min_score = float(layer_a_min_score)
        self.layer_b_base_notional = min(self.max_order, max(0.10, float(layer_b_base_notional)))
        self.layer_b_max_notional = min(self.max_order, max(self.layer_b_base_notional, float(layer_b_max_notional)))
        self.layer_b_min_score = float(layer_b_min_score)

        self.start_sec = float(start_sec)
        self.stop_sec = min(240.0, float(stop_sec))
        self.min_score = float(min_score)
        self.max_depth_participation = min(0.25, max(0.01, float(max_depth_participation)))
        self.hard_cutoff_seconds = max(self.HARD_CUTOFF_SECONDS, float(hard_cutoff_seconds))
        self.min_trade_gap_seconds = max(0.0, float(min_trade_gap_seconds))

        self._last_trade_at = None
        self._last_reset_at = None

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

        spread = max(0.0, ask - bid) if bid is not None else 0.02
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
            return min(points, key=lambda item: abs((now - item[0]) - seconds_ago))[1]

        p30 = nearest(30.0)
        p10 = nearest(10.0)
        momentum = ask - p30
        acceleration = (ask - p10) - (p10 - p30)
        return spread, momentum, acceleration

    def _base_score(self, ask, depth, spread, momentum, acceleration, elapsed):
        momentum_score = self._clamp(0.5 + momentum * 8.0)
        acceleration_score = self._clamp(0.5 + acceleration * 10.0)
        extremeness = self._clamp(abs(ask - 0.50) / 0.50)
        time_score = self._clamp((elapsed - self.start_sec) / max(1.0, self.stop_sec - self.start_sec))
        spread_score = self._clamp(1.0 - max(0.0, spread - 0.01) / 0.05)
        depth_dollars = max(0.0, float(depth)) * max(ask, 0.01)
        depth_score = self._clamp(depth_dollars / 20.0)
        return self._clamp(
            0.25 * momentum_score
            + 0.16 * acceleration_score
            + 0.18 * extremeness
            + 0.13 * time_score
            + 0.14 * depth_score
            + 0.14 * spread_score
        )

    def _regime_trigger_score(self, regime, base_score, momentum, acceleration, elapsed):
        score = base_score
        if regime == "CHEAP":
            score += 0.10 * self._clamp((-momentum + 0.002) / 0.04)
            score += 0.05 * self._clamp((-acceleration + 0.002) / 0.04)
            if elapsed >= 120:
                score += 0.025
        elif regime == "MID":
            # Intentionally neutral/intermediate. MID must remain observable.
            score += 0.08 * self._clamp(1.0 - abs(momentum - 0.002) / 0.045)
            score += 0.02 * self._clamp(1.0 - abs(acceleration) / 0.03)
        elif regime == "CORE":
            score += 0.12 * self._clamp((momentum - 0.006) / 0.04)
            score += 0.05 * self._clamp((acceleration - 0.000) / 0.03)
        elif regime == "HIGH":
            score += 0.15 * self._clamp((momentum - 0.006) / 0.03)
            score += 0.05 * self._clamp((acceleration + 0.002) / 0.02)
            if elapsed >= 150:
                score += 0.07
        return self._clamp(score)

    def _allowed(self, regime, score, momentum, acceleration, spread, depth, elapsed, entry_count):
        if depth <= 0 or spread > 0.05:
            return False

        if regime == "CHEAP":
            return (
                score >= max(self.layer_a_min_score, 0.49)
                and momentum <= 0.015
                and acceleration <= 0.020
            )

        if regime == "MID":
            return (
                score >= max(self.min_score, 0.56)
                and -0.020 <= momentum <= 0.035
                and -0.020 <= acceleration <= 0.025
                and spread <= 0.05
            )

        if regime == "CORE":
            # More selective than V6: CORE should represent real confirmation.
            return (
                score >= 0.79
                and momentum >= 0.012
                and spread <= 0.035
            )

        if regime == "HIGH":
            # Expensive trades need strong confirmation and later timing.
            return (
                score >= max(self.layer_b_min_score, 0.88)
                and momentum >= 0.012
                and elapsed >= 150
                and spread <= 0.030
            )

        return False

    def _size(self, regime, price, score, entry_count=0, elapsed_since_first_entry=0.0, add_on_allowed=True):
        p = self._clamp(price, 0.01, 0.995)
        s = self._clamp(score)
        n = max(0, int(entry_count))
        age = max(0.0, float(elapsed_since_first_entry))

        # Smooth entry-count curve, disabled when signal persistence is lost.
        if not add_on_allowed and n > 0:
            n_factor = 1.0
        else:
            n_factor = 1.0 + 0.045 * min(n, 4) + 0.015 * min(max(0, n - 4), 8)

        if regime == "CHEAP":
            x = self._clamp((p - 0.01) / 0.29)
            base = 0.30 + 0.28 * (x ** 0.65) * (0.85 + 0.15 * s)
            size = base * (0.95 + 0.10 * n_factor)
        elif regime == "MID":
            x = self._clamp((p - 0.30) / 0.40)
            base = 1.20 + 0.85 * (x ** 0.70) * (0.85 + 0.15 * s)
            size = base * (0.98 + 0.06 * n_factor)
        elif regime == "CORE":
            x = self._clamp((p - 0.70) / 0.20)
            base = 3.55 + 1.45 * (x ** 0.60) * (0.85 + 0.15 * s)
            size = base * (0.90 + 0.10 * n_factor)
        elif regime == "HIGH":
            x = self._clamp((p - 0.90) / 0.095)
            starter = 4.20 + 1.55 * (x ** 0.50) * (0.85 + 0.15 * s)
            if n == 0:
                size = starter
            else:
                size = starter + 1.55 * min(n, 4) + 0.45 * min(max(0, n - 4), 4)
                if age >= 90:
                    size += 0.30
        else:
            return 0.0

        return min(self.max_order, max(0.10, size))

    def _reset_allowed(self, thesis_side, thesis_price, candidate_side, candidate_price, elapsed, now, candidate_score, same_side_score):
        if not thesis_side or thesis_side == candidate_side:
            return True
        if thesis_price is None:
            return False
        if self._last_reset_at is not None and now - self._last_reset_at < 25.0:
            return False
        jump = abs(float(candidate_price) - float(thesis_price))
        # A reset must be a major probability move and the new side must be
        # convincingly stronger than the old side, not merely barely valid.
        if jump < 0.50:
            return False
        if candidate_score < same_side_score + 0.08:
            return False
        return float(elapsed) >= 75.0

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
        now = time.time() if now is None else float(now)
        elapsed = float(elapsed)

        if elapsed < self.start_sec or elapsed >= self.stop_sec:
            return None
        if 300.0 - elapsed <= self.hard_cutoff_seconds:
            return None
        if self._last_trade_at is not None and now - self._last_trade_at < self.min_trade_gap_seconds:
            return None

        remaining = min(
            self.max_market_exposure - float(current_exposure),
            self.max_asset_exposure - float(asset_exposure),
            self.max_total_exposure - float(total_exposure),
            float(available_cash),
            self.max_order,
        )
        if remaining < 0.10:
            return None

        candidates = []
        for side, ask, bid, depth, hist in (
            ("Up", up_ask, up_bid, up_depth, up_history),
            ("Down", down_ask, down_bid, down_depth, down_history),
        ):
            f = self._features(ask, bid, hist, now)
            if f is None:
                continue
            spread, momentum, acceleration = f
            ask = float(ask)
            depth = max(0.0, float(depth))
            regime = self._regime(ask)
            if regime is None:
                continue

            base = self._base_score(ask, depth, spread, momentum, acceleration, elapsed)
            score = self._regime_trigger_score(regime, base, momentum, acceleration, elapsed)

            if not self._allowed(regime, score, momentum, acceleration, spread, depth, elapsed, market_entry_count):
                continue

            direction_bias = {
                # CHEAP should win when weakness is real; a flat cheap quote
                # remains possible, but should not beat a strongly-confirmed
                # CORE/HIGH candidate solely because it is cheap.
                "CHEAP": (-momentum * 1.10) - (0.05 if momentum >= -0.002 else 0.0),
                "MID": -abs(momentum - 0.002) * 0.05,
                "CORE": momentum * 1.10,
                "HIGH": momentum * 1.35,
            }[regime]

            candidates.append({
                "side": side,
                "ask": ask,
                "depth": depth,
                "regime": regime,
                "score": score,
                "ranking": score + direction_bias,
                "momentum": momentum,
                "acceleration": acceleration,
                "spread": spread,
            })

        if not candidates:
            return None

        # Prefer staying on the existing thesis when both sides are viable.
        same = [c for c in candidates if not thesis_side or c["side"] == thesis_side]
        other = [c for c in candidates if thesis_side and c["side"] != thesis_side]
        same_best = max(same, key=lambda c: c["ranking"]) if same else None
        other_best = max(other, key=lambda c: c["ranking"]) if other else None

        if same_best is not None:
            best = same_best
            reset = False
        else:
            best = other_best
            reset = False

        if other_best is not None and thesis_side and other_best["side"] != thesis_side:
            same_score = same_best["score"] if same_best else 0.0
            if self._reset_allowed(
                thesis_side, thesis_price, other_best["side"], other_best["ask"],
                elapsed, now, other_best["score"], same_score
            ):
                best = other_best
                reset = True
            elif same_best is None:
                # No valid same-side entry means WAIT, not automatic reversal.
                return None

        # Once inside a market, stop scaling when the signal has materially
        # deteriorated versus a starter-level threshold.
        starter_floor = {
            "CHEAP": max(self.layer_a_min_score, 0.49),
            "MID": max(self.min_score, 0.56),
            "CORE": 0.79,
            "HIGH": max(self.layer_b_min_score, 0.88),
        }[best["regime"]]
        add_on_allowed = best["score"] >= starter_floor + 0.015

        desired = self._size(
            best["regime"],
            best["ask"],
            best["score"],
            market_entry_count,
            seconds_since_first_entry,
            add_on_allowed=add_on_allowed,
        )

        size = min(desired, remaining)
        depth_cap = best["depth"] * best["ask"] * self.max_depth_participation
        size = min(
            size,
            depth_cap,
            self.max_order,
            self.max_market_exposure - float(current_exposure),
            self.max_asset_exposure - float(asset_exposure),
            self.max_total_exposure - float(total_exposure),
            float(available_cash),
        )

        if size < 0.10:
            return None

        if reset:
            self._last_reset_at = now

        self._last_trade_at = now
        mode = "STARTER" if market_entry_count == 0 else "ADD_ON"
        if reset:
            mode = "RESET"
        if not add_on_allowed and market_entry_count > 0 and not reset:
            mode = "HOLD_CONFIRMATION"

        reason = (
            f"V7 regime={best['regime']} mode={mode} "
            f"entry_count={int(market_entry_count)} score={best['score']:.3f} "
            f"momentum={best['momentum']:+.4f} accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} depth={best['depth']:.2f} "
            f"elapsed={elapsed:.1f}s since_first={float(seconds_since_first_entry):.1f}s "
            f"seconds_left={300.0-elapsed:.1f} global_exposure={float(total_exposure):.2f} "
            f"global_cap={self.max_total_exposure:.2f} add_on_allowed={add_on_allowed} reset={reset}"
        )

        return Signal(
            best["side"],
            best["ask"],
            best["score"],
            round(size, 2),
            reason,
        )

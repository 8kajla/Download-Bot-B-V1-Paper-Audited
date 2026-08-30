
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
    """BOT B V4 — distribution + payoff research model.

    Design objective:
      - Reproduce the trader's broad execution shape without hard-coding
        a single historical hour.
      - Increase CHEAP and MID participation.
      - Reduce CORE over-selection.
      - Keep HIGH less frequent but materially larger.
      - Use the full $300 research exposure capacity.
      - Preserve BUY-only, repeated entries, depth protection and the
        existing 60-second market cutoff.

    This is a research model, not a claim of the trader's hidden trigger.
    """

    VERSION = "V4"

    CHEAP_MIN, CHEAP_MAX = 0.01, 0.30
    MID_MIN, MID_MAX = 0.30, 0.70
    CORE_MIN, CORE_MAX = 0.70, 0.90
    HIGH_MIN, HIGH_MAX = 0.90, 0.995

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_TOTAL_EXPOSURE = 300.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0

    # Structural prior from the longer matched trader sample.
    # We do not force exact quotas; this is used as a ranking prior.
    TARGETS = {
        "CHEAP": 0.476,
        "MID": 0.347,
        "CORE": 0.105,
        "HIGH": 0.072,
    }

    # V4 deliberately softens those targets so that one-hour noise does not
    # force the bot into artificial quotas.
    SOFT_TARGETS = {
        "CHEAP": 0.42,
        "MID": 0.34,
        "CORE": 0.16,
        "HIGH": 0.08,
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

        self.hard_cutoff_seconds = max(
            60.0,
            float(hard_cutoff_seconds),
        )

        self.min_trade_gap_seconds = max(
            0.0,
            float(min_trade_gap_seconds),
        )

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

        # Local decision counters. These are not persisted and are not used
        # to fake outcomes; they only control distribution within one run.
        self._regime_counts = {r: 0 for r in self.TARGETS}
        self._total_decisions = 0
        self._last_trade_at = None

    @staticmethod
    def _clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(x)))

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
                key=lambda x: abs((now - x[0]) - seconds),
            )[1]

        p30 = nearest(30)
        p10 = nearest(10)

        momentum = ask - p30
        acceleration = (
            (ask - p10)
            - (p10 - p30)
        )

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
        ms = self._clamp(
            0.5 + momentum * 8.0
        )
        acs = self._clamp(
            0.5 + acceleration * 10.0
        )
        extremeness = self._clamp(
            abs(ask - 0.5) / 0.5
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

        # Normalize depth against the maximum usable $10 paper order.
        depth_dollars = max(0.0, float(depth)) * max(
            ask,
            0.01,
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

        if score < self.min_score:
            return False

        if regime == "CHEAP":
            # Much broader than V3.3 to recover the trader's high CHEAP
            # participation. Do not require positive momentum.
            return (
                score >= max(
                    self.layer_a_min_score,
                    0.50,
                )
                and momentum >= -0.010
                and acceleration >= -0.012
                and spread <= 0.05
            )

        if regime == "MID":
            # Moderate gate: less restrictive than V3.3 because MID was
            # under-represented in the latest hour relative to the trader.
            return (
                score >= max(
                    self.min_score,
                    0.61,
                )
                and momentum >= -0.001
                and acceleration >= -0.004
                and spread <= 0.04
            )

        if regime == "CORE":
            # Core remains tradeable, but requires a meaningfully stronger
            # setup so it cannot dominate the distribution again.
            return (
                score >= 0.79
                and momentum >= 0.010
                and acceleration >= 0.004
                and spread <= 0.03
            )

        if regime == "HIGH":
            # High is intentionally selective. When it qualifies, it gets
            # a larger size.
            return (
                score >= max(
                    self.layer_b_min_score,
                    0.84,
                )
                and momentum >= 0.004
                and acceleration >= -0.001
                and spread <= 0.03
            )

        return False

    def _distribution_bias(self, regime):
        total = self._total_decisions

        if total < 10:
            return {
                "CHEAP": 0.10,
                "MID": 0.04,
                "CORE": -0.05,
                "HIGH": -0.01,
            }[regime]

        observed = (
            self._regime_counts[regime] / total
        )

        target = self.SOFT_TARGETS[regime]

        # Positive when under target, negative when over target.
        delta = target - observed

        # Keep the correction modest. This is a steering mechanism,
        # not a hard quota.
        return self._clamp(
            delta * 1.5,
            -0.16,
            0.16,
        )

    def _size(self, regime, price, score):
        p = self._clamp(
            price,
            0.01,
            0.995,
        )
        s = self._clamp(score)

        if regime == "CHEAP":
            # Small but less tiny than V3.3.
            x = self._clamp(
                (p - 0.01) / 0.29
            )
            return min(
                self.max_order,
                0.35
                + 0.90
                * (x ** 0.60)
                * (0.80 + 0.20 * s),
            )

        if regime == "MID":
            # Trader observed roughly $2-ish median behavior.
            x = self._clamp(
                (p - 0.30) / 0.40
            )
            return min(
                self.max_order,
                1.00
                + 2.60
                * (x ** 0.65)
                * (0.80 + 0.20 * s),
            )

        if regime == "CORE":
            # Fewer trades, but meaningful size.
            x = self._clamp(
                (p - 0.70) / 0.20
            )
            return min(
                self.max_order,
                6.00,
                2.75
                + 4.00
                * (x ** 0.65)
                * (0.82 + 0.18 * s),
            )

        if regime == "HIGH":
            # The trader's median was above our $10 ceiling. Strong HIGH
            # opportunities therefore saturate at $10 when allowed.
            x = self._clamp(
                (p - 0.90) / 0.095
            )
            return min(
                self.max_order,
                2.50
                + 8.50
                * (x ** 0.50)
                * (0.85 + 0.15 * s),
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

        # Honor the existing 2-second trade gap. This controls burst timing,
        # not the aggregate number of trades available over a full market.
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

            (
                spread,
                momentum,
                acceleration,
            ) = features

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

            # V4 has two components:
            #   1) signal quality
            #   2) distribution steering
            #
            # A regime that is under-represented becomes easier to select;
            # an over-represented regime needs a stronger signal.
            distribution_bias = self._distribution_bias(
                regime
            )

            candidates.append(
                {
                    "side": side,
                    "ask": ask,
                    "depth": depth,
                    "regime": regime,
                    "score": score,
                    "ranking": score + distribution_bias,
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "spread": spread,
                    "distribution_bias": distribution_bias,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: x["ranking"],
        )

        size = min(
            self._size(
                best["regime"],
                best["ask"],
                best["score"],
            ),
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

        self._regime_counts[
            best["regime"]
        ] += 1
        self._total_decisions += 1
        self._last_trade_at = now

        reason = (
            f"V4 regime={best['regime']} "
            f"score={best['score']:.3f} "
            f"momentum={best['momentum']:+.4f} "
            f"accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} "
            f"depth={best['depth']:.2f} "
            f"dist_bias={best['distribution_bias']:+.3f} "
            f"elapsed={elapsed:.1f}s "
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

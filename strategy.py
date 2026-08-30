
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
    """BOT B V3.3 — research replica.

    This version incorporates the strongest findings from the matched
    trader/V3 study:

      1) V3 was badly under-represented in CHEAP and over-represented in MID.
      2) V3 under-sized every regime, especially HIGH.
      3) The global research exposure ceiling is raised to $300 so the bot
         can continue taking opportunities without being blocked too early.

    The private trader trigger is unknown. The implementation therefore
    targets the observed execution distribution, not an invented trigger.
    """

    VERSION = "V3.3"

    CHEAP_MIN, CHEAP_MAX = 0.01, 0.30
    MID_MIN, MID_MAX = 0.30, 0.70
    CORE_MIN, CORE_MAX = 0.70, 0.90
    HIGH_MIN, HIGH_MAX = 0.90, 0.995

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_TOTAL_EXPOSURE = 300.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0

    # Observed target trade shares from the matched 8-hour comparison.
    # Reference/documentation constant only; not active adaptive logic.
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
        self.stop_sec = float(stop_sec)

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

        if not 0 < ask < 1:
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

                if 0 < p < 1:
                    pts.append((t, p))

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                pass

        if not pts:
            return spread, 0.0, 0.0

        def nearest(sec):
            return min(
                pts,
                key=lambda x: abs((now - x[0]) - sec),
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
            abs(ask - 0.50) / 0.50
        )

        ts = self._clamp(
            (elapsed - self.start_sec)
            / max(
                1.0,
                self.stop_sec - self.start_sec,
            )
        )

        ss = self._clamp(
            1.0
            - max(0.0, spread - 0.01)
            / 0.05
        )

        dr = max(
            1.0,
            self.max_order / max(ask, 0.01),
        )

        ds = self._clamp(
            float(depth) / dr
        )

        return self._clamp(
            0.30 * ms
            + 0.18 * acs
            + 0.18 * extremeness
            + 0.14 * ts
            + 0.12 * ds
            + 0.08 * ss
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

        # CHEAP is deliberately permissive.
        #
        # The matched data showed that V3 was dramatically
        # under-represented here while CHEAP was profitable
        # in the observed sample.
        if regime == "CHEAP":
            return (
                score
                >= min(
                    self.layer_a_min_score,
                    0.45,
                )
                and momentum >= -0.0030
                and acceleration >= -0.0045
                and spread <= 0.05
            )

        # MID is deliberately stricter because it was both
        # over-traded and the largest negative contributor.
        if regime == "MID":
            return (
                score >= max(
                    self.min_score,
                    0.70,
                )
                and momentum >= 0.0030
                and acceleration >= 0.0005
                and spread <= 0.035
            )

        if regime == "CORE":
            return (
                score >= 0.74
                and momentum >= 0.0040
                and acceleration >= 0.0010
                and spread <= 0.035
            )

        if regime == "HIGH":
            return (
                score
                >= max(
                    self.layer_b_min_score,
                    0.82,
                )
                and momentum >= 0.0010
                and acceleration >= -0.0005
                and spread <= 0.035
                and elapsed < self.stop_sec
            )

        return False

    def _size(
        self,
        regime,
        price,
        score,
    ):
        """Price/score sizing calibrated toward observed regime medians.

        The $10 research order ceiling remains authoritative.
        """

        p = self._clamp(
            price,
            0.01,
            0.995,
        )

        s = self._clamp(score)

        if regime == "CHEAP":
            # Target median around $0.81 while retaining
            # many sub-dollar fills.
            x = self._clamp(
                (p - 0.01) / 0.29
            )

            return min(
                self.max_order,
                0.30
                + 0.90
                * (x ** 0.65)
                * (0.75 + 0.25 * s),
            )

        if regime == "MID":
            # Target median around $2.02.
            x = self._clamp(
                (p - 0.30) / 0.40
            )

            return min(
                self.max_order,
                0.80
                + 2.40
                * (x ** 0.75)
                * (0.75 + 0.25 * s),
            )

        if regime == "CORE":
            # Target median around $5.03.
            #
            # Keep a $6 CORE sub-cap to prevent excessive
            # concentration in this regime.
            x = self._clamp(
                (p - 0.70) / 0.20
            )

            return min(
                self.max_order,
                6.00,
                2.50
                + 4.00
                * (x ** 0.70)
                * (0.80 + 0.20 * s),
            )

        if regime == "HIGH":
            # Reference trader median is approximately
            # $14.47, but our paper order ceiling is $10.
            #
            # Therefore strong HIGH signals can saturate
            # at the $10 order ceiling.
            x = self._clamp(
                (p - 0.90) / 0.095
            )

            return min(
                self.max_order,
                1.50
                + 8.50
                * (x ** 0.60)
                * (0.75 + 0.25 * s),
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

        # Trading window.
        if elapsed < self.start_sec:
            return None

        if elapsed >= self.stop_sec:
            return None

        # Final hard cutoff.
        if (
            max(
                0.0,
                300.0 - elapsed,
            )
            <= self.hard_cutoff_seconds
        ):
            return None

        # Determine remaining available capacity.
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

        for (
            side,
            ask,
            bid,
            depth,
            hist,
        ) in (
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
            depth = max(
                0.0,
                float(depth),
            )

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

            # Ranking bias:
            #
            # CHEAP receives a positive bias because it was
            # under-represented in V3.
            #
            # MID and CORE are penalized because they were
            # over-represented in V3.
            bias = {
                "CHEAP": 0.12,
                "MID": -0.08,
                "CORE": -0.12,
                "HIGH": -0.03,
            }[regime]

            candidates.append(
                {
                    "side": side,
                    "ask": ask,
                    "depth": depth,
                    "regime": regime,
                    "score": score,
                    "ranking": score + bias,
                    "momentum": momentum,
                    "acceleration": acceleration,
                    "spread": spread,
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

        reason = (
            f"V3.3 regime={best['regime']} "
            f"score={best['score']:.3f} "
            f"momentum={best['momentum']:+.4f} "
            f"accel={best['acceleration']:+.4f} "
            f"spread={best['spread']:.4f} "
            f"depth={best['depth']:.2f} "
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

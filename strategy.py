from __future__ import annotations

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


class CapitalFirstStrategy:
    """
    V10 hierarchical market x fine-price model.

    The model is deliberately structured around observed behavior:
    - separate logic for every coarse regime;
    - separate 90-95 and 95-100 behavior;
    - market-specific overlays for BTC/ETH/SOL/BNB;
    - price-conditioned capital;
    - entry-sequence-aware sizing;
    - burst cadence and side persistence;
    - passive best-bid execution proxy.

    Exact private trader triggers remain unknown.
    """

    VERSION = "V10_HIERARCHICAL_MARKET_SEGMENT"

    MARKET_NAMES = {"BTC", "ETH", "SOL", "BNB"}

    BAND_TABLE = (
        ("C00_05", 0.00, 0.05, "CHEAP"),
        ("C05_10", 0.05, 0.10, "CHEAP"),
        ("C10_15", 0.10, 0.15, "CHEAP"),
        ("C15_20", 0.15, 0.20, "CHEAP"),
        ("C20_30", 0.20, 0.30, "CHEAP"),
        ("M30_40", 0.30, 0.40, "MID"),
        ("M40_50", 0.40, 0.50, "MID"),
        ("M50_60", 0.50, 0.60, "MID"),
        ("M60_70", 0.60, 0.70, "MID"),
        ("R70_80", 0.70, 0.80, "CORE"),
        ("R80_90", 0.80, 0.90, "CORE"),
        ("H90_95", 0.90, 0.95, "HIGH"),
        ("H95_100", 0.95, 0.995, "HIGH"),
    )

    # Empirical mean/notional curve from the 84-day data's fine buckets.
    PRICE_CAPITAL_ANCHORS = (
        (0.025, 0.41),
        (0.075, 0.51),
        (0.125, 0.68),
        (0.175, 0.83),
        (0.25, 1.07),
        (0.35, 1.48),
        (0.45, 2.17),
        (0.55, 3.08),
        (0.65, 3.93),
        (0.75, 5.25),
        (0.85, 8.28),
        (0.925, 14.99),
        (0.975, 42.49),
    )

    # Each market gets independent overlays. These are deliberately modest:
    # they alter availability/quality requirements and target capital rather
    # than inventing a different hidden signal for each asset.
    MARKET_PROFILE = {
        "BTC": {
            "freq_weight": {"CHEAP": 1.00, "MID": 1.03, "CORE": 0.96, "HIGH": 1.00},
            "depth_mult": 1.00,
            "spread_mult": 1.00,
            "momentum_mult": 1.00,
            "capital_mult": 1.00,
            "reset_jump": 0.32,
        },
        "ETH": {
            "freq_weight": {"CHEAP": 1.06, "MID": 1.02, "CORE": 0.94, "HIGH": 0.98},
            "depth_mult": 0.90,
            "spread_mult": 1.05,
            "momentum_mult": 0.95,
            "capital_mult": 1.00,
            "reset_jump": 0.30,
        },
        "SOL": {
            "freq_weight": {"CHEAP": 1.15, "MID": 0.96, "CORE": 0.90, "HIGH": 0.94},
            "depth_mult": 1.10,
            "spread_mult": 0.95,
            "momentum_mult": 1.05,
            "capital_mult": 0.92,
            "reset_jump": 0.34,
        },
        "BNB": {
            "freq_weight": {"CHEAP": 1.08, "MID": 1.00, "CORE": 1.02, "HIGH": 0.90},
            "depth_mult": 1.05,
            "spread_mult": 1.00,
            "momentum_mult": 1.10,
            "capital_mult": 0.96,
            "reset_jump": 0.33,
        },
    }

    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_TOTAL_EXPOSURE = 300.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0

    DEFAULT_SLICE_CAP = {
        "CHEAP": 0.40,
        "MID": 0.85,
        "CORE": 2.50,
        "HIGH": 5.00,
    }

    def __init__(
        self,
        bankroll=1000,
        max_market_exposure=100,
        max_order=10,
        max_asset_exposure=35,
        max_total_exposure=300,
        start_sec=0,
        stop_sec=240,
        hard_cutoff_seconds=60,
        max_depth_participation=0.20,
        min_trade_gap_seconds=2.0,
        min_bid_depth=1.0,
        state_reset_jump=0.35,
        state_reset_cooldown=30.0,
        state_min_age=45.0,
        **_,
    ):
        self.bankroll = float(bankroll)
        self.max_market_exposure = min(float(max_market_exposure), self.HARD_MAX_MARKET_EXPOSURE)
        self.max_order = min(float(max_order), self.HARD_MAX_ORDER)
        self.max_asset_exposure = min(float(max_asset_exposure), self.HARD_MAX_ASSET_EXPOSURE)
        self.max_total_exposure = min(float(max_total_exposure), self.HARD_MAX_TOTAL_EXPOSURE)
        self.start_sec = max(0.0, float(start_sec))
        self.stop_sec = min(240.0, float(stop_sec))
        self.hard_cutoff_seconds = max(60.0, float(hard_cutoff_seconds))
        self.max_depth_participation = min(0.20, max(0.01, float(max_depth_participation)))
        self.min_trade_gap_seconds = max(0.0, float(min_trade_gap_seconds))
        self.min_bid_depth = max(0.0, float(min_bid_depth))
        self.state_reset_jump = max(0.20, float(state_reset_jump))
        self.state_reset_cooldown = max(0.0, float(state_reset_cooldown))
        self.state_min_age = max(0.0, float(state_min_age))
        self._last_trade_at = None
        self._last_reset_at = None

    @staticmethod
    def _clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(v)))

    @classmethod
    def normalize_market(cls, asset):
        x = str(asset or "").upper()
        for name in cls.MARKET_NAMES:
            if name in x:
                return name
        return "BTC"

    @classmethod
    def fine_band(cls, price):
        p = float(price)
        for name, lo, hi, regime in cls.BAND_TABLE:
            if lo <= p < hi:
                return name, regime
        if 0.995 <= p < 1:
            return "H95_100", "HIGH"
        return None, None

    @classmethod
    def _interpolate_capital(cls, price):
        p = float(price)
        a = cls.PRICE_CAPITAL_ANCHORS
        if p <= a[0][0]:
            return a[0][1]
        if p >= a[-1][0]:
            return a[-1][1]
        for (p0, c0), (p1, c1) in zip(a, a[1:]):
            if p0 <= p <= p1:
                w = ((p - p0) / max(1e-9, p1 - p0)) ** 0.92
                return c0 + (c1 - c0) * w
        return a[-1][1]

    def desired_capital(self, price, market=None, regime=None, entry_count=0):
        market = self.normalize_market(market)
        band, inferred = self.fine_band(price)
        regime = regime or inferred
        if not regime:
            return 0.0

        base = self._interpolate_capital(price)
        mult = self.MARKET_PROFILE[market]["capital_mult"]

        # Entry-sequence growth: starts near observed starter sizes and rises
        # without creating runaway HIGH ladders.
        if entry_count <= 0:
            seq_mult = 0.92
        elif entry_count == 1:
            seq_mult = 0.98
        elif entry_count <= 3:
            seq_mult = 1.00
        elif entry_count <= 7:
            seq_mult = 1.06
        else:
            seq_mult = 1.12

        # The highest 5c needs high aggregate capital in the trader's data,
        # but it is still gated by the separate HIGH trigger.
        value = base * mult * seq_mult
        return max(0.20, min(42.49 * mult, value))

    @staticmethod
    def _points(history, now):
        pts = []
        for x in history or []:
            try:
                if isinstance(x, dict):
                    t = float(x["ts"])
                    p = float(x.get("best_bid", x.get("mid")))
                else:
                    t = float(x[0])
                    p = float(x[1])
                if 0 < p < 1 and t <= now:
                    pts.append((t, p))
            except (TypeError, ValueError, KeyError, IndexError):
                continue
        return sorted(pts)

    @classmethod
    def _movement(cls, price, history, now):
        pts = cls._points(history, now)
        if not pts:
            return {"m1": 0.0, "m3": 0.0, "m5": 0.0, "m10": 0.0, "m30": 0.0, "accel": 0.0, "samples": 0}

        price = float(price)

        def at(sec):
            target = now - sec
            return min(pts, key=lambda x: abs(x[0] - target))[1]

        p1, p3, p5, p10, p30 = (at(s) for s in (1, 3, 5, 10, 30))
        return {
            "m1": price - p1,
            "m3": price - p3,
            "m5": price - p5,
            "m10": price - p10,
            "m30": price - p30,
            "accel": (price - p3) - (p3 - p10),
            "samples": len(pts),
        }

    def _book_quality(self, bid, ask, bid_depth, market, regime):
        bid = float(bid)
        depth = max(0.0, float(bid_depth or 0.0))
        if ask is None:
            spread = 0.0
        else:
            spread = max(0.0, float(ask) - bid)

        profile = self.MARKET_PROFILE[market]
        depth_score = 1.0 - 1.0 / (1.0 + depth / (8.0 * profile["depth_mult"]))
        max_spread = {
            "CHEAP": 0.055,
            "MID": 0.045,
            "CORE": 0.032,
            "HIGH": 0.020,
        }[regime] * profile["spread_mult"]

        spread_score = self._clamp(1.0 - spread / max(1e-6, max_spread))
        return 0.55 * depth_score + 0.45 * spread_score, spread, max_spread

    def _segment_check(self, market, regime, band, movement, same_side_fit, book_quality, depth, spread, max_spread, entry_count, burst_age):
        profile = self.MARKET_PROFILE[market]
        depth_need = {
            "CHEAP": 1.0 * profile["depth_mult"],
            "MID": 2.0 * profile["depth_mult"],
            "CORE": 3.0 * profile["depth_mult"],
            "HIGH": 6.0 * profile["depth_mult"],
        }[regime]

        if depth < depth_need:
            return None

        if spread > max_spread:
            return None

        # Four different decision families.
        if regime == "CHEAP":
            weakness = self._clamp((-movement["m10"] + 0.005) / 0.045)
            stability = self._clamp(1.0 - abs(movement["m1"]) / 0.025)
            # Avoid blindly averaging into acceleration-down collapses.
            crash_penalty = self._clamp((-movement["accel"] - 0.015) / 0.05)
            fit = 0.52 * weakness + 0.28 * stability + 0.20 * same_side_fit
            if crash_penalty > 0.70:
                return None
            rank = 0.52 * book_quality + 0.33 * fit + 0.15 * profile["freq_weight"]["CHEAP"] / 1.15
            path = "CHEAP_LIQUIDITY_WEAKNESS"

        elif regime == "MID":
            neutral = self._clamp(1.0 - abs(movement["m10"]) / 0.055)
            smooth = self._clamp(1.0 - abs(movement["accel"]) / 0.045)
            fit = 0.40 * neutral + 0.30 * smooth + 0.30 * same_side_fit
            rank = 0.50 * book_quality + 0.35 * fit + 0.15 * profile["freq_weight"]["MID"] / 1.15
            if fit < 0.35:
                return None
            path = "MID_STABLE_BOOK"

        elif regime == "CORE":
            strength = self._clamp((movement["m10"] + 0.002) / (0.035 * profile["momentum_mult"]))
            continuation = self._clamp((movement["m5"] + 0.001) / (0.025 * profile["momentum_mult"]))
            fit = 0.55 * strength + 0.25 * continuation + 0.20 * same_side_fit
            rank = 0.45 * book_quality + 0.43 * fit + 0.12 * profile["freq_weight"]["CORE"] / 1.15
            if fit < 0.42:
                return None
            path = "CORE_STRENGTH_CONTINUATION"

        else:
            # HIGH is the strictest path, split internally at 95c by requiring
            # progressively stronger evidence.
            strength = self._clamp((movement["m5"] + 0.002) / (0.025 * profile["momentum_mult"]))
            persistence = self._clamp(entry_count / 6.0)
            maturity = self._clamp(float(burst_age) / 120.0)
            fit = 0.56 * strength + 0.24 * same_side_fit + 0.10 * persistence + 0.10 * maturity

            if band == "H90_95":
                threshold = 0.70
                min_mom = 0.012
            else:
                threshold = 0.80
                min_mom = 0.020

            if (
                not same_side_fit
                or entry_count < 2
                or burst_age < self.state_min_age
                or movement["m5"] < min_mom
                or fit < threshold
            ):
                return None

            rank = 0.40 * book_quality + 0.49 * fit + 0.11 * profile["freq_weight"]["HIGH"] / 1.15
            path = "HIGH_ESTABLISHED_STRENGTH"

        return {
            "rank": self._clamp(rank),
            "fit": self._clamp(fit),
            "path": path,
        }

    def _can_reset(self, market, thesis_side, thesis_price, other, now, burst_age):
        if not thesis_side or thesis_side == other["side"]:
            return False
        if thesis_price is None:
            return True
        if burst_age < self.state_min_age:
            return False

        jump = self.MARKET_PROFILE[market]["reset_jump"]
        if self._last_reset_at is not None and now - self._last_reset_at < self.state_reset_cooldown:
            return False

        return abs(float(other["bid"]) - float(thesis_price)) >= jump and other["rank"] >= 0.70

    def _candidate(
        self,
        market,
        side,
        bid,
        ask,
        depth,
        history,
        now,
        thesis_side,
        entry_count,
        burst_age,
    ):
        if bid is None:
            return None

        try:
            bid = float(bid)
            ask = None if ask is None else float(ask)
            depth = max(0.0, float(depth or 0.0))
        except (TypeError, ValueError):
            return None

        if not 0 < bid < 1:
            return None

        band, regime = self.fine_band(bid)
        if not regime:
            return None

        movement = self._movement(bid, history, now)
        quality, spread, max_spread = self._book_quality(bid, ask, depth, market, regime)

        same_side_fit = (
            1.0 if thesis_side and side == thesis_side
            else 0.0 if thesis_side
            else 0.5
        )

        check = self._segment_check(
            market, regime, band, movement, same_side_fit,
            quality, depth, spread, max_spread, entry_count, burst_age
        )
        if not check:
            return None

        return {
            "market": market,
            "side": side,
            "bid": bid,
            "ask": ask,
            "depth": depth,
            "spread": spread,
            "max_spread": max_spread,
            "band": band,
            "regime": regime,
            "movement": movement,
            "book_quality": quality,
            "state_fit": check["fit"],
            "rank": check["rank"],
            "path": check["path"],
            "target_capital": self.desired_capital(
                bid, market=market, regime=regime, entry_count=entry_count
            ),
        }

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
        up_depth=0,
        down_depth=0,
        now=None,
        asset_exposure=0,
        total_exposure=0,
        market_entry_count=0,
        seconds_since_first_entry=0,
        thesis_side=None,
        thesis_price=None,
        asset=None,
        market=None,
    ):
        now = time.time() if now is None else float(now)
        elapsed = float(elapsed)
        market = self.normalize_market(market or asset)

        if elapsed < self.start_sec or elapsed >= self.stop_sec:
            return None

        # Hard stop before the final 60 seconds.
        if self.stop_sec - elapsed <= self.hard_cutoff_seconds:
            return None

        if (
            self._last_trade_at is not None
            and now - self._last_trade_at < self.min_trade_gap_seconds
        ):
            return None

        candidates = [
            c for c in (
                self._candidate(
                    market, "Up", up_bid, up_ask, up_depth, up_history,
                    now, thesis_side, market_entry_count, seconds_since_first_entry
                ),
                self._candidate(
                    market, "Down", down_bid, down_ask, down_depth, down_history,
                    now, thesis_side, market_entry_count, seconds_since_first_entry
                ),
            ) if c
        ]

        if not candidates:
            return None

        same = [c for c in candidates if thesis_side and c["side"] == thesis_side]
        other = [c for c in candidates if thesis_side and c["side"] != thesis_side]

        same_best = max(same, key=lambda c: c["rank"]) if same else None
        other_best = max(other, key=lambda c: c["rank"]) if other else None
        reset = False

        if same_best:
            best = same_best
            if other_best and self._can_reset(
                market, thesis_side, thesis_price, other_best, now, float(seconds_since_first_entry)
            ):
                best = other_best
                reset = True
        elif thesis_side:
            if not other_best:
                return None
            if not self._can_reset(
                market, thesis_side, thesis_price, other_best, now, float(seconds_since_first_entry)
            ):
                return None
            best = other_best
            reset = True
        else:
            # Market-specific frequency prior prevents HIGH from dominating
            # simply because it has a large capital target.
            best = max(
                candidates,
                key=lambda c: c["rank"]
                + 0.06 * math.log(max(0.10, self.MARKET_PROFILE[market]["freq_weight"][c["regime"]]))
            )

        target = self.desired_capital(
            best["bid"],
            market=market,
            regime=best["regime"],
            entry_count=market_entry_count,
        )

        starter_target = target * (
            0.88 if market_entry_count == 0 and best["regime"] != "HIGH"
            else 0.78 if market_entry_count == 0
            else 1.0
        )

        remaining_target = (
            starter_target
            if market_entry_count == 0
            else max(0.0, target - max(0.0, float(current_exposure)))
        )

        if remaining_target < 0.10:
            return None

        room = min(
            remaining_target,
            self.max_order,
            max(0.0, self.max_market_exposure - float(current_exposure)),
            max(0.0, self.max_asset_exposure - float(asset_exposure)),
            max(0.0, self.max_total_exposure - float(total_exposure)),
            max(0.0, float(available_cash)),
            max(0.0, best["depth"] * best["bid"] * self.max_depth_participation),
            self.DEFAULT_SLICE_CAP[best["regime"]],
        )

        if room < 0.10:
            return None

        # Additional anti-averaging guard in CHEAP: if the same market is
        # already deeply accumulated and the price keeps falling rapidly,
        # stop adding rather than mechanically doubling down.
        if (
            best["regime"] == "CHEAP"
            and market_entry_count >= 8
            and best["movement"]["m10"] < -0.04
        ):
            return None

        if reset:
            self._last_reset_at = now

        self._last_trade_at = now

        mode = (
            "STATE_RESET"
            if reset
            else "STARTER"
            if market_entry_count == 0
            else "ADD_ON"
        )

        mv = best["movement"]

        reason = (
            f"V10 market={market} band={best['band']} regime={best['regime']} "
            f"path={best['path']} mode={mode} passive=bid "
            f"target_capital=${target:.2f} current_exposure=${float(current_exposure):.2f} "
            f"remaining_target=${remaining_target:.2f} entry_count={int(market_entry_count)} "
            f"burst_age={float(seconds_since_first_entry):.1f}s "
            f"bid={best['bid']:.4f} ask={best['ask'] if best['ask'] is not None else 0:.4f} "
            f"spread={best['spread']:.4f}/{best['max_spread']:.4f} depth={best['depth']:.2f} "
            f"book_quality={best['book_quality']:.3f} state_fit={best['state_fit']:.3f} "
            f"m1={mv['m1']:+.4f} m3={mv['m3']:+.4f} m5={mv['m5']:+.4f} "
            f"m10={mv['m10']:+.4f} m30={mv['m30']:+.4f} accel={mv['accel']:+.4f} "
            f"elapsed={elapsed:.1f}s left={self.stop_sec-elapsed:.1f}s reset={reset}"
        )

        return Signal(
            side=best["side"],
            price=best["bid"],
            score=self._clamp(best["rank"]),
            notional=round(room, 2),
            reason=reason,
        )

    def size(self, price, regime=None, market=None, entry_count=0, **_):
        return self.desired_capital(
            price, market=market, regime=regime, entry_count=entry_count
        )

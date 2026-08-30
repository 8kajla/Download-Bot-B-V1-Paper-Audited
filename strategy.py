from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Signal:
    side: str
    price: float
    score: float
    notional: float
    reason: str


class CapitalFirstStrategy:
    """
    Evidence-constrained behavioral replica.

    This module deliberately separates:
      - CONFIRMED behavioral observations
      - execution/safety constraints
      - UNKNOWN private trigger mechanics

    No synthetic momentum threshold, composite alpha score, or invented
    state-reset rule is used.

    The directional preference is based only on the measured price-trajectory
    distribution by coarse regime:
      CHEAP: buy while falling 56.4%, rising 20.0%, flat 23.6%
      MID:   buy while falling 45.0%, rising 41.0%, flat 14.0%
      CORE:  buy while rising 54.2%, falling 33.4%, flat 12.4%
      HIGH:  buy while rising 61.0%, falling 17.1%, flat 21.9%

    These are used as empirical directional likelihoods, not as proof of a
    causal trading trigger.
    """

    VERSION = "V13_50PCT_SIZE_ONLY"

    BANDS: Tuple[Tuple[str, float, float, str], ...] = (
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
        ("H95_100", 0.95, 1.00, "HIGH"),
    )

    # Existing empirically-derived smooth price/capital curve retained.
    # Entry-count multipliers below are now derived directly from the verified
    # combined 12,337-trade entry-state medians.
    BASE_CAPITAL: Dict[str, float] = {
        "C00_05": 0.41,
        "C05_10": 0.51,
        "C10_15": 0.68,
        "C15_20": 0.83,
        "C20_30": 0.95,
        "M30_40": 1.48,
        "M40_50": 2.17,
        "M50_60": 3.08,
        "M60_70": 3.93,
        "R70_80": 5.25,
        "R80_90": 8.28,
        "H90_95": 14.99,
        "H95_100": 30.78,
    }

    # Confirmed median notional by regime and entry state:
    # (first, entries 2-3, entries 4+).
    ENTRY_MEDIANS = {
        "CHEAP": (0.40, 0.52, 0.61),
        "MID":   (1.92, 1.92, 2.07),
        "CORE":  (4.34, 4.61, 5.03),
        "HIGH":  (5.70, 12.96, 13.29),
    }

    # Confirmed combined-sample trade-count priors by asset x coarse regime.
    MARKET_REGIME_WEIGHT = {
        "BTC": {"CHEAP": 0.39, "MID": 0.40, "CORE": 0.12, "HIGH": 0.09},
        "ETH": {"CHEAP": 0.54, "MID": 0.28, "CORE": 0.09, "HIGH": 0.08},
        "SOL": {"CHEAP": 0.65, "MID": 0.22, "CORE": 0.07, "HIGH": 0.05},
        "BNB": {"CHEAP": 0.57, "MID": 0.27, "CORE": 0.11, "HIGH": 0.05},
    }

    # Empirical price-trajectory shares from 9,651 consecutive same-market
    # pairs. Keys are movement classes observed immediately before a buy.
    TRAJECTORY_SHARE = {
        "CHEAP": {"rising": 0.200, "falling": 0.564, "flat": 0.236},
        "MID":   {"rising": 0.410, "falling": 0.450, "flat": 0.140},
        "CORE":  {"rising": 0.542, "falling": 0.334, "flat": 0.124},
        "HIGH":  {"rising": 0.610, "falling": 0.171, "flat": 0.219},
    }

    # Execution-only checks. These are not claimed to be trader rules.
    MARKET_CHECKS = {
        "BTC": {
            "depth": {"CHEAP": 1.0, "MID": 2.0, "CORE": 4.0, "HIGH": 8.0},
            "spread": {"CHEAP": 0.060, "MID": 0.050, "CORE": 0.035, "HIGH": 0.025},
        },
        "ETH": {
            "depth": {"CHEAP": 1.0, "MID": 2.0, "CORE": 3.0, "HIGH": 6.0},
            "spread": {"CHEAP": 0.065, "MID": 0.055, "CORE": 0.040, "HIGH": 0.027},
        },
        "SOL": {
            "depth": {"CHEAP": 1.0, "MID": 2.0, "CORE": 4.0, "HIGH": 8.0},
            "spread": {"CHEAP": 0.060, "MID": 0.050, "CORE": 0.035, "HIGH": 0.025},
        },
        "BNB": {
            "depth": {"CHEAP": 1.0, "MID": 2.0, "CORE": 4.0, "HIGH": 8.0},
            "spread": {"CHEAP": 0.060, "MID": 0.050, "CORE": 0.035, "HIGH": 0.025},
        },
    }

    SIZE_SCALE = 0.50
    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET = 100.0
    HARD_MAX_ASSET = 35.0
    HARD_MAX_TOTAL = 300.0
    HARD_CUTOFF = 60.0

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
        max_depth_participation=0.25,
        min_trade_gap_seconds=0,
        min_bid_depth=1,
        **_,
    ):
        self.bankroll = float(bankroll)
        self.max_market_exposure = min(float(max_market_exposure), self.HARD_MAX_MARKET)
        self.max_order = min(float(max_order), self.HARD_MAX_ORDER)
        self.max_asset_exposure = min(float(max_asset_exposure), self.HARD_MAX_ASSET)
        self.max_total_exposure = min(float(max_total_exposure), self.HARD_MAX_TOTAL)
        self.start_sec = max(0.0, float(start_sec))
        self.stop_sec = min(300.0, float(stop_sec))
        self.hard_cutoff_seconds = max(60.0, float(hard_cutoff_seconds))
        self.max_depth_participation = min(0.25, max(0.01, float(max_depth_participation)))
        # Infrastructure rate limit only; 0 means no behavioral distortion.
        self.min_trade_gap_seconds = max(0.0, float(min_trade_gap_seconds))
        self.min_bid_depth = max(0.0, float(min_bid_depth))
        self._last_trade_at: Optional[float] = None

    @staticmethod
    def normalize_market(x) -> str:
        s = str(x or "").upper()
        for m in ("BTC", "ETH", "SOL", "BNB"):
            if m in s:
                return m
        return "BTC"

    @classmethod
    def fine_band(cls, price):
        p = float(price)
        for band, lo, hi, regime in cls.BANDS:
            if lo <= p < hi:
                return band, regime
        if p == 1.0:
            return "H95_100", "HIGH"
        return None, None

    @classmethod
    def capital_target(cls, price, market="BTC", entry_count=0):
        band, regime = cls.fine_band(price)
        if not band:
            return 0.0

        base = float(cls.BASE_CAPITAL[band])
        first, second_third, fourth_plus = cls.ENTRY_MEDIANS[regime]

        # Entry-state ratios are data-derived, not hand-chosen.
        if entry_count <= 0:
            ratio = 1.0
        elif entry_count <= 3:
            ratio = second_third / first
        else:
            ratio = fourth_plus / first

        return max(0.10, cls.SIZE_SCALE * max(0.20, base * ratio))

    @classmethod
    def entry_target(cls, price, market="BTC", entry_count=0):
        return cls.capital_target(price, market, entry_count)

    def desired_capital(self, price, regime=None, market="BTC", entry_count=0):
        return self.entry_target(price, market, entry_count)

    @staticmethod
    def _points(history: Iterable) -> List[Tuple[float, float]]:
        out = []
        for item in history or []:
            try:
                if isinstance(item, dict):
                    ts = float(item["ts"])
                    price = float(item.get("best_bid", item.get("mid")))
                else:
                    ts, price = float(item[0]), float(item[1])
                if 0.0 < price < 1.0:
                    out.append((ts, price))
            except (TypeError, ValueError, KeyError, IndexError):
                continue
        return sorted(out)

    @classmethod
    def movement(cls, price, history, now):
        pts = cls._points(history)
        result = {}
        for seconds in (1, 3, 5, 10, 30):
            prior = [p for t, p in pts if t <= float(now) - seconds]
            result[f"m{seconds}"] = float(price) - prior[-1] if prior else 0.0
        return result

    @staticmethod
    def _trajectory_class(delta: float) -> str:
        if delta > 0:
            return "rising"
        if delta < 0:
            return "falling"
        return "flat"

    def _book_ok(self, market, regime, bid, ask, depth):
        m = self.normalize_market(market)
        req = max(self.min_bid_depth, self.MARKET_CHECKS[m]["depth"][regime])
        if float(depth or 0.0) < req:
            return False, 0.0

        spread = 0.0 if ask is None else max(0.0, float(ask) - float(bid))
        allowed = self.MARKET_CHECKS[m]["spread"][regime]
        return spread <= allowed, spread

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
        entries,
        burst_age,
    ):
        if bid is None:
            return None
        try:
            bid = float(bid)
            depth = max(0.0, float(depth or 0.0))
            ask = None if ask is None else float(ask)
        except (TypeError, ValueError):
            return None

        if not 0.0 < bid < 1.0:
            return None

        band, regime = self.fine_band(bid)
        if not regime:
            return None

        book_ok, spread = self._book_ok(market, regime, bid, ask, depth)
        if not book_ok:
            return None

        movement = self.movement(bid, history, now)
        trajectory = self._trajectory_class(movement["m5"])
        likelihood = self.TRAJECTORY_SHARE[regime][trajectory]

        same_side = bool(thesis_side and side == thesis_side)

        return {
            "market": self.normalize_market(market),
            "side": side,
            "bid": bid,
            "ask": ask,
            "depth": depth,
            "spread": spread,
            "band": band,
            "regime": regime,
            "trajectory": trajectory,
            "trajectory_likelihood": likelihood,
            "same_side": same_side,
            "target": self.entry_target(bid, market, entries),
            "movement": movement,
            "entries": int(entries),
            "burst_age": float(burst_age),
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
        m = self.normalize_market(market or asset)

        if elapsed < self.start_sec or elapsed >= self.stop_sec:
            return None
        if self.stop_sec - elapsed <= self.hard_cutoff_seconds:
            return None
        if self._last_trade_at is not None and self.min_trade_gap_seconds:
            if now - self._last_trade_at < self.min_trade_gap_seconds:
                return None

        candidates = [
            c for c in (
                self._candidate(
                    m, "Up", up_bid, up_ask, up_depth, up_history, now,
                    thesis_side, market_entry_count, seconds_since_first_entry
                ),
                self._candidate(
                    m, "Down", down_bid, down_ask, down_depth, down_history, now,
                    thesis_side, market_entry_count, seconds_since_first_entry
                ),
            )
            if c is not None
        ]
        if not candidates:
            return None

        # Confirmed side persistence is implemented as a preference, not an
        # invented jump/cooldown reset rule.
        same_side = [c for c in candidates if c["same_side"]]
        if thesis_side and same_side:
            best = max(
                same_side,
                key=lambda c: (c["trajectory_likelihood"], c["depth"], -c["spread"])
            )
        else:
            # With no known thesis, choose using only the measured trajectory
            # distribution, then use execution quality as a tie-breaker.
            best = max(
                candidates,
                key=lambda c: (c["trajectory_likelihood"], c["depth"], -c["spread"])
            )

        target = float(best["target"])

        # Individual entry size is now conditioned directly on entry state;
        # no synthetic "remaining target" or arbitrary slice cap is applied.
        room = min(
            target,
            self.max_order,
            max(0.0, self.max_market_exposure - float(current_exposure)),
            max(0.0, self.max_asset_exposure - float(asset_exposure)),
            max(0.0, self.max_total_exposure - float(total_exposure)),
            max(0.0, float(available_cash)),
            max(
                0.0,
                float(best["depth"]) * float(best["bid"]) * self.max_depth_participation,
            ),
        )
        if room < 0.10:
            return None

        self._last_trade_at = now
        mv = best["movement"]

        reason = (
            f"V13 market={m} band={best['band']} regime={best['regime']} "
            f"behavior=EMPIRICAL_TRAJECTORY preference="
            f"{best['trajectory']} likelihood={best['trajectory_likelihood']:.3f} "
            f"side_persistence={'same' if best['same_side'] else 'no_thesis'} "
            f"passive=bid target=${target:.2f} entry_count={market_entry_count} "
            f"burst_age={float(seconds_since_first_entry):.1f}s "
            f"bid={best['bid']:.4f} "
            f"ask={best['ask'] if best['ask'] is not None else 0:.4f} "
            f"spread={best['spread']:.4f} depth={best['depth']:.2f} "
            f"m1={mv['m1']:+.4f} m3={mv['m3']:+.4f} m5={mv['m5']:+.4f} "
            f"m10={mv['m10']:+.4f} m30={mv['m30']:+.4f} "
            f"elapsed={elapsed:.1f}s left={self.stop_sec-elapsed:.1f}s"
        )

        return Signal(
            side=best["side"],
            price=best["bid"],
            score=best["trajectory_likelihood"],
            notional=round(room, 2),
            reason=reason,
        )

    def size(self, price, regime=None, market="BTC", entry_count=0, **_):
        return self.entry_target(price, market, entry_count)

from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    side: str
    price: float
    score: float
    notional: float
    reason: str


class ConvergenceStrategy:
    """Bot B behavioral replica.

    The model deliberately uses observable market microstructure only:
    price level, short-term momentum/acceleration, time remaining and
    executable depth. It does not read or copy the reference wallet.
    """

    def __init__(self, bankroll=1000, max_market_exposure=25, max_order=10,
                 min_price=.10, strong_price=.82, late_price=.90,
                 start_sec=90, aggressive_sec=180, stop_sec=240,
                 min_score=.50, max_depth_participation=.25, max_asset_exposure=50):
        self.bankroll = float(bankroll)
        self.max_market_exposure = float(max_market_exposure)
        self.max_order = float(max_order)
        self.min_price = float(min_price)
        self.strong_price = float(strong_price)
        self.late_price = float(late_price)
        self.start_sec = float(start_sec)
        self.aggressive_sec = float(aggressive_sec)
        self.stop_sec = float(stop_sec)
        self.min_score = float(min_score)
        self.max_depth_participation = float(max_depth_participation)
        self.max_asset_exposure = float(max_asset_exposure)

    @staticmethod
    def _features(ask, bid, hist, now):
        if ask is None or not 0 < ask < 1:
            return None
        spread = max(0.0, ask - bid) if bid is not None else 0.02
        hist = [(float(t), float(p)) for t, p in (hist or []) if 0 < float(p) < 1]
        momentum = 0.0
        acceleration = 0.0
        if hist:
            # Compare current ask with observations roughly 30s and 10s back.
            p30 = min(hist, key=lambda x: abs((now - x[0]) - 30))[1]
            p10 = min(hist, key=lambda x: abs((now - x[0]) - 10))[1]
            momentum = ask - p30
            acceleration = (ask - p10) - (p10 - p30)
        return spread, momentum, acceleration

    def decide(self, elapsed, up_ask, down_ask, up_bid, down_bid,
               up_history, down_history, current_exposure, available_cash,
               up_depth=0.0, down_depth=0.0, now=None, asset_exposure=0.0):
        if now is None:
            import time
            now = time.time()
        if not (self.start_sec <= elapsed < self.stop_sec):
            return None
        remaining_budget = min(
            self.max_market_exposure - float(current_exposure),
            self.max_asset_exposure - float(asset_exposure),
        )
        if remaining_budget <= 1e-9 or available_cash <= 1e-9:
            return None

        candidates = []
        for side, ask, bid, depth, hist in (
            ('Up', up_ask, up_bid, up_depth, up_history),
            ('Down', down_ask, down_bid, down_depth, down_history),
        ):
            f = self._features(ask, bid, hist, now)
            if f is None or ask < self.min_price:
                continue
            spread, momentum, acceleration = f

            # Positive momentum/acceleration are the primary convergence evidence.
            momentum_score = max(0.0, min(1.0, 0.5 + momentum * 8.0))
            accel_score = max(0.0, min(1.0, 0.5 + acceleration * 10.0))

            # Price is useful as a confirmation, not a hard directional rule.
            price_score = max(0.0, min(1.0, (ask - 0.50) / 0.50))

            # Time score rises as the market approaches the observed active window,
            # but the final stop prevents resolution gambling.
            span = max(1.0, self.stop_sec - self.start_sec)
            time_score = max(0.0, min(1.0, (elapsed - self.start_sec) / span))

            spread_score = max(0.0, min(1.0, 1.0 - max(0.0, spread - 0.01) / 0.05))
            depth_score = max(0.0, min(1.0, float(depth) / max(1.0, self.max_order / max(ask, 0.01))))

            score = (
                0.30 * momentum_score
                + 0.18 * accel_score
                + 0.18 * price_score
                + 0.14 * time_score
                + 0.12 * depth_score
                + 0.08 * spread_score
            )

            # Strong late convergence receives an additional boost, matching the
            # historical fingerprint without making price alone sufficient.
            if ask >= self.late_price and elapsed >= self.aggressive_sec:
                score += 0.08

            candidates.append({
                'score': score,
                'side': side,
                'ask': ask,
                'depth': max(0.0, float(depth)),
                'momentum': momentum,
                'acceleration': acceleration,
                'spread': spread,
                'price_score': price_score,
                'time_score': time_score,
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x['score'])
        if best['score'] < self.min_score:
            return None

        ask = best['ask']
        if ask >= self.late_price and elapsed >= self.aggressive_sec:
            base = self.max_order
            regime = 'late-convergence'
        elif ask >= self.strong_price or best['score'] >= .72:
            base = min(self.max_order, 5.0)
            regime = 'strong-convergence'
        else:
            base = min(self.max_order, 2.0)
            regime = 'probe-convergence'

        # Do not consume more than a fraction of visible executable depth.
        depth_cap = best['depth'] * ask * self.max_depth_participation
        notional = min(
            self.max_order,
            base,
            remaining_budget,
            float(available_cash),
            depth_cap,
        )
        if notional <= 0:
            return None

        reason = (
            f'{regime} score={best["score"]:.3f} '
            f'momentum={best["momentum"]:+.4f} '
            f'accel={best["acceleration"]:+.4f} '
            f'spread={best["spread"]:.4f} '
            f'depth={best["depth"]:.2f}'
        )
        return Signal(best['side'], ask, best['score'], round(notional, 2), reason)

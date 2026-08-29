import time
import tempfile
from pathlib import Path

from strategy import ConvergenceStrategy
from paper_ledger import PaperLedger
from research_logger import ResearchLogger


def make_strategy(**kwargs):
    kwargs.setdefault("max_order", 10)
    return ConvergenceStrategy(**kwargs)


def history(price, seconds_ago=30):
    return [(time.time() - seconds_ago, price)]


def decide(strategy, up=0.20, down=0.80, ub=0.19, db=0.79, up_depth=100, down_depth=100, elapsed=150, **kwargs):
    return strategy.decide(
        elapsed,
        up,
        down,
        ub,
        db,
        history(up),
        history(down),
        0,
        1000,
        up_depth=up_depth,
        down_depth=down_depth,
        **kwargs,
    )


# ------------------------- basic behavior -------------------------

def test_valid_cheap_signal():
    s = make_strategy()
    signal = decide(s, up=0.20, down=0.80, ub=0.19, db=0.79)
    assert signal is not None
    assert signal.side == "Up"
    assert signal.price == 0.20
    assert signal.reason.startswith("V3.1 ")
    assert "regime=CHEAP" in signal.reason
    assert signal.notional >= 0.10
    assert signal.notional <= 1.0


def test_buy_only_signal_model():
    s = make_strategy()
    # The strategy only emits Up/Down BUY candidates; no SELL field exists.
    signal = decide(s, up=0.20, down=0.80)
    assert signal is None or signal.side in {"Up", "Down"}


def test_invalid_price_range_rejected():
    s = make_strategy()
    signal = decide(s, up=0.9999, down=0.0001, ub=0.999, db=0.00001)
    assert signal is None


# ------------------------- empirical sizing -------------------------

def test_reference_size_curve_is_monotonic():
    s = make_strategy()
    prices = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95]
    sizes = [s._reference_size(p) for p in prices]
    assert all(b >= a for a, b in zip(sizes, sizes[1:]))


def test_reference_size_curve_reaches_research_cap_near_95c():
    s = make_strategy(max_order=10)
    assert s._reference_size(0.95) <= 10.0 + 1e-9
    assert s._reference_size(0.95) > s._reference_size(0.85)


def test_high_sizing_is_no_longer_flat_3_dollars():
    s = make_strategy(max_order=10)
    low = s._size("HIGH", 0.90, 0.90)
    high = s._size("HIGH", 0.95, 0.90)
    assert high > low
    assert high <= 10


def test_sizing_never_exceeds_order_limit():
    s = make_strategy(max_order=2)
    for regime, price in (("CHEAP", 0.20), ("MID", 0.50), ("CORE", 0.80), ("HIGH", 0.95)):
        assert s._size(regime, price, 0.95) <= 2.0


def test_score_multiplier_does_not_dominate_price_curve():
    s = make_strategy(max_order=10)
    low_score = s._size("HIGH", 0.90, 0.50)
    high_score = s._size("HIGH", 0.90, 1.00)
    assert high_score >= low_score
    assert high_score / low_score < 1.4


# ------------------------- exposure/cash -------------------------

def test_cash_is_upper_bound():
    s = make_strategy()
    signal = s.decide(150, 0.95, 0.05, 0.94, 0.04, history(0.95), history(0.05), 0, 0.20, up_depth=100, down_depth=100)
    assert signal is None or signal.notional <= 0.20 + 1e-9


def test_market_exposure_limit():
    s = make_strategy(max_market_exposure=100)
    signal = decide(s, current_exposure=99.95) if False else s.decide(150, 0.20, 0.80, 0.19, 0.79, history(0.20), history(0.80), 99.95, 1000, up_depth=100, down_depth=100)
    assert signal is None or signal.notional <= 0.05 + 1e-9


def test_asset_exposure_limit():
    s = make_strategy(max_asset_exposure=35)
    signal = s.decide(150, 0.20, 0.80, 0.19, 0.79, history(0.20), history(0.80), 0, 1000, up_depth=100, down_depth=100, asset_exposure=34.95)
    assert signal is None or signal.notional <= 0.05 + 1e-9


def test_total_exposure_limit():
    s = make_strategy(max_total_exposure=100)
    signal = s.decide(150, 0.20, 0.80, 0.19, 0.79, history(0.20), history(0.80), 0, 1000, up_depth=100, down_depth=100, total_exposure=99.95)
    assert signal is None or signal.notional <= 0.05 + 1e-9


def test_constructor_hard_caps():
    s = ConvergenceStrategy(max_order=50, max_market_exposure=500, max_asset_exposure=500, max_total_exposure=500)
    assert s.max_order == 10.0
    assert s.max_market_exposure == 100.0
    assert s.max_asset_exposure == 35.0
    assert s.max_total_exposure == 100.0


# ------------------------- depth -------------------------

def test_zero_depth_rejects_signal():
    s = make_strategy()
    signal = s.decide(150, 0.20, 0.80, 0.19, 0.79, history(0.20), history(0.80), 0, 1000, up_depth=0, down_depth=0)
    assert signal is None


def test_depth_participation_is_respected():
    s = make_strategy(max_depth_participation=0.25)
    signal = s.decide(150, 0.20, 0.80, 0.19, 0.79, history(0.20), history(0.80), 0, 1000, up_depth=2, down_depth=0)
    if signal is not None:
        assert signal.notional <= 2 * signal.price * 0.25 + 0.01


# ------------------------- score/gating -------------------------

def test_min_score_is_honored():
    s = make_strategy(min_score=0.99)
    signal = s.decide(150, 0.80, 0.20, 0.79, 0.19, history(0.80), history(0.20), 0, 1000, up_depth=100, down_depth=0)
    assert signal is None


def test_high_requires_strong_score():
    s = make_strategy()
    signal = s.decide(150, 0.91, 0.49, 0.90, 0.48, history(0.89), history(0.49), 0, 1000, up_depth=100, down_depth=0)
    assert signal is None


def test_high_requires_positive_momentum():
    s = make_strategy()
    signal = s.decide(150, 0.95, 0.50, 0.94, 0.49, history(0.99), history(0.50), 0, 1000, up_depth=100, down_depth=0)
    assert signal is None


# ------------------------- timing -------------------------

def test_before_start_is_blocked():
    s = make_strategy(start_sec=10)
    assert decide(s, elapsed=5) is None


def test_stop_second_is_blocked():
    s = make_strategy(stop_sec=240)
    assert decide(s, elapsed=240) is None


def test_final_minute_is_blocked():
    s = make_strategy(hard_cutoff_seconds=60)
    assert decide(s, elapsed=240) is None
    assert decide(s, elapsed=241) is None


def test_just_before_cutoff_can_still_signal():
    s = make_strategy()
    signal = decide(s, elapsed=239)
    assert signal is not None


# ------------------------- resolution regression -------------------------

def _market():
    return {
        "id": "m1", "condition": "c1", "slug": "btc-updown-5m-test",
        "asset": "BTC", "market": "BTC Up or Down", "start_ts": 1000.0,
        "end_ts": 1300.0,
    }


def test_resolution_logger_handles_winning_dict_settlement():
    with tempfile.TemporaryDirectory() as td:
        m = _market()
        ledger = PaperLedger(Path(td) / "paper_state.json", 1000)
        ledger.buy("c1", "up-token", m["market"], "Up", 0.20, 1.0, 1010, meta={"asset":"BTC", "slug":m["slug"], "market_id":m["id"], "start_ts":m["start_ts"], "end_ts":m["end_ts"]})
        closed = ledger.settle("c1", "up-token")
        assert len(closed) == 1
        assert closed[0]["pnl"] == 4.0
        logger = ResearchLogger(td)
        logger.record_resolution(ts=1302, market=m, winner="Up", winner_token="up-token", closed=closed)
        assert "RESOLVED" in (Path(td) / "resolutions.csv").read_text()


def test_resolution_logger_handles_losing_dict_settlement():
    with tempfile.TemporaryDirectory() as td:
        m = _market()
        ledger = PaperLedger(Path(td) / "paper_state.json", 1000)
        ledger.buy("c1", "down-token", m["market"], "Down", 0.80, 2.0, 1010, meta={"asset":"BTC", "slug":m["slug"], "market_id":m["id"], "start_ts":m["start_ts"], "end_ts":m["end_ts"]})
        closed = ledger.settle("c1", "up-token")
        assert len(closed) == 1
        assert closed[0]["pnl"] == -2.0
        logger = ResearchLogger(td)
        logger.record_resolution(ts=1302, market=m, winner="Up", winner_token="up-token", closed=closed)
        assert "RESOLVED" in (Path(td) / "resolutions.csv").read_text()


# ------------------------- paper ledger invariants -------------------------

def test_settlement_updates_realized_and_removes_position():
    with tempfile.TemporaryDirectory() as td:
        ledger = PaperLedger(Path(td) / "paper_state.json", 1000)
        ledger.buy("c1", "up-token", "BTC", "Up", 0.50, 5.0, 1010)
        closed = ledger.settle("c1", "up-token")
        assert len(closed) == 1
        assert not ledger.positions
        assert ledger.realized == 5.0
        assert ledger.cash == 1005.0


def test_multiple_positions_settle_atomically():
    with tempfile.TemporaryDirectory() as td:
        ledger = PaperLedger(Path(td) / "paper_state.json", 1000)
        ledger.buy("c1", "up-token", "BTC", "Up", 0.50, 2.0, 1010)
        ledger.buy("c1", "down-token", "BTC", "Down", 0.50, 3.0, 1011)
        closed = ledger.settle("c1", "up-token")
        assert len(closed) == 2
        assert not ledger.positions
        assert ledger.realized == -1.0

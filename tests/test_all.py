
import json
import time
from pathlib import Path

from strategy import ConvergenceStrategy
from paper_ledger import PaperLedger


def make_strategy(**kw):
    kw.setdefault("max_order", 10)
    kw.setdefault("max_market_exposure", 100)
    kw.setdefault("max_total_exposure", 300)
    kw.setdefault("max_asset_exposure", 35)
    return ConvergenceStrategy(**kw)


def history(price, seconds_ago=30, now=None):
    now = time.time() if now is None else now
    return [(now - seconds_ago, price)]


def test_regime_boundaries():
    s = make_strategy()
    assert s._regime(0.01) == "CHEAP"
    assert s._regime(0.299) == "CHEAP"
    assert s._regime(0.30) == "MID"
    assert s._regime(0.699) == "MID"
    assert s._regime(0.70) == "CORE"
    assert s._regime(0.899) == "CORE"
    assert s._regime(0.90) == "HIGH"
    assert s._regime(0.994) == "HIGH"
    assert s._regime(0.995) is None


def test_final_minute_block():
    s = make_strategy()
    assert s.decide(
        240, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, 100, 100, now=1000,
    ) is None
    assert s.decide(
        241, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, 100, 100, now=1000,
    ) is None


def test_cheap_flat_signal_is_allowed():
    s = make_strategy()
    sig = s.decide(
        120, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, 100, 100, now=1000,
    )
    assert sig is not None
    assert sig.side == "Up"
    assert sig.price == .20
    assert "V6 regime=CHEAP" in sig.reason
    assert .10 <= sig.notional <= 1.20


def test_cheap_prefers_weakness():
    s = make_strategy()
    falling = s.decide(
        150, .20, .80, .19, .79,
        [(900,.26),(970,.23),(990,.21)],
        [(900,.80),(970,.80),(990,.80)],
        0, 1000, 100, 100, now=1000,
    )
    assert falling is not None
    assert falling.side == "Up"


def test_mid_flat_is_rejected():
    s = make_strategy()
    sig = s.decide(
        150, .50, .49, .49, .48,
        history(.50), history(.50),
        0, 1000, 100, 0, now=1000,
    )
    assert sig is None


def test_mid_positive_confirmation_can_trade():
    s = make_strategy()
    now = 1000
    sig = s.decide(
        150, .50, .49, .49, .48,
        [(now-30,.46),(now-10,.48)],
        [], 0, 1000,
        up_depth=100, down_depth=0, now=now,
    )
    assert sig is not None
    assert sig.side == "Up"
    assert "regime=MID" in sig.reason


def test_core_prefers_strength():
    s = make_strategy()
    now = 1000
    sig = s.decide(
        130, .80, .20, .79, .19,
        [(now-60,.70),(now-30,.75),(now-10,.79)],
        [(now-60,.20),(now-30,.20),(now-10,.20)],
        0, 1000, 100, 100, now=now,
    )
    assert sig is not None
    assert sig.side == "Up"
    assert "regime=CORE" in sig.reason


def test_high_is_late_and_strength_oriented():
    s = make_strategy()
    now = 1000
    sig = s.decide(
        180, .95, .05, .94, .04,
        [(now-60,.88),(now-30,.90),(now-10,.94)],
        [(now-60,.05),(now-30,.05),(now-10,.05)],
        0, 1000, 100, 100, now=now,
    )
    assert sig is not None
    assert sig.side == "Up"
    assert "regime=HIGH" in sig.reason


def test_old_size_api_is_backward_compatible():
    s = make_strategy()
    for regime, price, score in [
        ("CHEAP", .20, .80),
        ("MID", .50, .80),
        ("CORE", .80, .80),
        ("HIGH", .95, .90),
    ]:
        value = s._size(regime, price, score)
        assert .10 <= value <= 10


def test_entry_count_scaling():
    s = make_strategy()
    assert s._size("CHEAP", .20, .90, 2, 60) > s._size("CHEAP", .20, .90, 0, 0)
    assert s._size("CORE", .80, .90, 5, 120) > s._size("CORE", .80, .90, 0, 0)
    assert s._size("HIGH", .95, .90, 3, 120) > s._size("HIGH", .95, .90, 0, 0)


def test_regime_sizing_increases_with_price():
    s = make_strategy()
    vals = [
        s._size("CHEAP", .20, .80),
        s._size("MID", .50, .80),
        s._size("CORE", .80, .80),
        s._size("HIGH", .95, .90),
    ]
    assert vals[0] < vals[1] < vals[2] < vals[3]


def test_high_add_on_is_order_capped():
    s = make_strategy()
    assert s._size("HIGH", .95, .90, 3, 120) <= 10
    assert s._size("HIGH", .95, .90, 0, 0) < s._size("HIGH", .95, .90, 3, 120)


def test_side_persistence():
    s = make_strategy()
    sig = s.decide(
        140, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, 100, 100, now=1000,
        thesis_side="Up",
        thesis_price=.20,
    )
    assert sig is not None
    assert sig.side == "Up"


def test_small_opposite_state_does_not_flip_thesis():
    s = make_strategy()
    sig = s.decide(
        140, .80, .20, .79, .19,
        [(970,.70),(990,.75)],
        [(970,.20),(990,.20)],
        0, 1000, 100, 100, now=1000,
        thesis_side="Up",
        thesis_price=.35,
    )
    assert sig is None or sig.side == "Up"


def test_global_exposure_limit():
    s = make_strategy(max_total_exposure=100)
    sig = s.decide(
        120, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, 100, 100, now=1000,
        total_exposure=99.95,
    )
    assert sig is None or sig.notional <= .05


def test_market_exposure_limit():
    s = make_strategy(max_market_exposure=100)
    sig = s.decide(
        120, .20, .80, .19, .79,
        history(.20), history(.80),
        99.95, 1000, 100, 100, now=1000,
    )
    assert sig is None or sig.notional <= .05


def test_asset_exposure_limit():
    s = make_strategy(max_asset_exposure=35)
    sig = s.decide(
        120, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, 100, 100, now=1000,
        asset_exposure=34.95,
    )
    assert sig is None or sig.notional <= .05


def test_depth_cap():
    s = make_strategy()
    sig = s.decide(
        120, .20, .80, .19, .79,
        history(.20), history(.80),
        0, 1000, .1, 100, now=1000,
    )
    assert sig is None or sig.notional <= .005 + 1e-9


def test_cash_limit():
    s = make_strategy()
    sig = s.decide(
        120, .20, .80, .19, .79,
        history(.20), history(.80),
        0, .15, 100, 100, now=1000,
    )
    assert sig is None or sig.notional <= .15


def test_invalid_prices_rejected():
    s = make_strategy()
    assert s._regime(0.0) is None
    assert s._regime(1.0) is None


def test_bot_wiring_contains_v6_state_arguments():
    text = Path("bot.py").read_text(encoding="utf-8")
    assert "market_entry_count=entry_count" in text
    assert "seconds_since_first_entry=seconds_since_first" in text
    assert "thesis_side=thesis_side" in text
    assert "thesis_price=thesis_price" in text
    assert "MAX_TOTAL_EXPOSURE" in text
    assert "PAPER_TRADING" in text


def test_ledger_settlement_is_auditable(tmp_path):
    p = PaperLedger(tmp_path / "state.json", 1000)
    p.buy("c1", "u", "m", "Up", .20, 2, 1)
    p.buy("c1", "d", "m", "Down", .80, 3, 2)
    closed = p.settle("c1", "u")
    expected = (2/.20 - 2) + (0 - 3)
    assert len(closed) == 2
    assert abs(p.realized - expected) < 1e-9
    assert abs(p.realized - p._settlement_total(p.trades)) < 1e-9
    assert p.total_open_cost() == 0
    assert abs(p.cash - (1000 - 5 + 2/.20)) < 1e-9


def test_ledger_reconciles_persisted_realized(tmp_path):
    path = tmp_path / "state.json"
    p = PaperLedger(path, 1000)
    p.buy("c1", "u", "m", "Up", .50, 5, 1)
    p.settle("c1", "u")
    d = json.loads(path.read_text())
    d["realized"] = -999
    path.write_text(json.dumps(d))
    q = PaperLedger(path, 1000)
    assert q.realized == q._settlement_total(q.trades)
    assert q.realized != -999


def test_ledger_losing_settlement():
    path = Path("/tmp/test_ledger_loss_v6_final.json")
    p = PaperLedger(path, 1000)
    try:
        p.buy("c2", "u", "m", "Up", .50, 5, 1)
        p.settle("c2", "d")
        assert abs(p.realized + 5) < 1e-9
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_ledger_no_position_is_noop(tmp_path):
    p = PaperLedger(tmp_path / "state.json", 1000)
    assert p.settle("missing", "u") == []
    assert p.realized == 0

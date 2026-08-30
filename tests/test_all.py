import json
import time
from pathlib import Path

from strategy import ConvergenceStrategy
from paper_ledger import PaperLedger


def make_strategy(**kw):
    kw.setdefault("max_order", 10)
    kw.setdefault("max_market_exposure", 100)
    kw.setdefault("max_total_exposure", 100)
    return ConvergenceStrategy(**kw)


def hist(price, now=None, seconds_ago=30):
    now = time.time() if now is None else now
    return [(now - seconds_ago, price)]


def args_for(price, *, now=1000.0, score_side="up"):
    if score_side == "up":
        return (120, price, 1-price, price-0.01, 1-price-0.01,
                hist(price, now), hist(1-price, now), 0, 1000,
                100, 100, now, 0, 0)
    return (120, price, 1-price, price-0.01, 1-price-0.01,
            hist(price, now), hist(1-price, now), 0, 1000,
            100, 100, now, 0, 0)


def test_regime_boundaries():
    s=make_strategy()
    assert s._regime(.01)=="CHEAP" and s._regime(.299)=="CHEAP"
    assert s._regime(.30)=="MID" and s._regime(.699)=="MID"
    assert s._regime(.70)=="CORE" and s._regime(.899)=="CORE"
    assert s._regime(.90)=="HIGH" and s._regime(.994)=="HIGH"
    assert s._regime(.995) is None


def test_final_minute_block():
    s=make_strategy()
    assert s.decide(240, .20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000) is None
    assert s.decide(241, .20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000) is None


def test_cheap_is_permissive_and_sizes_under_two():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000)
    assert sig is not None and "regime=CHEAP" in sig.reason and .10 <= sig.notional <= 1.20


def test_mid_is_strict():
    s=make_strategy()
    # Flat MID setup: score can be high from price/depth, but momentum is zero.
    sig=s.decide(120,.50,.50,.49,.49,hist(.50),hist(.50),0,1000,100,100,now=1000)
    assert sig is None


def test_high_can_size_up_to_order_cap():
    s=make_strategy()
    size=s._size("HIGH",.994,.99)
    assert 1.5 <= size <= 10


def test_total_exposure_is_hard_ceiling():
    s=make_strategy(max_total_exposure=100)
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000,total_exposure=99.95)
    assert sig is None or sig.notional <= .05


def test_asset_exposure_is_hard_ceiling():
    s=make_strategy(max_asset_exposure=35)
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000,asset_exposure=34.95)
    assert sig is None or sig.notional <= .05


def test_market_exposure_is_hard_ceiling():
    s=make_strategy(max_market_exposure=100)
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),99.95,1000,100,100,now=1000)
    assert sig is None or sig.notional <= .05


def test_depth_cap():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,.1,100,now=1000)
    assert sig is None or sig.notional <= .10*.20*.25 + 1e-9


def test_invalid_price_rejected():
    s=make_strategy()
    assert s._regime(0.0) is None and s._regime(1.0) is None


def test_ledger_settlement_is_atomic_and_auditable(tmp_path):
    p=PaperLedger(tmp_path/"state.json",1000)
    p.buy("c1","u","m","Up",.20,2,1)
    p.buy("c1","d","m","Down",.80,3,2)
    closed=p.settle("c1","u")
    assert len(closed)==2
    expected=(2/.20-2) + (0-3)
    assert abs(p.realized-expected)<1e-9
    assert abs(p.realized-p._settlement_total(p.trades))<1e-9
    assert p.total_open_cost()==0
    assert abs(p.cash-(1000-5+2/.20))<1e-9


def test_ledger_reconciles_persisted_realized_from_settlements(tmp_path):
    path=tmp_path/"state.json"
    p=PaperLedger(path,1000)
    p.buy("c1","u","m","Up",.50,5,1)
    p.settle("c1","u")
    d=json.loads(path.read_text())
    d["realized"]=-999
    path.write_text(json.dumps(d))
    q=PaperLedger(path,1000)
    assert q.realized==q._settlement_total(q.trades)
    assert q.realized != -999


def test_ledger_losing_settlement():
    p=PaperLedger(Path("/tmp/test_ledger_loss.json"),1000)
    try:
        p.buy("c2","u","m","Up",.50,5,1)
        p.settle("c2","d")
        assert abs(p.realized + 5)<1e-9
    finally:
        try: Path("/tmp/test_ledger_loss.json").unlink()
        except FileNotFoundError: pass


def test_ledger_no_position_is_noop(tmp_path):
    p=PaperLedger(tmp_path/"state.json",1000)
    assert p.settle("missing","u")==[] and p.realized==0


def test_price_sizing_increases_across_regimes():
    s=make_strategy()
    vals=[s._size("CHEAP",.20,.8),s._size("MID",.50,.8),s._size("CORE",.80,.8),s._size("HIGH",.95,.9)]
    assert vals[0] < vals[1] < vals[2] < vals[3]


def test_buy_side_selection_returns_valid_signal():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000)
    assert sig and sig.side in ("Up","Down") and 0 < sig.price < 1 and sig.notional >= .10


def test_cash_limit():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,.15,100,100,now=1000)
    assert sig is None or sig.notional <= .15

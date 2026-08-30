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
    kw.setdefault("min_trade_gap_seconds", 0)
    return ConvergenceStrategy(**kw)


def hist(price, now=1000, seconds_ago=30):
    return [(now-seconds_ago, price), (now-10, price)]


def test_regime_boundaries():
    s=make_strategy()
    assert s._regime(.01)=="CHEAP" and s._regime(.299)=="CHEAP"
    assert s._regime(.30)=="MID" and s._regime(.699)=="MID"
    assert s._regime(.70)=="CORE" and s._regime(.899)=="CORE"
    assert s._regime(.90)=="HIGH" and s._regime(.994)=="HIGH"
    assert s._regime(.995) is None


def test_final_minute_block():
    s=make_strategy()
    assert s.decide(240,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000) is None
    assert s.decide(241,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000) is None


def test_cheap_flat_still_allowed_but_has_no_unfair_priority():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000)
    assert sig is not None and sig.side=="Up" and "V7 regime=CHEAP" in sig.reason


def test_cheap_weakness_is_preferred():
    s=make_strategy()
    sig=s.decide(150,.20,.80,.19,.79,[(900,.25),(970,.22),(990,.20)],hist(.80),0,1000,100,100,now=1000)
    assert sig is not None and sig.side=="Up"


def test_mid_is_observable_on_neutral_state():
    s=make_strategy()
    sig=s.decide(150,.50,.99,.49,.98,hist(.50),[(970,.99),(990,.99)],0,1000,100,0,now=1000)
    assert sig is not None and sig.side=="Up" and "regime=MID" in sig.reason


def test_mid_flat_without_competing_side_is_allowed():
    s=make_strategy()
    sig=s.decide(150,.50,None,.49,None,hist(.50),[],0,1000,100,0,now=1000)
    assert sig is not None and sig.side=="Up"


def test_core_prefers_strength_over_flat_cheap_candidate():
    s=make_strategy()
    sig=s.decide(
        130,.80,.20,.79,.19,
        [(940,.70),(970,.75),(990,.79)],
        [(940,.20),(970,.20),(990,.20)],
        0,1000,100,100,now=1000,
    )
    assert sig is not None and sig.side=="Up" and "regime=CORE" in sig.reason


def test_high_is_late_and_strength_oriented():
    s=make_strategy()
    sig=s.decide(
        180,.95,.05,.94,.04,
        [(940,.88),(970,.90),(990,.94)],
        [(940,.05),(970,.05),(990,.05)],
        0,1000,100,100,now=1000,
    )
    assert sig is not None and sig.side=="Up" and "regime=HIGH" in sig.reason


def test_old_size_api_is_backward_compatible():
    s=make_strategy()
    for regime,price,score in [("CHEAP",.20,.8),("MID",.50,.8),("CORE",.80,.8),("HIGH",.95,.9)]:
        v=s._size(regime,price,score)
        assert .10 <= v <= 10


def test_entry_count_scaling():
    s=make_strategy()
    assert s._size("CHEAP",.20,.90,2,60) > s._size("CHEAP",.20,.90,0,0)
    assert s._size("CORE",.80,.90,5,120) > s._size("CORE",.80,.90,0,0)
    assert s._size("HIGH",.95,.90,3,120) > s._size("HIGH",.95,.90,0,0)


def test_add_on_stops_scaling_when_signal_degrades():
    s=make_strategy()
    strong=s._size("CORE",.80,.90,4,60,add_on_allowed=True)
    weak=s._size("CORE",.80,.90,4,60,add_on_allowed=False)
    assert weak < strong


def test_high_starter_is_smaller_than_later_high():
    s=make_strategy()
    starter=s._size("HIGH",.95,.90,0,0)
    later=s._size("HIGH",.95,.90,3,120)
    assert later > starter and later <= 10


def test_regime_sizing_increases_with_price():
    s=make_strategy()
    vals=[s._size("CHEAP",.20,.8),s._size("MID",.50,.8),s._size("CORE",.80,.8),s._size("HIGH",.95,.9)]
    assert vals[0] < vals[1] < vals[2] < vals[3]


def test_side_persistence():
    s=make_strategy()
    sig=s.decide(140,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000,thesis_side="Up",thesis_price=.20)
    assert sig is not None and sig.side=="Up"


def test_small_opposite_state_does_not_flip():
    s=make_strategy()
    sig=s.decide(140,.80,.20,.79,.19,[(970,.70),(990,.75)],[(970,.20),(990,.20)],0,1000,100,100,now=1000,thesis_side="Up",thesis_price=.35)
    assert sig is None or sig.side=="Up"


def test_reset_cooldown_blocks_ping_pong():
    s=make_strategy()
    s._last_reset_at=1000
    sig=s.decide(180,.95,.05,.94,.04,[(940,.88),(970,.90),(990,.94)],[(940,.05),(970,.05),(990,.05)],0,1000,100,100,now=1010,thesis_side="Up",thesis_price=.10)
    assert sig is None or sig.side=="Up"


def test_global_exposure_limit():
    s=make_strategy(max_total_exposure=100)
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000,total_exposure=99.95)
    assert sig is None or sig.notional <= .05


def test_market_exposure_limit():
    s=make_strategy(max_market_exposure=100)
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),99.95,1000,100,100,now=1000)
    assert sig is None or sig.notional <= .05


def test_asset_exposure_limit():
    s=make_strategy(max_asset_exposure=35)
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000,asset_exposure=34.95)
    assert sig is None or sig.notional <= .05


def test_depth_cap():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,.1,100,now=1000)
    assert sig is None or sig.notional <= .005+1e-9


def test_cash_limit():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,.15,100,100,now=1000)
    assert sig is None or sig.notional <= .15


def test_invalid_prices_rejected():
    s=make_strategy()
    assert s._regime(0.0) is None and s._regime(1.0) is None


def test_buy_side_selection_returns_valid_signal():
    s=make_strategy()
    sig=s.decide(120,.20,.80,.19,.79,hist(.20),hist(.80),0,1000,100,100,now=1000)
    assert sig and sig.side in ("Up","Down") and 0<sig.price<1 and sig.notional>=.10


def test_bot_wiring_state_args():
    bot=Path("bot.py").read_text()
    assert "market_entry_count=entry_count" in bot
    assert "seconds_since_first_entry=seconds_since_first" in bot
    assert "thesis_side=thesis_side" in bot
    assert "thesis_price=thesis_price" in bot
    assert "MAX_TOTAL_EXPOSURE" in bot
    assert "PAPER_TRADING" in bot


def test_ledger_settlement_is_auditable(tmp_path):
    p=PaperLedger(tmp_path/"state.json",1000)
    p.buy("c1","u","m","Up",.20,2,1)
    p.buy("c1","d","m","Down",.80,3,2)
    closed=p.settle("c1","u")
    expected=(2/.20-2)+(0-3)
    assert len(closed)==2
    assert abs(p.realized-expected)<1e-9
    assert abs(p.realized-p._settlement_total(p.trades))<1e-9
    assert p.total_open_cost()==0
    assert abs(p.cash-(1000-5+2/.20))<1e-9


def test_ledger_reconciles_persisted_realized(tmp_path):
    path=tmp_path/"state.json"
    p=PaperLedger(path,1000)
    p.buy("c1","u","m","Up",.50,5,1)
    p.settle("c1","u")
    d=json.loads(path.read_text())
    d["realized"]=-999
    path.write_text(json.dumps(d))
    q=PaperLedger(path,1000)
    assert q.realized==q._settlement_total(q.trades)
    assert q.realized!=-999


def test_ledger_losing_settlement():
    path=Path("/tmp/test_ledger_loss_v7.json")
    p=PaperLedger(path,1000)
    try:
        p.buy("c2","u","m","Up",.50,5,1)
        p.settle("c2","d")
        assert abs(p.realized+5)<1e-9
    finally:
        try:path.unlink()
        except FileNotFoundError:pass


def test_ledger_no_position_is_noop(tmp_path):
    p=PaperLedger(tmp_path/"state.json",1000)
    assert p.settle("missing","u")==[] and p.realized==0

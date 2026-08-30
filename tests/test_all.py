import json
from strategy import CapitalFirstStrategy
from paper_ledger import PaperLedger

def S(**kw):
    kw.setdefault('min_trade_gap_seconds',0); kw.setdefault('max_market_exposure',100); kw.setdefault('max_total_exposure',300); kw.setdefault('max_asset_exposure',35); kw.setdefault('max_order',10)
    return CapitalFirstStrategy(**kw)
def H(p,now=1000): return [(now-60,p),(now-30,p),(now-10,p),(now-1,p)]
def test_version_and_regimes():
    s=S(); assert s.VERSION.startswith('V8'); assert [s._regime(x) for x in (.01,.30,.70,.90,.995)]==['CHEAP','MID','CORE','HIGH',None]
def test_capital_curve():
    s=S(); assert s.desired_capital(.05)<s.desired_capital(.50)<s.desired_capital(.80)<s.desired_capital(.95)
def test_cheap_is_small_and_bounded():
    s=S(); assert s.desired_capital(.05)<=.81 and s.desired_capital(.25)<=.81
def test_high_is_large():
    s=S(); assert s.desired_capital(.95)>10 and s.desired_capital(.95)>10*s.desired_capital(.20)
def test_passive_uses_bid_and_capital_priority():
    s=S(); x=s.decide(180,.30,.80,.29,.79,H(.29),H(.79),0,1000,100,100,now=1000); assert x and x.price==.79 and 'passive=bid' in x.reason
def test_no_bid_side_is_ignored():
    s=S(); x=s.decide(180,.30,.80,None,.79,H(.29),H(.79),0,1000,100,100,now=1000); assert x and x.side=='Down'
def test_target_stops_repeated_adds():
    s=S(); assert s.decide(180,.30,None,.29,None,H(.29),[],.81,1000,100,0,now=1000) is None
def test_target_limits_remaining_add():
    s=S(); x=s.decide(180,.30,None,.29,None,H(.29),[],.70,1000,100,0,now=1000); assert x and x.notional<=.11
def test_high_uses_multiple_fill_slices():
    s=S(); x=s.decide(180,.96,.04,.95,.03,H(.95),H(.03),0,1000,100,100,now=1000); assert x and x.notional<=10 and s.desired_capital(.95)>10
def test_reset_cooldown():
    s=S(); s._last_reset_at=1000; x=s.decide(180,.20,.90,.19,.89,H(.19),H(.89),0,1000,100,100,now=1010,thesis_side='Up',thesis_price=.75,seconds_since_first_entry=60); assert x is None or x.side=='Up'
def test_final_cutoff():
    assert S().decide(181,.30,.80,.29,.79,H(.29),H(.79),0,1000,100,100,now=1000) is None
def test_global_market_asset_caps():
    s=S(); assert s.decide(180,.96,.04,.95,.03,H(.95),H(.03),0,1000,100,100,now=1000,total_exposure=299.95).notional<=.05 if s.decide(180,.96,.04,.95,.03,H(.95),H(.03),0,1000,100,100,now=1000,total_exposure=299.95) else True
    s=S(); assert s.decide(180,.96,.04,.95,.03,H(.95),H(.03),99.95,1000,100,100,now=1000) .notional<=.05 if s.decide(180,.96,.04,.95,.03,H(.95),H(.03),99.95,1000,100,100,now=1000) else True
def test_asset_cap():
    s=S(); x=s.decide(180,.96,.04,.95,.03,H(.95),H(.03),0,1000,100,100,now=1000,asset_exposure=34.95); assert x is None or x.notional<=.05
def test_depth_cap():
    s=S(); x=s.decide(180,.96,.04,.95,.03,H(.95),H(.03),0,1000,.1,.1,now=1000); assert x is None
def test_ledger_reconcile(tmp_path):
    p=PaperLedger(tmp_path/'s.json',1000); p.buy('c','u','m','Up',.5,1,1); p.settle('c','u'); assert p.realized==p._settlement_total(p.trades)

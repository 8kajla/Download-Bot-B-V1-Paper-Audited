
from strategy import CapitalFirstStrategy

def make():
    return CapitalFirstStrategy(
        bankroll=1000,
        max_market_exposure=100,
        max_order=10,
        max_asset_exposure=35,
        max_total_exposure=300,
        start_sec=0,
        stop_sec=240,
        hard_cutoff_seconds=60,
        max_depth_participation=0.20,
        min_trade_gap_seconds=0,
        min_bid_depth=1,
        state_reset_jump=0.30,
        state_reset_cooldown=30,
        state_min_age=45,
    )

def H(p, now=1000):
    return [
        {"ts":now-30, "best_bid":p},
        {"ts":now-10, "best_bid":p},
        {"ts":now-5, "best_bid":p},
        {"ts":now-3, "best_bid":p},
        {"ts":now, "best_bid":p},
    ]

def test_all_price_bands_map():
    x=make()
    expected=[
        (.02,"C00_05","CHEAP"), (.07,"C05_10","CHEAP"),
        (.12,"C10_15","CHEAP"), (.17,"C15_20","CHEAP"),
        (.25,"C20_30","CHEAP"), (.35,"M30_40","MID"),
        (.45,"M40_50","MID"), (.55,"M50_60","MID"),
        (.65,"M60_70","MID"), (.75,"R70_80","CORE"),
        (.85,"R80_90","CORE"), (.925,"H90_95","HIGH"),
        (.975,"H95_100","HIGH"),
    ]
    for p,b,r in expected:
        assert x.fine_band(p)==(b,r)

def test_price_capital_curve_is_increasing():
    x=make()
    ps=[.025,.075,.125,.175,.25,.35,.45,.55,.65,.75,.85,.925,.975]
    vals=[x.desired_capital(p,market="BTC") for p in ps]
    assert all(b>=a for a,b in zip(vals,vals[1:]))

def test_market_overlays_are_distinct():
    x=make()
    vals={m:x.desired_capital(.55,market=m) for m in ("BTC","ETH","SOL","BNB")}
    assert len(set(round(v,6) for v in vals.values()))>1

def test_cheap_path_is_independent():
    x=make()
    c=x._candidate("BTC","Up",.20,.21,50,H(.20),1000,None,0,0)
    assert c and c["regime"]=="CHEAP"
    assert c["path"]=="CHEAP_LIQUIDITY_WEAKNESS"

def test_mid_path_is_independent():
    x=make()
    c=x._candidate("ETH","Up",.50,.51,50,H(.50),1000,None,0,0)
    assert c and c["regime"]=="MID"
    assert c["path"]=="MID_STABLE_BOOK"

def test_core_path_requires_quality():
    x=make()
    h=[{"ts":970,"best_bid":.74},{"ts":990,"best_bid":.78},
       {"ts":995,"best_bid":.79},{"ts":1000,"best_bid":.80}]
    c=x._candidate("SOL","Up",.80,.81,50,h,1000,"Up",2,60)
    assert c and c["path"]=="CORE_STRENGTH_CONTINUATION"

def test_high_needs_established_state():
    x=make()
    h=[{"ts":970,"best_bid":.88},{"ts":990,"best_bid":.89},
       {"ts":995,"best_bid":.90},{"ts":1000,"best_bid":.925}]
    no_state=x._candidate("BNB","Up",.925,.94,50,h,1000,None,0,0)
    assert no_state is None
    good=x._candidate("BNB","Up",.925,.94,50,h,1000,"Up",3,90)
    assert good and good["regime"]=="HIGH"

def test_95_100_has_stricter_requirement():
    x=make()
    h90=[{"ts":970,"best_bid":.89},{"ts":990,"best_bid":.90},
         {"ts":995,"best_bid":.912},{"ts":1000,"best_bid":.925}]
    c90=x._candidate("BTC","Up",.925,.94,50,h90,1000,"Up",3,90)
    assert c90 is not None
    h95=[{"ts":970,"best_bid":.95},{"ts":990,"best_bid":.952},
         {"ts":995,"best_bid":.956},{"ts":1000,"best_bid":.975}]
    c95=x._candidate("BTC","Up",.975,.985,50,h95,1000,"Up",3,90)
    assert c95 is None or c95["band"]=="H95_100"

def test_entry_sequence_increases_target():
    x=make()
    a=x.desired_capital(.80,market="BTC",entry_count=0)
    b=x.desired_capital(.80,market="BTC",entry_count=5)
    c=x.desired_capital(.80,market="BTC",entry_count=20)
    assert a<=b<=c

def test_same_side_persistence():
    x=make()
    up=[{"ts":970,"best_bid":.74},{"ts":990,"best_bid":.78},
        {"ts":995,"best_bid":.795},{"ts":1000,"best_bid":.80}]
    down=H(.50)
    sig=x.decide(
        120,.81,.51,.80,.50,up,down,2,1000,
        up_depth=50,down_depth=50,now=1000,asset="ETH",market="ETH",
        market_entry_count=1,seconds_since_first_entry=90,
        thesis_side="Up",thesis_price=.78
    )
    assert sig and sig.side=="Up"

def test_final_minute_cutoff():
    x=make()
    assert x.decide(
        180,.51,.21,.50,.20,H(.50),H(.20),0,1000,
        up_depth=50,down_depth=50,now=1000,asset="BTC"
    ) is None

def test_cheap_collapse_guard():
    x=make()
    hist=[{"ts":970,"best_bid":.26},{"ts":990,"best_bid":.22},
          {"ts":995,"best_bid":.205},{"ts":1000,"best_bid":.20}]
    assert x.decide(
        80,.21,.06,.20,.05,hist,hist,5,1000,
        up_depth=50,down_depth=50,now=1000,asset="BTC",
        market_entry_count=9,seconds_since_first_entry=100,
        thesis_side="Up",thesis_price=.28
    ) is None

def test_signal_contains_market_and_fine_band():
    x=make()
    sig=x.decide(
        40,.51,.21,.50,.20,H(.50),H(.20),0,1000,
        up_depth=50,down_depth=50,now=1000,asset="ETH",market="ETH"
    )
    assert sig
    assert "V10 market=ETH" in sig.reason
    assert "band=M50_60" in sig.reason

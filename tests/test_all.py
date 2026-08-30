from strategy import CapitalFirstStrategy

def S():
    return CapitalFirstStrategy(
        min_trade_gap_seconds=0,
        max_market_exposure=100,
        max_order=10,
        max_asset_exposure=35,
        max_total_exposure=300,
        hard_cutoff_seconds=60,
        min_bid_depth=1,
        max_depth_participation=.20,
        state_min_age=45,
    )

def H(p,n=1000):
    return [
        {"ts":n-30,"best_bid":p},
        {"ts":n-10,"best_bid":p},
        {"ts":n-5,"best_bid":p},
        {"ts":n,"best_bid":p},
    ]

def test_bands():
    s=S()
    expected=[(.02,"C00_05","CHEAP"),(.07,"C05_10","CHEAP"),(.12,"C10_15","CHEAP"),
              (.17,"C15_20","CHEAP"),(.25,"C20_30","CHEAP"),(.35,"M30_40","MID"),
              (.45,"M40_50","MID"),(.55,"M50_60","MID"),(.65,"M60_70","MID"),
              (.75,"R70_80","CORE"),(.85,"R80_90","CORE"),(.925,"H90_95","HIGH"),
              (.975,"H95_100","HIGH")]
    for p,b,r in expected: assert s.fine_band(p)==(b,r)

def test_capital_curve():
    s=S()
    v=[s.capital_target(p,"BTC") for p in (.025,.075,.125,.175,.25,.35,.45,.55,.65,.75,.85,.925,.975)]
    assert all(b>=a for a,b in zip(v,v[1:]))

def test_market_profiles_distinct():
    s=S()
    v=[s.capital_target(.55,m) for m in ("BTC","ETH","SOL","BNB")]
    assert len(set(round(x,6) for x in v))>1

def test_cheap_profile():
    c=S()._candidate("BTC","Up",.20,.21,50,H(.20),1000,None,0,0)
    assert c and c["regime"]=="CHEAP"

def test_mid_profile():
    c=S()._candidate("ETH","Up",.50,.51,50,H(.50),1000,None,0,0)
    assert c and c["regime"]=="MID"

def test_core_requires_strength():
    assert S()._candidate("BTC","Up",.80,.81,50,H(.80),1000,None,0,0) is None

def test_high_requires_state():
    s=S()
    h=[{"ts":970,"best_bid":.88},{"ts":990,"best_bid":.90},{"ts":995,"best_bid":.90},{"ts":1000,"best_bid":.925}]
    assert s._candidate("BNB","Up",.925,.94,50,h,1000,None,0,0) is None
    assert s._candidate("BNB","Up",.925,.94,50,h,1000,"Up",3,90) is not None

def test_entry_growth_bounded():
    s=S()
    a=s.capital_target(.80,"BTC",0); b=s.capital_target(.80,"BTC",5); c=s.capital_target(.80,"BTC",20)
    assert a<=b<=c<10

def test_side_persistence():
    s=S()
    up=[{"ts":970,"best_bid":.74},{"ts":990,"best_bid":.78},{"ts":995,"best_bid":.795},{"ts":1000,"best_bid":.80}]
    x=s.decide(120,.81,.50,.80,.49,up,H(.49),2,1000,up_depth=50,down_depth=50,now=1000,
               asset="ETH",market="ETH",market_entry_count=1,seconds_since_first_entry=90,
               thesis_side="Up",thesis_price=.78)
    assert x and x.side=="Up"

def test_cutoff():
    assert S().decide(180,.51,.21,.50,.20,H(.50),H(.20),0,1000,50,50,now=1000,asset="BTC") is None

def test_cheap_collapse_guard():
    s=S()
    h=[{"ts":970,"best_bid":.28},{"ts":990,"best_bid":.21},{"ts":995,"best_bid":.205},{"ts":1000,"best_bid":.20}]
    assert s.decide(80,.21,.06,.20,.05,h,h,5,1000,50,50,now=1000,asset="BTC",
                    market_entry_count=10,seconds_since_first_entry=100,thesis_side="Up",
                    thesis_price=.28) is None

def test_signal_metadata():
    s=S()
    x=s.decide(40,.61,None,.60,None,H(.60),[],0,1000,100,0,now=1000,asset="ETH",market="ETH")
    assert x and "V11 market=ETH" in x.reason and "band=M60_70" in x.reason

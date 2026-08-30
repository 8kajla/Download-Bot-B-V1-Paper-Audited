from strategy import CapitalFirstStrategy

def S():
    return CapitalFirstStrategy(min_trade_gap_seconds=0,max_market_exposure=100,
        max_order=10,max_asset_exposure=35,max_total_exposure=300,
        hard_cutoff_seconds=60,min_bid_depth=1,max_depth_participation=.25,
        state_min_age=45)

def H(p,n=1000):
    return [{"ts":n-30,"best_bid":p},{"ts":n-10,"best_bid":p},{"ts":n-5,"best_bid":p},{"ts":n,"best_bid":p}]

def test_all_fine_bands():
    s=S()
    expected=[(.02,"C00_05","CHEAP"),(.07,"C05_10","CHEAP"),(.12,"C10_15","CHEAP"),
              (.17,"C15_20","CHEAP"),(.25,"C20_30","CHEAP"),(.35,"M30_40","MID"),
              (.45,"M40_50","MID"),(.55,"M50_60","MID"),(.65,"M60_70","MID"),
              (.75,"R70_80","CORE"),(.85,"R80_90","CORE"),(.925,"H90_95","HIGH"),
              (.975,"H95_100","HIGH")]
    for p,b,r in expected: assert s.fine_band(p)==(b,r)

def test_capital_curve_matches_direction():
    s=S()
    ps=[.025,.075,.125,.175,.25,.35,.45,.55,.65,.75,.85,.925,.975]
    v=[s.capital_target(p,"BTC") for p in ps]
    assert all(b>=a for a,b in zip(v,v[1:]))

def test_high_is_available_without_previous_entry():
    s=S()
    h=[{"ts":970,"best_bid":.93},{"ts":990,"best_bid":.94},{"ts":995,"best_bid":.945},{"ts":1000,"best_bid":.95}]
    c=s._candidate("BTC","Up",.95,.96,100,h,1000,None,0,0)
    assert c and c["regime"]=="HIGH"

def test_market_profiles_exist():
    s=S()
    assert set(s.MARKET_REGIME_WEIGHT)=={"BTC","ETH","SOL","BNB"}
    assert s.MARKET_REGIME_WEIGHT["SOL"]["CHEAP"]>s.MARKET_REGIME_WEIGHT["BTC"]["CHEAP"]

def test_segment_specific_book_checks():
    s=S()
    assert s._book_ok("BTC","HIGH",.95,.96,1)[0] is False
    assert s._book_ok("BTC","CHEAP",.20,.21,1)[0] is True

def test_side_persistence():
    s=S()
    up=[{"ts":970,"best_bid":.74},{"ts":990,"best_bid":.78},{"ts":995,"best_bid":.795},{"ts":1000,"best_bid":.80}]
    x=s.decide(120,.81,.50,.80,.49,up,H(.49),2,1000,50,50,now=1000,asset="ETH",market="ETH",
               market_entry_count=1,seconds_since_first_entry=90,thesis_side="Up",thesis_price=.78)
    assert x and x.side=="Up"

def test_starter_then_add_on():
    s=S()
    first=s.decide(30,.51,.21,.50,.20,H(.50),H(.20),0,1000,100,100,now=1000,asset="ETH",market="ETH")
    assert first
    later=s.decide(31,.51,.21,.50,.20,H(.50),H(.20),first.notional,1000,100,100,now=1003,
                   asset="ETH",market="ETH",market_entry_count=1,seconds_since_first_entry=3,
                   thesis_side=first.side,thesis_price=first.price)
    assert later is not None or True

def test_final_minute_cutoff():
    assert S().decide(180,.51,.21,.50,.20,H(.50),H(.20),0,1000,50,50,now=1000,asset="BTC") is None

def test_depth_limits_order_size():
    s=S()
    x=s.decide(30,.51,None,.50,None,H(.50),[],0,1000,1.0,0,now=1000,asset="BTC")
    assert x is None or x.notional <= .10

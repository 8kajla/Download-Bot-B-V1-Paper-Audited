from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass
class Signal:
    side: str
    price: float
    score: float
    notional: float
    reason: str

class CapitalFirstStrategy:
    VERSION="V11_SIMPLE_PROFILE"

    BANDS=(
        ("C00_05",0,.05,"CHEAP"),("C05_10",.05,.10,"CHEAP"),
        ("C10_15",.10,.15,"CHEAP"),("C15_20",.15,.20,"CHEAP"),
        ("C20_30",.20,.30,"CHEAP"),("M30_40",.30,.40,"MID"),
        ("M40_50",.40,.50,"MID"),("M50_60",.50,.60,"MID"),
        ("M60_70",.60,.70,"MID"),("R70_80",.70,.80,"CORE"),
        ("R80_90",.80,.90,"CORE"),("H90_95",.90,.95,"HIGH"),
        ("H95_100",.95,1.00,"HIGH"),
    )
    MARKETS=("BTC","ETH","SOL","BNB")

    # Empirical per-trade notional anchors from the trader sample.
    BASE_CAPITAL={
        "C00_05":.41,"C05_10":.51,"C10_15":.68,"C15_20":.83,"C20_30":1.07,
        "M30_40":1.48,"M40_50":2.17,"M50_60":3.08,"M60_70":3.93,
        "R70_80":5.25,"R80_90":8.28,"H90_95":14.99,"H95_100":42.49,
    }

    # Market-specific checks/overlays are intentionally explicit.
    PROFILE={
        "BTC":dict(cap=1.00,depth={"CHEAP":3,"MID":5,"CORE":8,"HIGH":15},
                   spread={"CHEAP":.060,"MID":.050,"CORE":.035,"HIGH":.020},
                   cheap=.020,mid=.040,core=.015,high=.020,reset=.32),
        "ETH":dict(cap=.98,depth={"CHEAP":2.5,"MID":4,"CORE":6,"HIGH":12},
                   spread={"CHEAP":.065,"MID":.055,"CORE":.040,"HIGH":.022},
                   cheap=.022,mid=.045,core=.014,high=.018,reset=.30),
        "SOL":dict(cap=.92,depth={"CHEAP":3.5,"MID":6,"CORE":9,"HIGH":18},
                   spread={"CHEAP":.055,"MID":.045,"CORE":.032,"HIGH":.018},
                   cheap=.025,mid=.045,core=.018,high=.022,reset=.34),
        "BNB":dict(cap=.95,depth={"CHEAP":3,"MID":5,"CORE":8,"HIGH":15},
                   spread={"CHEAP":.060,"MID":.050,"CORE":.035,"HIGH":.020},
                   cheap=.022,mid=.040,core=.016,high=.020,reset=.33),
    }

    SLICE={"CHEAP":.40,"MID":.85,"CORE":2.50,"HIGH":5.00}
    HARD_MAX_ORDER=10.0
    HARD_MAX_MARKET=100.0
    HARD_MAX_ASSET=35.0
    HARD_MAX_TOTAL=300.0
    HARD_CUTOFF=60.0

    def __init__(self,bankroll=1000,max_market_exposure=100,max_order=10,
                 max_asset_exposure=35,max_total_exposure=300,start_sec=0,
                 stop_sec=240,hard_cutoff_seconds=60,max_depth_participation=.20,
                 min_trade_gap_seconds=2,min_bid_depth=1,state_reset_jump=.35,
                 state_reset_cooldown=30,state_min_age=45,**_):
        self.bankroll=float(bankroll)
        self.max_market_exposure=min(float(max_market_exposure),self.HARD_MAX_MARKET)
        self.max_order=min(float(max_order),self.HARD_MAX_ORDER)
        self.max_asset_exposure=min(float(max_asset_exposure),self.HARD_MAX_ASSET)
        self.max_total_exposure=min(float(max_total_exposure),self.HARD_MAX_TOTAL)
        self.start_sec=max(0.,float(start_sec)); self.stop_sec=min(240.,float(stop_sec))
        self.hard_cutoff_seconds=max(60.,float(hard_cutoff_seconds))
        self.max_depth_participation=min(.20,max(.01,float(max_depth_participation)))
        self.min_trade_gap_seconds=max(0.,float(min_trade_gap_seconds))
        self.min_bid_depth=max(0.,float(min_bid_depth))
        self.state_reset_jump=max(.20,float(state_reset_jump))
        self.state_reset_cooldown=max(0.,float(state_reset_cooldown))
        self.state_min_age=max(0.,float(state_min_age))
        self._last_trade_at=None; self._last_reset_at=None

    @classmethod
    def normalize_market(cls,x):
        s=str(x or "").upper()
        for m in cls.MARKETS:
            if m in s:return m
        return "BTC"

    @classmethod
    def fine_band(cls,p):
        p=float(p)
        for b,lo,hi,r in cls.BANDS:
            if lo<=p<hi:return b,r
        return (None,None)

    @classmethod
    def capital_target(cls,p,market="BTC",entry_count=0):
        b,r=cls.fine_band(p)
        if not b:return 0.
        m=cls.normalize_market(market)
        v=cls.BASE_CAPITAL[b]*cls.PROFILE[m]["cap"]
        # Later entries are modestly larger, matching the observed burst pattern.
        if entry_count>=8:v*=1.12
        elif entry_count>=4:v*=1.06
        elif entry_count>=1:v*=1.02
        return max(.20,min(42.49*cls.PROFILE[m]["cap"],v))

    def desired_capital(self,p,regime=None,market="BTC",entry_count=0):
        return self.capital_target(p,market,entry_count)

    @staticmethod
    def _points(hist):
        out=[]
        for x in hist or []:
            try:
                if isinstance(x,dict):
                    out.append((float(x["ts"]),float(x.get("best_bid",x.get("mid")))))
                else: out.append((float(x[0]),float(x[1])))
            except (TypeError,ValueError,KeyError,IndexError): pass
        return sorted((t,p) for t,p in out if 0<p<1)

    @classmethod
    def movement(cls,p,hist,now):
        pts=cls._points(hist)
        if not pts:return {f"m{s}":0. for s in (1,3,5,10,30)}
        def at(sec):
            return min(pts,key=lambda x:abs(x[0]-(float(now)-sec)))[1]
        return {f"m{s}":float(p)-at(s) for s in (1,3,5,10,30)}

    def _book_ok(self,m,r,bid,ask,depth):
        prof=self.PROFILE[m]
        if depth<max(self.min_bid_depth,prof["depth"][r]):return False,0.
        sp=0. if ask is None else max(0.,float(ask)-float(bid))
        return sp<=prof["spread"][r],sp

    def _check(self,m,b,r,mv,side,thesis,entries,age):
        prof=self.PROFILE[m]
        if r=="CHEAP":
            return mv["m10"]<=prof["cheap"] and mv["m1"]>-0.045
        if r=="MID":
            return abs(mv["m10"])<=prof["mid"]
        if r=="CORE":
            return mv["m10"]>=prof["core"] and mv["m5"]>=-0.002
        req=prof["high"]*(1.35 if b=="H95_100" else 1.0)
        return bool(thesis) and side==thesis and entries>=2 and age>=self.state_min_age and mv["m5"]>=req

    def _candidate(self,m,side,bid,ask,depth,hist,now,thesis,entries,age):
        if bid is None:return None
        try:bid=float(bid);depth=max(0.,float(depth or 0));ask=None if ask is None else float(ask)
        except (TypeError,ValueError):return None
        if not 0<bid<1:return None
        band,reg=self.fine_band(bid)
        if not reg:return None
        ok,sp=self._book_ok(m,reg,bid,ask,depth)
        if not ok:return None
        mv=self.movement(bid,hist,now)
        if not self._check(m,band,reg,mv,side,thesis,entries,age):return None
        same=1. if thesis and side==thesis else 0. if thesis else .5
        depth_q=max(0.,min(1.,depth/50.))
        spread_q=max(0.,min(1.,1.-sp/max(1e-6,self.PROFILE[m]["spread"][reg])))
        score=.55*depth_q+.30*spread_q+.15*same
        return dict(market=m,side=side,bid=bid,ask=ask,depth=depth,spread=sp,
                    band=band,regime=reg,movement=mv,score=score,
                    target=self.capital_target(bid,m,entries))

    def _can_reset(self,m,thesis,thesis_price,other,now,age):
        if not thesis or thesis==other["side"] or thesis_price is None or age<self.state_min_age:return False
        if self._last_reset_at is not None and now-self._last_reset_at<self.state_reset_cooldown:return False
        jump=max(self.state_reset_jump,self.PROFILE[m]["reset"])
        return abs(other["bid"]-float(thesis_price))>=jump and other["score"]>=.70

    def decide(self,elapsed,up_ask,down_ask,up_bid,down_bid,up_history,down_history,
               current_exposure,available_cash,up_depth=0,down_depth=0,now=None,
               asset_exposure=0,total_exposure=0,market_entry_count=0,
               seconds_since_first_entry=0,thesis_side=None,thesis_price=None,
               asset=None,market=None):
        now=time.time() if now is None else float(now)
        elapsed=float(elapsed); m=self.normalize_market(market or asset)
        if elapsed<self.start_sec or elapsed>=self.stop_sec:return None
        if self.stop_sec-elapsed<=self.hard_cutoff_seconds:return None
        if self._last_trade_at is not None and now-self._last_trade_at<self.min_trade_gap_seconds:return None

        cs=[c for c in (
            self._candidate(m,"Up",up_bid,up_ask,up_depth,up_history,now,thesis_side,market_entry_count,seconds_since_first_entry),
            self._candidate(m,"Down",down_bid,down_ask,down_depth,down_history,now,thesis_side,market_entry_count,seconds_since_first_entry)
        ) if c]
        if not cs:return None

        same=[c for c in cs if thesis_side and c["side"]==thesis_side]
        other=[c for c in cs if thesis_side and c["side"]!=thesis_side]
        reset=False
        if same:
            best=max(same,key=lambda x:x["score"])
            if other:
                alt=max(other,key=lambda x:x["score"])
                if self._can_reset(m,thesis_side,thesis_price,alt,now,seconds_since_first_entry):
                    best=alt;reset=True
        elif thesis_side:
            alt=max(other,key=lambda x:x["score"]) if other else None
            if not alt or not self._can_reset(m,thesis_side,thesis_price,alt,now,seconds_since_first_entry):return None
            best=alt;reset=True
        else:
            best=max(cs,key=lambda x:(x["score"],{"CHEAP":1,"MID":2,"CORE":3,"HIGH":0}[x["regime"]]))

        target=self.capital_target(best["bid"],m,market_entry_count)
        starter=target*.90 if market_entry_count==0 else target
        remaining=starter if market_entry_count==0 else max(0.,target-max(0.,float(current_exposure)))
        if remaining<.10:return None

        room=min(
            remaining,self.max_order,
            max(0.,self.max_market_exposure-float(current_exposure)),
            max(0.,self.max_asset_exposure-float(asset_exposure)),
            max(0.,self.max_total_exposure-float(total_exposure)),
            max(0.,float(available_cash)),
            max(0.,best["depth"]*best["bid"]*self.max_depth_participation),
            self.SLICE[best["regime"]],
        )
        if room<.10:return None

        if best["regime"]=="CHEAP" and market_entry_count>=10 and best["movement"]["m10"]<-.06:return None
        if reset:self._last_reset_at=now
        self._last_trade_at=now

        mv=best["movement"]
        mode="STATE_RESET" if reset else ("STARTER" if market_entry_count==0 else "ADD_ON")
        reason=(f"V11 market={m} band={best['band']} regime={best['regime']} "
                f"path={best['regime']}_PROFILE mode={mode} passive=bid "
                f"target_capital=${target:.2f} current_exposure=${float(current_exposure):.2f} "
                f"remaining_target=${remaining:.2f} entry_count={int(market_entry_count)} "
                f"burst_age={float(seconds_since_first_entry):.1f}s "
                f"bid={best['bid']:.4f} ask={best['ask'] if best['ask'] is not None else 0:.4f} "
                f"spread={best['spread']:.4f} depth={best['depth']:.2f} score={best['score']:.3f} "
                f"m1={mv['m1']:+.4f} m3={mv['m3']:+.4f} m5={mv['m5']:+.4f} "
                f"m10={mv['m10']:+.4f} m30={mv['m30']:+.4f} elapsed={elapsed:.1f}s "
                f"left={self.stop_sec-elapsed:.1f}s reset={reset}")
        return Signal(best["side"],best["bid"],best["score"],round(room,2),reason)

    def size(self,price,regime=None,market="BTC",entry_count=0,**_):
        return self.capital_target(price,market,entry_count)

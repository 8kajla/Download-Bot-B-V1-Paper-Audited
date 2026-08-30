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
    """
    V12 TRADER-BEHAVIOR COPY.

    This version intentionally contains no invented "alpha" trigger. It encodes
    only measured behavioral dimensions:
      * market-specific profiles (BTC/ETH/SOL/BNB)
      * fine price bands
      * price-conditioned capital
      * segment-specific depth/spread requirements
      * observed side persistence
      * starter/add-on sizing
      * burst cadence
      * passive bid proxy
      * final-minute cutoff
      * exposure caps

    Unknown private trigger mechanics are not fabricated here.
    """

    VERSION="V12_TRADER_BEHAVIOR_COPY"

    BANDS=(
        ("C00_05",0.00,0.05,"CHEAP"),
        ("C05_10",0.05,0.10,"CHEAP"),
        ("C10_15",0.10,0.15,"CHEAP"),
        ("C15_20",0.15,0.20,"CHEAP"),
        ("C20_30",0.20,0.30,"CHEAP"),
        ("M30_40",0.30,0.40,"MID"),
        ("M40_50",0.40,0.50,"MID"),
        ("M50_60",0.50,0.60,"MID"),
        ("M60_70",0.60,0.70,"MID"),
        ("R70_80",0.70,0.80,"CORE"),
        ("R80_90",0.80,0.90,"CORE"),
        ("H90_95",0.90,0.95,"HIGH"),
        ("H95_100",0.95,1.00,"HIGH"),
    )

    # Median/representative trader notionals from the verified price buckets.
    # The top tail is intentionally allowed to be much larger.
    BASE_CAPITAL={
        "C00_05":0.41,
        "C05_10":0.51,
        "C10_15":0.68,
        "C15_20":0.83,
        "C20_30":0.95,
        "M30_40":1.48,
        "M40_50":2.17,
        "M50_60":3.08,
        "M60_70":3.93,
        "R70_80":5.25,
        "R80_90":8.28,
        "H90_95":14.99,
        "H95_100":30.78,
    }

    # Observed trade-count shares by market x coarse regime from the combined
    # sample. These are behavioral priors, not hard quotas.
    MARKET_REGIME_WEIGHT={
        "BTC":{"CHEAP":0.39,"MID":0.40,"CORE":0.12,"HIGH":0.09},
        "ETH":{"CHEAP":0.54,"MID":0.28,"CORE":0.09,"HIGH":0.08},
        "SOL":{"CHEAP":0.65,"MID":0.22,"CORE":0.07,"HIGH":0.05},
        "BNB":{"CHEAP":0.57,"MID":0.27,"CORE":0.11,"HIGH":0.05},
    }

    # Segment-specific book requirements. These are execution filters, not an
    # invented predictive trigger.
    MARKET_CHECKS={
        "BTC":{"depth":{"CHEAP":1.0,"MID":2.0,"CORE":4.0,"HIGH":8.0},
               "spread":{"CHEAP":0.060,"MID":0.050,"CORE":0.035,"HIGH":0.025}},
        "ETH":{"depth":{"CHEAP":1.0,"MID":2.0,"CORE":3.0,"HIGH":6.0},
               "spread":{"CHEAP":0.065,"MID":0.055,"CORE":0.040,"HIGH":0.027}},
        "SOL":{"depth":{"CHEAP":1.0,"MID":2.0,"CORE":4.0,"HIGH":8.0},
               "spread":{"CHEAP":0.060,"MID":0.050,"CORE":0.035,"HIGH":0.025}},
        "BNB":{"depth":{"CHEAP":1.0,"MID":2.0,"CORE":4.0,"HIGH":8.0},
               "spread":{"CHEAP":0.060,"MID":0.050,"CORE":0.035,"HIGH":0.025}},
    }

    SLICE_CAP={"CHEAP":0.40,"MID":0.85,"CORE":2.50,"HIGH":5.00}

    HARD_MAX_ORDER=10.0
    HARD_MAX_MARKET=100.0
    HARD_MAX_ASSET=35.0
    HARD_MAX_TOTAL=300.0
    HARD_CUTOFF=60.0

    def __init__(
        self, bankroll=1000,max_market_exposure=100,max_order=10,
        max_asset_exposure=35,max_total_exposure=300,start_sec=0,
        stop_sec=240,hard_cutoff_seconds=60,max_depth_participation=.25,
        min_trade_gap_seconds=2,min_bid_depth=1,state_reset_jump=.35,
        state_reset_cooldown=30,state_min_age=45,**_,
    ):
        self.bankroll=float(bankroll)
        self.max_market_exposure=min(float(max_market_exposure),100.0)
        self.max_order=min(float(max_order),10.0)
        self.max_asset_exposure=min(float(max_asset_exposure),35.0)
        self.max_total_exposure=min(float(max_total_exposure),300.0)
        self.start_sec=max(0.,float(start_sec))
        self.stop_sec=min(240.,float(stop_sec))
        self.hard_cutoff_seconds=max(60.,float(hard_cutoff_seconds))
        self.max_depth_participation=min(.25,max(.01,float(max_depth_participation)))
        self.min_trade_gap_seconds=max(0.,float(min_trade_gap_seconds))
        self.min_bid_depth=max(0.,float(min_bid_depth))
        self.state_reset_jump=max(.20,float(state_reset_jump))
        self.state_reset_cooldown=max(0.,float(state_reset_cooldown))
        self.state_min_age=max(0.,float(state_min_age))
        self._last_trade_at=None
        self._last_reset_at=None

    @staticmethod
    def normalize_market(x):
        s=str(x or "").upper()
        for m in ("BTC","ETH","SOL","BNB"):
            if m in s:return m
        return "BTC"

    @classmethod
    def fine_band(cls,p):
        p=float(p)
        for b,lo,hi,r in cls.BANDS:
            if lo<=p<hi:return b,r
        return None,None

    @classmethod
    def capital_target(cls,p,market="BTC",entry_count=0):
        b,r=cls.fine_band(p)
        if not b:return 0.
        m=cls.normalize_market(market)
        v=cls.BASE_CAPITAL[b]
        # Use the observed market differences conservatively: the distribution
        # evidence is much stronger than any claim about different hidden alpha.
        if entry_count>=8:v*=1.10
        elif entry_count>=4:v*=1.05
        elif entry_count>=1:v*=1.02
        return max(.20,v)

    def desired_capital(self,p,regime=None,market="BTC",entry_count=0):
        return self.capital_target(p,market,entry_count)

    @staticmethod
    def _points(hist):
        out=[]
        for x in hist or []:
            try:
                if isinstance(x,dict):
                    out.append((float(x["ts"]),float(x.get("best_bid",x.get("mid")))))
                else:
                    out.append((float(x[0]),float(x[1])))
            except (TypeError,ValueError,KeyError,IndexError):
                continue
        return sorted((t,p) for t,p in out if 0<p<1)

    @classmethod
    def movement(cls,p,hist,now):
        pts=cls._points(hist)
        if not pts:return {f"m{s}":0.0 for s in (1,3,5,10,30)}
        def at(sec):
            return min(pts,key=lambda x:abs(x[0]-(float(now)-sec)))[1]
        return {f"m{s}":float(p)-at(s) for s in (1,3,5,10,30)}

    def _book_ok(self,m,r,bid,ask,depth):
        req=max(self.min_bid_depth,self.MARKET_CHECKS[m]["depth"][r])
        if depth<req:return False,0.0
        spread=0.0 if ask is None else max(0.,float(ask)-float(bid))
        return spread<=self.MARKET_CHECKS[m]["spread"][r],spread

    def _candidate(self,m,side,bid,ask,depth,hist,now,thesis_side,entries,burst_age):
        if bid is None:return None
        try:
            bid=float(bid);depth=max(0.,float(depth or 0.))
            ask=None if ask is None else float(ask)
        except (TypeError,ValueError):
            return None
        if not 0<bid<1:return None

        band,reg=self.fine_band(bid)
        if not reg:return None

        ok,spread=self._book_ok(m,reg,bid,ask,depth)
        if not ok:return None

        # We do NOT use momentum as a hidden alpha score. The only directional
        # behavior encoded is the verified side-persistence preference.
        same_side=1.0 if thesis_side and side==thesis_side else 0.0 if thesis_side else 0.5

        depth_q=min(1.,depth/20.)
        spread_q=max(0.,1.-spread/max(1e-6,self.MARKET_CHECKS[m]["spread"][reg]))
        weight=self.MARKET_REGIME_WEIGHT[m][reg]

        # This ranks valid trader-like opportunities according to observed
        # frequency, while capital remains controlled separately.
        score=.45*depth_q+.25*spread_q+.20*weight+.10*same_side

        return {
            "market":m,"side":side,"bid":bid,"ask":ask,"depth":depth,
            "spread":spread,"band":band,"regime":reg,"score":score,
            "target":self.capital_target(bid,m,entries),
            "movement":self.movement(bid,hist,now),
        }

    def _can_reset(self,m,thesis_side,thesis_price,other,now,burst_age):
        if not thesis_side or thesis_side==other["side"] or thesis_price is None:return False
        if burst_age<self.state_min_age:return False
        if self._last_reset_at is not None and now-self._last_reset_at<self.state_reset_cooldown:return False
        jump=max(self.state_reset_jump,self.MARKET_CHECKS[m]["depth"]["CHEAP"]*.1)
        return abs(other["bid"]-float(thesis_price))>=jump and other["score"]>=.65

    def decide(
        self,elapsed,up_ask,down_ask,up_bid,down_bid,up_history,down_history,
        current_exposure,available_cash,up_depth=0,down_depth=0,now=None,
        asset_exposure=0,total_exposure=0,market_entry_count=0,
        seconds_since_first_entry=0,thesis_side=None,thesis_price=None,
        asset=None,market=None,
    ):
        now=time.time() if now is None else float(now)
        elapsed=float(elapsed)
        m=self.normalize_market(market or asset)

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
            best=max(same,key=lambda c:c["score"])
            if other:
                alt=max(other,key=lambda c:c["score"])
                if self._can_reset(m,thesis_side,thesis_price,alt,now,float(seconds_since_first_entry)):
                    best=alt
                    reset=True
        elif thesis_side:
            # The trader flips occasionally; only a large state change permits
            # a flip, but HIGH does not require a pre-existing position.
            alt=max(other,key=lambda c:c["score"]) if other else None
            if not alt:return None
            if not self._can_reset(m,thesis_side,thesis_price,alt,now,float(seconds_since_first_entry)):
                return None
            best=alt
            reset=True
        else:
            best=max(cs,key=lambda c:c["score"])

        target=best["target"]
        starter=target*0.90 if market_entry_count==0 else target
        remaining=starter if market_entry_count==0 else max(0.,target-max(0.,float(current_exposure)))
        if remaining<.10:return None

        room=min(
            remaining,self.max_order,
            max(0.,self.max_market_exposure-float(current_exposure)),
            max(0.,self.max_asset_exposure-float(asset_exposure)),
            max(0.,self.max_total_exposure-float(total_exposure)),
            max(0.,float(available_cash)),
            max(0.,best["depth"]*best["bid"]*self.max_depth_participation),
            self.SLICE_CAP[best["regime"]],
        )
        if room<.10:return None

        if reset:self._last_reset_at=now
        self._last_trade_at=now

        mv=best["movement"]
        mode="STATE_RESET" if reset else ("STARTER" if market_entry_count==0 else "ADD_ON")
        reason=(
            f"V12 market={m} band={best['band']} regime={best['regime']} "
            f"profile=TRADER_BEHAVIOR mode={mode} passive=bid "
            f"target=${target:.2f} current=${float(current_exposure):.2f} "
            f"remaining=${remaining:.2f} entry_count={int(market_entry_count)} "
            f"burst_age={float(seconds_since_first_entry):.1f}s "
            f"bid={best['bid']:.4f} ask={best['ask'] if best['ask'] is not None else 0:.4f} "
            f"spread={best['spread']:.4f} depth={best['depth']:.2f} "
            f"weight={self.MARKET_REGIME_WEIGHT[m][best['regime']]:.3f} "
            f"score={best['score']:.3f} "
            f"m1={mv['m1']:+.4f} m3={mv['m3']:+.4f} m5={mv['m5']:+.4f} "
            f"m10={mv['m10']:+.4f} m30={mv['m30']:+.4f} "
            f"elapsed={elapsed:.1f}s left={self.stop_sec-elapsed:.1f}s reset={reset}"
        )
        return Signal(best["side"],best["bid"],best["score"],round(room,2),reason)

    def size(self,price,regime=None,market="BTC",entry_count=0,**_):
        return self.capital_target(price,market,entry_count)

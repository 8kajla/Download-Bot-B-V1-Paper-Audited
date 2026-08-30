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
    VERSION = "V8_CAPITAL_FIRST"
    CHEAP_MIN, CHEAP_MAX = 0.01, 0.30
    MID_MIN, MID_MAX = 0.30, 0.70
    CORE_MIN, CORE_MAX = 0.70, 0.90
    HIGH_MIN, HIGH_MAX = 0.90, 0.995
    HARD_MAX_ORDER = 10.0
    HARD_MAX_MARKET_EXPOSURE = 100.0
    HARD_MAX_TOTAL_EXPOSURE = 300.0
    HARD_MAX_ASSET_EXPOSURE = 35.0
    HARD_CUTOFF_SECONDS = 60.0
    CAPITAL_ANCHORS = (
        (0.01,0.35),(0.05,0.41),(0.15,0.65),(0.295,0.81),
        (0.30,1.20),(0.50,2.02),(0.695,2.30),(0.70,3.50),
        (0.80,5.03),(0.895,6.00),(0.90,5.70),(0.95,14.47),
        (0.98,20.00),(0.995,28.00),
    )
    def __init__(self, bankroll=1000, max_market_exposure=100, max_order=10,
                 max_asset_exposure=35, max_total_exposure=300,
                 start_sec=0, stop_sec=240, hard_cutoff_seconds=60,
                 max_depth_participation=0.25, min_trade_gap_seconds=2,
                 min_bid_depth=1, state_reset_jump=0.35,
                 state_reset_cooldown=30, state_min_age=45, **_):
        self.bankroll=float(bankroll)
        self.max_market_exposure=min(max(0,float(max_market_exposure)),100)
        self.max_order=min(max(0,float(max_order)),10)
        self.max_asset_exposure=min(max(0,float(max_asset_exposure)),35)
        self.max_total_exposure=min(max(0,float(max_total_exposure)),300)
        self.start_sec=max(0,float(start_sec)); self.stop_sec=min(240,float(stop_sec))
        self.hard_cutoff_seconds=max(60,float(hard_cutoff_seconds))
        self.max_depth_participation=min(.25,max(.01,float(max_depth_participation)))
        self.min_trade_gap_seconds=max(0,float(min_trade_gap_seconds))
        self.min_bid_depth=max(0,float(min_bid_depth))
        self.state_reset_jump=max(.20,float(state_reset_jump))
        self.state_reset_cooldown=max(0,float(state_reset_cooldown))
        self.state_min_age=max(0,float(state_min_age))
        self._last_trade_at=None; self._last_reset_at=None
    @staticmethod
    def _clamp(v,lo=0,hi=1): return max(lo,min(hi,float(v)))
    def _regime(self,p):
        p=float(p)
        if .01<=p<.30:return "CHEAP"
        if .30<=p<.70:return "MID"
        if .70<=p<.90:return "CORE"
        if .90<=p<.995:return "HIGH"
        return None
    @staticmethod
    def _points(history,now):
        out=[]
        for x in history or []:
            try:
                if isinstance(x,dict): t=float(x["ts"]); p=float(x.get("best_bid",x.get("mid")))
                else: t=float(x[0]); p=float(x[1])
                if 0<p<1 and t<=now: out.append((t,p))
            except (TypeError,ValueError,KeyError,IndexError): pass
        return sorted(out)
    @classmethod
    def _movement(cls,price,history,now):
        pts=cls._points(history,now)
        if not pts:return 0,0,0
        def at(s): return min(pts,key=lambda x:abs(x[0]-(now-s)))[1]
        p10,p30=at(10),at(30); price=float(price)
        return price-p30,(price-p10)-(p10-p30),len(pts)
    @classmethod
    def _interpolate_capital(cls,p):
        p=float(p); a=cls.CAPITAL_ANCHORS
        if p<=a[0][0]:return a[0][1]
        if p>=a[-1][0]:return a[-1][1]
        for (p0,c0),(p1,c1) in zip(a,a[1:]):
            if p0<=p<=p1:
                w=((p-p0)/max(1e-9,p1-p0))**.85
                return c0+(c1-c0)*w
        return a[-1][1]
    def desired_capital(self,p,regime=None):
        r=regime or self._regime(p)
        if not r:return 0
        c=self._interpolate_capital(p)
        bounds={"CHEAP":(.35,.81),"MID":(1.20,2.30),"CORE":(3.50,6.00),"HIGH":(5.70,28.00)}[r]
        return max(bounds[0],min(bounds[1],c))
    @staticmethod
    def _quality(bid,bid_size,ask):
        q=1-1/(1+max(0,float(bid_size or 0))/5)
        sp=max(0,float(ask)-float(bid)) if ask is not None else .01
        sq=max(0,1-max(0,sp-.01)/.04)
        return .60*q+.40*sq
    def _candidate(self,side,bid,ask,depth,hist,now):
        if bid is None:return None
        try: bid=float(bid); depth=max(0,float(depth or 0)); ask=None if ask is None else float(ask)
        except (TypeError,ValueError):return None
        if not 0<bid<1 or depth<self.min_bid_depth:return None
        if ask is not None and ask<bid:return None
        reg=self._regime(bid)
        if not reg:return None
        sp=(ask-bid) if ask is not None else .01
        mom,acc,n=self._movement(bid,hist,now)
        quality=self._quality(bid,depth,ask)
        if reg=="CHEAP": fit=self._clamp((-mom+.003)/.03)
        elif reg in ("CORE","HIGH"): fit=self._clamp((mom+.001)/.03)
        else: fit=self._clamp(1-abs(mom)/.04)
        target=self.desired_capital(bid,reg)
        rank=.58*quality+.27*fit+.15*self._clamp(target/28)
        return {"side":side,"bid":bid,"ask":ask,"depth":depth,"spread":sp,"regime":reg,
                "momentum":mom,"acceleration":acc,"history_samples":n,"quality":quality,
                "state_fit":fit,"target_capital":target,"rank":rank}
    def _can_reset(self,thesis_side,thesis_price,other,now,age):
        if not thesis_side or thesis_side==other["side"]:return False
        if thesis_price is None:return True
        if age<self.state_min_age:return False
        if self._last_reset_at is not None and now-self._last_reset_at<self.state_reset_cooldown:return False
        return abs(other["bid"]-float(thesis_price))>=self.state_reset_jump and other["rank"]>=.70
    def decide(self,elapsed,up_ask,down_ask,up_bid,down_bid,up_history,down_history,
               current_exposure,available_cash,up_depth=0,down_depth=0,now=None,
               asset_exposure=0,total_exposure=0,market_entry_count=0,
               seconds_since_first_entry=0,thesis_side=None,thesis_price=None):
        now=time.time() if now is None else float(now); elapsed=float(elapsed)
        if elapsed<self.start_sec or elapsed>=self.stop_sec:return None
        if self.stop_sec-elapsed<self.hard_cutoff_seconds:return None
        if self._last_trade_at is not None and now-self._last_trade_at<self.min_trade_gap_seconds:return None
        candidates=[c for c in (
            self._candidate("Up",up_bid,up_ask,up_depth,up_history,now),
            self._candidate("Down",down_bid,down_ask,down_depth,down_history,now)) if c]
        if not candidates:return None
        same=[c for c in candidates if thesis_side and c["side"]==thesis_side]
        other=[c for c in candidates if thesis_side and c["side"]!=thesis_side]
        same_best=max(same,key=lambda c:c["rank"]) if same else None
        other_best=max(other,key=lambda c:c["rank"]) if other else None
        reset=False
        if same_best is not None:
            best=same_best
            if other_best and self._can_reset(thesis_side,thesis_price,other_best,now,float(seconds_since_first_entry)):
                best=other_best; reset=True
        elif thesis_side:
            if not other_best or not self._can_reset(thesis_side,thesis_price,other_best,now,float(seconds_since_first_entry)):return None
            best=other_best; reset=True
        else: best=max(candidates,key=lambda c:c["rank"])
        remaining_target=max(0,best["target_capital"]-max(0,float(current_exposure)))
        if remaining_target<.10:return None
        room=min(remaining_target,self.max_order,max(0,self.max_market_exposure-current_exposure),
                 max(0,self.max_asset_exposure-asset_exposure),max(0,self.max_total_exposure-total_exposure),
                 max(0,float(available_cash)),max(0,best["depth"]*best["bid"]*self.max_depth_participation))
        if room<.10:return None
        if reset:self._last_reset_at=now
        self._last_trade_at=now
        mode="STATE_RESET" if reset else ("STARTER" if market_entry_count==0 else "ADD_ON")
        reason=(f"V8 regime={best['regime']} mode={mode} passive=bid target_capital=${best['target_capital']:.2f} "
                f"current_exposure=${float(current_exposure):.2f} remaining_target=${remaining_target:.2f} "
                f"entry_count={int(market_entry_count)} bid={best['bid']:.4f} ask={best['ask'] if best['ask'] is not None else 0:.4f} "
                f"spread={best['spread']:.4f} bid_depth={best['depth']:.2f} book_quality={best['quality']:.3f} "
                f"state_fit={best['state_fit']:.3f} momentum={best['momentum']:+.4f} accel={best['acceleration']:+.4f} "
                f"history_samples={best['history_samples']} elapsed={elapsed:.1f}s left={self.stop_sec-elapsed:.1f}s reset={reset}")
        return Signal(best["side"],best["bid"],self._clamp(best["rank"]),round(room,2),reason)

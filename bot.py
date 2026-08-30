import os,time,traceback,logging,shutil
from pathlib import Path
from market_discovery import discover,book,resolve
from strategy import CapitalFirstStrategy
from paper_ledger import PaperLedger
from research_logger import ResearchLogger
logging.basicConfig(level=logging.INFO,format="%(asctime)s UTC %(levelname)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
log=logging.getLogger("bot")

def prepare_fresh_data_dir():
    d=Path(os.getenv("DATA_DIR","/app/data")).expanduser(); fresh=os.getenv("FRESH_START","true").lower() in ("1","true","yes","on")
    if str(d) in ("/",".",""): raise RuntimeError(f"Refusing to wipe unsafe DATA_DIR={d!r}")
    d.mkdir(parents=True,exist_ok=True)
    if fresh:
        for c in d.iterdir(): shutil.rmtree(c) if c.is_dir() else c.unlink()
    return d
DATA=prepare_fresh_data_dir()
if os.getenv("PAPER_TRADING","true").lower()!="true": raise SystemExit("SAFETY LOCK: PAPER_TRADING must be true")
strategy=CapitalFirstStrategy(
    bankroll=float(os.getenv("STARTING_CAPITAL","1000")),max_market_exposure=float(os.getenv("MAX_MARKET_EXPOSURE","100")),
    max_order=float(os.getenv("MAX_ORDER_USD","10")),max_asset_exposure=float(os.getenv("MAX_ASSET_EXPOSURE","35")),
    max_total_exposure=float(os.getenv("MAX_TOTAL_EXPOSURE","300")),start_sec=float(os.getenv("START_TRADING_SECOND","0")),
    stop_sec=float(os.getenv("STOP_TRADING_SECOND","240")),hard_cutoff_seconds=float(os.getenv("HARD_CUTOFF_SECONDS","60")),
    max_depth_participation=float(os.getenv("MAX_DEPTH_PARTICIPATION","0.25")),min_trade_gap_seconds=float(os.getenv("MIN_TRADE_GAP_SECONDS","2")),
    min_bid_depth=float(os.getenv("MIN_BID_DEPTH","1")),state_reset_jump=float(os.getenv("STATE_RESET_JUMP","0.35")),
    state_reset_cooldown=float(os.getenv("STATE_RESET_COOLDOWN","30")),state_min_age=float(os.getenv("STATE_MIN_AGE","45")))
ledger=PaperLedger(DATA/"paper_state.json",strategy.bankroll); ledger.save(); research=ResearchLogger(DATA,ledger)
markets={};histories={};pending={};last_disc=last_report=last_maintenance=0.;last_trade={};ob_last={};decision_last={};consecutive_errors=0

def asset_exposure(asset): return sum(float(p.get("cost",0)) for p in ledger.positions.values() if p.get("asset")==asset)
def p(msg): log.info(msg)
def startup_data_check():
    req=["decisions.jsonl","orderbooks.jsonl","trades.csv","markets.csv","resolutions.csv","pnl_1min.csv","paper_state.json"]
    miss=[x for x in req if not (DATA/x).exists()]
    if miss: raise RuntimeError(f"DATA STORE INITIALIZATION FAILED: {miss}")
def resolve_pending(now):
    for c,m in list(pending.items()):
        if now<float(m.get("end_ts",0))+2: continue
        try:
            token,outcome,status=resolve(m)
            if token:
                closed=ledger.settle(c,token); pnl=sum(float(x["pnl"]) for x in closed)
                research.record_resolution(ts=now,market=m,winner=outcome or token,winner_token=token,closed=closed)
                p(f"RESOLUTION | asset={m['asset']} | slug={m['slug']} | winner={outcome or token} | pnl={pnl:+.4f} | closed={len(closed)}")
                pending.pop(c,None);markets.pop(c,None);histories.pop(c,None)
            elif status=="CLOSED_UNRESOLVED": research.record_resolution_error(ts=now,market=m,status=status)
        except Exception as e:
            research.record_resolution_error(ts=now,market=m,status=f"ERROR:{type(e).__name__}"); p(f"RESOLUTION ERROR | {m['slug']} | {type(e).__name__}: {e}")
def report(books):
    global last_report
    now=time.time()
    if now-last_report<60:return
    last_report=now;m=ledger.mark(books);m["positions"]=len(ledger.positions);research.record_pnl(now,m)
    p(f"P&L ours ${m['pnl']:+.2f} | realized ${m['realized']:+.2f} | unrealized ${m['unrealized']:+.2f} | cash ${m['cash']:.2f} | open ${m['open_cost']:.2f} | positions {m['positions']}")
def market_entry_state(c,now):
    e=[t for t in ledger.trades if t.get("action")=="BUY" and t.get("condition")==c]
    if not e:return 0,0.,None,None
    f=min(float(t.get("ts",now)) for t in e);l=max(e,key=lambda t:float(t.get("ts",now)))
    return len(e),max(0,now-f),l.get("side"),l.get("price")
def main():
    global last_disc,last_maintenance,consecutive_errors
    startup_data_check();p("BOT B | PAPER ONLY | V8 CAPITAL-FIRST PASSIVE MODEL")
    while True:
        try:
            now=time.time()
            if now-last_disc>=20:
                for m in discover():markets[m["condition"]]=m
                for c,m in list(markets.items()):
                    if any(q.get("condition")==c for q in ledger.positions.values()): pending[c]=m
                    elif m["end_ts"]<now-30:markets.pop(c,None)
                last_disc=now;p(f"MARKETS | active={len(markets)} | pending_resolution={len(pending)}")
            resolve_pending(now);books={}
            for m in list(markets.values()):
                if not m.get("end_ts") or m["end_ts"]<now-30:continue
                elapsed=now-m["start_ts"];left=m["end_ts"]-now
                if left<=0 or elapsed<0 or elapsed>300:continue
                try:ub,ua,ubs,uas=book(m["up"]);db,da,dbs,das=book(m["down"])
                except Exception as e:p(f"BOOK ERROR | {m['asset']} | {m['slug']} | {type(e).__name__}: {e}");continue
                books[m["up"]]=ub;books[m["down"]]=db;h=histories.setdefault(m["condition"],{"Up":[],"Down":[]})
                if ub is not None:h["Up"].append((now,ub));h["Up"]=h["Up"][-60:]
                if db is not None:h["Down"].append((now,db));h["Down"]=h["Down"][-60:]
                if now-ob_last.get(m["condition"],0)>=float(os.getenv("ORDERBOOK_SAMPLE_SECONDS","15")):
                    research.record_orderbook(ts=now,market=m,elapsed=elapsed,left=left,up_bid=ub,up_ask=ua,up_depth=ubs,down_bid=db,down_ask=da,down_depth=dbs);ob_last[m["condition"]]=now
                if not m["accepting_orders"]:continue
                exp=ledger.exposure(m["condition"]);aexp=asset_exposure(m["asset"]);total=ledger.total_open_cost();ec,sfirst,tside,tprice=market_entry_state(m["condition"],now)
                sig=strategy.decide(elapsed,ua,da,ub,db,h["Up"],h["Down"],exp,ledger.cash,up_depth=ubs,down_depth=dbs,now=now,asset_exposure=aexp,total_exposure=total,market_entry_count=ec,seconds_since_first_entry=sfirst,thesis_side=tside,thesis_price=tprice,asset=m["asset"],market=m["asset"])
                interval=float(os.getenv("DECISION_SAMPLE_SECONDS","10"))
                if sig is not None or now-decision_last.get(m["condition"],0)>=interval:
                    research.record_decision(ts=now,market=m,elapsed=elapsed,left=left,up_bid=ub,up_ask=ua,up_depth=ubs,down_bid=db,down_ask=da,down_depth=dbs,signal=sig,exposure=exp,cash=ledger.cash);decision_last[m["condition"]]=now
                if not sig or left<=strategy.hard_cutoff_seconds or now-last_trade.get(m["condition"],0)<strategy.min_trade_gap_seconds:continue
                token=m["up"] if sig.side=="Up" else m["down"];bid_size=ubs if sig.side=="Up" else dbs
                depth_cap=max(0,float(bid_size)*float(sig.price)*strategy.max_depth_participation);rem_market=max(0,strategy.max_market_exposure-ledger.exposure(m["condition"]));rem_asset=max(0,strategy.max_asset_exposure-asset_exposure(m["asset"]));rem_total=max(0,strategy.max_total_exposure-ledger.total_open_cost())
                notion=min(sig.notional,strategy.max_order,depth_cap,rem_market,rem_asset,rem_total,max(0,ledger.cash))
                if notion<float(os.getenv("MIN_PAPER_FILL_USD","0.10")):continue
                meta={"slug":m["slug"],"asset":m["asset"],"start_ts":m["start_ts"],"end_ts":m["end_ts"],"market_id":m["id"],"up_token":m["up"],"down_token":m["down"],"model_version":"V11_SIMPLE_PROFILE","entry_count_before":ec,"seconds_since_first_entry":sfirst,"regime":strategy.fine_band(sig.price)[1], "fine_band":strategy.fine_band(sig.price)[0],"execution_mode":"PASSIVE_BID_PROXY","target_capital":strategy.desired_capital(sig.price),"bid_size":bid_size}
                t=ledger.buy(m["condition"],token,m["market"],sig.side,sig.price,notion,now,meta);pending[m["condition"]]=m;last_trade[m["condition"]]=now
                p(f"TRADE PAPER | V11 SIMPLE | asset={m['asset']} | side={sig.side} | notional=${notion:.2f} | bid=${sig.price:.4f} | target=${meta['target_capital']:.2f} | entry_count={ec} | {sig.reason}")
                research.record_trade(ts=now,market=m,elapsed=elapsed,left=left,up_bid=ub,up_ask=ua,up_depth=ubs,down_bid=db,down_ask=da,down_depth=dbs,trade=t,score=sig.score,momentum=None,reason=sig.reason,cash_after=ledger.cash,exposure_after=ledger.exposure(m["condition"]));ledger.save()
            report(books)
            if now-last_maintenance>=float(os.getenv("DATA_MAINTENANCE_SECONDS","3600")):research.maintenance();last_maintenance=now
            consecutive_errors=0;time.sleep(max(.05,float(os.getenv("LOOP_SECONDS","1"))))
        except KeyboardInterrupt: break
        except Exception as e:
            consecutive_errors+=1;p(f"LOOP ERROR | {type(e).__name__}: {e}");traceback.print_exc()
            if consecutive_errors>=10:raise
            time.sleep(2)
if __name__=="__main__":main()

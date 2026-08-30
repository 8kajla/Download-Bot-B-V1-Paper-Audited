import csv, json, os, threading, time, uuid
from pathlib import Path
from collections import defaultdict

SCHEMAS={
"trades.csv":["trade_id","timestamp","market_id","condition","slug","asset","market","side","token","price","shares","notional","seconds_into_market","seconds_remaining","up_bid","up_ask","up_depth","down_bid","down_ask","down_depth","spread","score","momentum","signal_reason","cash_after","market_exposure_after"],
"markets.csv":["market_id","condition","slug","asset","market","start_ts","end_ts","winner","entries","total_cost","total_shares","avg_entry","first_entry","last_entry","max_exposure","up_cost","down_cost","up_shares","down_shares","winning_cost","losing_cost","payout","realized_pnl","roi","resolved_ts"],
"resolutions.csv":["timestamp","market_id","condition","slug","asset","winner","winner_token","entries","cost","payout","pnl","roi","status"],
"pnl_1min.csv":["timestamp","equity","total_pnl","realized_pnl","unrealized_pnl","cash","open_cost","market_value","drawdown","positions","marked"]}

class ResearchLogger:
    def __init__(self,data_dir,ledger=None):
        self.root=Path(data_dir); self.root.mkdir(parents=True,exist_ok=True); self.lock=threading.Lock(); self.last_resolution_error={}; self._trade_cache=defaultdict(list)
        self.market_stats=defaultdict(lambda:{"entries":0,"cost":0.,"shares":0.,"first_entry":None,"last_entry":None,"max_exposure":0.,"asset":"","market":"","up_cost":0.,"down_cost":0.,"up_shares":0.,"down_shares":0.,"slug":"","market_id":"","start_ts":0.,"end_ts":0.})
        self._ensure_files()
        if ledger is not None:self.rebuild_from_ledger(ledger)
    def _ensure_files(self):
        for fn,fields in SCHEMAS.items():
            p=self.root/fn
            if not p.exists() or p.stat().st_size==0:
                with p.open("w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(fields)
        for fn in ("decisions.jsonl","orderbooks.jsonl"): (self.root/fn).touch(exist_ok=True)
    def _append_csv(self,fn,row):
        with self.lock,(self.root/fn).open("a",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=SCHEMAS[fn],extrasaction="ignore").writerow(row); f.flush()
    def _append_jsonl(self,fn,obj):
        with self.lock,(self.root/fn).open("a",encoding="utf-8") as f: f.write(json.dumps(obj,separators=(",",":"),ensure_ascii=False)+"\n"); f.flush()
    def record_decision(self,**kw):
        m=kw["market"]; sig=kw.get("signal"); ts=kw["ts"]; action=sig.side if sig else "WAIT"; ub,ua,db,da=kw.get("up_bid"),kw.get("up_ask"),kw.get("down_bid"),kw.get("down_ask")
        self._append_jsonl("decisions.jsonl",{"t":round(ts,3),"m":m["id"],"c":m["condition"],"s":m["slug"],"a":m["asset"],"e":round(kw["elapsed"],1),"r":round(kw["left"],1),"ub":ub,"ua":ua,"ud":kw.get("up_depth"),"db":db,"da":da,"dd":kw.get("down_depth"),"us":ua-ub if ua is not None and ub is not None else None,"ds":da-db if da is not None and db is not None else None,"x":action,"p":sig.price if sig else None,"score":sig.score if sig else None,"n":sig.notional if sig else 0.,"reason":sig.reason if sig else "no_signal","ex":kw.get("exposure",0.),"cash":kw.get("cash",0.)})
    def record_orderbook(self,**kw):
        m=kw["market"]; self._append_jsonl("orderbooks.jsonl",{"t":round(kw["ts"],3),"m":m["id"],"c":m["condition"],"s":m["slug"],"a":m["asset"],"e":round(kw["elapsed"],1),"r":round(kw["left"],1),"ub":kw.get("up_bid"),"ua":kw.get("up_ask"),"ud":kw.get("up_depth"),"db":kw.get("down_bid"),"da":kw.get("down_ask"),"dd":kw.get("down_depth")})
    def record_trade(self,**kw):
        t,m=kw["trade"],kw["market"]; tid=str(t.get("trade_id") or f"paper-{uuid.uuid4().hex}"); row={"trade_id":tid,"timestamp":t["ts"],"market_id":t.get("market_id",m["id"]),"condition":t["condition"],"slug":t.get("slug",m["slug"]),"asset":t.get("asset",m["asset"]),"market":t.get("market",m["market"]),"side":t["side"],"token":t["token"],"price":t["price"],"shares":t["shares"],"notional":t["notional"],"seconds_into_market":kw["elapsed"],"seconds_remaining":kw["left"],"up_bid":kw.get("up_bid"),"up_ask":kw.get("up_ask"),"up_depth":kw.get("up_depth"),"down_bid":kw.get("down_bid"),"down_ask":kw.get("down_ask"),"down_depth":kw.get("down_depth"),"spread":(kw["up_ask"]-kw["up_bid"]) if t["side"]=="Up" and kw.get("up_ask") is not None and kw.get("up_bid") is not None else ((kw["down_ask"]-kw["down_bid"]) if t["side"]=="Down" and kw.get("down_ask") is not None and kw.get("down_bid") is not None else None),"score":kw.get("score"),"momentum":kw.get("momentum"),"signal_reason":kw.get("reason"),"cash_after":kw.get("cash_after"),"market_exposure_after":kw.get("exposure_after")}; self._append_csv("trades.csv",row); self._trade_cache[t["condition"]].append(t); s=self.market_stats[t["condition"]]; s["entries"]+=1;s["cost"]+=float(t["notional"]);s["shares"]+=float(t["shares"]);s["first_entry"]=t["ts"] if s["first_entry"] is None else min(s["first_entry"],t["ts"]);s["last_entry"]=t["ts"];s["max_exposure"]=max(s["max_exposure"],float(kw.get("exposure_after",0)));s["asset"]=t.get("asset",m["asset"]);s["market"]=t.get("market",m["market"]);s["slug"]=m["slug"];s["market_id"]=m["id"];s["start_ts"]=m["start_ts"];s["end_ts"]=m["end_ts"]
        if t["side"]=="Up":s["up_cost"]+=float(t["notional"]);s["up_shares"]+=float(t["shares"])
        else:s["down_cost"]+=float(t["notional"]);s["down_shares"]+=float(t["shares"])
    def record_resolution(self,**kw):
        m=kw["market"]; closed=kw["closed"]; s=self.market_stats[m["condition"]]; cost=float(s["cost"]); pnl=sum(float(x["pnl"]) for x in closed); payout=cost+pnl; avg=cost/s["shares"] if s["shares"] else 0.; wc=sum(float(t.get("notional",0)) for t in self._trade_cache[m["condition"]] if t.get("token")==kw["winner_token"]); self._append_csv("resolutions.csv",{"timestamp":kw["ts"],"market_id":m["id"],"condition":m["condition"],"slug":m["slug"],"asset":m["asset"],"winner":kw["winner"],"winner_token":kw["winner_token"],"entries":s["entries"],"cost":cost,"payout":payout,"pnl":pnl,"roi":pnl/cost if cost else 0.,"status":"RESOLVED"}); self._append_csv("markets.csv",{"market_id":m["id"],"condition":m["condition"],"slug":m["slug"],"asset":m["asset"],"market":m["market"],"start_ts":m["start_ts"],"end_ts":m["end_ts"],"winner":kw["winner"],"entries":s["entries"],"total_cost":cost,"total_shares":s["shares"],"avg_entry":avg,"first_entry":s["first_entry"],"last_entry":s["last_entry"],"max_exposure":s["max_exposure"],"up_cost":s["up_cost"],"down_cost":s["down_cost"],"up_shares":s["up_shares"],"down_shares":s["down_shares"],"winning_cost":wc,"losing_cost":cost-wc,"payout":payout,"realized_pnl":pnl,"roi":pnl/cost if cost else 0.,"resolved_ts":kw["ts"]}); self.market_stats.pop(m["condition"],None); self._trade_cache.pop(m["condition"],None)
    def record_resolution_error(self,**kw):
        m=kw["market"]; key=m["condition"]; status=kw["status"]; prev=self.last_resolution_error.get(key)
        if prev and prev[0]==status and kw["ts"]-prev[1]<60:return
        self.last_resolution_error[key]=(status,kw["ts"]); s=self.market_stats[key]; self._append_csv("resolutions.csv",{"timestamp":kw["ts"],"market_id":m["id"],"condition":key,"slug":m["slug"],"asset":m["asset"],"winner":"","winner_token":"","entries":s["entries"],"cost":s["cost"],"payout":"","pnl":"","roi":"","status":status})
    def record_pnl(self,ts,m): self._append_csv("pnl_1min.csv",{"timestamp":ts,"equity":m["equity"],"total_pnl":m["pnl"],"realized_pnl":m["realized"],"unrealized_pnl":m["unrealized"],"cash":m["cash"],"open_cost":m["open_cost"],"market_value":m["market_value"],"drawdown":m["drawdown"],"positions":m.get("positions",""),"marked":m["marked"]})
    def rebuild_from_ledger(self,ledger):
        for t in ledger.trades:
            if t.get("action")!="BUY" or not t.get("condition"):continue
            c=t["condition"];s=self.market_stats[c];s["entries"]+=1;s["cost"]+=float(t.get("notional",0));s["shares"]+=float(t.get("shares",0));self._trade_cache[c].append(t)
    def maintenance(self): return None

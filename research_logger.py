import csv, json, os, threading, time, uuid
from pathlib import Path
from collections import defaultdict

SCHEMAS = {
    "trades.csv": [
        "trade_id","timestamp","market_id","condition","slug","asset","market","side",
        "token","price","shares","notional","seconds_into_market","seconds_remaining",
        "up_bid","up_ask","up_depth","down_bid","down_ask","down_depth",
        "spread","score","momentum","signal_reason","cash_after","market_exposure_after"
    ],
    "markets.csv": [
        "market_id","condition","slug","asset","market","start_ts","end_ts","winner",
        "entries","total_cost","total_shares","avg_entry","first_entry","last_entry",
        "max_exposure","up_cost","down_cost","up_shares","down_shares","winning_cost","losing_cost",
        "payout","realized_pnl","roi","resolved_ts"
    ],
    "resolutions.csv": [
        "timestamp","market_id","condition","slug","asset","winner","winner_token",
        "entries","cost","payout","pnl","roi","status"
    ],
    "pnl_1min.csv": [
        "timestamp","equity","total_pnl","realized_pnl","unrealized_pnl","cash",
        "open_cost","market_value","drawdown","positions","marked"
    ],
}

class ResearchLogger:
    """
    Storage-conscious research writer.

    Permanent/important:
      trades.csv, markets.csv, resolutions.csv, pnl_1min.csv, paper_state.json

    Automatically retained for a limited period:
      decisions.jsonl, orderbooks.jsonl

    No raw order-book archive is kept forever. This is deliberate: order-book
    snapshots are the fastest-growing research artifact.
    """
    def __init__(self, data_dir, ledger=None):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.last_resolution_error = {}
        self._trade_cache = defaultdict(list)
        self.market_stats = defaultdict(lambda: {
            "entries": 0, "cost": 0.0, "shares": 0.0, "first_entry": None,
            "last_entry": None, "max_exposure": 0.0, "asset": "", "market": "",
            "up_cost": 0.0, "down_cost": 0.0, "up_shares": 0.0, "down_shares": 0.0,
            "slug": "", "market_id": "", "start_ts": 0.0, "end_ts": 0.0,
        })
        self._ensure_files()
        if ledger is not None:
            self.rebuild_from_ledger(ledger)

    def _ensure_files(self):
        for filename, fields in SCHEMAS.items():
            path = self.root / filename
            if not path.exists() or path.stat().st_size == 0:
                tmp = path.with_suffix(path.suffix + ".tmp")
                with tmp.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(fields)
                os.replace(tmp, path)
        for filename in ("decisions.jsonl", "orderbooks.jsonl"):
            (self.root / filename).touch(exist_ok=True)

    def _append_csv(self, filename, row):
        fields = SCHEMAS[filename]
        path = self.root / filename
        with self.lock, path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)
            f.flush()

    def _append_jsonl(self, filename, obj):
        path = self.root / filename
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        with self.lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def record_decision(self, *, ts, market, elapsed, left, up_bid, up_ask, up_depth,
                        down_bid, down_ask, down_depth, signal, exposure, cash):
        spread_up = (up_ask-up_bid) if up_ask is not None and up_bid is not None else None
        spread_down = (down_ask-down_bid) if down_ask is not None and down_bid is not None else None
        action = signal.side if signal else "WAIT"
        self._append_jsonl("decisions.jsonl", {
            "t": round(ts,3), "m": market["id"], "c": market["condition"], "s": market["slug"],
            "a": market["asset"], "e": round(elapsed,1), "r": round(left,1),
            "ub": up_bid, "ua": up_ask, "ud": up_depth, "db": down_bid, "da": down_ask,
            "dd": down_depth, "us": spread_up, "ds": spread_down, "x": action,
            "p": signal.price if signal else None, "score": signal.score if signal else None,
            "n": signal.notional if signal else 0.0,
            "reason": signal.reason if signal else "no_signal",
            "ex": exposure, "cash": cash,
        })

    def record_orderbook(self, *, ts, market, elapsed, left, up_bid, up_ask, up_depth,
                         down_bid, down_ask, down_depth):
        self._append_jsonl("orderbooks.jsonl", {
            "t": round(ts,3), "m": market["id"], "c": market["condition"], "s": market["slug"],
            "a": market["asset"], "e": round(elapsed,1), "r": round(left,1),
            "ub": up_bid, "ua": up_ask, "ud": up_depth,
            "db": down_bid, "da": down_ask, "dd": down_depth
        })

    def record_trade(self, *, trade, market, elapsed, left, up_bid, up_ask, up_depth,
                     down_bid, down_ask, down_depth, score, momentum, reason,
                     cash_after, exposure_after):
        trade_id = str(trade.get("trade_id") or f"paper-{uuid.uuid4().hex}")
        row = {
            "trade_id": trade_id, "timestamp": trade["ts"], "market_id": trade.get("market_id", market["id"]),
            "condition": trade["condition"], "slug": trade.get("slug", market["slug"]),
            "asset": trade.get("asset", market["asset"]), "market": trade.get("market", market["market"]),
            "side": trade["side"], "token": trade["token"], "price": trade["price"],
            "shares": trade["shares"], "notional": trade["notional"], "seconds_into_market": elapsed,
            "seconds_remaining": left, "up_bid": up_bid, "up_ask": up_ask, "up_depth": up_depth,
            "down_bid": down_bid, "down_ask": down_ask, "down_depth": down_depth,
            "spread": (up_ask-up_bid) if trade["side"]=="Up" and up_ask is not None and up_bid is not None
                      else ((down_ask-down_bid) if trade["side"]=="Down" and down_ask is not None and down_bid is not None else None),
            "score": score, "momentum": momentum, "signal_reason": reason,
            "cash_after": cash_after, "market_exposure_after": exposure_after,
        }
        self._append_csv("trades.csv", row)
        self._trade_cache[trade["condition"]].append(trade)
        s = self.market_stats[trade["condition"]]
        s["entries"] += 1; s["cost"] += float(trade["notional"]); s["shares"] += float(trade["shares"])
        if trade["side"] == "Up":
            s["up_cost"] += float(trade["notional"]); s["up_shares"] += float(trade["shares"])
        else:
            s["down_cost"] += float(trade["notional"]); s["down_shares"] += float(trade["shares"])
        s["first_entry"] = trade["ts"] if s["first_entry"] is None else min(s["first_entry"], trade["ts"])
        s["last_entry"] = trade["ts"]
        s["max_exposure"] = max(s["max_exposure"], float(exposure_after))
        for k in ("asset","market","slug","market_id"):
            s[k] = trade.get(k, market.get(k, s[k]))
        s["start_ts"], s["end_ts"] = market["start_ts"], market["end_ts"]

    def record_resolution(self, *, ts, market, winner, winner_token, closed):
        s = self.market_stats[market["condition"]]
        entries = int(s["entries"]); cost = float(s["cost"])
        pnl = sum(float(x[1]) for x in closed); payout = cost + pnl
        avg = cost / s["shares"] if s["shares"] else 0.0
        winning_cost = sum(float(t.get("notional",0)) for t in self._trade_cache.get(market["condition"], [])
                           if t.get("token") == winner_token)
        losing_cost = cost - winning_cost
        roi = pnl / cost if cost else 0.0
        self._append_csv("resolutions.csv", {
            "timestamp": ts, "market_id": market["id"], "condition": market["condition"],
            "slug": market["slug"], "asset": market["asset"], "winner": winner,
            "winner_token": winner_token, "entries": entries, "cost": cost,
            "payout": payout, "pnl": pnl, "roi": roi, "status": "RESOLVED"
        })
        self._append_csv("markets.csv", {
            "market_id": market["id"], "condition": market["condition"], "slug": market["slug"],
            "asset": market["asset"], "market": market["market"], "start_ts": market["start_ts"],
            "end_ts": market["end_ts"], "winner": winner, "entries": entries, "total_cost": cost,
            "total_shares": s["shares"], "avg_entry": avg, "first_entry": s["first_entry"],
            "last_entry": s["last_entry"], "max_exposure": s["max_exposure"],
            "up_cost": s["up_cost"], "down_cost": s["down_cost"], "up_shares": s["up_shares"],
            "down_shares": s["down_shares"], "winning_cost": winning_cost, "losing_cost": losing_cost,
            "payout": payout, "realized_pnl": pnl, "roi": roi, "resolved_ts": ts
        })
        self.market_stats.pop(market["condition"], None)
        self._trade_cache.pop(market["condition"], None)

    def record_resolution_error(self, *, ts, market, status):
        key = market["condition"]
        prev = self.last_resolution_error.get(key, (None, 0.0))
        if prev[0] == status and ts - prev[1] < 60:
            return
        self.last_resolution_error[key] = (status, ts)
        s = self.market_stats[key]
        self._append_csv("resolutions.csv", {
            "timestamp": ts, "market_id": market["id"], "condition": key,
            "slug": market["slug"], "asset": market["asset"], "winner": "", "winner_token": "",
            "entries": s["entries"], "cost": s["cost"], "payout": "",
            "pnl": "", "roi": "", "status": status
        })

    def record_pnl(self, ts, m):
        self._append_csv("pnl_1min.csv", {
            "timestamp": ts, "equity": m["equity"], "total_pnl": m["pnl"],
            "realized_pnl": m["realized"], "unrealized_pnl": m["unrealized"], "cash": m["cash"],
            "open_cost": m["open_cost"], "market_value": m["market_value"],
            "drawdown": m["drawdown"], "positions": m.get("positions",""), "marked": m["marked"]
        })

    def rebuild_from_ledger(self, ledger):
        for t in ledger.trades:
            if t.get("action") != "BUY": continue
            condition=t.get("condition")
            if not condition: continue
            s=self.market_stats[condition]
            s["entries"] += 1; s["cost"] += float(t.get("notional",0)); s["shares"] += float(t.get("shares",0))
            if t.get("side") == "Up":
                s["up_cost"] += float(t.get("notional",0)); s["up_shares"] += float(t.get("shares",0))
            elif t.get("side") == "Down":
                s["down_cost"] += float(t.get("notional",0)); s["down_shares"] += float(t.get("shares",0))
            self._trade_cache[condition].append(t)
            ts=float(t.get("ts",0)); s["first_entry"]=ts if s["first_entry"] is None else min(s["first_entry"],ts); s["last_entry"]=ts
            for k in ("asset","market","slug","market_id","start_ts","end_ts"):
                if t.get(k) is not None: s[k]=t[k]
        for p in ledger.positions.values():
            c=p.get("condition")
            if c in self.market_stats:
                self.market_stats[c]["max_exposure"]=max(self.market_stats[c]["max_exposure"],float(p.get("cost",0)))

    def _prune_jsonl(self, filename, days):
        if days <= 0: return
        path = self.root / filename
        if not path.exists(): return
        cutoff = time.time() - days * 86400
        tmp = path.with_suffix(path.suffix + ".tmp")
        kept = 0
        try:
            with self.lock, path.open("r", encoding="utf-8", errors="replace") as src, \
                 tmp.open("w", encoding="utf-8") as dst:
                for line in src:
                    try:
                        obj=json.loads(line)
                        ts=float(obj.get("t", obj.get("timestamp", 0)))
                        if ts >= cutoff:
                            dst.write(line); kept += 1
                    except Exception:
                        # Never let one malformed historical line break maintenance.
                        continue
                dst.flush()
                os.replace(tmp, path)
        except Exception:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass

    def maintenance(self):
        """Delete old high-volume research artifacts. Important CSVs are retained."""
        self._prune_jsonl("decisions.jsonl", float(os.getenv("DECISION_RETENTION_DAYS","7")))
        self._prune_jsonl("orderbooks.jsonl", float(os.getenv("ORDERBOOK_RETENTION_DAYS","2")))

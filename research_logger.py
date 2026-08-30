import csv
import json
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

REGIMES = ("CHEAP", "MID", "CORE", "HIGH")

SCHEMAS = {
    "trades.csv": [
        "trade_id","timestamp","market_id","condition","slug","asset","market",
        "side","token","price","shares","notional","seconds_into_market",
        "seconds_remaining","up_bid","up_ask","up_depth","down_bid","down_ask",
        "down_depth","spread","score","momentum","signal_reason","cash_after",
        "market_exposure_after"
    ],
    "markets.csv": [
        "market_id","condition","slug","asset","market","start_ts","end_ts",
        "winner","entries","total_cost","total_shares","avg_entry","first_entry",
        "last_entry","max_exposure","up_cost","down_cost","up_shares","down_shares",
        "winning_cost","losing_cost","payout","realized_pnl","roi","resolved_ts"
    ],
    "resolutions.csv": [
        "timestamp","market_id","condition","slug","asset","winner","winner_token",
        "entries","cost","payout","pnl","roi","status"
    ],
    "settlement_details.csv": [
        "timestamp","market_id","condition","slug","asset","trade_id","side",
        "token","regime","price","shares","cost","settlement_per_share",
        "payout","pnl","roi","outcome"
    ],
    "regime_1min.csv": [
        "timestamp","regime","trades","notional","trade_share","settled_trades",
        "wins","losses","win_rate","settled_cost","settled_pnl","settled_roi",
        "avg_settled_pnl","open_cost"
    ],
    "trade_details.csv": [
        "trade_id","timestamp","market_id","condition","slug","asset","market",
        "side","token","regime","price","shares","notional","seconds_into_market",
        "seconds_remaining","spread","score","momentum","cash_after",
        "market_exposure_after","up_bid","up_ask","up_depth","down_bid",
        "down_ask","down_depth","signal_reason"
    ],
    "pnl_1min.csv": [
        "timestamp","equity","total_pnl","realized_pnl","unrealized_pnl","cash",
        "open_cost","market_value","drawdown","positions","marked"
    ],
}

class ResearchLogger:
    """Detailed research logger.

    This version preserves the existing ResearchLogger API used by bot.py.
    It additionally emits human-readable one-minute regime reports and
    settlement attribution directly from record_pnl()/record_resolution(),
    so bot.py does not need to change.
    """

    def __init__(self, data_dir, ledger=None):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.last_resolution_error = {}
        self._trade_cache = defaultdict(list)
        self.regime_stats = {
            r: {
                "trades":0, "notional":0.0, "settled_trades":0, "wins":0,
                "losses":0, "settled_cost":0.0, "settled_pnl":0.0,
                "open_cost":0.0
            } for r in REGIMES
        }
        self.market_stats = defaultdict(lambda: {
            "entries":0,"cost":0.0,"shares":0.0,"first_entry":None,"last_entry":None,
            "max_exposure":0.0,"asset":"","market":"","up_cost":0.0,"down_cost":0.0,
            "up_shares":0.0,"down_shares":0.0,"slug":"","market_id":"",
            "start_ts":0.0,"end_ts":0.0
        })
        self._ensure_files()
        if ledger is not None:
            self.rebuild_from_ledger(ledger)

    def _ensure_files(self):
        for fn, fields in SCHEMAS.items():
            p = self.root / fn
            if not p.exists() or p.stat().st_size == 0:
                with p.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(fields)
        for fn in ("decisions.jsonl", "orderbooks.jsonl"):
            (self.root / fn).touch(exist_ok=True)

    def _append_csv(self, fn, row):
        with self.lock, (self.root / fn).open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SCHEMAS[fn], extrasaction="ignore").writerow(row)
            f.flush()

    def _append_jsonl(self, fn, obj):
        with self.lock, (self.root / fn).open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()

    @staticmethod
    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _regime(price):
        p = ResearchLogger._safe_float(price, -1.0)
        if 0.01 <= p < 0.30: return "CHEAP"
        if 0.30 <= p < 0.70: return "MID"
        if 0.70 <= p < 0.90: return "CORE"
        if 0.90 <= p < 0.995: return "HIGH"
        return "OTHER"

    def record_decision(self, **kw):
        m = kw["market"]; sig = kw.get("signal")
        ts = kw["ts"]
        ub, ua = kw.get("up_bid"), kw.get("up_ask")
        db, da = kw.get("down_bid"), kw.get("down_ask")
        self._append_jsonl("decisions.jsonl", {
            "t": round(ts,3), "m":m["id"], "c":m["condition"], "s":m["slug"],
            "a":m["asset"], "e":round(kw["elapsed"],1), "r":round(kw["left"],1),
            "ub":ub,"ua":ua,"ud":kw.get("up_depth"),"db":db,"da":da,
            "dd":kw.get("down_depth"),
            "us":ua-ub if ua is not None and ub is not None else None,
            "ds":da-db if da is not None and db is not None else None,
            "x":sig.side if sig else "WAIT",
            "p":sig.price if sig else None,
            "score":sig.score if sig else None,
            "n":sig.notional if sig else 0.0,
            "reason":sig.reason if sig else "no_signal",
            "ex":kw.get("exposure",0.0), "cash":kw.get("cash",0.0)
        })

    def record_orderbook(self, **kw):
        m = kw["market"]
        self._append_jsonl("orderbooks.jsonl", {
            "t":round(kw["ts"],3),"m":m["id"],"c":m["condition"],"s":m["slug"],
            "a":m["asset"],"e":round(kw["elapsed"],1),"r":round(kw["left"],1),
            "ub":kw.get("up_bid"),"ua":kw.get("up_ask"),"ud":kw.get("up_depth"),
            "db":kw.get("down_bid"),"da":kw.get("down_ask"),"dd":kw.get("down_depth")
        })

    def record_trade(self, **kw):
        t, m = kw["trade"], kw["market"]
        tid = str(t.get("trade_id") or f"paper-{uuid.uuid4().hex}")
        regime = self._regime(t.get("price"))
        t["trade_id"] = tid
        t["regime"] = regime

        ub, ua = kw.get("up_bid"), kw.get("up_ask")
        db, da = kw.get("down_bid"), kw.get("down_ask")
        spread = (
            ua-ub if t["side"] == "Up" and ua is not None and ub is not None
            else da-db if t["side"] == "Down" and da is not None and db is not None
            else None
        )

        row = {
            "trade_id":tid,"timestamp":t["ts"],"market_id":t.get("market_id",m["id"]),
            "condition":t["condition"],"slug":t.get("slug",m["slug"]),
            "asset":t.get("asset",m["asset"]),"market":t.get("market",m["market"]),
            "side":t["side"],"token":t["token"],"price":t["price"],
            "shares":t["shares"],"notional":t["notional"],
            "seconds_into_market":kw["elapsed"],"seconds_remaining":kw["left"],
            "up_bid":ub,"up_ask":ua,"up_depth":kw.get("up_depth"),
            "down_bid":db,"down_ask":da,"down_depth":kw.get("down_depth"),
            "spread":spread,"score":kw.get("score"),"momentum":kw.get("momentum"),
            "signal_reason":kw.get("reason"),"cash_after":kw.get("cash_after"),
            "market_exposure_after":kw.get("exposure_after")
        }
        self._append_csv("trades.csv", row)

        self._append_csv("trade_details.csv", {
            "trade_id":tid,"timestamp":t["ts"],"market_id":t.get("market_id",m["id"]),
            "condition":t["condition"],"slug":t.get("slug",m["slug"]),
            "asset":t.get("asset",m["asset"]),"market":t.get("market",m["market"]),
            "side":t["side"],"token":t["token"],"regime":regime,
            "price":t["price"],"shares":t["shares"],"notional":t["notional"],
            "seconds_into_market":kw["elapsed"],"seconds_remaining":kw["left"],
            "spread":spread,"score":kw.get("score"),"momentum":kw.get("momentum"),
            "cash_after":kw.get("cash_after"),
            "market_exposure_after":kw.get("exposure_after"),
            "up_bid":ub,"up_ask":ua,"up_depth":kw.get("up_depth"),
            "down_bid":db,"down_ask":da,"down_depth":kw.get("down_depth"),
            "signal_reason":kw.get("reason")
        })

        self._trade_cache[t["condition"]].append(t)
        s = self.market_stats[t["condition"]]
        n = self._safe_float(t["notional"])
        sh = self._safe_float(t["shares"])
        s["entries"] += 1
        s["cost"] += n
        s["shares"] += sh
        s["first_entry"] = t["ts"] if s["first_entry"] is None else min(s["first_entry"], t["ts"])
        s["last_entry"] = t["ts"]
        s["max_exposure"] = max(s["max_exposure"], self._safe_float(kw.get("exposure_after")))
        s["asset"] = t.get("asset",m["asset"])
        s["market"] = t.get("market",m["market"])
        s["slug"] = m["slug"]; s["market_id"] = m["id"]
        s["start_ts"] = m["start_ts"]; s["end_ts"] = m["end_ts"]
        if t["side"] == "Up":
            s["up_cost"] += n; s["up_shares"] += sh
        else:
            s["down_cost"] += n; s["down_shares"] += sh

        if regime in self.regime_stats:
            r = self.regime_stats[regime]
            r["trades"] += 1
            r["notional"] += n
            r["open_cost"] += n

    def record_resolution(self, **kw):
        m = kw["market"]
        closed = kw["closed"]
        condition = m["condition"]
        s = self.market_stats[condition]
        trades = self._trade_cache[condition]
        cost = self._safe_float(s["cost"])
        pnl = sum(self._safe_float(x.get("pnl")) for x in closed)
        payout = cost + pnl
        avg_entry = cost / s["shares"] if s["shares"] else 0.0
        wc = sum(
            self._safe_float(t.get("notional"))
            for t in trades
            if t.get("token") == kw["winner_token"]
        )

        by_regime = {}
        for trade in trades:
            regime = trade.get("regime") or self._regime(trade.get("price"))
            tc = self._safe_float(trade.get("notional"))
            shares = self._safe_float(trade.get("shares"))
            win = trade.get("token") == kw["winner_token"]
            tpnl = (shares - tc) if win else -tc
            payout_trade = shares if win else 0.0
            roi = tpnl/tc if tc else 0.0

            by_regime.setdefault(regime, {"trades":0,"wins":0,"losses":0,"cost":0.0,"pnl":0.0})
            br = by_regime[regime]
            br["trades"] += 1
            br["wins"] += int(win)
            br["losses"] += int(not win)
            br["cost"] += tc
            br["pnl"] += tpnl

            self._append_csv("settlement_details.csv", {
                "timestamp":kw["ts"],"market_id":m["id"],"condition":condition,
                "slug":m["slug"],"asset":m["asset"],"trade_id":trade.get("trade_id",""),
                "side":trade.get("side",""),"token":trade.get("token",""),
                "regime":regime,"price":trade.get("price",0.0),"shares":shares,
                "cost":tc,"settlement_per_share":1.0 if win else 0.0,
                "payout":payout_trade,"pnl":tpnl,"roi":roi,
                "outcome":"WIN" if win else "LOSS"
            })

            if regime in self.regime_stats:
                r = self.regime_stats[regime]
                r["settled_trades"] += 1
                r["settled_cost"] += tc
                r["settled_pnl"] += tpnl
                r["open_cost"] = max(0.0, r["open_cost"] - tc)
                if win: r["wins"] += 1
                else: r["losses"] += 1

        self._append_csv("resolutions.csv", {
            "timestamp":kw["ts"],"market_id":m["id"],"condition":condition,
            "slug":m["slug"],"asset":m["asset"],"winner":kw["winner"],
            "winner_token":kw["winner_token"],"entries":s["entries"],
            "cost":cost,"payout":payout,"pnl":pnl,
            "roi":pnl/cost if cost else 0.0,"status":"RESOLVED"
        })

        self._append_csv("markets.csv", {
            "market_id":m["id"],"condition":condition,"slug":m["slug"],
            "asset":m["asset"],"market":m["market"],"start_ts":m["start_ts"],
            "end_ts":m["end_ts"],"winner":kw["winner"],"entries":s["entries"],
            "total_cost":cost,"total_shares":s["shares"],
            "avg_entry":avg_entry,"first_entry":s["first_entry"],
            "last_entry":s["last_entry"],"max_exposure":s["max_exposure"],
            "up_cost":s["up_cost"],"down_cost":s["down_cost"],
            "up_shares":s["up_shares"],"down_shares":s["down_shares"],
            "winning_cost":wc,"losing_cost":cost-wc,"payout":payout,
            "realized_pnl":pnl,"roi":pnl/cost if cost else 0.0,
            "resolved_ts":kw["ts"]
        })

        # Console: detailed settlement attribution.
        print(f"SETTLEMENT | {m['asset']} | {m['slug']} | winner={kw['winner']} | total_pnl=${pnl:+.4f} | cost=${cost:.4f}")
        for regime in REGIMES:
            b = by_regime.get(regime)
            if not b:
                continue
            wr = b["wins"] / b["trades"] * 100 if b["trades"] else 0.0
            roi = b["pnl"] / b["cost"] * 100 if b["cost"] else 0.0
            print(
                f"  {regime:<5} | trades={b['trades']} | "
                f"W/L={b['wins']}/{b['losses']} ({wr:.0f}%) | "
                f"cost=${b['cost']:.2f} | pnl=${b['pnl']:+.4f} | ROI={roi:+.2f}%"
            )
        print(
            f"  TOTAL | trades={sum(x['trades'] for x in by_regime.values())} | "
            f"W/L={sum(x['wins'] for x in by_regime.values())}/"
            f"{sum(x['losses'] for x in by_regime.values())} | "
            f"pnl=${pnl:+.4f}"
        )

        self.market_stats.pop(condition, None)
        self._trade_cache.pop(condition, None)

    def record_resolution_error(self, **kw):
        m = kw["market"]; condition = m["condition"]; status = kw["status"]
        prev = self.last_resolution_error.get(condition)
        if prev and prev[0] == status and kw["ts"] - prev[1] < 60:
            return
        self.last_resolution_error[condition] = (status, kw["ts"])
        s = self.market_stats[condition]
        self._append_csv("resolutions.csv", {
            "timestamp":kw["ts"],"market_id":m["id"],"condition":condition,
            "slug":m["slug"],"asset":m["asset"],"winner":"","winner_token":"",
            "entries":s["entries"],"cost":s["cost"],"payout":"","pnl":"","roi":"",
            "status":status
        })

    def _regime_rows(self, ts):
        total_trades = sum(x["trades"] for x in self.regime_stats.values())
        rows = []
        for regime in REGIMES:
            r = self.regime_stats[regime]
            settled = r["settled_trades"]
            wr = r["wins"] / settled if settled else 0.0
            roi = r["settled_pnl"] / r["settled_cost"] if r["settled_cost"] else 0.0
            avg = r["settled_pnl"] / settled if settled else 0.0
            row = {
                "timestamp":ts,"regime":regime,"trades":r["trades"],
                "notional":r["notional"],
                "trade_share":r["trades"] / total_trades if total_trades else 0.0,
                "settled_trades":settled,"wins":r["wins"],"losses":r["losses"],
                "win_rate":wr,"settled_cost":r["settled_cost"],
                "settled_pnl":r["settled_pnl"],"settled_roi":roi,
                "avg_settled_pnl":avg,"open_cost":r["open_cost"]
            }
            self._append_csv("regime_1min.csv", row)
            rows.append(row)
        return rows

    def record_pnl(self, ts, m):
        self._append_csv("pnl_1min.csv", {
            "timestamp":ts,"equity":m["equity"],"total_pnl":m["pnl"],
            "realized_pnl":m["realized"],"unrealized_pnl":m["unrealized"],
            "cash":m["cash"],"open_cost":m["open_cost"],
            "market_value":m["market_value"],"drawdown":m["drawdown"],
            "positions":m.get("positions",""),"marked":m["marked"]
        })

        rows = self._regime_rows(ts)

        # Console: detailed one-minute research snapshot.
        print(
            f"RESEARCH 1MIN | total_trades={sum(r['trades'] for r in rows)} | "
            f"total_notional=${sum(r['notional'] for r in rows):.2f} | "
            f"settled={sum(r['settled_trades'] for r in rows)}"
        )

        for r in rows:
            print(
                f"  {r['regime']:<5} | "
                f"trades={r['trades']} | share={r['trade_share']*100:.1f}% | "
                f"notional=${r['notional']:.2f} | "
                f"settled={r['settled_trades']} | "
                f"W/L={r['wins']}/{r['losses']} ({r['win_rate']*100:.0f}%) | "
                f"pnl=${r['settled_pnl']:+.4f} | "
                f"ROI={r['settled_roi']*100:+.2f}% | "
                f"avg=${r['avg_settled_pnl']:+.4f} | "
                f"open=${r['open_cost']:.2f}"
            )

        settled_pnl = sum(r["settled_pnl"] for r in rows)
        reconciliation = settled_pnl - self._safe_float(m.get("realized"))
        print(
            f"  TOTAL | settled_pnl=${settled_pnl:+.4f} | "
            f"ledger_realized=${self._safe_float(m.get('realized')):+.4f} | "
            f"RECONCILIATION=${reconciliation:+.4f}"
        )

    def rebuild_from_ledger(self, ledger):
        for trade in ledger.trades:
            if trade.get("action") != "BUY" or not trade.get("condition"):
                continue
            condition = trade["condition"]
            regime = trade.get("regime") or self._regime(trade.get("price"))
            trade["regime"] = regime
            s = self.market_stats[condition]
            n = self._safe_float(trade.get("notional"))
            sh = self._safe_float(trade.get("shares"))
            s["entries"] += 1
            s["cost"] += n
            s["shares"] += sh
            self._trade_cache[condition].append(trade)
            if regime in self.regime_stats:
                self.regime_stats[regime]["trades"] += 1
                self.regime_stats[regime]["notional"] += n
                self.regime_stats[regime]["open_cost"] += n

    def maintenance(self):
        return None

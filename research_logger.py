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
        "trade_id",
        "timestamp",
        "market_id",
        "condition",
        "slug",
        "asset",
        "market",
        "side",
        "token",
        "price",
        "shares",
        "notional",
        "seconds_into_market",
        "seconds_remaining",
        "up_bid",
        "up_ask",
        "up_depth",
        "down_bid",
        "down_ask",
        "down_depth",
        "spread",
        "score",
        "momentum",
        "signal_reason",
        "cash_after",
        "market_exposure_after",
    ],
    "markets.csv": [
        "market_id",
        "condition",
        "slug",
        "asset",
        "market",
        "start_ts",
        "end_ts",
        "winner",
        "entries",
        "total_cost",
        "total_shares",
        "avg_entry",
        "first_entry",
        "last_entry",
        "max_exposure",
        "up_cost",
        "down_cost",
        "up_shares",
        "down_shares",
        "winning_cost",
        "losing_cost",
        "payout",
        "realized_pnl",
        "roi",
        "resolved_ts",
    ],
    "resolutions.csv": [
        "timestamp",
        "market_id",
        "condition",
        "slug",
        "asset",
        "winner",
        "winner_token",
        "entries",
        "cost",
        "payout",
        "pnl",
        "roi",
        "status",
    ],
    "settlement_details.csv": [
        "timestamp",
        "market_id",
        "condition",
        "slug",
        "asset",
        "trade_id",
        "side",
        "token",
        "regime",
        "price",
        "shares",
        "cost",
        "settlement_per_share",
        "payout",
        "pnl",
        "roi",
        "outcome",
    ],
    "regime_1min.csv": [
        "timestamp",
        "regime",
        "trades",
        "notional",
        "trade_share",
        "settled_trades",
        "wins",
        "losses",
        "win_rate",
        "settled_cost",
        "settled_pnl",
        "settled_roi",
        "avg_settled_pnl",
        "open_cost",
    ],
    "trade_details.csv": [
        "trade_id",
        "timestamp",
        "market_id",
        "condition",
        "slug",
        "asset",
        "market",
        "side",
        "token",
        "regime",
        "price",
        "shares",
        "notional",
        "seconds_into_market",
        "seconds_remaining",
        "spread",
        "score",
        "momentum",
        "cash_after",
        "market_exposure_after",
        "up_bid",
        "up_ask",
        "up_depth",
        "down_bid",
        "down_ask",
        "down_depth",
        "signal_reason",
    ],
    "pnl_1min.csv": [
        "timestamp",
        "equity",
        "total_pnl",
        "realized_pnl",
        "unrealized_pnl",
        "cash",
        "open_cost",
        "market_value",
        "drawdown",
        "positions",
        "marked",
    ],
}


class ResearchLogger:
    """
    Research logger for BOT B.

    Keeps the existing CSV interfaces used by bot.py and adds:
      - trade_details.csv
      - settlement_details.csv
      - regime_1min.csv

    The new files provide regime-level attribution for:
      CHEAP  = 0.01 <= price < 0.30
      MID    = 0.30 <= price < 0.70
      CORE   = 0.70 <= price < 0.90
      HIGH   = 0.90 <= price < 0.995

    This logger does not change strategy decisions.
    """

    def __init__(self, data_dir, ledger=None):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.last_resolution_error = {}
        self._trade_cache = defaultdict(list)

        self.market_stats = defaultdict(
            lambda: {
                "entries": 0,
                "cost": 0.0,
                "shares": 0.0,
                "first_entry": None,
                "last_entry": None,
                "max_exposure": 0.0,
                "asset": "",
                "market": "",
                "up_cost": 0.0,
                "down_cost": 0.0,
                "up_shares": 0.0,
                "down_shares": 0.0,
                "slug": "",
                "market_id": "",
                "start_ts": 0.0,
                "end_ts": 0.0,
            }
        )

        # Lifetime research counters for the current DATA_DIR session.
        self.regime_stats = {
            regime: {
                "trades": 0,
                "notional": 0.0,
                "settled_trades": 0,
                "wins": 0,
                "losses": 0,
                "settled_cost": 0.0,
                "settled_pnl": 0.0,
                "open_cost": 0.0,
            }
            for regime in REGIMES
        }

        self._ensure_files()

        if ledger is not None:
            self.rebuild_from_ledger(ledger)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _ensure_files(self):
        for filename, fields in SCHEMAS.items():
            path = self.root / filename

            if not path.exists() or path.stat().st_size == 0:
                with path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    csv.writer(handle).writerow(fields)

        for filename in ("decisions.jsonl", "orderbooks.jsonl"):
            (self.root / filename).touch(exist_ok=True)

    def _append_csv(self, filename, row):
        with self.lock:
            with (self.root / filename).open(
                "a",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=SCHEMAS[filename],
                    extrasaction="ignore",
                )
                writer.writerow(row)
                handle.flush()

    def _append_jsonl(self, filename, obj):
        with self.lock:
            with (self.root / filename).open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        obj,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _regime(price):
        try:
            p = float(price)
        except (TypeError, ValueError):
            return "UNKNOWN"

        if 0.01 <= p < 0.30:
            return "CHEAP"
        if 0.30 <= p < 0.70:
            return "MID"
        if 0.70 <= p < 0.90:
            return "CORE"
        if 0.90 <= p < 0.995:
            return "HIGH"

        return "OTHER"

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Decision/order-book logging
    # ------------------------------------------------------------------

    def record_decision(self, **kw):
        market = kw["market"]
        signal = kw.get("signal")

        timestamp = kw["ts"]
        action = signal.side if signal else "WAIT"

        up_bid = kw.get("up_bid")
        up_ask = kw.get("up_ask")
        down_bid = kw.get("down_bid")
        down_ask = kw.get("down_ask")

        self._append_jsonl(
            "decisions.jsonl",
            {
                "t": round(timestamp, 3),
                "m": market["id"],
                "c": market["condition"],
                "s": market["slug"],
                "a": market["asset"],
                "e": round(kw["elapsed"], 1),
                "r": round(kw["left"], 1),
                "ub": up_bid,
                "ua": up_ask,
                "ud": kw.get("up_depth"),
                "db": down_bid,
                "da": down_ask,
                "dd": kw.get("down_depth"),
                "us": (
                    up_ask - up_bid
                    if up_ask is not None and up_bid is not None
                    else None
                ),
                "ds": (
                    down_ask - down_bid
                    if down_ask is not None and down_bid is not None
                    else None
                ),
                "x": action,
                "p": signal.price if signal else None,
                "score": signal.score if signal else None,
                "n": signal.notional if signal else 0.0,
                "reason": (
                    signal.reason
                    if signal
                    else "no_signal"
                ),
                "ex": kw.get("exposure", 0.0),
                "cash": kw.get("cash", 0.0),
            },
        )

    def record_orderbook(self, **kw):
        market = kw["market"]

        self._append_jsonl(
            "orderbooks.jsonl",
            {
                "t": round(kw["ts"], 3),
                "m": market["id"],
                "c": market["condition"],
                "s": market["slug"],
                "a": market["asset"],
                "e": round(kw["elapsed"], 1),
                "r": round(kw["left"], 1),
                "ub": kw.get("up_bid"),
                "ua": kw.get("up_ask"),
                "ud": kw.get("up_depth"),
                "db": kw.get("down_bid"),
                "da": kw.get("down_ask"),
                "dd": kw.get("down_depth"),
            },
        )

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def record_trade(self, **kw):
        trade = kw["trade"]
        market = kw["market"]

        trade_id = str(
            trade.get("trade_id")
            or f"paper-{uuid.uuid4().hex}"
        )

        regime = self._regime(trade.get("price"))

        trade["trade_id"] = trade_id
        trade["regime"] = regime

        up_bid = kw.get("up_bid")
        up_ask = kw.get("up_ask")
        down_bid = kw.get("down_bid")
        down_ask = kw.get("down_ask")

        if trade["side"] == "Up":
            spread = (
                up_ask - up_bid
                if up_ask is not None and up_bid is not None
                else None
            )
        else:
            spread = (
                down_ask - down_bid
                if down_ask is not None and down_bid is not None
                else None
            )

        # Keep existing trades.csv schema compatible.
        row = {
            "trade_id": trade_id,
            "timestamp": trade["ts"],
            "market_id": trade.get("market_id", market["id"]),
            "condition": trade["condition"],
            "slug": trade.get("slug", market["slug"]),
            "asset": trade.get("asset", market["asset"]),
            "market": trade.get("market", market["market"]),
            "side": trade["side"],
            "token": trade["token"],
            "price": trade["price"],
            "shares": trade["shares"],
            "notional": trade["notional"],
            "seconds_into_market": kw["elapsed"],
            "seconds_remaining": kw["left"],
            "up_bid": up_bid,
            "up_ask": up_ask,
            "up_depth": kw.get("up_depth"),
            "down_bid": down_bid,
            "down_ask": down_ask,
            "down_depth": kw.get("down_depth"),
            "spread": spread,
            "score": kw.get("score"),
            "momentum": kw.get("momentum"),
            "signal_reason": kw.get("reason"),
            "cash_after": kw.get("cash_after"),
            "market_exposure_after": kw.get("exposure_after"),
        }

        self._append_csv("trades.csv", row)

        # Separate detailed row with regime and all fields needed for analysis.
        self._append_csv(
            "trade_details.csv",
            {
                "trade_id": trade_id,
                "timestamp": trade["ts"],
                "market_id": trade.get("market_id", market["id"]),
                "condition": trade["condition"],
                "slug": trade.get("slug", market["slug"]),
                "asset": trade.get("asset", market["asset"]),
                "market": trade.get("market", market["market"]),
                "side": trade["side"],
                "token": trade["token"],
                "regime": regime,
                "price": trade["price"],
                "shares": trade["shares"],
                "notional": trade["notional"],
                "seconds_into_market": kw["elapsed"],
                "seconds_remaining": kw["left"],
                "spread": spread,
                "score": kw.get("score"),
                "momentum": kw.get("momentum"),
                "cash_after": kw.get("cash_after"),
                "market_exposure_after": kw.get("exposure_after"),
                "up_bid": up_bid,
                "up_ask": up_ask,
                "up_depth": kw.get("up_depth"),
                "down_bid": down_bid,
                "down_ask": down_ask,
                "down_depth": kw.get("down_depth"),
                "signal_reason": kw.get("reason"),
            },
        )

        self._trade_cache[trade["condition"]].append(trade)

        stats = self.market_stats[trade["condition"]]

        stats["entries"] += 1
        stats["cost"] += self._safe_float(trade["notional"])
        stats["shares"] += self._safe_float(trade["shares"])

        if stats["first_entry"] is None:
            stats["first_entry"] = trade["ts"]
        else:
            stats["first_entry"] = min(
                stats["first_entry"],
                trade["ts"],
            )

        stats["last_entry"] = trade["ts"]

        stats["max_exposure"] = max(
            stats["max_exposure"],
            self._safe_float(
                kw.get("exposure_after")
            ),
        )

        stats["asset"] = trade.get(
            "asset",
            market["asset"],
        )
        stats["market"] = trade.get(
            "market",
            market["market"],
        )
        stats["slug"] = market["slug"]
        stats["market_id"] = market["id"]
        stats["start_ts"] = market["start_ts"]
        stats["end_ts"] = market["end_ts"]

        notional = self._safe_float(trade["notional"])
        shares = self._safe_float(trade["shares"])

        if trade["side"] == "Up":
            stats["up_cost"] += notional
            stats["up_shares"] += shares
        else:
            stats["down_cost"] += notional
            stats["down_shares"] += shares

        if regime in self.regime_stats:
            regime_stats = self.regime_stats[regime]
            regime_stats["trades"] += 1
            regime_stats["notional"] += notional
            regime_stats["open_cost"] += notional

    # ------------------------------------------------------------------
    # Settlement / resolution logging
    # ------------------------------------------------------------------

    def record_resolution(self, **kw):
        market = kw["market"]
        closed = kw["closed"]

        condition = market["condition"]
        stats = self.market_stats[condition]
        trades = self._trade_cache[condition]

        cost = self._safe_float(stats["cost"])
        pnl = sum(
            self._safe_float(item.get("pnl"))
            for item in closed
        )
        payout = cost + pnl

        total_shares = self._safe_float(stats["shares"])
        avg_entry = (
            cost / total_shares
            if total_shares
            else 0.0
        )

        winning_cost = sum(
            self._safe_float(trade.get("notional"))
            for trade in trades
            if trade.get("token") == kw["winner_token"]
        )

        # Attribute every settled trade to its entry regime.
        for trade in trades:
            regime = (
                trade.get("regime")
                or self._regime(trade.get("price"))
            )

            shares = self._safe_float(
                trade.get("shares")
            )
            trade_cost = self._safe_float(
                trade.get("notional")
            )

            is_win = (
                trade.get("token")
                == kw["winner_token"]
            )

            trade_payout = (
                shares
                if is_win
                else 0.0
            )

            trade_pnl = (
                trade_payout
                - trade_cost
            )

            trade_roi = (
                trade_pnl / trade_cost
                if trade_cost
                else 0.0
            )

            self._append_csv(
                "settlement_details.csv",
                {
                    "timestamp": kw["ts"],
                    "market_id": market["id"],
                    "condition": condition,
                    "slug": market["slug"],
                    "asset": market["asset"],
                    "trade_id": trade.get(
                        "trade_id",
                        "",
                    ),
                    "side": trade.get("side", ""),
                    "token": trade.get("token", ""),
                    "regime": regime,
                    "price": trade.get("price", 0.0),
                    "shares": shares,
                    "cost": trade_cost,
                    "settlement_per_share": (
                        1.0
                        if is_win
                        else 0.0
                    ),
                    "payout": trade_payout,
                    "pnl": trade_pnl,
                    "roi": trade_roi,
                    "outcome": (
                        "WIN"
                        if is_win
                        else "LOSS"
                    ),
                },
            )

            if regime in self.regime_stats:
                regime_stats = self.regime_stats[regime]

                regime_stats["settled_trades"] += 1
                regime_stats["settled_cost"] += trade_cost
                regime_stats["settled_pnl"] += trade_pnl
                regime_stats["open_cost"] = max(
                    0.0,
                    regime_stats["open_cost"]
                    - trade_cost,
                )

                if is_win:
                    regime_stats["wins"] += 1
                else:
                    regime_stats["losses"] += 1

        # Market-level resolution file remains unchanged.
        self._append_csv(
            "resolutions.csv",
            {
                "timestamp": kw["ts"],
                "market_id": market["id"],
                "condition": condition,
                "slug": market["slug"],
                "asset": market["asset"],
                "winner": kw["winner"],
                "winner_token": kw["winner_token"],
                "entries": stats["entries"],
                "cost": cost,
                "payout": payout,
                "pnl": pnl,
                "roi": pnl / cost if cost else 0.0,
                "status": "RESOLVED",
            },
        )

        self._append_csv(
            "markets.csv",
            {
                "market_id": market["id"],
                "condition": condition,
                "slug": market["slug"],
                "asset": market["asset"],
                "market": market["market"],
                "start_ts": market["start_ts"],
                "end_ts": market["end_ts"],
                "winner": kw["winner"],
                "entries": stats["entries"],
                "total_cost": cost,
                "total_shares": total_shares,
                "avg_entry": avg_entry,
                "first_entry": stats["first_entry"],
                "last_entry": stats["last_entry"],
                "max_exposure": stats["max_exposure"],
                "up_cost": stats["up_cost"],
                "down_cost": stats["down_cost"],
                "up_shares": stats["up_shares"],
                "down_shares": stats["down_shares"],
                "winning_cost": winning_cost,
                "losing_cost": cost - winning_cost,
                "payout": payout,
                "realized_pnl": pnl,
                "roi": pnl / cost if cost else 0.0,
                "resolved_ts": kw["ts"],
            },
        )

        self.market_stats.pop(condition, None)
        self._trade_cache.pop(condition, None)

    def record_resolution_error(self, **kw):
        market = kw["market"]
        condition = market["condition"]
        status = kw["status"]

        previous = self.last_resolution_error.get(
            condition
        )

        if (
            previous
            and previous[0] == status
            and kw["ts"] - previous[1] < 60
        ):
            return

        self.last_resolution_error[
            condition
        ] = (
            status,
            kw["ts"],
        )

        stats = self.market_stats[condition]

        self._append_csv(
            "resolutions.csv",
            {
                "timestamp": kw["ts"],
                "market_id": market["id"],
                "condition": condition,
                "slug": market["slug"],
                "asset": market["asset"],
                "winner": "",
                "winner_token": "",
                "entries": stats["entries"],
                "cost": stats["cost"],
                "payout": "",
                "pnl": "",
                "roi": "",
                "status": status,
            },
        )

    # ------------------------------------------------------------------
    # 1-minute reporting / analytics
    # ------------------------------------------------------------------

    def _record_regime_snapshot(self, timestamp):
        total_trades = sum(
            stats["trades"]
            for stats in self.regime_stats.values()
        )

        rows = []

        for regime in REGIMES:
            stats = self.regime_stats[regime]

            settled_trades = int(
                stats["settled_trades"]
            )
            wins = int(stats["wins"])
            losses = int(stats["losses"])

            trade_share = (
                stats["trades"] / total_trades
                if total_trades
                else 0.0
            )

            win_rate = (
                wins / settled_trades
                if settled_trades
                else 0.0
            )

            settled_roi = (
                stats["settled_pnl"]
                / stats["settled_cost"]
                if stats["settled_cost"]
                else 0.0
            )

            avg_settled_pnl = (
                stats["settled_pnl"]
                / settled_trades
                if settled_trades
                else 0.0
            )

            row = {
                "timestamp": timestamp,
                "regime": regime,
                "trades": stats["trades"],
                "notional": stats["notional"],
                "trade_share": trade_share,
                "settled_trades": settled_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "settled_cost": stats["settled_cost"],
                "settled_pnl": stats["settled_pnl"],
                "settled_roi": settled_roi,
                "avg_settled_pnl": avg_settled_pnl,
                "open_cost": stats["open_cost"],
            }

            self._append_csv(
                "regime_1min.csv",
                row,
            )

            rows.append(row)

        return rows

    def record_pnl(self, ts, m):
        self._append_csv(
            "pnl_1min.csv",
            {
                "timestamp": ts,
                "equity": m["equity"],
                "total_pnl": m["pnl"],
                "realized_pnl": m["realized"],
                "unrealized_pnl": m["unrealized"],
                "cash": m["cash"],
                "open_cost": m["open_cost"],
                "market_value": m["market_value"],
                "drawdown": m["drawdown"],
                "positions": m.get("positions", ""),
                "marked": m["marked"],
            },
        )

        # Every normal one-minute P&L sample also writes the
        # current regime analytics snapshot.
        self._record_regime_snapshot(ts)

    # ------------------------------------------------------------------
    # Persistence / rebuild
    # ------------------------------------------------------------------

    def rebuild_from_ledger(self, ledger):
        for trade in ledger.trades:
            if (
                trade.get("action") != "BUY"
                or not trade.get("condition")
            ):
                continue

            condition = trade["condition"]
            regime = (
                trade.get("regime")
                or self._regime(trade.get("price"))
            )

            trade["regime"] = regime

            stats = self.market_stats[condition]
            stats["entries"] += 1
            stats["cost"] += self._safe_float(
                trade.get("notional")
            )
            stats["shares"] += self._safe_float(
                trade.get("shares")
            )

            self._trade_cache[
                condition
            ].append(trade)

            if regime in self.regime_stats:
                self.regime_stats[regime]["trades"] += 1
                self.regime_stats[regime]["notional"] += (
                    self._safe_float(
                        trade.get("notional")
                    )
                )
                self.regime_stats[regime]["open_cost"] += (
                    self._safe_float(
                        trade.get("notional")
                    )
                )

    def maintenance(self):
        return None

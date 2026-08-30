import logging
import os
import shutil
import time
import traceback
from pathlib import Path

from market_discovery import discover, book, resolve
from paper_ledger import PaperLedger
from research_logger import ResearchLogger
from strategy import CapitalFirstStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")


def prepare_fresh_data_dir():
    data_dir = Path(os.getenv("DATA_DIR", "/app/data")).expanduser()
    fresh = os.getenv("FRESH_START", "true").lower() in ("1", "true", "yes", "on")

    if str(data_dir) in ("/", ".", ""):
        raise RuntimeError(f"Refusing to wipe unsafe DATA_DIR={data_dir!r}")

    data_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for child in data_dir.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    return data_dir


DATA = prepare_fresh_data_dir()

if os.getenv("PAPER_TRADING", "true").lower() != "true":
    raise SystemExit("SAFETY LOCK: PAPER_TRADING must be true")

strategy = CapitalFirstStrategy(
    bankroll=float(os.getenv("STARTING_CAPITAL", "1000")),
    max_market_exposure=float(os.getenv("MAX_MARKET_EXPOSURE", "100")),
    max_order=float(os.getenv("MAX_ORDER_USD", "10")),
    max_asset_exposure=float(os.getenv("MAX_ASSET_EXPOSURE", "35")),
    max_total_exposure=float(os.getenv("MAX_TOTAL_EXPOSURE", "300")),
    start_sec=float(os.getenv("START_TRADING_SECOND", "0")),
    stop_sec=float(os.getenv("STOP_TRADING_SECOND", "240")),
    hard_cutoff_seconds=float(os.getenv("HARD_CUTOFF_SECONDS", "60")),
    max_depth_participation=float(os.getenv("MAX_DEPTH_PARTICIPATION", "0.25")),
    # Zero here because the trader's median 2s cadence is not a hard rule.
    min_trade_gap_seconds=float(os.getenv("MIN_TRADE_GAP_SECONDS", "0")),
    min_bid_depth=float(os.getenv("MIN_BID_DEPTH", "1")),
)

ledger = PaperLedger(DATA / "paper_state.json", strategy.bankroll)
# The ledger file is created here so startup validation never treats an
# expected first-run artifact as a fatal missing dependency.
ledger.save()
research = ResearchLogger(DATA, ledger)

markets = {}
histories = {}
pending = {}
last_disc = 0.0
last_report = 0.0
last_maintenance = 0.0
last_trade = {}
ob_last = {}
decision_last = {}
consecutive_errors = 0

# P90 of the observed trader intertrade distribution. This is recorded as a
# descriptive burst boundary only; it is NOT used as a trade trigger.
BURST_GAP_SECONDS = float(os.getenv("BURST_GAP_SECONDS", "18"))


def asset_exposure(asset):
    return sum(
        float(p.get("cost", 0))
        for p in ledger.positions.values()
        if p.get("asset") == asset
    )


def prepare_histories(history_map, now, window_seconds=60.0):
    for side in ("Up", "Down"):
        history_map[side] = [
            point for point in history_map.get(side, [])
            if float(point[0]) >= now - window_seconds
        ]


def market_entry_state(condition, now):
    entries = [
        t for t in ledger.trades
        if t.get("action") == "BUY"
        and t.get("condition") == condition
    ]
    if not entries:
        return {
            "count": 0,
            "seconds_since_first": 0.0,
            "seconds_since_previous": None,
            "side": None,
            "price": None,
            "burst_position": 0,
        }

    ordered = sorted(entries, key=lambda t: float(t.get("ts", now)))
    first_ts = float(ordered[0].get("ts", now))
    previous_ts = float(ordered[-1].get("ts", now))
    gaps = [float(cur.get("ts", now)) - float(prev.get("ts", now))
            for prev, cur in zip(ordered, ordered[1:])]

    burst_position = 1
    for gap in reversed(gaps):
        if gap <= BURST_GAP_SECONDS:
            burst_position += 1
        else:
            break

    latest = ordered[-1]
    return {
        "count": len(ordered),
        "seconds_since_first": max(0.0, now - first_ts),
        "seconds_since_previous": max(0.0, now - previous_ts),
        "side": latest.get("side"),
        "price": latest.get("price"),
        "burst_position": burst_position,
    }


def p(message):
    log.info(message)


def startup_data_check():
    required = [
        "decisions.jsonl",
        "orderbooks.jsonl",
        "trades.csv",
        "markets.csv",
        "resolutions.csv",
        "pnl_1min.csv",
        "paper_state.json",
    ]
    missing = [name for name in required if not (DATA / name).exists()]
    if missing:
        raise RuntimeError(f"DATA STORE INITIALIZATION FAILED: {missing}")


def resolve_pending(now):
    for condition, market in list(pending.items()):
        if now < float(market.get("end_ts", 0)) + 2:
            continue

        try:
            token, outcome, status = resolve(market)
            if token:
                closed = ledger.settle(condition, token)
                pnl = sum(float(x["pnl"]) for x in closed)

                research.record_resolution(
                    ts=now,
                    market=market,
                    winner=outcome or token,
                    winner_token=token,
                    closed=closed,
                )

                p(
                    f"RESOLUTION | asset={market['asset']} | slug={market['slug']} "
                    f"| winner={outcome or token} | pnl={pnl:+.4f} | closed={len(closed)}"
                )

                pending.pop(condition, None)
                markets.pop(condition, None)
                histories.pop(condition, None)
            elif status == "CLOSED_UNRESOLVED":
                research.record_resolution_error(
                    ts=now, market=market, status=status
                )
        except Exception as exc:
            research.record_resolution_error(
                ts=now,
                market=market,
                status=f"ERROR:{type(exc).__name__}",
            )
            p(
                f"RESOLUTION ERROR | {market['slug']} | "
                f"{type(exc).__name__}: {exc}"
            )


def report(books):
    global last_report

    now = time.time()
    interval = float(os.getenv("REPORT_INTERVAL_SECONDS", "60"))
    if now - last_report < interval:
        return

    last_report = now
    metrics = ledger.mark(books)
    metrics["positions"] = len(ledger.positions)
    research.record_pnl(now, metrics)

    p(
        f"P&L ours ${metrics['pnl']:+.2f} | realized ${metrics['realized']:+.2f} "
        f"| unrealized ${metrics['unrealized']:+.2f} | cash ${metrics['cash']:.2f} "
        f"| open ${metrics['open_cost']:.2f} | positions {metrics['positions']}"
    )


def main():
    global last_disc, last_maintenance, consecutive_errors

    startup_data_check()
    p("BOT B | PAPER ONLY | V13 EVIDENCE-CONSTRAINED BEHAVIORAL MODEL")

    while True:
        try:
            now = time.time()

            if now - last_disc >= 20:
                for market in discover():
                    markets[market["condition"]] = market

                for condition, market in list(markets.items()):
                    if any(
                        position.get("condition") == condition
                        for position in ledger.positions.values()
                    ):
                        pending[condition] = market
                    elif market["end_ts"] < now - 30:
                        markets.pop(condition, None)

                last_disc = now
                p(
                    f"MARKETS | active={len(markets)} "
                    f"| pending_resolution={len(pending)}"
                )

            resolve_pending(now)
            books = {}

            for market in list(markets.values()):
                if not market.get("end_ts") or market["end_ts"] < now - 30:
                    continue

                elapsed = now - market["start_ts"]
                left = market["end_ts"] - now

                if left <= 0 or elapsed < 0 or elapsed > 300:
                    continue

                try:
                    up_bid, up_ask, up_bid_depth, up_ask_depth = book(market["up"])
                    down_bid, down_ask, down_bid_depth, down_ask_depth = book(market["down"])
                except Exception as exc:
                    p(
                        f"BOOK ERROR | {market['asset']} | {market['slug']} "
                        f"| {type(exc).__name__}: {exc}"
                    )
                    continue

                books[market["up"]] = up_bid
                books[market["down"]] = down_bid

                history = histories.setdefault(
                    market["condition"], {"Up": [], "Down": []}
                )

                if up_bid is not None:
                    history["Up"].append((now, up_bid))
                if down_bid is not None:
                    history["Down"].append((now, down_bid))
                prepare_histories(history, now, 60.0)

                orderbook_interval = float(
                    os.getenv("ORDERBOOK_SAMPLE_SECONDS", "1")
                )
                if (
                    now - ob_last.get(market["condition"], 0)
                    >= orderbook_interval
                ):
                    research.record_orderbook(
                        ts=now,
                        market=market,
                        elapsed=elapsed,
                        left=left,
                        up_bid=up_bid,
                        up_ask=up_ask,
                        up_depth=up_bid_depth,
                        down_bid=down_bid,
                        down_ask=down_ask,
                        down_depth=down_bid_depth,
                        up_ask_depth=up_ask_depth,
                        down_ask_depth=down_ask_depth,
                        up_history=history["Up"],
                        down_history=history["Down"],
                    )
                    ob_last[market["condition"]] = now

                if not market["accepting_orders"]:
                    continue

                exposure = ledger.exposure(market["condition"])
                asset_exp = asset_exposure(market["asset"])
                total_exp = ledger.total_open_cost()
                state = market_entry_state(market["condition"], now)

                signal = strategy.decide(
                    elapsed,
                    up_ask,
                    down_ask,
                    up_bid,
                    down_bid,
                    history["Up"],
                    history["Down"],
                    exposure,
                    ledger.cash,
                    up_depth=up_bid_depth,
                    down_depth=down_bid_depth,
                    now=now,
                    asset_exposure=asset_exp,
                    total_exposure=total_exp,
                    market_entry_count=state["count"],
                    seconds_since_first_entry=state["seconds_since_first"],
                    thesis_side=state["side"],
                    thesis_price=state["price"],
                    asset=market["asset"],
                    market=market["asset"],
                )

                decision_interval = float(
                    os.getenv("DECISION_SAMPLE_SECONDS", "1")
                )
                if (
                    signal is not None
                    or now - decision_last.get(market["condition"], 0)
                    >= decision_interval
                ):
                    research.record_decision(
                        ts=now,
                        market=market,
                        elapsed=elapsed,
                        left=left,
                        up_bid=up_bid,
                        up_ask=up_ask,
                        up_depth=up_bid_depth,
                        down_bid=down_bid,
                        down_ask=down_ask,
                        down_depth=down_bid_depth,
                        signal=signal,
                        exposure=exposure,
                        cash=ledger.cash,
                        entry_count=state["count"],
                        burst_position=state["burst_position"],
                        seconds_since_previous=state["seconds_since_previous"],
                        up_history=history["Up"],
                        down_history=history["Down"],
                    )
                    decision_last[market["condition"]] = now

                if signal is None:
                    continue

                if left <= strategy.hard_cutoff_seconds:
                    continue

                if (
                    strategy.min_trade_gap_seconds
                    and now - last_trade.get(market["condition"], 0)
                    < strategy.min_trade_gap_seconds
                ):
                    continue

                token = market["up"] if signal.side == "Up" else market["down"]
                bid_depth = up_bid_depth if signal.side == "Up" else down_bid_depth

                depth_cap = max(
                    0.0,
                    float(bid_depth)
                    * float(signal.price)
                    * strategy.max_depth_participation,
                )

                remaining_market = max(
                    0.0,
                    strategy.max_market_exposure - exposure,
                )
                remaining_asset = max(
                    0.0,
                    strategy.max_asset_exposure - asset_exp,
                )
                remaining_total = max(
                    0.0,
                    strategy.max_total_exposure - total_exp,
                )

                # IMPORTANT: the fill target is entry-state-conditioned size,
                # not synthetic remaining capital.
                target = strategy.entry_target(
                    signal.price,
                    market["asset"],
                    state["count"],
                )

                notion = min(
                    target,
                    signal.notional,
                    strategy.max_order,
                    depth_cap,
                    remaining_market,
                    remaining_asset,
                    remaining_total,
                    max(0.0, ledger.cash),
                )

                if notion < float(os.getenv("MIN_PAPER_FILL_USD", "0.10")):
                    continue

                band, regime = strategy.fine_band(signal.price)

                meta = {
                    "slug": market["slug"],
                    "asset": market["asset"],
                    "start_ts": market["start_ts"],
                    "end_ts": market["end_ts"],
                    "market_id": market["id"],
                    "up_token": market["up"],
                    "down_token": market["down"],
                    "model_version": strategy.VERSION,
                    "entry_count_before": state["count"],
                    "burst_position": state["burst_position"],
                    "seconds_since_first_entry": state["seconds_since_first"],
                    "seconds_since_previous_trade": state["seconds_since_previous"],
                    "regime": regime,
                    "fine_band": band,
                    "execution_mode": "PASSIVE_BID_PROXY",
                    "target_capital": target,
                    "bid_size": bid_depth,
                    "trajectory_likelihood": signal.score,
                }

                trade = ledger.buy(
                    market["condition"],
                    token,
                    market["market"],
                    signal.side,
                    signal.price,
                    notion,
                    now,
                    meta,
                )

                pending[market["condition"]] = market
                last_trade[market["condition"]] = now

                p(
                    f"TRADE PAPER | V13 50PCT SIZE | asset={market['asset']} "
                    f"| side={signal.side} | notional=${notion:.2f} "
                    f"| bid=${signal.price:.4f} | target=${target:.2f} "
                    f"| entry_count={state['count']} | burst={state['burst_position']} "
                    f"| {signal.reason}"
                )

                research.record_trade(
                    ts=now,
                    market=market,
                    elapsed=elapsed,
                    left=left,
                    up_bid=up_bid,
                    up_ask=up_ask,
                    up_depth=up_bid_depth,
                    down_bid=down_bid,
                    down_ask=down_ask,
                    down_depth=down_bid_depth,
                    trade=trade,
                    score=signal.score,
                    momentum=None,
                    reason=signal.reason,
                    cash_after=ledger.cash,
                    exposure_after=ledger.exposure(market["condition"]),
                    entry_count_before=state["count"],
                    burst_position=state["burst_position"],
                    seconds_since_previous=state["seconds_since_previous"],
                    up_history=history["Up"],
                    down_history=history["Down"],
                )
                ledger.save()

            report(books)

            maintenance_interval = float(
                os.getenv("DATA_MAINTENANCE_SECONDS", "3600")
            )
            if now - last_maintenance >= maintenance_interval:
                research.maintenance()
                last_maintenance = now

            consecutive_errors = 0
            time.sleep(max(0.05, float(os.getenv("LOOP_SECONDS", "1"))))

        except KeyboardInterrupt:
            break
        except Exception as exc:
            consecutive_errors += 1
            p(f"LOOP ERROR | {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if consecutive_errors >= 10:
                raise
            time.sleep(2)


if __name__ == "__main__":
    main()

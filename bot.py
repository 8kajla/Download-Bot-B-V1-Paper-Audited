
import os
import time
import traceback
import logging
import shutil

from pathlib import Path

from market_discovery import discover, book, resolve
from strategy import ConvergenceStrategy
from paper_ledger import PaperLedger
from research_logger import ResearchLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("bot")


def prepare_fresh_data_dir():
    """
    Clear DATA_DIR on every deployment when FRESH_START=true.
    """

    data_dir = Path(
        os.getenv("DATA_DIR", "/app/data")
    ).expanduser()

    fresh = (
        os.getenv(
            "FRESH_START",
            "true",
        )
        .strip()
        .lower()
        in ("1", "true", "yes", "on")
    )

    if str(data_dir) in ("/", ".", ""):
        raise RuntimeError(
            f"Refusing to wipe unsafe DATA_DIR={data_dir!r}"
        )

    if fresh:
        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for child in list(data_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        print(
            f"DATA | fresh_start=ON | cleared={data_dir}"
        )

    else:
        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    return data_dir


DATA = prepare_fresh_data_dir()


if (
    os.getenv(
        "PAPER_TRADING",
        "true",
    ).lower()
    != "true"
):
    raise SystemExit(
        "SAFETY LOCK: PAPER_TRADING must be true"
    )


# ----------------------------------------------------------------------
# TRADER-REPLICATION STRATEGY
# ----------------------------------------------------------------------
#
# The strategy now reflects the observable behavior found in the trader
# dataset:
#
# Layer A: 0.01 - 0.30
#     Small, high-frequency positions.
#
# Layer B: 0.90 - 0.995
#     Larger capital allocation.
#
# Rapid re-entry is allowed.
# No artificial two-trade-per-market restriction.
# No forced selling/hedging.
# Final 60 seconds are blocked.
# Dollar exposure remains the safety constraint.
#
# The internal trader trigger is unknown, so strategy.py still uses
# observable market microstructure as a proxy.
# ----------------------------------------------------------------------

strategy = ConvergenceStrategy(
    bankroll=float(
        os.getenv(
            "STARTING_CAPITAL",
            "1000",
        )
    ),

    max_market_exposure=float(
        os.getenv(
            "MAX_MARKET_EXPOSURE",
            "25",
        )
    ),

    max_order=float(
        os.getenv(
            "MAX_ORDER_USD",
            "10",
        )
    ),

    layer_a_min_price=float(
        os.getenv(
            "LAYER_A_MIN_PRICE",
            "0.01",
        )
    ),

    layer_a_max_price=float(
        os.getenv(
            "LAYER_A_MAX_PRICE",
            "0.30",
        )
    ),

    layer_b_min_price=float(
        os.getenv(
            "LAYER_B_MIN_PRICE",
            "0.90",
        )
    ),

    layer_b_max_price=float(
        os.getenv(
            "LAYER_B_MAX_PRICE",
            "0.995",
        )
    ),

    layer_a_base_notional=float(
        os.getenv(
            "LAYER_A_BASE_NOTIONAL",
            "0.15",
        )
    ),

    layer_a_max_notional=float(
        os.getenv(
            "LAYER_A_MAX_NOTIONAL",
            "1.00",
        )
    ),

    layer_b_base_notional=float(
        os.getenv(
            "LAYER_B_BASE_NOTIONAL",
            "2.00",
        )
    ),

    layer_b_max_notional=float(
        os.getenv(
            "LAYER_B_MAX_NOTIONAL",
            "3.00",
        )
    ),

    start_sec=float(
        os.getenv(
            "START_TRADING_SECOND",
            "0",
        )
    ),

    stop_sec=float(
        os.getenv(
            "STOP_TRADING_SECOND",
            "240",
        )
    ),

    min_score=float(
        os.getenv(
            "MIN_SIGNAL_SCORE",
            "0.50",
        )
    ),

    layer_a_min_score=float(
        os.getenv(
            "LAYER_A_MIN_SCORE",
            "0.50",
        )
    ),

    layer_b_min_score=float(
        os.getenv(
            "LAYER_B_MIN_SCORE",
            "0.82",
        )
    ),

    max_depth_participation=float(
        os.getenv(
            "MAX_DEPTH_PARTICIPATION",
            "0.25",
        )
    ),

    max_asset_exposure=float(
        os.getenv(
            "MAX_ASSET_EXPOSURE",
            "35",
        )
    ),

    max_total_exposure=float(
        os.getenv(
            "MAX_TOTAL_EXPOSURE",
            "25",
        )
    ),
)


ledger = PaperLedger(
    DATA / "paper_state.json",
    strategy.bankroll,
)

ledger.save()

research = ResearchLogger(
    DATA,
    ledger,
)


markets = {}
histories = {}
pending = {}


# Recover positions from persistent paper state.
for _p0 in ledger.positions.values():

    if (
        _p0.get("condition")
        and _p0.get("slug")
        and _p0.get("start_ts")
        and _p0.get("end_ts")
    ):

        pending[_p0["condition"]] = {
            "condition": _p0["condition"],
            "id": _p0.get(
                "market_id",
                _p0["condition"],
            ),
            "market": _p0.get(
                "market",
                _p0["slug"],
            ),
            "slug": _p0["slug"],
            "asset": _p0.get(
                "asset",
                "?",
            ),
            "up": _p0.get(
                "up_token",
                _p0["token"]
                if _p0.get("side") == "Up"
                else "",
            ),
            "down": _p0.get(
                "down_token",
                _p0["token"]
                if _p0.get("side") == "Down"
                else "",
            ),
            "start_ts": _p0["start_ts"],
            "end_ts": _p0["end_ts"],
            "accepting_orders": False,
            "enable_order_book": True,
        }


last_disc = 0.0
last_report = 0.0
last_trade = {}
ob_last = {}
decision_last = {}
last_maintenance = 0.0
consecutive_errors = 0


def asset_exposure(asset):
    """
    Total open cost for a given underlying asset.
    """

    return sum(
        float(p.get("cost", 0))
        for p in ledger.positions.values()
        if p.get("asset") == asset
    )


def p(msg):
    log.info(msg)


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

    missing = [
        f
        for f in required
        if not (DATA / f).exists()
    ]

    if missing:
        raise RuntimeError(
            "DATA STORE INITIALIZATION FAILED: "
            f"{missing}"
        )

    p(
        f"DATA | directory={DATA} | "
        "files=OK | "
        "persistent_research_store=READY"
    )


def resolve_pending(now):

    for condition, m in list(
        pending.items()
    ):

        if (
            now
            < float(m.get("end_ts", 0))
            + 2
        ):
            continue

        try:

            token, outcome, status = resolve(m)

            if token:

                closed = ledger.settle(
                    condition,
                    token,
                )

                pnl = sum(
                    x[1]
                    for x in closed
                )

                research.record_resolution(
                    ts=now,
                    market=m,
                    winner=outcome or token,
                    winner_token=token,
                    closed=closed,
                )

                p(
                    f'RESOLUTION | '
                    f'asset={m["asset"]} | '
                    f'slug={m["slug"]} | '
                    f'winner={outcome or token} | '
                    f'pnl={pnl:+.4f} | '
                    f'closed={len(closed)}'
                )

                pending.pop(
                    condition,
                    None,
                )

                markets.pop(
                    condition,
                    None,
                )

                histories.pop(
                    condition,
                    None,
                )

            elif status == "CLOSED_UNRESOLVED":

                research.record_resolution_error(
                    ts=now,
                    market=m,
                    status=status,
                )

                p(
                    f'CLOSED_UNRESOLVED | '
                    f'asset={m["asset"]} | '
                    f'slug={m["slug"]}'
                )

        except Exception as e:

            research.record_resolution_error(
                ts=now,
                market=m,
                status=f"ERROR:{type(e).__name__}",
            )

            p(
                f'RESOLUTION ERROR | '
                f'{m["slug"]} | '
                f'{type(e).__name__}: {e}'
            )


def report(books):

    global last_report

    now = time.time()

    if now - last_report < 60:
        return

    last_report = now

    m = ledger.mark(books)

    m["positions"] = len(
        ledger.positions
    )

    research.record_pnl(
        now,
        m,
    )

    p(
        f'MINUTE P&L | '
        f'equity=${m["equity"]:.2f} | '
        f'total={m["pnl"]:+.2f} | '
        f'realized={m["realized"]:+.2f} | '
        f'unrealized={m["unrealized"]:+.2f} | '
        f'cash=${m["cash"]:.2f} | '
        f'open_cost=${m["open_cost"]:.2f} | '
        f'DD={m["drawdown"]:+.2f} | '
        f'positions={len(ledger.positions)} | '
        f'marked={m["marked"]}'
    )


def main():

    global markets, last_disc, consecutive_errors, last_maintenance
    
    startup_data_check()

    p(
        "BOT B | PAPER ONLY | "
        "TRADER BEHAVIORAL REPLICA | "
        "NO COPY | "
        "FRESH START ENABLED"
    )

    while True:

        try:

            now = time.time()

            # ----------------------------------------------------------
            # MARKET DISCOVERY
            # ----------------------------------------------------------

            if now - last_disc >= 20:

                for m in discover():
                    markets[
                        m["condition"]
                    ] = m

                # Do not remove markets containing positions.
                for condition, m in list(
                    markets.items()
                ):

                    if any(
                        p0.get("condition")
                        == condition
                        for p0 in ledger.positions.values()
                    ):
                        pending[condition] = m

                    elif (
                        m["end_ts"]
                        < now - 30
                    ):
                        markets.pop(
                            condition,
                            None,
                        )

                last_disc = now

                assets = (
                    ",".join(
                        sorted(
                            {
                                m["asset"]
                                for m in markets.values()
                            }
                        )
                    )
                    if markets
                    else "NONE"
                )

                p(
                    f"MARKETS | "
                    f"active={len(markets)} | "
                    f"pending_resolution={len(pending)} | "
                    f"assets={assets}"
                )

            # ----------------------------------------------------------
            # RESOLUTION
            # ----------------------------------------------------------

            resolve_pending(now)

            books = {}

            # ----------------------------------------------------------
            # MARKET LOOP
            # ----------------------------------------------------------

            for m in list(
                markets.values()
            ):

                if (
                    not m.get("end_ts")
                    or m["end_ts"]
                    < now - 30
                ):
                    continue

                elapsed = (
                    now
                    - m["start_ts"]
                )

                left = (
                    m["end_ts"]
                    - now
                )

                if (
                    left <= 0
                    or elapsed < 0
                    or elapsed > 300
                ):
                    continue

                # ------------------------------------------------------
                # ORDER BOOK
                # ------------------------------------------------------

                try:

                    (
                        ub,
                        ua,
                        ubs,
                        uas,
                    ) = book(
                        m["up"]
                    )

                    (
                        db,
                        da,
                        dbs,
                        das,
                    ) = book(
                        m["down"]
                    )

                except Exception as e:

                    p(
                        f'BOOK ERROR | '
                        f'{m["asset"]} | '
                        f'{m["slug"]} | '
                        f'{type(e).__name__}: {e}'
                    )

                    continue

                books[m["up"]] = ub
                books[m["down"]] = db

                # ------------------------------------------------------
                # HISTORY
                # ------------------------------------------------------

                histories.setdefault(
                    m["condition"],
                    {
                        "Up": [],
                        "Down": [],
                    },
                )

                # ------------------------------------------------------
                # ORDER BOOK RESEARCH SAMPLING
                # ------------------------------------------------------

                if (
                    now
                    - ob_last.get(
                        m["condition"],
                        0,
                    )
                    >= float(
                        os.getenv(
                            "ORDERBOOK_SAMPLE_SECONDS",
                            "15",
                        )
                    )
                ):

                    research.record_orderbook(
                        ts=now,
                        market=m,
                        elapsed=elapsed,
                        left=left,
                        up_bid=ub,
                        up_ask=ua,
                        up_depth=uas,
                        down_bid=db,
                        down_ask=da,
                        down_depth=das,
                    )

                    ob_last[
                        m["condition"]
                    ] = now

                if ua is not None:

                    histories[
                        m["condition"]
                    ]["Up"].append(
                        (
                            now,
                            ua,
                        )
                    )

                    histories[
                        m["condition"]
                    ]["Up"] = histories[
                        m["condition"]
                    ]["Up"][-60:]

                if da is not None:

                    histories[
                        m["condition"]
                    ]["Down"].append(
                        (
                            now,
                            da,
                        )
                    )

                    histories[
                        m["condition"]
                    ]["Down"] = histories[
                        m["condition"]
                    ]["Down"][-60:]

                # ------------------------------------------------------
                # MARKET ORDER ACCEPTANCE
                # ------------------------------------------------------

                if not m[
                    "accepting_orders"
                ]:
                    continue

                # ------------------------------------------------------
                # EXPOSURE
                # ------------------------------------------------------

                exp = ledger.exposure(
                    m["condition"]
                )

                aexp = asset_exposure(
                    m["asset"]
                )

                total_exp = ledger.total_open_cost()

                # ------------------------------------------------------
                # STRATEGY DECISION
                # ------------------------------------------------------

                sig = strategy.decide(
                    elapsed,
                    ua,
                    da,
                    ub,
                    db,
                    histories[
                        m["condition"]
                    ]["Up"],
                    histories[
                        m["condition"]
                    ]["Down"],
                    exp,
                    ledger.cash,
                    up_depth=uas,
                    down_depth=das,
                    now=now,
                    asset_exposure=aexp,
                    total_exposure=total_exp,
                )

                # ------------------------------------------------------
                # DECISION RESEARCH LOGGING
                # ------------------------------------------------------

                decision_interval = float(
                    os.getenv(
                        "DECISION_SAMPLE_SECONDS",
                        "10",
                    )
                )

                should_record_decision = (
                    sig is not None
                    or (
                        now
                        - decision_last.get(
                            m["condition"],
                            0,
                        )
                        >= decision_interval
                    )
                )

                if should_record_decision:

                    research.record_decision(
                        ts=now,
                        market=m,
                        elapsed=elapsed,
                        left=left,
                        up_bid=ub,
                        up_ask=ua,
                        up_depth=uas,
                        down_bid=db,
                        down_ask=da,
                        down_depth=das,
                        signal=sig,
                        exposure=exp,
                        cash=ledger.cash,
                    )

                    decision_last[
                        m["condition"]
                    ] = now

                # ------------------------------------------------------
                # TRADE ENTRY
                #
                # Trader behavior:
                # rapid repeated entries are allowed.
                #
                # We therefore use a 2-second gap instead of the old
                # 15-second restriction.
                # ------------------------------------------------------

                if (
                    sig
                    and (
                        now
                        - last_trade.get(
                            m["condition"],
                            0,
                        )
                        >= float(
                            os.getenv(
                                "MIN_TRADE_GAP_SECONDS",
                                "2",
                            )
                        )
                    )
                ):

                    # --------------------------------------------------
                    # FINAL 60-SECOND HARD CUTOFF
                    #
                    # No new positions during the final minute.
                    # This is deliberately enforced here as the last
                    # gate before the paper order is created.
                    # --------------------------------------------------

                    hard_cutoff = float(
                        os.getenv(
                            "HARD_CUTOFF_SECONDS",
                            "60",
                        )
                    )

                    if left <= hard_cutoff:

                        p(
                            f'ENTRY BLOCKED | '
                            f'asset={m["asset"]} | '
                            f'side={sig.side} | '
                            f'left={left:.1f}s | '
                            f'reason=HARD_CUTOFF'
                        )

                        continue

                    # --------------------------------------------------
                    # TOKEN
                    # --------------------------------------------------

                    token = (
                        m["up"]
                        if sig.side == "Up"
                        else m["down"]
                    )

                    ask_size = (
                        uas
                        if sig.side == "Up"
                        else das
                    )

                    # --------------------------------------------------
                    # DEPTH CAP
                    # --------------------------------------------------

                    depth_cap = max(
                        0.0,
                        float(ask_size)
                        * float(sig.price),
                    )

                    # Re-read all risk limits immediately before creating
                    # the paper order. This is the final portfolio gate.
                    total_exp = ledger.total_open_cost()
                    market_exp = ledger.exposure(
                        m["condition"]
                    )
                    asset_exp = asset_exposure(
                        m["asset"]
                    )

                    global_remaining = max(
                        0.0,
                        float(strategy.max_total_exposure)
                        - float(total_exp),
                    )
                    market_remaining = max(
                        0.0,
                        float(strategy.max_market_exposure)
                        - float(market_exp),
                    )
                    asset_remaining = max(
                        0.0,
                        float(strategy.max_asset_exposure)
                        - float(asset_exp),
                    )

                    exec_notional = min(
                        float(sig.notional),
                        depth_cap,
                        float(strategy.max_order),
                        global_remaining,
                        market_remaining,
                        asset_remaining,
                        max(0.0, float(ledger.cash)),
                    )

                    # --------------------------------------------------
                    # MINIMUM PAPER FILL
                    # --------------------------------------------------

                    min_fill = float(
                        os.getenv(
                            "MIN_PAPER_FILL_USD",
                            "0.25",
                        )
                    )

                    if (
                        exec_notional
                        < min_fill
                    ):
                        continue

                    # --------------------------------------------------
                    # POSITION METADATA
                    # --------------------------------------------------

                    meta = {
                        "slug": m["slug"],
                        "asset": m["asset"],
                        "start_ts": m[
                            "start_ts"
                        ],
                        "end_ts": m[
                            "end_ts"
                        ],
                        "market_id": m["id"],
                        "up_token": m[
                            "up"
                        ],
                        "down_token": m[
                            "down"
                        ],
                    }

                    # --------------------------------------------------
                    # FINAL EXECUTION SAFETY RECHECK
                    # --------------------------------------------------

                    if (
                        float(m["end_ts"]) - time.time()
                        <= hard_cutoff
                    ):
                        p(
                            f'ENTRY BLOCKED | '
                            f'asset={m["asset"]} | '
                            f'side={sig.side} | '
                            f'reason=HARD_CUTOFF_FINAL_RECHECK'
                        )
                        continue

                    # --------------------------------------------------
                    # PAPER ENTRY
                    # --------------------------------------------------

                    t = ledger.buy(
                        m["condition"],
                        token,
                        m["market"],
                        sig.side,
                        sig.price,
                        exec_notional,
                        now,
                        meta,
                    )

                    pending[
                        m["condition"]
                    ] = m

                    last_trade[
                        m["condition"]
                    ] = now

                    # --------------------------------------------------
                    # RESEARCH TRADE LOG
                    # --------------------------------------------------

                    momentum = 0.0

                    if (
                        "momentum="
                        in sig.reason
                    ):
                        try:
                            momentum = float(
                                sig.reason.split(
                                    "momentum="
                                )[-1].split(
                                    " "
                                )[0]
                            )
                        except (
                            ValueError,
                            IndexError,
                        ):
                            momentum = 0.0

                    research.record_trade(
                        trade=t,
                        market=m,
                        elapsed=elapsed,
                        left=left,
                        up_bid=ub,
                        up_ask=ua,
                        up_depth=uas,
                        down_bid=db,
                        down_ask=da,
                        down_depth=das,
                        score=sig.score,
                        momentum=momentum,
                        reason=sig.reason,
                        cash_after=ledger.cash,
                        exposure_after=ledger.exposure(
                            m["condition"]
                        ),
                    )

                    # --------------------------------------------------
                    # TRADE LOG
                    # --------------------------------------------------

                    p(
                        f'TRADE PAPER | '
                        f'asset={m["asset"]} | '
                        f'side={sig.side} | '
                        f'notional=${t["notional"]:.2f} | '
                        f'price=${t["price"]:.4f} | '
                        f'shares={t["shares"]:.4f} | '
                        f'score={sig.score:.3f} | '
                        f't={elapsed:.0f}s | '
                        f'left={left:.0f}s | '
                        f'ask_size={ask_size:.2f} | '
                        f'depth_cap=${depth_cap:.2f} | '
                        f'asset_exposure='
                        f'${asset_exposure(m["asset"]):.2f} | '
                        f'cash=${ledger.cash:.2f} | '
                        f'{sig.reason}'
                    )

            # ----------------------------------------------------------
            # DATA MAINTENANCE
            # ----------------------------------------------------------

            if (
                now
                - last_maintenance
                >= float(
                    os.getenv(
                        "DATA_MAINTENANCE_SECONDS",
                        "3600",
                    )
                )
            ):

                try:

                    research.maintenance()

                    last_maintenance = now

                    p(
                        "DATA | maintenance=OK | "
                        "old high-volume research "
                        "data pruned"
                    )

                except Exception as e:

                    p(
                        f'DATA | maintenance error | '
                        f'{type(e).__name__}: {e}'
                    )

            # ----------------------------------------------------------
            # REPORT
            # ----------------------------------------------------------

            report(books)

            consecutive_errors = 0

            time.sleep(
                float(
                    os.getenv(
                        "LOOP_SECONDS",
                        "1",
                    )
                )
            )

        except KeyboardInterrupt:
            return

        except Exception as e:

            consecutive_errors += 1

            log.error(
                "LOOP ERROR #%d | %s: %s",
                consecutive_errors,
                type(e).__name__,
                e,
                exc_info=True,
            )

            time.sleep(
                min(
                    30,
                    3 * consecutive_errors,
                )
            )


if __name__ == "__main__":
    main()

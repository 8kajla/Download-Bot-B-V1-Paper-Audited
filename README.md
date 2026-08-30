# Polymarket Bot B — V13 Evidence-Constrained Behavioral Replica

Paper-only behavioral research engine. It does **not** place live orders and does not read/copy the reference trader's private activity.

## What changed in V13

V13 removes several unsupported strategy assumptions from the previous branch:

- No synthetic composite alpha score.
- No arbitrary `+2.5c`/`35c` momentum or state-reset thresholds.
- No synthetic "remaining target" capital ladder.
- No arbitrary 1.02/1.05/1.10 entry multipliers.
- No 2-second minimum enforced as trader behavior.
- Entry size is conditioned on price band + observed entry-state medians.
- Side persistence is retained as a preference because ~89.3% of consecutive same-market pairs stayed on the same side.
- The measured weakness/strength gradient is used only as an empirical directional likelihood.
- The final 60-second cutoff remains hard because violations were ~0.04%.
- Research capture defaults to 1-second snapshots and retains a real time-based 60-second history.
- Research records explicit trade-candidate vs non-trade observations plus depth imbalance and movement features.

## Evidence boundary

The following remain **UNKNOWN** and are not hardcoded:

- the reference trader's exact private trigger;
- whether price movement is causal or consequential;
- exact passive-order placement distance;
- exact side-reset mechanism;
- exact market/asset-specific hidden alpha.

## Railway variables

```text
PAPER_TRADING=true
STARTING_CAPITAL=1000
MAX_MARKET_EXPOSURE=100
MAX_ASSET_EXPOSURE=35
MAX_TOTAL_EXPOSURE=300
MAX_ORDER_USD=10
MIN_PAPER_FILL_USD=0.10

START_TRADING_SECOND=0
STOP_TRADING_SECOND=240
HARD_CUTOFF_SECONDS=60

MAX_DEPTH_PARTICIPATION=0.25
MIN_BID_DEPTH=1

MIN_TRADE_GAP_SECONDS=0
LOOP_SECONDS=1
REPORT_INTERVAL_SECONDS=60

DECISION_SAMPLE_SECONDS=1
ORDERBOOK_SAMPLE_SECONDS=1

DATA_DIR=/app/data
DATA_MAINTENANCE_SECONDS=3600
BURST_GAP_SECONDS=18
FRESH_START=true
```

`MIN_TRADE_GAP_SECONDS` is an infrastructure throttle only. It defaults to `0` so it does not distort the observed ~2-second trader cadence.

## Research files

Permanent:

- `trades.csv`
- `trade_details.csv`
- `markets.csv`
- `resolutions.csv`
- `settlement_details.csv`
- `pnl_1min.csv`
- `regime_1min.csv`
- `paper_state.json`

High-volume:

- `decisions.jsonl`
- `orderbooks.jsonl`

The research stream records both candidate events and non-trade observations. Book snapshots include bid/ask, bid/ask depth, spread, and depth imbalance. Decision records additionally include 1/3/5/10/30-second movement features, entry state, burst position, and time since the previous paper entry.

## Accounting

`paper_ledger.py` remains unchanged from the audited branch. Realized P&L is derived from settlement records and reconciled on load/save.

## Validation

Run:

```bash
pytest -q
python -m py_compile strategy.py bot.py paper_ledger.py market_discovery.py research_logger.py
```

The tests cover fine-band boundaries, entry-state sizing, side persistence, empirical trajectory preference, final-minute cutoff, depth limits, resolution accounting, and research logging.

# Polymarket Bot B — Behavioral Replica (Paper Only)

Bot B is derived from the audited V6 paper-trading infrastructure. It does **not** read or copy the reference trader's live activity. It uses observable market microstructure to approximate the behavioral pattern found in the historical research: broad scanning, small incremental entries, stronger sizing as convergence strengthens, liquidity-aware execution, and a hard pre-resolution cutoff.

## Safety

- `PAPER_TRADING` must be exactly `true` (case-insensitive).
- No private key is required.
- There is no live order-placement code in this version.

## Railway variables

```text
PAPER_TRADING=true
STARTING_CAPITAL=1000
MAX_MARKET_EXPOSURE=25
MAX_ASSET_EXPOSURE=50
MAX_ORDER_USD=10
MIN_PAPER_FILL_USD=0.25
MIN_ENTRY_PRICE=0.10
STRONG_ENTRY_PRICE=0.82
LATE_ENTRY_PRICE=0.90
MIN_SIGNAL_SCORE=0.50
MAX_DEPTH_PARTICIPATION=0.25
START_TRADING_SECOND=90
AGGRESSIVE_SECOND=180
STOP_TRADING_SECOND=240
MIN_TRADE_GAP_SECONDS=5
LOOP_SECONDS=1
REPORT_INTERVAL_SECONDS=60
DATA_DIR=/app/data
DECISION_SAMPLE_SECONDS=10
ORDERBOOK_SAMPLE_SECONDS=15
DECISION_RETENTION_DAYS=7
ORDERBOOK_RETENTION_DAYS=2
DATA_MAINTENANCE_SECONDS=3600
FRESH_START=true
```

### Fresh-start behavior

`FRESH_START=true` clears `/app/data` at process startup and creates a new $STARTING_CAPITAL paper account. Because Railway can restart a container without a new deployment, set it to `true` for a deliberate fresh experiment and then set it to `false` if you want restart recovery to preserve the experiment.

## Strategy

Bot B scores each side using:

- short-term price momentum — 30%
- acceleration — 18%
- probability/price confirmation — 18%
- time-to-resolution — 14%
- executable depth — 12%
- spread quality — 8%

A late high-probability convergence receives a modest additional score boost. Position sizing is incremental: probe orders are about $2, strong signals about $5, and late/very strong signals up to `MAX_ORDER_USD`, subject to market, asset, cash, and visible-depth limits.

The strategy is a research hypothesis inferred from observed data, not a claim of access to the target trader's private algorithm.

## Logging / storage

Permanent files:

- `trades.csv` — every paper fill and signal context
- `markets.csv` — resolved-market summaries
- `resolutions.csv` — resolution and settlement results
- `pnl_1min.csv` — minute-level equity/P&L
- `paper_state.json` — paper account state

High-volume files are pruned automatically:

- `decisions.jsonl` — default 7 days
- `orderbooks.jsonl` — default 2 days

Mount a Railway Volume at `/app/data`.

## Expected logs

```text
BOT B | PAPER ONLY | BEHAVIORAL REPLICA | NO COPY | FRESH START ENABLED
MARKETS | active=4 | pending_resolution=0 | assets=BNB,BTC,ETH,SOL
TRADE PAPER | asset=BTC | side=Up | notional=$2.00 | price=$0.8120 | ...
MINUTE P&L | equity=$1001.73 | total=+1.73 | realized=+0.00 | unrealized=+1.73 | ...
RESOLUTION | asset=BTC | ... | pnl=+2.14 | closed=1
```

## Validation

The repository includes unit tests for strategy thresholds, dynamic sizing, depth caps, asset exposure, ledger settlement, market discovery, resolution handling, research logging, and storage initialization.

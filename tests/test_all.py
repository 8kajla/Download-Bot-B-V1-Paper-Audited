import tempfile
from pathlib import Path

import pytest

from strategy import CapitalFirstStrategy
from paper_ledger import PaperLedger
from research_logger import ResearchLogger


def strategy():
    return CapitalFirstStrategy(
        min_trade_gap_seconds=0,
        max_market_exposure=100,
        max_order=10,
        max_asset_exposure=35,
        max_total_exposure=300,
        hard_cutoff_seconds=60,
        min_bid_depth=1,
        max_depth_participation=0.25,
    )


def history(price, now=1000):
    return [
        {"ts": now - 30, "best_bid": price},
        {"ts": now - 10, "best_bid": price},
        {"ts": now - 5, "best_bid": price},
        {"ts": now, "best_bid": price},
    ]


def test_all_fine_bands():
    s = strategy()
    expected = [
        (.02, "C00_05", "CHEAP"),
        (.07, "C05_10", "CHEAP"),
        (.12, "C10_15", "CHEAP"),
        (.17, "C15_20", "CHEAP"),
        (.25, "C20_30", "CHEAP"),
        (.35, "M30_40", "MID"),
        (.45, "M40_50", "MID"),
        (.55, "M50_60", "MID"),
        (.65, "M60_70", "MID"),
        (.75, "R70_80", "CORE"),
        (.85, "R80_90", "CORE"),
        (.925, "H90_95", "HIGH"),
        (.975, "H95_100", "HIGH"),
    ]
    for price, band, regime in expected:
        assert s.fine_band(price) == (band, regime)


def test_entry_state_ratios_are_data_derived():
    s = strategy()
    cheap0 = s.capital_target(.25, "BTC", 0)
    cheap2 = s.capital_target(.25, "BTC", 2)
    cheap4 = s.capital_target(.25, "BTC", 4)
    assert cheap2 / cheap0 == pytest.approx(0.52 / 0.40)
    assert cheap4 / cheap0 == pytest.approx(0.61 / 0.40)

    high0 = s.capital_target(.95, "BTC", 0)
    high2 = s.capital_target(.95, "BTC", 2)
    high4 = s.capital_target(.95, "BTC", 4)
    assert high2 / high0 == pytest.approx(12.96 / 5.70)
    assert high4 / high0 == pytest.approx(13.29 / 5.70)


def test_high_is_available_without_previous_entry():
    s = strategy()
    h = [
        {"ts": 970, "best_bid": .93},
        {"ts": 990, "best_bid": .94},
        {"ts": 995, "best_bid": .945},
        {"ts": 1000, "best_bid": .95},
    ]
    c = s._candidate("BTC", "Up", .95, .96, 100, h, 1000, None, 0, 0)
    assert c and c["regime"] == "HIGH"


def test_market_profiles_exist():
    s = strategy()
    assert set(s.MARKET_REGIME_WEIGHT) == {"BTC", "ETH", "SOL", "BNB"}
    assert s.MARKET_REGIME_WEIGHT["SOL"]["CHEAP"] > s.MARKET_REGIME_WEIGHT["BTC"]["CHEAP"]


def test_segment_specific_book_checks_are_execution_constraints():
    s = strategy()
    assert s._book_ok("BTC", "HIGH", .95, .96, 1)[0] is False
    assert s._book_ok("BTC", "CHEAP", .20, .21, 1)[0] is True


def test_side_persistence_prefers_same_side_without_fake_reset_threshold():
    s = strategy()
    up = [
        {"ts": 970, "best_bid": .74},
        {"ts": 990, "best_bid": .78},
        {"ts": 995, "best_bid": .795},
        {"ts": 1000, "best_bid": .80},
    ]
    down = history(.49)
    signal = s.decide(
        120, .81, .50, .80, .49, up, down,
        2, 1000, 50, 50, now=1000,
        asset="ETH", market="ETH",
        market_entry_count=1,
        seconds_since_first_entry=90,
        thesis_side="Up",
        thesis_price=.78,
    )
    assert signal and signal.side == "Up"


def test_trajectory_gradient_prefers_cheap_falling():
    s = strategy()
    falling = [
        {"ts": 970, "best_bid": .25},
        {"ts": 995, "best_bid": .24},
        {"ts": 1000, "best_bid": .20},
    ]
    rising = [
        {"ts": 970, "best_bid": .75},
        {"ts": 995, "best_bid": .79},
        {"ts": 1000, "best_bid": .80},
    ]
    cheap = s._candidate("BTC", "Up", .20, .21, 100, falling, 1000, None, 0, 0)
    core = s._candidate("BTC", "Up", .80, .81, 100, rising, 1000, None, 0, 0)
    assert cheap["trajectory"] == "falling"
    assert cheap["trajectory_likelihood"] == 0.564
    assert core["trajectory"] == "rising"
    assert core["trajectory_likelihood"] == 0.542


def test_final_minute_cutoff():
    assert strategy().decide(
        180, .51, .21, .50, .20,
        history(.50), history(.20),
        0, 1000, 50, 50, now=1000,
        asset="BTC", market="BTC"
    ) is None


def test_depth_limits_order_size():
    s = strategy()
    signal = s.decide(
        30, .51, None, .50, None,
        history(.50), [],
        0, 1000, 1.0, 0, now=1000,
        asset="BTC", market="BTC"
    )
    assert signal is None or signal.notional <= 0.10


def test_paper_resolution_and_research_logging():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ledger = PaperLedger(root / "paper_state.json", 1000)
        market = {
            "id": "m1",
            "condition": "c1",
            "slug": "btc-updown-5m-1000000000",
            "asset": "BTC",
            "market": "BTC Up or Down",
            "start_ts": 1000000000.0,
            "end_ts": 1000000300.0,
        }
        ledger.buy(
            "c1", "up-token", market["market"], "Up", 0.20, 1.0,
            1000000010,
            meta={
                "asset": "BTC",
                "slug": market["slug"],
                "market_id": market["id"],
                "start_ts": market["start_ts"],
                "end_ts": market["end_ts"],
                "fine_band": "C20_30",
            },
        )
        closed = ledger.settle("c1", "up-token")
        assert len(closed) == 1
        assert closed[0]["pnl"] == 4.0

        logger = ResearchLogger(root)
        logger.record_resolution(
            ts=1000000302,
            market=market,
            winner="Up",
            winner_token="up-token",
            closed=closed,
        )
        assert "RESOLVED" in (root / "resolutions.csv").read_text()
        assert ",4.0," in (root / "resolutions.csv").read_text()

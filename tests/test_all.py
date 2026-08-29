import time

from strategy import ConvergenceStrategy


def make_strategy(**kwargs):
    return ConvergenceStrategy(
        max_order=10,
        max_depth_participation=1.0,
        **kwargs,
    )


def history(price, seconds_ago=30):
    return [
        (
            time.time() - seconds_ago,
            price,
        )
    ]


# ============================================================
# BASIC SIGNAL TESTS
# ============================================================

def test_strategy_returns_signal_for_valid_cheap_setup():
    s = make_strategy()

    signal = s.decide(
        120,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is not None
    assert signal.side == "Up"
    assert signal.price == 0.20
    assert signal.notional >= 0.10
    assert signal.notional <= 1.00
    assert "V2 regime=CHEAP" in signal.reason


def test_strategy_rejects_price_outside_supported_ranges():
    s = make_strategy()

    signal = s.decide(
        150,
        0.995,
        0.005,
        0.994,
        0.004,
        history(0.995),
        history(0.005),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is None


# ============================================================
# CASH / EXPOSURE
# ============================================================

def test_cash_cap():
    s = make_strategy()

    signal = s.decide(
        220,
        0.94,
        0.06,
        0.93,
        0.05,
        [],
        [],
        0,
        0.20,
        up_depth=20,
        down_depth=20,
    )

    assert signal is not None
    assert signal.notional <= 0.20
    assert signal.notional >= 0.10


def test_market_exposure_cap():
    s = make_strategy(
        max_market_exposure=25,
    )

    signal = s.decide(
        150,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        24.95,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is None or signal.notional <= 0.05


def test_asset_exposure_cap():
    s = make_strategy(
        max_asset_exposure=35,
    )

    signal = s.decide(
        150,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=100,
        down_depth=100,
        asset_exposure=34.95,
    )

    assert signal is None or signal.notional <= 0.05


# ============================================================
# TIMING
# ============================================================

def test_strategy_does_not_trade_before_start():
    s = make_strategy()

    signal = s.decide(
        -1,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is None


def test_strategy_stops_at_trading_cutoff():
    s = make_strategy()

    signal = s.decide(
        240,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is None


def test_strategy_allows_trade_before_cutoff():
    s = make_strategy()

    signal = s.decide(
        180,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is not None


# ============================================================
# REGIME TESTS
# ============================================================

def test_cheap_regime():
    s = make_strategy()

    assert s._regime(0.05) == "CHEAP"
    assert s._regime(0.20) == "CHEAP"
    assert s._regime(0.299) == "CHEAP"


def test_mid_regime():
    s = make_strategy()

    assert s._regime(0.30) == "MID"
    assert s._regime(0.50) == "MID"
    assert s._regime(0.699) == "MID"


def test_core_regime():
    s = make_strategy()

    assert s._regime(0.70) == "CORE"
    assert s._regime(0.80) == "CORE"
    assert s._regime(0.899) == "CORE"


def test_high_regime():
    s = make_strategy()

    assert s._regime(0.90) == "HIGH"
    assert s._regime(0.95) == "HIGH"
    assert s._regime(0.994) == "HIGH"


def test_regime_boundaries():
    s = make_strategy()

    assert s._regime(0.01) == "CHEAP"
    assert s._regime(0.30) == "MID"
    assert s._regime(0.70) == "CORE"
    assert s._regime(0.90) == "HIGH"
    assert s._regime(0.995) is None


# ============================================================
# SIZING
# ============================================================

def test_cheap_size_respects_v2_limits():
    s = make_strategy()

    size = s._size(
        "CHEAP",
        0.20,
        0.80,
    )

    assert size >= 0.15
    assert size <= 1.00


def test_high_size_respects_v2_limits():
    s = make_strategy()

    size = s._size(
        "HIGH",
        0.95,
        0.90,
    )

    assert size >= 2.00
    assert size <= 3.00


def test_high_regime_is_not_required_to_be_larger_than_core():
    s = make_strategy()

    core = s._size(
        "CORE",
        0.86,
        0.90,
    )

    high = s._size(
        "HIGH",
        0.95,
        0.90,
    )

    assert core > 0
    assert high > 0
    assert high <= 3.00


# ============================================================
# HIGH REGIME SELECTIVITY
# ============================================================

def test_high_regime_rejects_weak_score():
    s = make_strategy()

    signal = s.decide(
        150,
        0.91,
        0.09,
        0.90,
        0.08,
        history(0.89),
        history(0.09),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    if signal is not None:
        assert signal.score >= 0.82


def test_high_regime_requires_positive_momentum():
    s = make_strategy()

    signal = s.decide(
        150,
        0.95,
        0.05,
        0.94,
        0.04,
        history(0.96),
        history(0.05),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is None or signal.side != "Up"


# ============================================================
# DEPTH
# ============================================================

def test_zero_depth_rejects_signal():
    s = make_strategy()

    signal = s.decide(
        150,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=0,
        down_depth=0,
    )

    assert signal is None


def test_depth_participation_is_respected():
    s = make_strategy(
        max_depth_participation=0.25,
    )

    signal = s.decide(
        150,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=1,
        down_depth=1,
    )

    if signal is not None:
        depth_cap = 1 * signal.price * 0.25
        assert signal.notional <= depth_cap + 0.01


# ============================================================
# SIGNAL QUALITY
# ============================================================

def test_weak_signal_with_high_global_threshold_is_rejected():
    s = make_strategy(
        min_score=0.90,
    )

    signal = s.decide(
        150,
        0.50,
        0.50,
        0.49,
        0.49,
        history(0.50),
        history(0.50),
        0,
        1000,
        up_depth=50,
        down_depth=50,
    )

    assert signal is None


def test_signal_reason_identifies_v2_regime():
    s = make_strategy()

    signal = s.decide(
        150,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=100,
        down_depth=100,
    )

    assert signal is not None
    assert "V2 regime=" in signal.reason
    assert "independent=true" in signal.reason


# ============================================================
# ENVIRONMENT-CONFIGURED BOUNDARIES
# ============================================================

def test_configured_cheap_boundaries_are_used():
    s = make_strategy(
        layer_a_min_price=0.05,
        layer_a_max_price=0.25,
    )

    assert s._regime(0.04) is None
    assert s._regime(0.05) == "CHEAP"
    assert s._regime(0.249) == "CHEAP"


def test_configured_high_boundaries_are_used():
    s = make_strategy(
        layer_b_min_price=0.85,
        layer_b_max_price=0.98,
    )

    assert s._regime(0.84) == "CORE"
    assert s._regime(0.85) == "HIGH"
    assert s._regime(0.979) == "HIGH"
    assert s._regime(0.98) is None

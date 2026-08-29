import time

from strategy import ConvergenceStrategy


def make_strategy(**kwargs):
    kwargs.setdefault("max_order", 10)
    return ConvergenceStrategy(**kwargs)


def history(price, seconds_ago=30):
    return [
        (
            time.time() - seconds_ago,
            price,
        )
    ]


# ============================================================
# BASIC SIGNALS
# ============================================================

def test_valid_cheap_signal():
    strategy = make_strategy()

    signal = strategy.decide(
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
    assert "V2 regime=CHEAP" in signal.reason
    assert signal.notional >= 0.10
    assert signal.notional <= 1.00


def test_no_signal_for_invalid_price():
    strategy = make_strategy()

    signal = strategy.decide(
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

def test_cash_is_an_upper_bound():
    strategy = make_strategy()

    available_cash = 0.20

    signal = strategy.decide(
        220,
        0.94,
        0.06,
        0.93,
        0.05,
        [],
        [],
        0,
        available_cash,
        up_depth=20,
        down_depth=20,
    )

    assert signal is not None
    assert signal.notional <= available_cash


def test_market_exposure_limit():
    strategy = make_strategy(
        max_market_exposure=25,
    )

    signal = strategy.decide(
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

    if signal is not None:
        assert signal.notional <= 0.05


def test_asset_exposure_limit():
    strategy = make_strategy(
        max_asset_exposure=35,
    )

    signal = strategy.decide(
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

    if signal is not None:
        assert signal.notional <= 0.05


# ============================================================
# TIMING
# ============================================================

def test_before_start_is_blocked():
    strategy = make_strategy(
        start_sec=0,
    )

    signal = strategy.decide(
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


def test_cutoff_is_blocked():
    strategy = make_strategy(
        stop_sec=240,
    )

    signal = strategy.decide(
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


def test_before_cutoff_can_trade():
    strategy = make_strategy(
        stop_sec=240,
    )

    signal = strategy.decide(
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
# REGIMES
# ============================================================

def test_cheap_regime():
    strategy = make_strategy()

    assert strategy._regime(0.01) == "CHEAP"
    assert strategy._regime(0.10) == "CHEAP"
    assert strategy._regime(0.20) == "CHEAP"
    assert strategy._regime(0.299) == "CHEAP"


def test_mid_regime():
    strategy = make_strategy()

    assert strategy._regime(0.30) == "MID"
    assert strategy._regime(0.50) == "MID"
    assert strategy._regime(0.699) == "MID"


def test_core_regime():
    strategy = make_strategy()

    assert strategy._regime(0.70) == "CORE"
    assert strategy._regime(0.80) == "CORE"
    assert strategy._regime(0.899) == "CORE"


def test_high_regime():
    strategy = make_strategy()

    assert strategy._regime(0.90) == "HIGH"
    assert strategy._regime(0.95) == "HIGH"
    assert strategy._regime(0.994) == "HIGH"


def test_regime_upper_boundary():
    strategy = make_strategy()

    assert strategy._regime(0.995) is None


# ============================================================
# SIZING
# ============================================================

def test_cheap_size_is_small():
    strategy = make_strategy()

    size = strategy._size(
        "CHEAP",
        0.20,
        0.80,
    )

    assert size >= 0.10
    assert size <= 1.00


def test_high_size_is_bounded():
    strategy = make_strategy()

    size = strategy._size(
        "HIGH",
        0.95,
        0.90,
    )

    assert size > 0
    assert size <= strategy.max_order


def test_sizing_never_exceeds_order_limit():
    strategy = make_strategy(
        max_order=2,
    )

    for regime, price, score in (
        ("CHEAP", 0.20, 0.90),
        ("MID", 0.50, 0.90),
        ("CORE", 0.80, 0.90),
        ("HIGH", 0.95, 0.90),
    ):
        size = strategy._size(
            regime,
            price,
            score,
        )

        assert size <= strategy.max_order


# ============================================================
# HIGH REGIME
# ============================================================

def test_high_regime_requires_strong_score():
    strategy = make_strategy()

    # Construct a candidate whose HIGH-side score is deliberately
    # below the required 0.82 threshold while the opposite side
    # is outside the cheap/high test.
    signal = strategy.decide(
        150,
        0.91,
        0.49,
        0.90,
        0.48,
        history(0.89),
        history(0.49),
        0,
        1000,
        up_depth=100,
        down_depth=0,
    )

    assert signal is None


def test_high_regime_requires_positive_momentum():
    strategy = make_strategy()

    signal = strategy.decide(
        150,
        0.95,
        0.50,
        0.94,
        0.49,
        history(0.99),
        history(0.50),
        0,
        1000,
        up_depth=100,
        down_depth=0,
    )

    assert signal is None


# ============================================================
# DEPTH
# ============================================================

def test_zero_depth_rejects_signal():
    strategy = make_strategy()

    signal = strategy.decide(
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
    strategy = make_strategy(
        max_depth_participation=0.25,
    )

    signal = strategy.decide(
        150,
        0.20,
        0.80,
        0.19,
        0.79,
        history(0.20),
        history(0.80),
        0,
        1000,
        up_depth=2,
        down_depth=0,
    )

    if signal is not None:
        depth_cap = (
            2
            * signal.price
            * 0.25
        )

        assert signal.notional <= depth_cap + 0.01


# ============================================================
# GLOBAL SCORE
# ============================================================

def test_min_score_configuration_is_honored_for_mid_and_core():
    strategy = make_strategy(
        min_score=0.99,
    )

    signal = strategy.decide(
        150,
        0.80,
        0.20,
        0.79,
        0.19,
        history(0.80),
        history(0.20),
        0,
        1000,
        up_depth=100,
        down_depth=0,
    )

    assert signal is None


# ============================================================
# SIGNAL METADATA
# ============================================================

def test_signal_reason_contains_v2_marker():
    strategy = make_strategy()

    signal = strategy.decide(
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
    assert signal.reason.startswith("V2 ")
    assert "regime=CHEAP" in signal.reason
    assert "independent=true" in signal.reason


# ============================================================
# HARD V2 CAPS
# ============================================================

def test_market_cap_cannot_exceed_25():
    strategy = ConvergenceStrategy(
        max_market_exposure=50,
    )

    assert strategy.max_market_exposure == 25.0


def test_asset_cap_cannot_exceed_35():
    strategy = ConvergenceStrategy(
        max_asset_exposure=50,
    )

    assert strategy.max_asset_exposure == 35.0


def test_order_cap_cannot_exceed_10():
    strategy = ConvergenceStrategy(
        max_order=50,
    )

    assert strategy.max_order == 10.0

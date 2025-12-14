"""
Test the simplified production stop loss functions.
"""

import pandas as pd
import numpy as np
from backtesting.stop_loss_production import calculate_stop_loss_gross, calculate_stop_loss_metrics


def test_calculate_stop_loss_gross_basic():
    """Test basic gross multiplier calculation with simplified logic."""
    # Daily PnL: Start at 0, lose $12k, recover $3k, recover $2k
    daily_pnl = pd.Series([0, -12000, 3000, 2000])
    levels = [(5000, 0.75), (10000, 0.50)]
    initial_capital = 100000

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital)

    # Day 0: $100k, $0 DD -> 100%
    # Day 1: $88k, $12k DD -> 50% (entered L2)
    # Day 2: $91k, $9k DD -> 100% (exited, DD < $10k threshold, jumped to no stop loss)
    # Day 3: $93k, $7k DD -> 75% (re-entered L1, DD >= $5k)

    assert gross.iloc[0] == 1.0
    assert gross.iloc[1] == 0.50
    assert gross.iloc[2] == 1.0  # Immediate exit, not 75%
    assert gross.iloc[3] == 0.75  # Re-entered L1 since DD >= $5k

    print("✓ Test passed: calculate_stop_loss_gross basic")


def test_calculate_stop_loss_gross_numpy():
    """Test with numpy array input."""
    daily_pnl = np.array([0, -12000, 3000, 2000])
    levels = [(5000, 0.75), (10000, 0.50)]
    initial_capital = 100000

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital)

    assert gross.iloc[0] == 1.0
    assert gross.iloc[1] == 0.50
    assert gross.iloc[2] == 1.0
    assert gross.iloc[3] == 0.75  # Re-entered L1

    print("✓ Test passed: calculate_stop_loss_gross with numpy")


def test_calculate_stop_loss_metrics():
    """Test detailed metrics calculation."""
    daily_pnl = pd.Series([0, -12000, 3000, 2000])
    levels = [(5000, 0.75), (10000, 0.50)]
    initial_capital = 100000

    metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital)

    # Check columns exist
    assert 'portfolio_value' in metrics.columns
    assert 'peak_value' in metrics.columns
    assert 'drawdown_dollar' in metrics.columns
    assert 'triggered_level' in metrics.columns
    assert 'gross_multiplier' in metrics.columns

    # Check values
    assert metrics.iloc[0]['portfolio_value'] == 100000
    assert metrics.iloc[0]['triggered_level'] is None
    assert metrics.iloc[0]['gross_multiplier'] == 1.0

    assert metrics.iloc[1]['portfolio_value'] == 88000
    assert metrics.iloc[1]['drawdown_dollar'] == 12000
    assert metrics.iloc[1]['triggered_level'] == 1  # Level 2 (0-indexed)
    assert metrics.iloc[1]['gross_multiplier'] == 0.50

    assert metrics.iloc[2]['portfolio_value'] == 91000
    assert metrics.iloc[2]['drawdown_dollar'] == 9000
    assert metrics.iloc[2]['triggered_level'] is None  # Cleared
    assert metrics.iloc[2]['gross_multiplier'] == 1.0

    print("✓ Test passed: calculate_stop_loss_metrics")


def test_validation_2_tuple_only():
    """Test that only 2-tuple format is accepted."""
    daily_pnl = pd.Series([0, -5000])
    initial_capital = 100000

    # 2-tuple should work
    levels = [(5000, 0.75)]
    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital)
    assert len(gross) == 2

    # 3-tuple should fail
    try:
        levels_3tuple = [(5000, 0.75, 2000)]
        calculate_stop_loss_gross(daily_pnl, levels_3tuple, initial_capital)
        assert False, "Should have raised ValueError for 3-tuple"
    except ValueError as e:
        assert "2-tuple" in str(e)

    print("✓ Test passed: Only 2-tuple validation")


def test_multiple_cycles():
    """Test multiple entry/exit cycles."""
    # Lose, recover, lose again pattern
    daily_pnl = pd.Series([0, -12000, 3000, 5000, -15000, 10000])
    levels = [(5000, 0.75), (10000, 0.50)]
    initial_capital = 100000

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital)

    # Day 0: $100k, $0 DD -> 100%
    # Day 1: $88k, $12k DD -> 50%
    # Day 2: $91k, $9k DD -> 100% (cleared)
    # Day 3: $96k, $4k DD -> 100% (still clear)
    # Day 4: $81k, $19k DD -> 50% (re-entered L2)
    # Day 5: $91k, $9k DD -> 100% (cleared again)

    assert gross.iloc[0] == 1.0
    assert gross.iloc[1] == 0.50
    assert gross.iloc[2] == 1.0
    assert gross.iloc[3] == 1.0
    assert gross.iloc[4] == 0.50
    assert gross.iloc[5] == 1.0

    print("✓ Test passed: Multiple cycles")


if __name__ == '__main__':
    test_calculate_stop_loss_gross_basic()
    test_calculate_stop_loss_gross_numpy()
    test_calculate_stop_loss_metrics()
    test_validation_2_tuple_only()
    test_multiple_cycles()
    print("\n✅ All production function tests passed!")

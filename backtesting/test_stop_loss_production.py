"""
Tests for production stop loss calculator.

These tests verify the standalone stop loss functions work correctly
with various input formats and scenarios.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.stop_loss_production import (
    calculate_stop_loss_gross,
    calculate_stop_loss_metrics
)


def test_basic_stop_loss_calculation():
    """Test basic stop loss calculation with simple PnL series."""
    print("\n" + "="*70)
    print("TEST: Basic Stop Loss Calculation")
    print("="*70)

    # Create a simple PnL series
    # Day 0: 0, Portfolio = 10,000
    # Day 1: -600, Portfolio = 9,400 ($600 loss, triggers level 1: 75%)
    # Day 2: -600, Portfolio = 8,800 ($1,200 loss, triggers level 2: 50%)
    # Day 3: +400, Portfolio = 9,200 ($800 recovery from 8,800, back to level 1: 75%)
    # Day 4: +800, Portfolio = 10,000 (back to peak, cleared: 100%)
    daily_pnl = pd.Series([0, -600, -600, 400, 800])

    levels = [
        (500, 0.75, 400),   # $500 loss → 75%, recover at $400
        (1000, 0.50, 600),  # $1000 loss → 50%, recover at $600
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000)

    print(f"\nDaily PnL: {list(daily_pnl)}")
    print(f"Portfolio values: {list(10000 + daily_pnl.cumsum())}")
    print(f"Gross multipliers: {list(gross)}")

    # Verify expected values
    expected = [1.0, 0.75, 0.50, 0.75, 1.0]
    assert list(gross) == expected, f"Expected {expected}, got {list(gross)}"

    print("\n✓ Basic stop loss calculation test passed!")
    print("="*70)


def test_with_dates():
    """Test stop loss calculation with datetime index."""
    print("\n" + "="*70)
    print("TEST: Stop Loss with Datetime Index")
    print("="*70)

    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    daily_pnl = pd.Series([0, -600, -600, 400, 800], index=dates)

    levels = [(500, 0.75, 400), (1000, 0.50, 600)]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000)

    print(f"\nDates: {list(dates)}")
    print(f"\nGross multipliers by date:")
    for date, mult in gross.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {mult:.2f}")

    # Verify index is preserved
    assert isinstance(gross.index, pd.DatetimeIndex)
    assert len(gross) == len(dates)

    print("\n✓ Datetime index test passed!")
    print("="*70)


def test_numpy_array_input():
    """Test with numpy array input."""
    print("\n" + "="*70)
    print("TEST: Numpy Array Input")
    print("="*70)

    daily_pnl = np.array([0, -600, -600, 400, 800])
    dates = pd.date_range('2023-01-01', periods=5, freq='D')

    levels = [(500, 0.75)]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000, dates=dates)

    print(f"\nInput type: {type(daily_pnl)}")
    print(f"Output type: {type(gross)}")
    print(f"Gross multipliers: {list(gross)}")

    assert isinstance(gross, pd.Series)
    assert len(gross) == len(daily_pnl)

    print("\n✓ Numpy array input test passed!")
    print("="*70)


def test_no_recovery_thresholds():
    """Test stop loss without recovery thresholds."""
    print("\n" + "="*70)
    print("TEST: Stop Loss Without Recovery")
    print("="*70)

    # PnL that drops then recovers, but without recovery threshold
    # Without recovery thresholds, levels are determined solely by current drawdown
    daily_pnl = pd.Series([0, -600, -600, 400, 400])

    levels = [
        (500, 0.75),   # No recovery threshold
        (1000, 0.50),  # No recovery threshold
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000)

    portfolio_values = 10000 + daily_pnl.cumsum()
    print(f"\nPortfolio values: {list(portfolio_values)}")
    print(f"Drawdowns: {[10000 - pv for pv in portfolio_values]}")
    print(f"Gross multipliers: {list(gross)}")

    # Without recovery thresholds, level is based on current drawdown from peak
    # Day 0: 10000, DD=0 → 100%
    # Day 1: 9400, DD=600 → 75% (level 1)
    # Day 2: 8800, DD=1200 → 50% (level 2)
    # Day 3: 9200, DD=800 → 75% (level 1, since DD < 1000)
    # Day 4: 9600, DD=400 → 100% (no level triggered, since DD < 500)
    expected = [1.0, 0.75, 0.50, 0.75, 1.0]
    assert list(gross) == expected, f"Expected {expected}, got {list(gross)}"

    print("\nNote: Without recovery thresholds, levels are based on current drawdown")
    print("      from peak, not on recovery from trough.")
    print("\n✓ No recovery test passed!")
    print("="*70)


def test_detailed_metrics():
    """Test detailed metrics function."""
    print("\n" + "="*70)
    print("TEST: Detailed Metrics")
    print("="*70)

    daily_pnl = pd.Series([0, -600, -600, 400, 800])
    levels = [(500, 0.75, 400), (1000, 0.50, 600)]

    metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=10000)

    print(f"\nMetrics DataFrame:")
    print(metrics.to_string())

    # Verify columns exist
    expected_cols = [
        'portfolio_value', 'peak_value', 'trough_value',
        'drawdown_dollar', 'recovery_dollar', 'triggered_level', 'gross_multiplier'
    ]
    for col in expected_cols:
        assert col in metrics.columns, f"Missing column: {col}"

    # Verify some key values
    assert metrics['portfolio_value'].iloc[0] == 10000
    assert metrics['portfolio_value'].iloc[-1] == 10000
    assert metrics['peak_value'].iloc[0] == 10000
    assert metrics['trough_value'].iloc[2] == 8800  # Lowest point
    assert metrics['drawdown_dollar'].iloc[2] == 1200  # Max drawdown

    print("\n✓ Detailed metrics test passed!")
    print("="*70)


def test_multiple_levels_recovery():
    """Test recovery through multiple levels."""
    print("\n" + "="*70)
    print("TEST: Recovery Through Multiple Levels")
    print("="*70)

    # Scenario: Drop through 3 levels, then recover through all of them
    daily_pnl = pd.Series([
        0,      # Day 0: 10,000
        -600,   # Day 1: 9,400 (Level 1: 75%)
        -600,   # Day 2: 8,800 (Level 2: 50%)
        -600,   # Day 3: 8,200 (Level 3: 25%)
        800,    # Day 4: 9,000 ($800 recovery, back to Level 2: 50%)
        600,    # Day 5: 9,600 ($1400 recovery >= $600, back to Level 1, but DD=$400 < $500, so cleared)
        400,    # Day 6: 10,000 (back to peak, cleared: 100%)
    ])

    levels = [
        (500, 0.75, 400),    # Level 1
        (1000, 0.50, 600),   # Level 2
        (1500, 0.25, 800),   # Level 3
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000)
    metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=10000)

    print(f"\nPortfolio progression:")
    for i in range(len(daily_pnl)):
        pv = metrics['portfolio_value'].iloc[i]
        dd = metrics['drawdown_dollar'].iloc[i]
        rec = metrics['recovery_dollar'].iloc[i]
        level = metrics['triggered_level'].iloc[i]
        mult = metrics['gross_multiplier'].iloc[i]
        print(f"  Day {i}: ${pv:,.0f}, DD: ${dd:,.0f}, Rec: ${rec:,.0f}, "
              f"Level: {level}, Gross: {mult:.0%}")

    # Verify the progression
    # Note: On day 5, recovery of $1400 >= $600 moves back to Level 1,
    # but since drawdown is only $400 (< $500), it clears completely
    expected = [1.0, 0.75, 0.50, 0.25, 0.50, 1.0, 1.0]
    assert list(gross) == expected, f"Expected {expected}, got {list(gross)}"

    print("\n✓ Multiple levels recovery test passed!")
    print("="*70)


def test_new_peak_clears_stop_loss():
    """Test that reaching a new peak clears the stop loss."""
    print("\n" + "="*70)
    print("TEST: New Peak Clears Stop Loss")
    print("="*70)

    daily_pnl = pd.Series([
        0,      # Day 0: 10,000 (peak)
        -600,   # Day 1: 9,400 (Level 1: 75%)
        1600,   # Day 2: 11,000 (new peak, cleared)
        -800,   # Day 3: 10,200 (Level 1: 75% again from new peak)
    ])

    levels = [(500, 0.75, 400)]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000)
    metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=10000)

    print(f"\nPortfolio progression:")
    for i in range(len(daily_pnl)):
        pv = metrics['portfolio_value'].iloc[i]
        peak = metrics['peak_value'].iloc[i]
        dd = metrics['drawdown_dollar'].iloc[i]
        mult = metrics['gross_multiplier'].iloc[i]
        print(f"  Day {i}: ${pv:,.0f}, Peak: ${peak:,.0f}, DD: ${dd:,.0f}, Gross: {mult:.0%}")

    expected = [1.0, 0.75, 1.0, 0.75]
    assert list(gross) == expected, f"Expected {expected}, got {list(gross)}"

    # Verify peak was updated
    assert metrics['peak_value'].iloc[2] == 11000
    assert metrics['peak_value'].iloc[3] == 11000

    print("\n✓ New peak clears stop loss test passed!")
    print("="*70)


def test_validation():
    """Test input validation."""
    print("\n" + "="*70)
    print("TEST: Input Validation")
    print("="*70)

    daily_pnl = pd.Series([0, -100, 100])

    # Test empty levels
    try:
        calculate_stop_loss_gross(daily_pnl, [], initial_capital=10000)
        assert False, "Should have raised ValueError for empty levels"
    except ValueError as e:
        print(f"✓ Caught expected error for empty levels: {e}")

    # Test invalid tuple length
    try:
        calculate_stop_loss_gross(daily_pnl, [(100,)], initial_capital=10000)
        assert False, "Should have raised ValueError for invalid tuple length"
    except ValueError as e:
        print(f"✓ Caught expected error for invalid tuple: {e}")

    # Test negative threshold
    try:
        calculate_stop_loss_gross(daily_pnl, [(-100, 0.75)], initial_capital=10000)
        assert False, "Should have raised ValueError for negative threshold"
    except ValueError as e:
        print(f"✓ Caught expected error for negative threshold: {e}")

    # Test invalid gross reduction
    try:
        calculate_stop_loss_gross(daily_pnl, [(100, 1.5)], initial_capital=10000)
        assert False, "Should have raised ValueError for invalid gross reduction"
    except ValueError as e:
        print(f"✓ Caught expected error for invalid gross reduction: {e}")

    print("\n✓ All validation tests passed!")
    print("="*70)


def test_realistic_scenario():
    """Test with a realistic trading scenario."""
    print("\n" + "="*70)
    print("TEST: Realistic Trading Scenario")
    print("="*70)

    # Simulate 20 days of trading with volatile PnL
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 20)  # 0.1% mean, 2% std
    initial_capital = 100000
    daily_pnl = pd.Series(daily_returns * initial_capital)
    daily_pnl.iloc[0] = 0  # First day no PnL

    # Add a crash event
    daily_pnl.iloc[10] = -8000  # Sudden $8k loss
    daily_pnl.iloc[11] = -5000  # Another $5k loss
    # Then recovery
    daily_pnl.iloc[12:15] = [2000, 2000, 2000]

    levels = [
        (5000, 0.75, 2500),
        (10000, 0.50, 5000),
        (15000, 0.25, 7500),
    ]

    metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=initial_capital)

    print(f"\nTrading simulation (20 days, ${initial_capital:,.0f} initial):")
    print("\nKey statistics:")
    print(f"  Starting value: ${metrics['portfolio_value'].iloc[0]:,.2f}")
    print(f"  Ending value: ${metrics['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  Peak value: ${metrics['peak_value'].max():,.2f}")
    print(f"  Max drawdown: ${metrics['drawdown_dollar'].max():,.2f}")
    print(f"  Days with stop loss active: {(metrics['triggered_level'].notna()).sum()}")

    print(f"\nDays when stop loss was triggered:")
    triggered_days = metrics[metrics['triggered_level'].notna()]
    for idx, row in triggered_days.iterrows():
        print(f"  Day {idx}: Level {int(row['triggered_level'])+1}, "
              f"Gross: {row['gross_multiplier']:.0%}, "
              f"Portfolio: ${row['portfolio_value']:,.0f}")

    # Verify we have valid results
    assert len(metrics) == len(daily_pnl)
    assert metrics['gross_multiplier'].min() >= 0
    assert metrics['gross_multiplier'].max() <= 1.0

    print("\n✓ Realistic scenario test passed!")
    print("="*70)


if __name__ == '__main__':
    test_basic_stop_loss_calculation()
    test_with_dates()
    test_numpy_array_input()
    test_no_recovery_thresholds()
    test_detailed_metrics()
    test_multiple_levels_recovery()
    test_new_peak_clears_stop_loss()
    test_validation()
    test_realistic_scenario()

    print("\n" + "="*70)
    print("ALL PRODUCTION STOP LOSS TESTS PASSED!")
    print("="*70)

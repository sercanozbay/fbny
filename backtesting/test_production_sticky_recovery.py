"""
Test production stop loss functions with sticky recovery logic.

Tests the new drawdown-based sticky recovery implementation.
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


def test_sticky_recovery_basic():
    """Test basic sticky recovery behavior."""
    print("\n" + "="*70)
    print("TEST: Sticky Recovery - Basic")
    print("="*70)

    # Scenario demonstrating sticky behavior
    # Starting: $100k
    # Day 1: $94k ($6k DD) → enters Level 1 (75%)
    # Day 2: $95k ($5k DD) → STAYS at Level 1 (sticky, DD > $2k exit)
    # Day 3: $96k ($4k DD) → STAYS at Level 1 (sticky, DD > $2k exit)
    # Day 4: $98.5k ($1.5k DD) → exits Level 1 (DD ≤ $2k), back to 100%
    daily_pnl = pd.Series([0, -6000, 1000, 1000, 2500])

    levels = [(5000, 0.75, 2000)]  # Enter $5k, Exit $2k

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print(f"\nConfiguration: Enter at $5k DD, Exit at $2k DD")
    print(f"\nDay-by-day:")
    portfolio_values = 100000 + daily_pnl.cumsum()
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        dd = max(0, 100000 - pv)
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.75, 0.75, 0.75, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Sticky recovery test passed!")
    print("  - Entered at $6k DD")
    print("  - Stayed at 75% through $5k and $4k DD (sticky!)")
    print("  - Exited when DD improved to $1.5k (≤ $2k)")
    print("="*70)


def test_multiple_levels_sticky():
    """Test sticky behavior across multiple levels."""
    print("\n" + "="*70)
    print("TEST: Multiple Levels with Sticky Recovery")
    print("="*70)

    # Start: $100k
    # Day 1: $88k ($12k DD) → Level 2 (50%)
    # Day 2: $91k ($9k DD) → STAYS Level 2 (DD > $5k exit)
    # Day 3: $93k ($7k DD) → STAYS Level 2 (DD > $5k exit)
    # Day 4: $95.5k ($4.5k DD) → Exits Level 2, enters Level 1 (75%)
    # Day 5: $96k ($4k DD) → STAYS Level 1 (DD > $2k exit)
    # Day 6: $98.5k ($1.5k DD) → Exits Level 1 (100%)
    daily_pnl = pd.Series([0, -12000, 3000, 2000, 2500, 500, 2500])

    levels = [
        (5000, 0.75, 2000),   # Level 1
        (10000, 0.50, 5000),  # Level 2
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k, Exit $2k, 75% gross")
    print(f"  Level 2: Enter $10k, Exit $5k, 50% gross")

    print(f"\nDay-by-day:")
    portfolio_values = 100000 + daily_pnl.cumsum()
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        dd = max(0, 100000 - pv)
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.50, 0.50, 0.50, 0.75, 0.75, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Multiple levels sticky recovery test passed!")
    print("  - Sticky at Level 2 (50%) from Day 1-3")
    print("  - Recovered to Level 1 (75%) on Day 4")
    print("  - Sticky at Level 1 from Day 4-5")
    print("  - Fully recovered to 100% on Day 6")
    print("="*70)


def test_new_peak_clears():
    """Test that new peak clears stop loss."""
    print("\n" + "="*70)
    print("TEST: New Peak Clears Stop Loss")
    print("="*70)

    # Start: $100k
    # Day 1: $94k ($6k DD) → Level 1 (75%)
    # Day 2: $101k (new peak) → Cleared (100%)
    # Day 3: $96k ($5k DD from new peak) → Level 1 again (75%)
    daily_pnl = pd.Series([0, -6000, 7000, -5000])

    levels = [(5000, 0.75, 2000)]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print(f"\nDay-by-day:")
    portfolio_values = 100000 + daily_pnl.cumsum()
    peak = 100000
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        if pv > peak:
            peak = pv
        dd = max(0, peak - pv)
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (Peak: ${peak:,.0f}, DD: ${dd:,.0f}) → {mult:.0%}")

    expected = [1.0, 0.75, 1.0, 0.75]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ New peak clears stop loss test passed!")
    print("="*70)


def test_no_recovery_threshold():
    """Test sticky behavior without recovery threshold."""
    print("\n" + "="*70)
    print("TEST: No Recovery Threshold (Only New Peak Clears)")
    print("="*70)

    # With no recovery threshold, should stay at level until new peak
    # Start: $100k
    # Day 1: $94k ($6k DD) → Level 1 (75%)
    # Day 2: $96k ($4k DD) → STAYS Level 1 (no recovery threshold)
    # Day 3: $98k ($2k DD) → STAYS Level 1 (no recovery threshold)
    # Day 4: $99k ($1k DD) → STAYS Level 1 (no recovery threshold)
    # Day 5: $101k (new peak) → Cleared (100%)
    daily_pnl = pd.Series([0, -6000, 2000, 2000, 1000, 2000])

    levels = [(5000, 0.75)]  # No recovery threshold

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print(f"\nConfiguration: Enter at $5k DD, NO recovery threshold")

    print(f"\nDay-by-day:")
    portfolio_values = 100000 + daily_pnl.cumsum()
    peak = 100000
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        if pv > peak:
            peak = pv
        dd = max(0, peak - pv)
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (Peak: ${peak:,.0f}, DD: ${dd:,.0f}) → {mult:.0%}")

    expected = [1.0, 0.75, 0.75, 0.75, 0.75, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ No recovery threshold test passed!")
    print("  - Stayed at 75% even as DD improved to $1k")
    print("  - Only cleared at new peak")
    print("="*70)


def test_metrics_function():
    """Test detailed metrics function."""
    print("\n" + "="*70)
    print("TEST: Detailed Metrics with Sticky Recovery")
    print("="*70)

    daily_pnl = pd.Series([0, -6000, 1000, 1000, 2500])
    levels = [(5000, 0.75, 2000)]

    metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=100000)

    print(f"\nMetrics DataFrame:")
    print(metrics.to_string())

    # Verify sticky behavior in metrics
    assert metrics['triggered_level'].iloc[1] == 0  # Day 1: Level 0
    assert metrics['triggered_level'].iloc[2] == 0  # Day 2: Still Level 0 (sticky)
    assert metrics['triggered_level'].iloc[3] == 0  # Day 3: Still Level 0 (sticky)
    assert metrics['triggered_level'].iloc[4] is None  # Day 4: Cleared

    assert metrics['gross_multiplier'].iloc[1] == 0.75
    assert metrics['gross_multiplier'].iloc[2] == 0.75  # Sticky!
    assert metrics['gross_multiplier'].iloc[3] == 0.75  # Sticky!
    assert metrics['gross_multiplier'].iloc[4] == 1.0

    print("\n✓ Metrics function test passed!")
    print("  - Correctly tracked triggered_level through sticky period")
    print("  - Correctly tracked gross_multiplier")
    print("="*70)


def test_validation():
    """Test input validation with new logic."""
    print("\n" + "="*70)
    print("TEST: Input Validation")
    print("="*70)

    daily_pnl = pd.Series([0, -100, 100])

    # Test recovery_drawdown >= drawdown_threshold
    try:
        calculate_stop_loss_gross(
            daily_pnl,
            [(5000, 0.75, 6000)],  # Recovery > Entry!
            initial_capital=10000
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    # Test recovery_drawdown == drawdown_threshold
    try:
        calculate_stop_loss_gross(
            daily_pnl,
            [(5000, 0.75, 5000)],  # Recovery == Entry!
            initial_capital=10000
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("\n✓ All validation tests passed!")
    print("="*70)


def test_rapid_fluctuations():
    """Test that sticky behavior prevents rapid level changes."""
    print("\n" + "="*70)
    print("TEST: Sticky Prevents Rapid Fluctuations")
    print("="*70)

    # Scenario with rapid up/down movements
    # Start: $100k
    # Day 1: $93k ($7k DD) → Level 1 (75%)
    # Day 2: $94k ($6k DD) → STAYS Level 1 (sticky)
    # Day 3: $93k ($7k DD) → STAYS Level 1 (sticky)
    # Day 4: $94.5k ($5.5k DD) → STAYS Level 1 (sticky, DD > $2k exit)
    # Day 5: $95k ($5k DD) → STAYS Level 1 (sticky, DD > $2k exit)
    # Day 6: $98.5k ($1.5k DD) → Exits Level 1
    daily_pnl = pd.Series([0, -7000, 1000, -1000, 1500, 500, 3500])

    levels = [(5000, 0.75, 2000)]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print(f"\nConfiguration: Enter $5k, Exit $2k")
    print(f"\nDay-by-day (demonstrating sticky behavior):")
    portfolio_values = 100000 + daily_pnl.cumsum()
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        dd = max(0, 100000 - pv)
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%}")

    # Should stay at 75% through all the fluctuations
    expected = [1.0, 0.75, 0.75, 0.75, 0.75, 0.75, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Rapid fluctuations test passed!")
    print("  - Stayed at 75% through $6k-$7k DD fluctuations")
    print("  - Prevents unnecessary trading from level changes")
    print("="*70)


if __name__ == '__main__':
    test_sticky_recovery_basic()
    test_multiple_levels_sticky()
    test_new_peak_clears()
    test_no_recovery_threshold()
    test_metrics_function()
    test_validation()
    test_rapid_fluctuations()

    print("\n" + "="*70)
    print("ALL STICKY RECOVERY PRODUCTION TESTS PASSED!")
    print("="*70)

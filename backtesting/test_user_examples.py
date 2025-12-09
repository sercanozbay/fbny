"""
Test using exact user-provided examples
"""

import pandas as pd
from backtesting.stop_loss_production import calculate_stop_loss_gross


def test_example_1():
    """Test Example 1 from user."""
    print("\n" + "="*70)
    print("TEST: User Example 1")
    print("="*70)

    # Levels: [(5000, 75%, 7500), (10000, 50%, 15000)]
    # Day 1: DD=$6k → 75%
    # Day 2: DD=$8k → 75%
    # Day 3: DD=$9k → 75%
    # Day 4: DD=$12k → 50%
    # Day 5: DD=$16k → 50%
    # Day 6: DD=$15.5k → 50%
    # Day 7: DD=$14k → 75%
    # Day 8: DD=$13k → 75%
    # Day 9: DD=$8k → 75%
    # Day 10: DD=$10k → 50%

    # Create PnL series that produces these DDs
    drawdowns = [0, 6000, 8000, 9000, 12000, 16000, 15500, 14000, 13000, 8000, 10000]
    daily_pnl = [0] + [-drawdowns[i] + drawdowns[i-1] if i > 0 else -drawdowns[i] for i in range(1, len(drawdowns))]
    daily_pnl = pd.Series(daily_pnl)

    levels = [
        (5000, 0.75, 7500),
        (10000, 0.50, 15000),
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print("\nExpected behavior:")
    for i, dd in enumerate(drawdowns):
        expected = None
        if i == 0:
            expected = 1.0
        elif i <= 3:
            expected = 0.75
        elif i <= 6:
            expected = 0.50
        elif i <= 9:
            expected = 0.75
        elif i == 10:
            expected = 0.50

        actual = gross.iloc[i]
        status = "✓" if abs(actual - expected) < 0.01 else "✗"
        print(f"  Day {i}: DD=${dd:,.0f} → Expected {expected:.0%}, Got {actual:.0%} {status}")

    expected = [1.0, 0.75, 0.75, 0.75, 0.50, 0.50, 0.50, 0.75, 0.75, 0.75, 0.50]
    actual = list(gross)

    print(f"\nExpected: {expected}")
    print(f"Actual:   {actual}")

    assert actual == expected, f"Test failed"
    print("\n✓ Example 1 passed!")
    print("="*70)


def test_example_2():
    """Test Example 2 from user."""
    print("\n" + "="*70)
    print("TEST: User Example 2")
    print("="*70)

    # Same as Example 1 through Day 8, then diverges
    # Day 9: DD=$15.5k → 50%
    # Day 10: DD=$17k → 50%

    drawdowns = [0, 6000, 8000, 9000, 12000, 16000, 15500, 14000, 13000, 15500, 17000]
    daily_pnl = [0] + [-drawdowns[i] + drawdowns[i-1] if i > 0 else -drawdowns[i] for i in range(1, len(drawdowns))]
    daily_pnl = pd.Series(daily_pnl)

    levels = [
        (5000, 0.75, 7500),
        (10000, 0.50, 15000),
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    print("\nExpected behavior:")
    for i, dd in enumerate(drawdowns):
        expected = None
        if i == 0:
            expected = 1.0
        elif i <= 3:
            expected = 0.75
        elif i <= 6:
            expected = 0.50
        elif i <= 8:
            expected = 0.75
        elif i >= 9:
            expected = 0.50

        actual = gross.iloc[i]
        status = "✓" if abs(actual - expected) < 0.01 else "✗"
        print(f"  Day {i}: DD=${dd:,.0f} → Expected {expected:.0%}, Got {actual:.0%} {status}")

    expected = [1.0, 0.75, 0.75, 0.75, 0.50, 0.50, 0.50, 0.75, 0.75, 0.50, 0.50]
    actual = list(gross)

    print(f"\nExpected: {expected}")
    print(f"Actual:   {actual}")

    assert actual == expected, f"Test failed"
    print("\n✓ Example 2 passed!")
    print("="*70)


if __name__ == "__main__":
    test_example_1()
    test_example_2()

    print("\n" + "="*70)
    print("ALL USER EXAMPLES PASSED!")
    print("="*70)
    print("\nKey behaviors demonstrated:")
    print("  1. Enter level when DD >= drawdown_threshold")
    print("  2. Stay at level while DD >= recovery_drawdown")
    print("  3. Exit when DD < recovery_drawdown")
    print("  4. Hysteresis: Re-enter only when DD >= recovery_drawdown")
    print("="*70)

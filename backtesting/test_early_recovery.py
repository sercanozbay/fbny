"""
Test Early Recovery Logic

Demonstrates the new stop loss behavior where:
- recovery_drawdown > drawdown_threshold (scale up early during recovery)
- recovery_drawdown defaults to drawdown_threshold when None (immediate scale up)
"""

import pandas as pd
from backtesting.stop_loss_production import calculate_stop_loss_gross


def test_immediate_scale_up():
    """Test immediate scale up when recovery_drawdown is None (defaults to drawdown_threshold)."""
    print("\n" + "="*70)
    print("TEST: Immediate Scale Up (No Recovery Threshold)")
    print("="*70)

    # Scenario: DD crosses thresholds, scales up immediately when DD improves below threshold
    # Day 0: $0 DD → 100%
    # Day 1: $12k DD → 50% (DD ≥ $10k entry)
    # Day 2: $9k DD → 75% (DD < $10k, DD ≥ $5k entry) - immediate scale up!
    # Day 3: $4k DD → 100% (DD < $5k entry) - immediate scale up!

    daily_pnl = pd.Series([0, -12000, 3000, 5000])
    levels = [
        (5000, 0.75),   # Enter at $5k DD, exit immediately when DD < $5k
        (10000, 0.50),  # Enter at $10k DD, exit immediately when DD < $10k
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    portfolio_values = 100000 + daily_pnl.cumsum()
    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k DD → 75% gross (immediate scale up)")
    print(f"  Level 2: Enter $10k DD → 50% gross (immediate scale up)")

    print(f"\nDay-by-day:")
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        dd = max(0, 100000 - pv) if pv < 100000 else 0
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.50, 0.75, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Immediate scale up test passed!")
    print("  - Entered 50% at $12k DD")
    print("  - IMMEDIATELY scaled to 75% at $9k DD (DD < $10k)")
    print("  - IMMEDIATELY scaled to 100% at $4k DD (DD < $5k)")
    print("="*70)


def test_early_recovery():
    """Test early recovery when recovery_drawdown > drawdown_threshold."""
    print("\n" + "="*70)
    print("TEST: Early Recovery (Scale Up Before Full Recovery)")
    print("="*70)

    # Scenario: Scale up early to recoup losses faster
    # Day 0: $0 DD → 100%
    # Day 1: $15k DD → 50% (DD ≥ $10k entry, DD >= $15k recovery)
    # Day 2: $12k DD → 75% (DD < $15k recovery → exit Level 2, DD ≥ $5k → enter Level 1)
    # Day 3: $14k DD → 75% (DD < $15k, DD ≥ $7.5k → stay Level 1)
    # Day 4: $6k DD → 100% (DD < $7.5k recovery → early exit from Level 1!)
    # Day 5: $2k DD → 100% (stay at 100%)

    daily_pnl = pd.Series([0, -15000, 3000, -2000, 8000, 4000])
    levels = [
        (5000, 0.75, 7500),   # Enter at $5k DD, scale up at $7.5k DD (worse)
        (10000, 0.50, 15000), # Enter at $10k DD, scale up at $15k DD (worse)
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    portfolio_values = 100000 + daily_pnl.cumsum()
    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k DD, Scale up at $7.5k DD → 75% gross")
    print(f"  Level 2: Enter $10k DD, Scale up at $15k DD → 50% gross")

    print(f"\nDay-by-day:")
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        # Track peak properly
        peak = 100000
        if i >= 1 and portfolio_values.iloc[:i+1].max() > peak:
            peak = portfolio_values.iloc[:i+1].max()
        dd = peak - pv if pv < peak else 0
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.50, 0.75, 0.75, 1.0, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Early recovery test passed!")
    print("  - Entered 50% at $15k DD")
    print("  - Scaled to 75% at $12k DD (DD < $15k recovery → early scale up!)")
    print("  - Stayed at 75% at $14k DD")
    print("  - Scaled to 100% at $6k DD (DD < $7.5k recovery → early scale up!)")
    print("  This demonstrates EARLY recovery: exited at $6k DD even though entry was $5k")
    print("="*70)


def test_very_early_recovery():
    """Test very early recovery to recoup losses as quickly as possible."""
    print("\n" + "="*70)
    print("TEST: Very Early Recovery (Aggressive Scale Up)")
    print("="*70)

    # Scenario: Recovery threshold >> entry threshold for aggressive scaling
    # Day 0: $0 DD → 100%
    # Day 1: $20k DD → 75% (DD ≥ $10k but DD < $25k recovery → exit Level 2, DD ≥ $5k → Level 1)
    # Day 2: $22k DD → 75% (DD < $25k, DD ≥ $15k → stay Level 1)
    # Day 3: $12k DD → 100% (DD < $15k recovery → exit Level 1)
    # Day 4: $4k DD → 100% (stay at 100%)

    daily_pnl = pd.Series([0, -20000, -2000, 10000, 8000])
    levels = [
        (5000, 0.75, 15000),  # Enter at $5k DD, scale up at $15k DD
        (10000, 0.50, 25000), # Enter at $10k DD, scale up at $25k DD
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    portfolio_values = 100000 + daily_pnl.cumsum()
    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k DD, Scale up at $15k DD → 75% gross")
    print(f"  Level 2: Enter $10k DD, Scale up at $25k DD → 50% gross")
    print(f"\nLogic: Very aggressive - scale up even while still in deep drawdown")

    print(f"\nDay-by-day:")
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        peak = 100000
        dd = peak - pv if pv < peak else 0
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.75, 0.75, 1.0, 1.0]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Very early recovery test passed!")
    print("  - Entered 75% at $20k DD (never hit Level 2 because DD < $25k recovery!)")
    print("  - Stayed at 75% at $22k DD")
    print("  - Scaled to 100% at $12k DD (DD < $15k recovery → very aggressive!)")
    print("  This shows VERY aggressive recovery: never even fully entered deeper levels")
    print("="*70)


if __name__ == "__main__":
    test_immediate_scale_up()
    test_early_recovery()
    test_very_early_recovery()

    print("\n" + "="*70)
    print("ALL EARLY RECOVERY TESTS PASSED!")
    print("="*70)

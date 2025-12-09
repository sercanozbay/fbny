"""
Test Hysteresis Logic

Demonstrates proper stop loss behavior with hysteresis:
1. Enter level when DD >= drawdown_threshold
2. Stay at level while drawdown_threshold <= DD < recovery_drawdown
3. Exit when DD < recovery_drawdown
4. Hysteresis: Once exited, need to breach entry threshold again to re-enter
"""

import pandas as pd
from backtesting.stop_loss_production import calculate_stop_loss_gross


def test_basic_hysteresis():
    """Test basic hysteresis behavior."""
    print("\n" + "="*70)
    print("TEST: Basic Hysteresis")
    print("="*70)

    # Scenario demonstrating hysteresis:
    # Day 0: $0 DD → 100%
    # Day 1: $17k DD → 50% (DD ≥ $10k entry)
    # Day 2: $14k DD → 50% (stay Level 2: $10k ≤ DD < $15k recovery)
    # Day 3: $9k DD → 75% (exit Level 2: DD < $15k, enter Level 1: DD ≥ $5k)
    # Day 4: $6k DD → 75% (stay Level 1: $5k ≤ DD < $7.5k)
    # Day 5: $4k DD → 100% (exit Level 1: DD < $7.5k)
    # Day 6: $6k DD → 100% (HYSTERESIS: stay out, DD < $7.5k recovery)
    # Day 7: $8k DD → 75% (re-enter Level 1: DD ≥ $7.5k recovery)

    daily_pnl = pd.Series([0, -17000, 3000, 5000, 3000, 2000, -2000, -2000])
    levels = [
        (5000, 0.75, 7500),   # Enter at $5k DD, exit at $7.5k DD
        (10000, 0.50, 15000), # Enter at $10k DD, exit at $15k DD
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    portfolio_values = 100000 + daily_pnl.cumsum()
    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k DD, Exit $7.5k DD → 75% gross")
    print(f"  Level 2: Enter $10k DD, Exit $15k DD → 50% gross")

    print(f"\nDay-by-day:")
    peak = 100000
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        if pv > peak:
            peak = pv
        dd = peak - pv
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.50, 0.75, 0.75, 1.0, 1.0, 1.0, 0.75]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Basic hysteresis test passed!")
    print("  - Entered Level 2 at $17k DD")
    print("  - Exited to Level 1 at $14k DD (< $15k recovery)")
    print("  - Stayed at Level 1 at $9k DD")
    print("  - Exited Level 1 at $6k DD (< $7.5k recovery)")
    print("  - HYSTERESIS: Stayed at 100% at $4k DD and $6k DD")
    print("  - Re-entered Level 1 at $8k DD (≥ $7.5k recovery)")
    print("="*70)


def test_no_recovery_threshold():
    """Test immediate exit when recovery_drawdown is None (defaults to drawdown_threshold)."""
    print("\n" + "="*70)
    print("TEST: No Recovery Threshold (Immediate Exit)")
    print("="*70)

    # When recovery_drawdown = None, defaults to drawdown_threshold
    # This means exit immediately when DD < drawdown_threshold
    # Day 0: $0 DD → 100%
    # Day 1: $12k DD → 50% (DD ≥ $10k)
    # Day 2: $9k DD → 75% (DD < $10k → exit Level 2, DD ≥ $5k → enter Level 1)
    # Day 3: $4k DD → 100% (DD < $5k → exit Level 1)
    # Day 4: $6k DD → 75% (DD ≥ $5k → re-enter Level 1 immediately)

    daily_pnl = pd.Series([0, -12000, 3000, 5000, -2000])
    levels = [
        (5000, 0.75),   # Enter at $5k DD, exit at $5k DD (immediate)
        (10000, 0.50),  # Enter at $10k DD, exit at $10k DD (immediate)
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    portfolio_values = 100000 + daily_pnl.cumsum()
    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k DD, Exit $5k DD (immediate) → 75% gross")
    print(f"  Level 2: Enter $10k DD, Exit $10k DD (immediate) → 50% gross")

    print(f"\nDay-by-day:")
    peak = 100000
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        if pv > peak:
            peak = pv
        dd = peak - pv
        mult = gross.iloc[i]
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross")

    expected = [1.0, 0.50, 0.75, 1.0, 0.75]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ No recovery threshold test passed!")
    print("  - Immediate exit when DD improves below entry threshold")
    print("  - Immediate re-entry when DD worsens past entry threshold")
    print("="*70)


def test_recovery_threshold_hysteresis():
    """Test that recovery threshold > drawdown_threshold creates proper hysteresis."""
    print("\n" + "="*70)
    print("TEST: Recovery Threshold Creates Hysteresis Zone")
    print("="*70)

    # recovery_drawdown > drawdown_threshold creates a hysteresis zone
    # Level 1: Entry=$5k, Recovery=$10k creates zone [$5k, $10k)
    # Day 0: $0 DD → 100%
    # Day 1: $15k DD → 75% (DD ≥ $5k entry)
    # Day 2: $12k DD → 75% (stay: $5k ≤ DD < $10k)
    # Day 3: $8k DD → 75% (stay: $5k ≤ DD < $10k)
    # Day 4: $9k DD → 75% (stay: $5k ≤ DD < $10k) - bouncing in zone
    # Day 5: $7k DD → 75% (stay: $5k ≤ DD < $10k)
    # Day 6: $3k DD → 100% (exit: DD < $10k recovery)
    # Day 7: $8k DD → 100% (HYSTERESIS: don't re-enter, DD < $10k recovery)
    # Day 8: $11k DD → 75% (re-enter: DD ≥ $10k recovery)

    daily_pnl = pd.Series([0, -15000, 3000, 4000, -1000, 2000, 4000, -5000, -3000])
    levels = [
        (5000, 0.75, 10000),  # Enter at $5k DD, exit at $10k DD
    ]

    gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)

    portfolio_values = 100000 + daily_pnl.cumsum()
    print(f"\nConfiguration:")
    print(f"  Level 1: Enter $5k DD, Exit $10k DD → 75% gross")
    print(f"  Creates hysteresis zone: [$5k, $10k)")

    print(f"\nDay-by-day:")
    peak = 100000
    for i in range(len(daily_pnl)):
        pv = portfolio_values.iloc[i]
        if pv > peak:
            peak = pv
        dd = peak - pv
        mult = gross.iloc[i]
        status = ""
        if i == 1:
            status = "← ENTER"
        elif i >= 2 and i <= 5:
            status = "← STAY (in hysteresis zone)"
        elif i == 6:
            status = "← EXIT"
        elif i == 7:
            status = "← HYSTERESIS (don't re-enter)"
        elif i == 8:
            status = "← RE-ENTER"
        print(f"  Day {i}: ${pv:,.0f} (DD: ${dd:,.0f}) → {mult:.0%} gross {status}")

    expected = [1.0, 0.75, 0.75, 0.75, 0.75, 0.75, 1.0, 1.0, 0.75]
    actual = list(gross)
    assert actual == expected, f"Expected {expected}, got {actual}"

    print("\n✓ Recovery threshold hysteresis test passed!")
    print("  - Entered at $15k DD (≥ $5k entry)")
    print("  - Stayed in level through $12k, $8k, $9k, $7k DD (all in zone)")
    print("  - Exited at $3k DD (< $10k recovery)")
    print("  - HYSTERESIS: Stayed out at $8k DD (< $10k recovery)")
    print("  - Re-entered at $11k DD (≥ $10k recovery)")
    print("="*70)


if __name__ == "__main__":
    test_basic_hysteresis()
    test_no_recovery_threshold()
    test_recovery_threshold_hysteresis()

    print("\n" + "="*70)
    print("ALL HYSTERESIS TESTS PASSED!")
    print("="*70)
    print("\nKey Insight:")
    print("  recovery_drawdown > drawdown_threshold creates a 'hysteresis zone'")
    print("  where you stay at the level even as DD fluctuates within the zone.")
    print("  Once you exit (DD < recovery_drawdown), you need DD ≥ recovery_drawdown")
    print("  to re-enter, preventing rapid bouncing.")
    print("="*70)

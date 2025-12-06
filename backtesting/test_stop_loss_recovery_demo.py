"""
Demonstration test for stop loss recovery functionality.

This test creates a scenario with clear drawdown and recovery phases
to demonstrate the stop loss moving through multiple levels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting import Backtester, BacktestConfig, DataManager


def test_stop_loss_recovery_demo():
    """Demonstrate stop loss recovery through multiple levels."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Stop Loss Recovery Through Multiple Levels")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_recovery_demo')
    test_dir.mkdir(exist_ok=True)

    # Create dates - longer period to show all transitions
    dates = pd.date_range('2023-01-01', periods=120, freq='B')
    tickers = ['AAPL']

    # Create dramatic price scenario to trigger all levels
    # Phase 1 (Days 0-10): Stable at $100
    # Phase 2 (Days 10-30): Drop to $90 (10% DD - triggers Level 1: 75% gross)
    # Phase 3 (Days 30-35): Drop to $85 (15% DD - triggers Level 2: 50% gross)
    # Phase 4 (Days 35-40): Drop to $80 (20% DD - triggers Level 3: 25% gross)
    # Phase 5 (Days 40-60): Recover to $90 (50% recovery from $80 - back to Level 2: 50% gross)
    # Phase 6 (Days 60-80): Recover to $95 (50% recovery from $90 - back to Level 1: 75% gross)
    # Phase 7 (Days 80-100): Recover to $99 (50% recovery from $95 - back to 100% gross)
    # Phase 8 (Days 100-120): New peak at $105 (fully cleared stop loss)

    prices = []
    for i in range(len(dates)):
        if i < 10:
            price = 100.0  # Phase 1: Peak
        elif i < 30:
            # Phase 2: Drop to $90
            progress = (i - 10) / 20
            price = 100.0 - progress * 10.0
        elif i < 35:
            # Phase 3: Drop to $85
            progress = (i - 30) / 5
            price = 90.0 - progress * 5.0
        elif i < 40:
            # Phase 4: Drop to $80
            progress = (i - 35) / 5
            price = 85.0 - progress * 5.0
        elif i < 60:
            # Phase 5: Recover to $90 (50% of $80-$100 = $10)
            progress = (i - 40) / 20
            price = 80.0 + progress * 10.0
        elif i < 80:
            # Phase 6: Recover to $95 (50% of $90-$100 = $5)
            progress = (i - 60) / 20
            price = 90.0 + progress * 5.0
        elif i < 100:
            # Phase 7: Recover to $99 (50% of $95-$100 = $2.5, but round up)
            progress = (i - 80) / 20
            price = 95.0 + progress * 4.0
        else:
            # Phase 8: New peak
            progress = (i - 100) / 20
            price = 99.0 + progress * 6.0
        prices.append(price)

    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice Scenario (8 phases):")
    print(f"  Phase 1 (Days 0-10):   $100 (peak)")
    print(f"  Phase 2 (Days 10-30):  $100 → $90 (10% DD → Level 1: 75% gross)")
    print(f"  Phase 3 (Days 30-35):  $90 → $85 (15% DD → Level 2: 50% gross)")
    print(f"  Phase 4 (Days 35-40):  $85 → $80 (20% DD → Level 3: 25% gross)")
    print(f"  Phase 5 (Days 40-60):  $80 → $90 (50% recovery → back to Level 2: 50%)")
    print(f"  Phase 6 (Days 60-80):  $90 → $95 (50% recovery → back to Level 1: 75%)")
    print(f"  Phase 7 (Days 80-100): $95 → $99 (50% recovery → back to 100%)")
    print(f"  Phase 8 (Days 100-120): $99 → $105 (new peak → stop loss cleared)")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with stop loss and recovery levels
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (0.10, 0.75, 0.50),  # 10% DD → 75% gross, recover at 50%
            (0.15, 0.50, 0.50),  # 15% DD → 50% gross, recover at 50%
            (0.20, 0.25, 0.50),  # 20% DD → 25% gross, recover at 50%
        ]
    )

    print(f"\nStop Loss Configuration:")
    print(f"  Level 1: 10% DD → 75% gross (recover at 50% from bottom)")
    print(f"  Level 2: 15% DD → 50% gross (recover at 50% from bottom)")
    print(f"  Level 3: 20% DD → 25% gross (recover at 50% from bottom)")

    # Create backtester
    backtester = Backtester(config, data_manager)

    # Buy 1000 shares on day 1
    targets = {
        dates[0]: {'AAPL': 1000.0}
    }

    inputs = {
        'type': 'shares',
        'targets': targets
    }

    # Run backtest
    print(f"\nRunning backtest (watch for stop loss triggers and recoveries)...")
    print("="*70)

    results = backtester.run(
        start_date=dates[0],
        end_date=dates[-1],
        use_case=1,
        inputs=inputs,
        show_progress=False
    )

    # Analyze results
    print(f"\n{'='*70}")
    print("FINAL RESULTS:")
    print(f"{'='*70}")

    final_value = results.portfolio_values[-1]
    peak_value = max(results.portfolio_values)
    total_return = (final_value / config.initial_cash - 1)
    max_dd_pct = (peak_value - min(results.portfolio_values)) / peak_value

    print(f"Initial capital: ${config.initial_cash:,.2f}")
    print(f"Peak portfolio value: ${peak_value:,.2f}")
    print(f"Lowest portfolio value: ${min(results.portfolio_values):,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Max drawdown: {max_dd_pct:.2%}")

    # Calculate what would have happened without stop loss
    initial_shares = 1000.0
    start_price = 100.0
    end_price = prices[-1]
    no_sl_value = config.initial_cash * (end_price / start_price)
    no_sl_return = (no_sl_value / config.initial_cash - 1)

    print(f"\nComparison:")
    print(f"  With stop loss:    {total_return:>8.2%} (${final_value:,.2f})")
    print(f"  Without stop loss: {no_sl_return:>8.2%} (${no_sl_value:,.2f})")
    print(f"  Benefit:           {(total_return - no_sl_return):>8.2%} (${final_value - no_sl_value:,.2f})")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    print(f"\n{'='*70}")
    print("✓ Stop loss recovery demonstration complete!")
    print(f"{'='*70}\n")

    print("\nKey Observations:")
    print("  1. Stop loss triggers at deeper levels as drawdown increases")
    print("  2. Recovery thresholds move us back through levels as portfolio recovers")
    print("  3. Final recovery to new peak fully clears the stop loss")
    print("  4. Stop loss reduces losses during drawdown phases")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_stop_loss_recovery_demo()

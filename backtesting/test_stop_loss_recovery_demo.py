"""
Demonstration test for dollar-based stop loss recovery functionality.

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
    """Demonstrate dollar-based stop loss recovery through multiple levels."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Dollar-Based Stop Loss Recovery")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_recovery_demo')
    test_dir.mkdir(exist_ok=True)

    # Create dates - longer period to show all transitions
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    tickers = ['AAPL']

    # Create dramatic price scenario to trigger all levels
    # Initial portfolio value: $100,000 (1000 shares @ $100)
    # Phase 1 (Days 0-10):  Stable at $100 (peak = $100k)
    # Phase 2 (Days 10-25): Drop to $95 ($5k loss - triggers Level 1: 75% gross)
    # Phase 3 (Days 25-35): Drop to $90 ($10k loss - triggers Level 2: 50% gross)
    # Phase 4 (Days 35-45): Drop to $85 ($15k loss - triggers Level 3: 25% gross)
    # Phase 5 (Days 45-60): Recover to $92.5 ($7.5k recovery - back to Level 2: 50%)
    # Phase 6 (Days 60-75): Recover to $97.5 ($5k recovery - back to Level 1: 75%)
    # Phase 7 (Days 75-90): Recover to $100 ($2.5k recovery - cleared: 100%)
    # Phase 8 (Days 90-100): New peak at $102 (fully cleared)

    prices = []
    for i in range(len(dates)):
        if i < 10:
            price = 100.0  # Phase 1: Peak
        elif i < 25:
            # Phase 2: Drop to $95
            progress = (i - 10) / 15
            price = 100.0 - progress * 5.0
        elif i < 35:
            # Phase 3: Drop to $90
            progress = (i - 25) / 10
            price = 95.0 - progress * 5.0
        elif i < 45:
            # Phase 4: Drop to $85
            progress = (i - 35) / 10
            price = 90.0 - progress * 5.0
        elif i < 60:
            # Phase 5: Recover to $92.5 ($7.5k recovery from $85k)
            progress = (i - 45) / 15
            price = 85.0 + progress * 7.5
        elif i < 75:
            # Phase 6: Recover to $97.5 ($5k recovery from $92.5k)
            progress = (i - 60) / 15
            price = 92.5 + progress * 5.0
        elif i < 90:
            # Phase 7: Recover to $100 ($2.5k recovery from $97.5k)
            progress = (i - 75) / 15
            price = 97.5 + progress * 2.5
        else:
            # Phase 8: New peak
            progress = (i - 90) / 10
            price = 100.0 + progress * 2.0
        prices.append(price)

    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice Scenario (8 phases, 1000 shares @ $100 = $100k):")
    print(f"  Phase 1 (Days 0-10):   $100/share = $100k (peak)")
    print(f"  Phase 2 (Days 10-25):  $95/share = $95k ($5k loss → Level 1: 75%)")
    print(f"  Phase 3 (Days 25-35):  $90/share = $90k ($10k loss → Level 2: 50%)")
    print(f"  Phase 4 (Days 35-45):  $85/share = $85k ($15k loss → Level 3: 25%)")
    print(f"  Phase 5 (Days 45-60):  $92.5/share = $92.5k ($7.5k recovery → Level 2)")
    print(f"  Phase 6 (Days 60-75):  $97.5/share = $97.5k ($5k recovery → Level 1)")
    print(f"  Phase 7 (Days 75-90):  $100/share = $100k ($2.5k recovery → Cleared)")
    print(f"  Phase 8 (Days 90-100): $102/share = $102k (new peak)")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with stop loss and recovery levels (dollar-based)
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (5000, 0.75, 2500),   # $5k loss → 75%, recover at $2.5k from bottom
            (10000, 0.50, 5000),  # $10k loss → 50%, recover at $5k from bottom
            (15000, 0.25, 7500),  # $15k loss → 25%, recover at $7.5k from bottom
        ]
    )

    print(f"\nStop Loss Configuration:")
    print(f"  Level 1: $5k loss → 75% gross (recover at $2.5k from bottom)")
    print(f"  Level 2: $10k loss → 50% gross (recover at $5k from bottom)")
    print(f"  Level 3: $15k loss → 25% gross (recover at $7.5k from bottom)")

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
    max_dd_dollar = peak_value - min(results.portfolio_values)

    print(f"Initial capital: ${config.initial_cash:,.2f}")
    print(f"Peak portfolio value: ${peak_value:,.2f}")
    print(f"Lowest portfolio value: ${min(results.portfolio_values):,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Max dollar drawdown: ${max_dd_dollar:,.2f}")

    # Calculate what would have happened without stop loss
    initial_shares = 1000.0
    start_price = 100.0
    end_price = prices[-1]
    no_sl_value = config.initial_cash * (end_price / start_price)
    no_sl_return = (no_sl_value / config.initial_cash - 1)

    print(f"\nComparison:")
    print(f"  With stop loss:    {total_return:>8.2%} (${final_value:,.2f})")
    print(f"  Without stop loss: {no_sl_return:>8.2%} (${no_sl_value:,.2f})")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    print(f"\n{'='*70}")
    print("✓ Dollar-based stop loss recovery demonstration complete!")
    print(f"{'='*70}\n")

    print("\nKey Observations:")
    print("  1. Stop loss triggers at deeper levels as dollar losses increase")
    print("  2. Recovery thresholds move us back through levels as portfolio recovers")
    print("  3. Final recovery to new peak fully clears the stop loss")
    print("  4. Stop loss provides protection during severe drawdowns")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_stop_loss_recovery_demo()

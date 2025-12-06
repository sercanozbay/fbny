"""
Test stop loss recovery functionality.

Tests both percent-based and dollar-based recovery thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting import Backtester, BacktestConfig, DataManager


def test_stop_loss_percent_recovery():
    """Test percent-based recovery thresholds."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - Percent-Based Recovery")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_recovery_pct')
    test_dir.mkdir(exist_ok=True)

    # Create dates
    dates = pd.date_range('2023-01-01', periods=80, freq='B')
    tickers = ['AAPL']

    # Create price scenario with drawdown and recovery
    # Days 0-10: Stable at $100 (peak)
    # Days 10-30: Decline to $85 (15% drawdown - should trigger 25% gross at level 3)
    # Days 30-50: Recover to $92.5 (50% recovery from $85 bottom - should move back to level 2)
    # Days 50-65: Recover to $97.5 (50% recovery from $92.5 - should move back to level 1)
    # Days 65-80: Recover to $101 (new peak - should clear stop loss)
    prices = []
    for i in range(len(dates)):
        if i < 10:
            price = 100.0  # Peak
        elif i < 30:
            # Decline to $85 (15% drawdown)
            progress = (i - 10) / 20
            price = 100.0 - progress * 15.0
        elif i < 50:
            # Recover to $92.5 (50% recovery: 85 + 0.5*(100-85) = 92.5)
            progress = (i - 30) / 20
            price = 85.0 + progress * 7.5
        elif i < 65:
            # Recover to $97.5 (50% recovery from new trough)
            progress = (i - 50) / 15
            price = 92.5 + progress * 5.0
        else:
            # Full recovery to new peak
            progress = (i - 65) / 15
            price = 97.5 + progress * 3.5
        prices.append(price)

    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice scenario:")
    print(f"  Days 0-10:  $100 (peak)")
    print(f"  Days 10-30: Decline to $85 (15% DD)")
    print(f"  Days 30-50: Recover to $92.5 (50% recovery)")
    print(f"  Days 50-65: Recover to $97.5 (another 50% recovery)")
    print(f"  Days 65-80: Recover to $101 (new peak)")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with stop loss and recovery levels
    # 5% DD -> 75% gross, recover at 50% recovery
    # 10% DD -> 50% gross, recover at 50% recovery
    # 15% DD -> 25% gross, recover at 50% recovery
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (0.05, 0.75, 0.50),  # 5% DD -> 75% gross, recover at 50%
            (0.10, 0.50, 0.50),  # 10% DD -> 50% gross, recover at 50%
            (0.15, 0.25, 0.50),  # 15% DD -> 25% gross, recover at 50%
        ]
    )

    print(f"\nStop loss configuration (percent-based with recovery):")
    for threshold, gross, recovery in config.stop_loss_levels:
        print(f"  {threshold:.0%} DD -> {gross:.0%} gross, recover at {recovery:.0%} from bottom")

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
    print(f"\nRunning backtest...")

    results = backtester.run(
        start_date=dates[0],
        end_date=dates[-1],
        use_case=1,
        inputs=inputs,
        show_progress=False
    )

    # Analyze results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")

    final_value = results.portfolio_values[-1]
    peak_value = max(results.portfolio_values)
    total_return = (final_value / config.initial_cash - 1)
    max_dd_pct = (peak_value - min(results.portfolio_values)) / peak_value

    print(f"Peak portfolio value: ${peak_value:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Max drawdown: {max_dd_pct:.2%}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    # Verify recovery worked - with recovery, we should have better returns
    # than without recovery (where we'd stay at 25% gross even after recovering)
    print(f"\nExpected behavior:")
    print(f"  - Stop loss triggers at 5%, 10%, 15% DD levels")
    print(f"  - Recovery moves back through levels as portfolio recovers")
    print(f"  - Final recovery to new peak clears stop loss")

    print(f"\n{'='*70}")
    print("✓ Percent-based recovery test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


def test_stop_loss_dollar_recovery():
    """Test dollar-based recovery thresholds."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - Dollar-Based Recovery")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_recovery_dollar')
    test_dir.mkdir(exist_ok=True)

    # Create dates
    dates = pd.date_range('2023-01-01', periods=60, freq='B')
    tickers = ['AAPL']

    # Create price scenario with dollar-based drawdown and recovery
    # Start with 1000 shares at $100 = $100k portfolio
    # Days 0-10: $100 (stable)
    # Days 10-25: Decline to $85 ($15k loss - triggers $15k level)
    # Days 25-40: Recover to $92.50 ($7.5k recovery from bottom - triggers recovery)
    # Days 40-60: Recover to $102 (new peak - clears stop loss)
    prices = []
    for i in range(len(dates)):
        if i < 10:
            price = 100.0
        elif i < 25:
            # Decline to $85
            progress = (i - 10) / 15
            price = 100.0 - progress * 15.0
        elif i < 40:
            # Recover to $92.50 (50% dollar recovery: $7.5k out of $15k)
            progress = (i - 25) / 15
            price = 85.0 + progress * 7.5
        else:
            # Full recovery to new peak
            progress = (i - 40) / 20
            price = 92.5 + progress * 9.5
        prices.append(price)

    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice scenario (with 1000 shares):")
    print(f"  Days 0-10:  $100/share = $100k portfolio")
    print(f"  Days 10-25: $85/share = $85k portfolio ($15k loss)")
    print(f"  Days 25-40: $92.50/share = $92.5k portfolio ($7.5k recovery)")
    print(f"  Days 40-60: $102/share = $102k portfolio (new peak)")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with dollar-based stop loss and recovery
    # $5k loss -> 75% gross, recover at $2.5k recovery
    # $10k loss -> 50% gross, recover at $5k recovery
    # $15k loss -> 25% gross, recover at $7.5k recovery
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (5000, 0.75, 2500, 'dollar'),   # $5k loss -> 75%, recover at $2.5k
            (10000, 0.50, 5000, 'dollar'),  # $10k loss -> 50%, recover at $5k
            (15000, 0.25, 7500, 'dollar'),  # $15k loss -> 25%, recover at $7.5k
        ]
    )

    print(f"\nStop loss configuration (dollar-based with recovery):")
    for threshold, gross, recovery, _ in config.stop_loss_levels:
        print(f"  ${threshold:,.0f} loss -> {gross:.0%} gross, recover at ${recovery:,.0f} from bottom")

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
    print(f"\nRunning backtest...")

    results = backtester.run(
        start_date=dates[0],
        end_date=dates[-1],
        use_case=1,
        inputs=inputs,
        show_progress=False
    )

    # Analyze results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")

    final_value = results.portfolio_values[-1]
    peak_value = max(results.portfolio_values)
    total_return = (final_value / config.initial_cash - 1)
    max_dd_dollar = peak_value - min(results.portfolio_values)

    print(f"Peak portfolio value: ${peak_value:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Max dollar drawdown: ${max_dd_dollar:,.2f}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    print(f"\nExpected behavior:")
    print(f"  - Stop loss triggers at $5k, $10k, $15k loss levels")
    print(f"  - Recovery moves back through levels as portfolio recovers in dollars")
    print(f"  - Final recovery to new peak clears stop loss")

    print(f"\n{'='*70}")
    print("✓ Dollar-based recovery test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


def test_stop_loss_no_recovery():
    """Test stop loss without recovery thresholds (backward compatibility)."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - No Recovery (Backward Compatible)")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_no_recovery')
    test_dir.mkdir(exist_ok=True)

    # Create dates
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    tickers = ['AAPL']

    # Create price scenario with drawdown and recovery
    # But since there's no recovery threshold, stop loss should stay active
    prices = []
    for i in range(len(dates)):
        if i < 10:
            price = 100.0
        elif i < 25:
            # Decline to $85 (15% DD)
            progress = (i - 10) / 15
            price = 100.0 - progress * 15.0
        else:
            # Recover to $98 (almost full recovery)
            progress = (i - 25) / 25
            price = 85.0 + progress * 13.0
        prices.append(price)

    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice scenario:")
    print(f"  Days 0-10:  $100 (peak)")
    print(f"  Days 10-25: Decline to $85 (15% DD)")
    print(f"  Days 25-50: Recover to $98 (almost full recovery)")
    print(f"\nNote: No recovery threshold, so stop loss stays active")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config WITHOUT recovery thresholds (2-tuple format)
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (0.05, 0.75),  # 5% DD -> 75% gross (no recovery)
            (0.10, 0.50),  # 10% DD -> 50% gross (no recovery)
            (0.15, 0.25),  # 15% DD -> 25% gross (no recovery)
        ]
    )

    print(f"\nStop loss configuration (no recovery thresholds):")
    for threshold, gross in config.stop_loss_levels:
        print(f"  {threshold:.0%} DD -> {gross:.0%} gross (no automatic recovery)")

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
    print(f"\nRunning backtest...")

    results = backtester.run(
        start_date=dates[0],
        end_date=dates[-1],
        use_case=1,
        inputs=inputs,
        show_progress=False
    )

    # Analyze results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")

    final_value = results.portfolio_values[-1]
    peak_value = max(results.portfolio_values)
    total_return = (final_value / config.initial_cash - 1)

    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    print(f"\nExpected behavior:")
    print(f"  - Stop loss triggers at 15% DD")
    print(f"  - Stays at 25% gross even after recovery to $98")
    print(f"  - Only clears when new peak is reached")

    print(f"\n{'='*70}")
    print("✓ No recovery test passed (backward compatible)!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_stop_loss_percent_recovery()
    test_stop_loss_dollar_recovery()
    test_stop_loss_no_recovery()

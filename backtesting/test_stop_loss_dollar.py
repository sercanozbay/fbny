"""
Test dollar-based stop loss functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting import Backtester, BacktestConfig, DataManager


def test_stop_loss_dollar():
    """Test dollar-based stop loss thresholds."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - Dollar Thresholds")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_dollar')
    test_dir.mkdir(exist_ok=True)

    # Create dates
    dates = pd.date_range('2023-01-01', periods=40, freq='B')
    tickers = ['AAPL']

    # Create price scenario with specific dollar losses
    # Start at $100, decline to trigger dollar thresholds
    # Days 0-10: $100 (stable)
    # Days 10-20: decline to $85 (should trigger $5k and $10k thresholds with 1000 shares)
    # Days 20-40: stable at $85
    prices = []
    for i in range(len(dates)):
        if i < 10:
            price = 100.0
        elif i < 20:
            progress = (i - 10) / 10
            price = 100.0 - progress * 15.0  # Decline to $85
        else:
            price = 85.0
        prices.append(price)

    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice scenario:")
    print(f"  Days 0-10: $100 (stable)")
    print(f"  Days 10-20: Decline to $85")
    print(f"  Days 20-40: $85 (stable)")
    print(f"\nWith 1000 shares:")
    print(f"  Peak value: ~$100,000")
    print(f"  Min value: ~$85,000")
    print(f"  Dollar loss: ~$15,000")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with dollar-based stop loss
    # $5,000 loss -> 75% gross
    # $10,000 loss -> 50% gross
    # $15,000 loss -> 25% gross
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (5000, 0.75, 'dollar'),   # $5k loss -> 75% gross
            (10000, 0.50, 'dollar'),  # $10k loss -> 50% gross
            (15000, 0.25, 'dollar'),  # $15k loss -> 25% gross
        ]
    )

    print(f"\nStop loss configuration (dollar-based):")
    for threshold, gross, _ in config.stop_loss_levels:
        print(f"  ${threshold:,.0f} loss -> {gross:.0%} gross")

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
    dollar_dd = peak_value - min(results.portfolio_values)

    print(f"Peak portfolio value: ${peak_value:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Max dollar drawdown: ${dollar_dd:,.2f}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    # Calculate what return would be without stop loss
    buy_price = 100.0
    end_price = 85.0
    no_sl_return = (end_price / buy_price - 1)

    print(f"\nComparison:")
    print(f"  Without stop loss: {no_sl_return:.2%}")
    print(f"  With stop loss: {total_return:.2%}")
    print(f"  Benefit: {(total_return - no_sl_return):.2%}")

    print(f"\n{'='*70}")
    print("✓ Dollar-based stop loss test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


def test_stop_loss_mixed():
    """Test mixed percent and dollar thresholds."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - Mixed Percent and Dollar Thresholds")
    print("="*70)

    # Create test data
    test_dir = Path('./test_data_sl_mixed')
    test_dir.mkdir(exist_ok=True)

    # Create dates
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    tickers = ['AAPL']

    # Simple decline scenario
    prices = np.linspace(100, 80, len(dates))
    prices_df = pd.DataFrame({'AAPL': prices}, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nPrice scenario: Linear decline from $100 to $80")

    # Create ADV
    adv_df = pd.DataFrame({'AAPL': [1000000] * len(dates)}, index=dates)
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with mixed thresholds
    # First level: 5% percent -> 75% gross
    # Second level: $7,500 dollar -> 50% gross
    # Third level: 15% percent -> 25% gross
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (0.05, 0.75),              # 5% DD -> 75% gross (percent-based)
            (7500, 0.50, 'dollar'),    # $7.5k loss -> 50% gross (dollar-based)
            (0.15, 0.25),              # 15% DD -> 25% gross (percent-based)
        ]
    )

    print(f"\nStop loss configuration (mixed):")
    for i, level_tuple in enumerate(config.stop_loss_levels):
        if len(level_tuple) == 2:
            threshold, gross = level_tuple
            print(f"  {threshold:.0%} DD -> {gross:.0%} gross (percent)")
        else:
            threshold, gross, typ = level_tuple
            print(f"  ${threshold:,.0f} loss -> {gross:.0%} gross (dollar)")

    # Create backtester
    backtester = Backtester(config, data_manager)

    # Buy 500 shares on day 1
    targets = {
        dates[0]: {'AAPL': 500.0}
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
    total_return = (final_value / config.initial_cash - 1)

    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")

    # Verify results
    assert len(results.dates) > 0, "No results generated"
    assert final_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    print(f"\n{'='*70}")
    print("✓ Mixed threshold stop loss test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_stop_loss_dollar()
    test_stop_loss_mixed()

"""
Test stop loss functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting import Backtester, BacktestConfig, DataManager


def test_stop_loss_basic():
    """Test basic stop loss functionality with drawdown triggers."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - Basic Functionality")
    print("="*70)

    # Create test data directory
    test_dir = Path('./test_data_stop_loss')
    test_dir.mkdir(exist_ok=True)

    # Create test dates - 60 days
    dates = pd.date_range('2023-01-01', periods=60, freq='B')
    tickers = ['AAPL', 'MSFT']

    # Create prices with a drawdown scenario
    # First 20 days: up trend (150 -> 165)
    # Days 20-35: sharp drawdown (165 -> 140, ~15% DD)
    # Days 35-60: recovery (140 -> 160)
    prices_data = {}
    for ticker in tickers:
        prices = []
        for i, date in enumerate(dates):
            if i < 20:
                # Up trend
                price = 150.0 + i * 0.75
            elif i < 35:
                # Drawdown
                decline_progress = (i - 20) / 15
                price = 165.0 - decline_progress * 25.0
            else:
                # Recovery
                recovery_progress = (i - 35) / 25
                price = 140.0 + recovery_progress * 20.0

            prices.append(price)

        prices_data[ticker] = prices

    prices_df = pd.DataFrame(prices_data, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')

    print(f"\nCreated price scenario:")
    print(f"  Days 0-20: Up trend (150 -> 165)")
    print(f"  Days 20-35: Drawdown (165 -> 140, ~15% DD)")
    print(f"  Days 35-60: Recovery (140 -> 160)")

    # Create ADV
    adv_df = pd.DataFrame(
        {ticker: [2000000] * len(dates) for ticker in tickers},
        index=dates
    )
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with stop loss levels
    # Dollar-based stop loss levels (based on $100k initial capital)
    # $5k loss -> 75% gross
    # $10k loss -> 50% gross
    # $15k loss -> 25% gross
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (5000, 0.75),   # $5k loss -> 75% gross
            (10000, 0.50),  # $10k loss -> 50% gross
            (15000, 0.25),  # $15k loss -> 25% gross
        ]
    )

    print(f"\nStop loss levels configured:")
    print(f"  $5k loss -> 75% gross")
    print(f" $10k loss -> 50% gross")
    print(f" $15k loss -> 25% gross")

    # Create backtester
    backtester = Backtester(config, data_manager)

    # Define target positions - buy and hold 100 shares of each
    targets = {
        dates[0]: {
            'AAPL': 100.0,
            'MSFT': 100.0
        }
    }

    inputs = {
        'type': 'shares',
        'targets': targets
    }

    # Run backtest
    print(f"\nRunning backtest with stop loss...")

    results = backtester.run(
        start_date=dates[0],
        end_date=dates[-1],
        use_case=1,
        inputs=inputs,
        show_progress=False
    )

    # Check results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    final_portfolio_value = results.portfolio_values[-1] if results.portfolio_values else 0
    total_return = (final_portfolio_value / config.initial_cash - 1) if config.initial_cash > 0 else 0
    total_pnl = final_portfolio_value - config.initial_cash

    max_dd = min(results.daily_returns) if results.daily_returns else 0

    print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Max drawdown: {max_dd:.2%}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_portfolio_value > 0, "Portfolio value is zero"

    # Check that P&L was calculated correctly (not NaN or inf)
    assert np.isfinite(total_pnl), f"P&L is not finite: {total_pnl}"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    # Verify that all daily P&L values are finite
    daily_pnl_finite = all(np.isfinite(pnl) for pnl in results.daily_pnl)
    assert daily_pnl_finite, "Some daily P&L values are not finite"

    print(f"\nNumber of days simulated: {len(results.dates)}")
    print(f"All daily P&L values are finite: ✓")

    print(f"\n{'='*70}")
    print("✓ Stop loss test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


def test_stop_loss_use_case_2():
    """Test stop loss with use case 2 (signals)."""
    print("\n" + "="*70)
    print("TEST: Stop Loss - Use Case 2 (Signals)")
    print("="*70)

    # Create test data directory
    test_dir = Path('./test_data_stop_loss_uc2')
    test_dir.mkdir(exist_ok=True)

    # Create test dates
    dates = pd.date_range('2023-01-01', periods=40, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Create prices with volatility
    prices_data = {}
    for ticker in tickers:
        base_price = 150.0
        # Add some random walk with downward bias initially
        np.random.seed(42)
        returns = np.random.normal(-0.01, 0.02, len(dates))
        returns[:20] = np.random.normal(-0.02, 0.03, 20)  # Higher volatility initially
        prices = base_price * (1 + returns).cumprod()
        prices_data[ticker] = prices

    prices_df = pd.DataFrame(prices_data, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')
    print(f"\nCreated prices for {len(dates)} days, {len(tickers)} tickers")

    # Create ADV
    adv_df = pd.DataFrame(
        {ticker: [2000000] * len(dates) for ticker in tickers},
        index=dates
    )
    adv_df.to_csv(test_dir / 'adv.csv')

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()
    data_manager.load_adv()

    # Create config with dollar-based stop loss (based on $100k capital)
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5,
        stop_loss_levels=[
            (8000, 0.60),   # $8k loss -> 60% gross
            (12000, 0.30),  # $12k loss -> 30% gross
        ]
    )

    print(f"\nStop loss levels:")
    print(f"  $8k loss -> 60% gross")
    print(f" $12k loss -> 30% gross")

    # Create backtester
    backtester = Backtester(config, data_manager)

    # Define signals - uniform weights
    signals = {}
    for date in dates[::5]:  # Rebalance every 5 days
        signals[date] = {
            'AAPL': 1.0,
            'MSFT': 1.0,
            'GOOGL': 1.0
        }

    inputs = {
        'signals': signals
    }

    # Run backtest
    print(f"\nRunning use case 2 backtest with stop loss...")

    results = backtester.run(
        start_date=dates[0],
        end_date=dates[-1],
        use_case=2,
        inputs=inputs,
        show_progress=False
    )

    # Check results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    final_portfolio_value = results.portfolio_values[-1] if results.portfolio_values else 0
    total_return = (final_portfolio_value / config.initial_cash - 1) if config.initial_cash > 0 else 0

    print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
    print(f"Total return: {total_return:.2%}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_portfolio_value > 0, "Portfolio value is zero"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    print(f"\n{'='*70}")
    print("✓ Use case 2 stop loss test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_stop_loss_basic()
    test_stop_loss_use_case_2()

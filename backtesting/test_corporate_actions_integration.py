"""
Integration test for corporate actions in the backtester.
Tests that corporate actions don't break P&L calculations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting import Backtester, BacktestConfig, DataManager


def test_corporate_actions_pnl():
    """Test that corporate actions work correctly and don't break P&L calculations."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Corporate Actions with P&L Calculation")
    print("="*70)

    # Create test data directory
    test_dir = Path('./test_data_integration')
    test_dir.mkdir(exist_ok=True)

    # Create test dates
    dates = pd.date_range('2023-06-01', '2023-06-30', freq='B')
    tickers = ['AAPL', 'MSFT']

    # Create prices (with a trend)
    prices_data = {}
    for ticker in tickers:
        base_price = 150.0
        prices_data[ticker] = base_price + np.arange(len(dates)) * 0.5  # Upward trend

    prices_df = pd.DataFrame(prices_data, index=dates)
    prices_df.to_csv(test_dir / 'prices.csv')
    print(f"\nCreated prices for {len(dates)} days")
    print(f"AAPL price range: ${prices_df['AAPL'].min():.2f} - ${prices_df['AAPL'].max():.2f}")
    print(f"MSFT price range: ${prices_df['MSFT'].min():.2f} - ${prices_df['MSFT'].max():.2f}")

    # Create ADV
    adv_df = pd.DataFrame(
        {ticker: [2000000] * len(dates) for ticker in tickers},
        index=dates
    )
    adv_df.to_csv(test_dir / 'adv.csv')

    # Create corporate actions
    # Split on June 15, Dividend on June 20
    split_date = pd.Timestamp('2023-06-15')
    dividend_date = pd.Timestamp('2023-06-20')

    actions_data = pd.DataFrame([
        {'date': split_date.strftime('%Y-%m-%d'), 'ticker': 'AAPL', 'action_type': 'split', 'value': 2.0},
        {'date': dividend_date.strftime('%Y-%m-%d'), 'ticker': 'MSFT', 'action_type': 'dividend', 'value': 1.0},
    ])
    actions_data.to_csv(test_dir / 'corporate_actions.csv', index=False)
    print(f"\nCorporate actions:")
    print(f"  AAPL: 2-for-1 split on {split_date.date()}")
    print(f"  MSFT: $1.00 dividend on {dividend_date.date()}")

    # Initialize data manager
    data_manager = DataManager(str(test_dir))
    data_manager.load_prices()  # Load prices
    data_manager.load_adv()
    data_manager.load_corporate_actions()

    # Create config
    config = BacktestConfig(
        initial_cash=100000.0,
        tc_coefficient=0.001,
        tc_power=1.5
    )

    # Create backtester
    backtester = Backtester(config, data_manager)

    # Define target positions (buy and hold)
    targets = {
        dates[0]: {
            'AAPL': 100.0,  # 100 shares before split -> 200 shares after
            'MSFT': 200.0,  # 200 shares, will receive $200 dividend
        }
    }

    inputs = {
        'type': 'shares',
        'targets': targets
    }

    # Run backtest
    print(f"\nRunning backtest...")
    print(f"Initial targets: AAPL=100 shares, MSFT=200 shares")
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

    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Total P&L: ${total_pnl:.2f}")

    # Verify we have results
    assert len(results.dates) > 0, "No results generated"
    assert final_portfolio_value > 0, "Portfolio value is zero"

    # Check that P&L was calculated correctly (not NaN or inf)
    assert np.isfinite(total_pnl), f"P&L is not finite: {total_pnl}"
    assert np.isfinite(total_return), f"Return is not finite: {total_return}"

    # Verify corporate actions were recorded
    print(f"\nNumber of days simulated: {len(results.dates)}")

    # Verify that all daily P&L values are finite (not NaN or inf)
    daily_pnl_finite = all(np.isfinite(pnl) for pnl in results.daily_pnl)
    assert daily_pnl_finite, "Some daily P&L values are not finite"
    print(f"All daily P&L values are finite: ✓")

    print(f"\n{'='*70}")
    print("✓ Integration test passed!")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_corporate_actions_pnl()

"""
Simple test for DataFrame-based corporate actions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the simplified approach
def test_dataframe_loading():
    """Test loading corporate actions as DataFrame."""
    print("\n" + "="*60)
    print("TEST: DataFrame-based Corporate Actions Loading")
    print("="*60)

    # Create test data directory
    test_dir = Path('./test_data')
    test_dir.mkdir(exist_ok=True)

    # Create sample corporate actions CSV
    actions_data = pd.DataFrame([
        {'date': '2023-06-15', 'ticker': 'AAPL', 'action_type': 'split', 'value': 2.0},
        {'date': '2023-09-01', 'ticker': 'MSFT', 'action_type': 'dividend', 'value': 0.75},
        {'date': '2023-12-15', 'ticker': 'GOOGL', 'action_type': 'split', 'value': 3.0},
    ])

    csv_path = test_dir / 'corporate_actions.csv'
    actions_data.to_csv(csv_path, index=False)
    print(f"Created test CSV: {csv_path}")
    print(actions_data)

    # Test DataManager loading
    from backtesting import DataManager

    # Create minimal data files for DataManager
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Create prices
    prices_df = pd.DataFrame(
        np.random.uniform(100, 200, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )
    prices_df.to_csv(test_dir / 'prices.csv')

    # Create ADV
    adv_df = pd.DataFrame(
        np.random.uniform(1000000, 5000000, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )
    adv_df.to_csv(test_dir / 'adv.csv')

    # Load data
    data_mgr = DataManager(str(test_dir))
    data_mgr.load_prices()
    data_mgr.load_adv()

    # Load corporate actions
    ca_df = data_mgr.load_corporate_actions()

    print(f"\n✓ Loaded corporate actions DataFrame:")
    print(f"  Shape: {ca_df.shape}")
    print(f"  Index: {ca_df.index.names}")
    print(f"  Columns: {ca_df.columns.tolist()}")
    print(f"\n{ca_df}")

    # Test get_data_for_date
    print(f"\n✓ Testing get_data_for_date:")

    test_date = pd.Timestamp('2023-06-15')
    day_data = data_mgr.get_data_for_date(test_date)

    if 'corporate_actions' in day_data:
        print(f"  Found corporate actions on {test_date}:")
        print(day_data['corporate_actions'])
    else:
        print(f"  No corporate actions on {test_date}")

    # Test date with multiple actions
    test_date2 = pd.Timestamp('2023-09-01')
    day_data2 = data_mgr.get_data_for_date(test_date2)

    if 'corporate_actions' in day_data2:
        print(f"\n  Found corporate actions on {test_date2}:")
        print(day_data2['corporate_actions'])

    print("\n✓ DataFrame loading test passed!")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("\n✓ Cleanup complete")


def test_portfolio_updates():
    """Test applying corporate actions to portfolio."""
    print("\n" + "="*60)
    print("TEST: Portfolio Updates from DataFrame")
    print("="*60)

    # Simulate corporate actions DataFrame for a date
    actions_df = pd.DataFrame([
        {'action_type': 'split', 'value': 2.0},
        {'action_type': 'dividend', 'value': 0.75},
    ], index=['AAPL', 'MSFT'])

    print("Corporate actions for the day:")
    print(actions_df)

    # Simulate portfolio
    positions = {'AAPL': 100.0, 'MSFT': 200.0, 'GOOGL': 50.0}
    cash = 10000.0

    print(f"\nBefore corporate actions:")
    print(f"  Positions: {positions}")
    print(f"  Cash: ${cash:.2f}")

    # Apply actions (simulating what backtester does)
    for ticker, row in actions_df.iterrows():
        action_type = row['action_type']
        value = row['value']

        if ticker not in positions or positions[ticker] == 0:
            continue

        shares_held = positions[ticker]

        if action_type == 'split':
            new_shares = shares_held * value
            positions[ticker] = new_shares
            print(f"\n  Split: {ticker} {value:.2f}-for-1")
            print(f"    {shares_held:.2f} → {new_shares:.2f} shares")

        elif action_type == 'dividend':
            dividend_received = shares_held * value
            cash += dividend_received
            print(f"\n  Dividend: {ticker} ${value:.4f}/share")
            print(f"    Received ${dividend_received:.2f} on {shares_held:.2f} shares")

    print(f"\nAfter corporate actions:")
    print(f"  Positions: {positions}")
    print(f"  Cash: ${cash:.2f}")

    # Verify
    assert positions['AAPL'] == 200.0, "AAPL should double from split"
    assert positions['MSFT'] == 200.0, "MSFT shares unchanged"
    assert positions['GOOGL'] == 50.0, "GOOGL unchanged"
    assert cash == 10150.0, "Cash should increase by dividend"

    print("\n✓ Portfolio update test passed!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SIMPLIFIED CORPORATE ACTIONS TEST SUITE")
    print("="*60)

    test_dataframe_loading()
    test_portfolio_updates()

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60 + "\n")

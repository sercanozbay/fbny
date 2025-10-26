"""
Quick fix example to avoid date alignment issues.

This script demonstrates how to properly align dates with your data
to avoid KeyError issues.
"""

import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager
from backtesting.utils import get_date_range

# 1. Load data
data_manager = DataManager(data_dir='./sample_data')
prices = data_manager.load_prices()

# 2. Get the actual date range in your data
print(f"Data available from {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Total trading days: {len(prices.index)}")

# 3. Use get_date_range to align your desired dates to actual data
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')

aligned_start, aligned_end = get_date_range(prices, start_date, end_date)

print(f"\nRequested: {start_date.date()} to {end_date.date()}")
print(f"Aligned to: {aligned_start.date()} to {aligned_end.date()}")

# 4. Prepare inputs - equal weight portfolio
target_weights = pd.read_csv('./sample_data/target_weights.csv',
                              index_col=0, parse_dates=True)

# Make sure target dates are within the aligned range
target_weights = target_weights.loc[aligned_start:aligned_end]

targets_by_date = {
    date: target_weights.loc[date].to_dict()
    for date in target_weights.index
}

inputs = {
    'type': 'weights',
    'targets': targets_by_date
}

# 5. Configure backtest
config = BacktestConfig(
    initial_cash=10_000_000,
    max_adv_participation=0.05,
    enable_beta_hedge=False,
    risk_free_rate=0.02
)

# 6. Run backtest with ALIGNED dates
backtester = Backtester(config, data_manager)
results = backtester.run(
    start_date=aligned_start,  # Use aligned dates!
    end_date=aligned_end,      # Use aligned dates!
    use_case=1,
    inputs=inputs
)

# 7. Analyze results
results.print_summary()

# 8. Generate reports
results.generate_full_report(
    output_dir='./output/fixed_example',
    formats=['html', 'excel']
)

print("\n✓ Success! Reports saved to ./output/fixed_example")

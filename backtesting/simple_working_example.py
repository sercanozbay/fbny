#!/usr/bin/env python
"""
SIMPLE WORKING EXAMPLE - No date errors!

This script uses the actual dates from your data, so it will ALWAYS work.
No date alignment needed - we just use what's actually in the data.
"""

import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager

print("=" * 70)
print("Running Simple Backtest Example")
print("=" * 70)

# Step 1: Generate sample data if needed
print("\n1. Checking for sample data...")
import os
if not os.path.exists('./sample_data/prices.csv'):
    print("   Generating sample data...")
    from generate_sample_data import generate_sample_data
    generate_sample_data(
        n_securities=100,
        n_days=252,
        n_factors=5,
        output_dir='./sample_data',
        seed=42
    )
    print("   ✓ Sample data generated")
else:
    print("   ✓ Sample data found")

# Step 2: Load data
print("\n2. Loading data...")
data_manager = DataManager(data_dir='./sample_data', use_float32=True)
prices = data_manager.load_prices()

# Step 3: Use ACTUAL dates from the data (NO ERRORS!)
print("\n3. Getting date range...")
print(f"   Data contains {len(prices)} trading days")
print(f"   From: {prices.index[0].date()}")
print(f"   To:   {prices.index[-1].date()}")

# Use the ACTUAL first and last dates from your data
start_date = prices.index[0]   # First date in data
end_date = prices.index[-1]    # Last date in data

print(f"\n   Using dates: {start_date.date()} to {end_date.date()}")

# Step 4: Load target weights
print("\n4. Loading target weights...")
target_weights = pd.read_csv('./sample_data/target_weights.csv',
                              index_col=0, parse_dates=True)

# Use only dates that are in BOTH datasets
common_dates = prices.index.intersection(target_weights.index)
start_date = common_dates[0]
end_date = common_dates[-1]

print(f"   Found {len(common_dates)} common trading days")
print(f"   Final date range: {start_date.date()} to {end_date.date()}")

# Filter target weights to common dates
target_weights = target_weights.loc[start_date:end_date]

# Convert to dictionary format
targets_by_date = {
    date: target_weights.loc[date].to_dict()
    for date in target_weights.index
}

inputs = {
    'type': 'weights',
    'targets': targets_by_date
}

# Step 5: Configure backtest
print("\n5. Configuring backtest...")
config = BacktestConfig(
    initial_cash=10_000_000,
    max_adv_participation=0.05,
    tc_power=1.5,
    tc_coefficient=0.01,
    enable_beta_hedge=False,
    enable_sector_hedge=False,
    risk_free_rate=0.02
)
print("   ✓ Configuration created")

# Step 6: Run backtest (THIS WILL WORK!)
print("\n6. Running backtest...")
print("   (This may take a minute...)")
backtester = Backtester(config, data_manager)

results = backtester.run(
    start_date=start_date,  # Using ACTUAL dates from data
    end_date=end_date,      # Using ACTUAL dates from data
    use_case=1,
    inputs=inputs,
    show_progress=True
)

# Step 7: Print results
print("\n7. Results:")
print("=" * 70)
results.print_summary()

# Step 8: Get key metrics
print("\n8. Key Metrics:")
metrics = results.calculate_metrics()
print(f"   Total Return:    {metrics['total_return']:>10.2%}")
print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
print(f"   Max Drawdown:    {metrics['max_drawdown']:>10.2%}")
print(f"   Win Rate:        {metrics['win_rate']:>10.2%}")
print(f"   Volatility:      {metrics['volatility']:>10.2%}")
print(f"   Calmar Ratio:    {metrics['calmar_ratio']:>10.2f}")

# Step 9: Generate reports
print("\n9. Generating reports...")
results.generate_full_report(
    output_dir='./output/simple_example',
    formats=['html', 'excel', 'csv']
)

print("\n" + "=" * 70)
print("✓ SUCCESS! Backtest completed without errors.")
print("=" * 70)
print("\nReports saved to: ./output/simple_example/")
print("  - backtest_report.html  (Open in browser)")
print("  - backtest_report.xlsx  (Open in Excel)")
print("  - backtest_results.csv  (Raw data)")
print("  - charts/               (All charts)")
print("\nHappy backtesting! 🚀")

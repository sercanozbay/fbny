# Quick Start Guide

Get up and running with the backtesting framework in 5 minutes!

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd backtesting

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Generate Sample Data

```bash
# Generate sample data for 100 securities over 252 days
python generate_sample_data.py --securities 100 --days 252 --output ./sample_data
```

This creates all necessary data files in the `sample_data` directory.

## Run Your First Backtest

### Option 1: Python Script

Create a file `my_first_backtest.py`:

```python
import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager
from backtesting.utils import get_date_range

# 1. Load data
data_manager = DataManager(data_dir='./sample_data')
prices = data_manager.load_prices()

# 2. Get actual date range (important to avoid date mismatch errors!)
print(f"Data available from {prices.index[0].date()} to {prices.index[-1].date()}")

# Align your desired dates to actual trading days in the data
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')
aligned_start, aligned_end = get_date_range(prices, start_date, end_date)

print(f"Using dates: {aligned_start.date()} to {aligned_end.date()}")

# 3. Configure backtest
config = BacktestConfig(
    initial_cash=10_000_000,
    max_adv_participation=0.05,
    enable_beta_hedge=False,
    risk_free_rate=0.02
)

# 4. Prepare inputs - equal weight portfolio
target_weights = pd.read_csv('./sample_data/target_weights.csv',
                              index_col=0, parse_dates=True)

# Filter to aligned date range
target_weights = target_weights.loc[aligned_start:aligned_end]

targets_by_date = {
    date: target_weights.loc[date].to_dict()
    for date in target_weights.index
}

inputs = {
    'type': 'weights',
    'targets': targets_by_date
}

# 5. Run backtest with ALIGNED dates
backtester = Backtester(config, data_manager)
results = backtester.run(
    start_date=aligned_start,  # Use aligned dates!
    end_date=aligned_end,      # Use aligned dates!
    use_case=1,  # Use case 1: target positions
    inputs=inputs
)

# 6. Analyze results
results.print_summary()

# 7. Generate full report (PDF, HTML, Excel, CSV)
results.generate_full_report(
    output_dir='./output/my_first_backtest',
    formats=['pdf', 'html', 'excel', 'csv']
)

print("\nDone! Check ./output/my_first_backtest for results.")
```

Run it:
```bash
python my_first_backtest.py
```

### Option 2: Jupyter Notebook

Start Jupyter:
```bash
jupyter notebook
```

Open and run: `notebooks/01_basic_setup_and_data_loading.ipynb`

## What You'll See

After running the backtest, you'll get:

1. **Console Output**:
   - Progress bar during simulation
   - Summary statistics (Sharpe ratio, drawdown, etc.)

2. **Output Directory** (`./output/my_first_backtest/`):
   - `backtest_report.html` - Interactive HTML report
   - `backtest_report.xlsx` - Excel workbook with multiple sheets
   - `backtest_results.csv` - Daily performance data
   - `trades.csv` - All trade records
   - `charts/` - Directory with all charts

## Understanding the Results

### Key Metrics

- **Total Return**: Overall return for the period
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
  - `< 0`: Losing strategy
  - `0-1`: Subpar
  - `1-2`: Good
  - `2-3`: Very good
  - `> 3`: Excellent
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable days

### Charts

1. **Cumulative Returns**: Portfolio value over time
2. **Drawdown**: How far portfolio is from its peak
3. **Rolling Sharpe**: Time-varying risk-adjusted performance
4. **Return Distribution**: Histogram of daily returns
5. **Exposures**: Gross and net exposure over time
6. **Transaction Costs**: Cumulative costs

## Next Steps

### Try Different Use Cases

#### Use Case 2: Signal-Based Trading

```python
# Load signals
signals_df = pd.read_csv('./sample_data/signals.csv', index_col=0, parse_dates=True)

signals_by_date = {
    date: signals_df.loc[date].to_dict()
    for date in signals_df.index
}

inputs = {'signals': signals_by_date}

# Run with beta hedging
config.enable_beta_hedge = True
config.target_beta = 0.0  # Market neutral

results = backtester.run(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31'),
    use_case=2,  # Signal-based
    inputs=inputs
)
```

#### Use Case 3: Risk-Managed Portfolio

```python
# Configure with risk constraints
config = BacktestConfig(
    initial_cash=10_000_000,
    max_portfolio_variance=0.0004,
    max_factor_exposure={
        'Factor1': 0.1,
        'Factor2': 0.15
    }
)

# External trades
external_trades = {
    pd.Timestamp('2023-01-01'): {'STOCK0001': 1000, 'STOCK0002': -500}
}

inputs = {'external_trades': external_trades}

results = backtester.run(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31'),
    use_case=3,  # Risk-managed
    inputs=inputs
)
```

### Customize Transaction Costs

```python
config = BacktestConfig(
    tc_power=2.0,          # Quadratic market impact
    tc_coefficient=0.02,   # Higher impact coefficient
    tc_fixed=0.0005       # 5 bps fixed cost
)
```

### Add Hedging

```python
config = BacktestConfig(
    enable_beta_hedge=True,
    beta_hedge_instrument='SPY',
    target_beta=0.0,           # Market neutral
    enable_sector_hedge=True   # Sector neutral
)
```

### Compare Strategies

```python
# Run multiple backtests
configs = [
    BacktestConfig(max_adv_participation=0.03),
    BacktestConfig(max_adv_participation=0.05),
    BacktestConfig(max_adv_participation=0.10)
]

results_list = []
for config in configs:
    backtester = Backtester(config, data_manager)
    results = backtester.run(...)
    results_list.append(results)

# Compare
for i, results in enumerate(results_list):
    metrics = results.calculate_metrics()
    print(f"Config {i}: Sharpe = {metrics['sharpe_ratio']:.2f}")
```

## Common Tasks

### Export Results to CSV

```python
# Export main results
results.save_to_csv('./output/results.csv')

# Export trades
trades_df = results.get_trades_dataframe()
trades_df.to_csv('./output/trades.csv', index=False)

# Export factor attribution
factor_pnl = results.get_factor_attribution()
if factor_pnl is not None:
    factor_pnl.to_csv('./output/factor_attribution.csv')
```

### Generate Charts Only

```python
results.generate_charts(output_dir='./output/charts')
```

### Get Metrics Programmatically

```python
metrics = results.calculate_metrics()

sharpe = metrics['sharpe_ratio']
max_dd = metrics['max_drawdown']
total_ret = metrics['total_return']

print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Total Return: {total_ret:.2%}")
```

### Compare to Benchmark

```python
# Load benchmark returns
benchmark_returns = pd.read_csv('benchmark.csv')['returns'].values

# Calculate comparison metrics
comparison = results.compare_to_benchmark(benchmark_returns)

print(f"Alpha: {comparison['alpha']:.2%}")
print(f"Beta: {comparison['beta']:.2f}")
print(f"Information Ratio: {comparison['information_ratio']:.2f}")
```

## Tips

1. **Start Small**: Test with 50-100 securities before scaling to 2000+
2. **Use Float32**: Set `use_float32=True` for large datasets
3. **Validate Data**: Always run `data_manager.validate_data()`
4. **Reasonable ADV**: Use 3-5% max ADV participation
5. **Transaction Costs**: Calibrate costs to your market
6. **Risk Limits**: Set realistic risk constraints
7. **Progress Bars**: Disable with `show_progress=False` for production

## Troubleshooting

### Import Error

```bash
# Make sure you're in the right directory
cd /path/to/backtesting

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/backtesting"
```

### Missing Data

```bash
# Regenerate sample data
python generate_sample_data.py
```

### Out of Memory

```python
# Use float32
data_manager = DataManager(data_dir='./sample_data', use_float32=True)

# Or reduce universe size
prices = prices.iloc[:, :500]  # First 500 securities only
```

### Slow Performance

```python
# Disable progress bar
results = backtester.run(..., show_progress=False)

# Use smaller date range for testing
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-03-31')  # Just Q1
```

## Getting Help

- **Examples**: Check the `notebooks/` directory
- **Documentation**: See `README.md` and `docs/data_schema.md`
- **Issues**: Open an issue on GitHub
- **Questions**: Check existing issues or ask a new question

## What's Next?

1. **Explore Notebooks**: Work through all example notebooks
2. **Use Your Data**: Replace sample data with your own
3. **Customize**: Modify configuration for your strategy
4. **Optimize**: Run parameter sweeps to find optimal settings
5. **Production**: Scale to full universe (2000-3000 securities)

Happy Backtesting! 🚀

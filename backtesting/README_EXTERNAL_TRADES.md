# External Trades with Tags - Complete Guide

This document provides an overview of the external trade features in the backtesting framework, including tag-based attribution and CSV loading.

## Quick Start

### Load trades from CSV and analyze by counterparty:

```python
from backtesting import DataManager, Backtester, BacktestConfig

# Initialize
data_manager = DataManager('../sample_data')

# Load external trades from CSV
trades_df = data_manager.load_external_trades('external_trades.csv')
trades_by_date = data_manager.get_external_trades_by_date()

# Run backtest
config = BacktestConfig(initial_cash=1_000_000, tc_fixed=0.001)
backtester = Backtester(config, data_manager)

results = backtester.run(
    start_date='2023-01-02',
    end_date='2023-12-31',
    use_case=3,
    inputs={'external_trades': trades_by_date}
)

# Analyze PnL by counterparty
pnl_summary = results.get_pnl_summary_by_tag()
print(pnl_summary)
```

## Features

### 1. Tag-Based Attribution

Track and analyze PnL by any dimension using tags:
- Counterparty attribution
- Strategy attribution
- Regime-based attribution
- Multi-dimensional tags (e.g., "High Vol / Goldman / Buy")

**Key Methods:**
- `get_pnl_by_tag(tag)` - Time series data
- `get_pnl_summary_by_tag()` - Summary statistics
- `get_trades_by_tag(tag)` - Trade filtering

### 2. CSV Data Loading

Load external trades from CSV files:
- Required columns: date, ticker, qty, price
- Optional column: tag (for attribution)
- Automatic validation and error handling
- Backward compatible (works without tags)

**Key Methods:**
- `load_external_trades(filename)` - Load from CSV
- `get_external_trades_by_date()` - Convert to backtest format

### 3. Dynamic Tag Generation

Generate tags programmatically based on market context:
- Assign counterparties based on trade size
- Route by market regime (volatility, momentum)
- Conditional logic for tag assignment

## CSV Format

Your CSV should look like this:

```csv
date,ticker,qty,price,tag
2023-01-02,AAPL,1000,150.25,Goldman Sachs
2023-01-02,MSFT,-500,200.50,Morgan Stanley
2023-01-03,GOOGL,750,95.75,JPMorgan
```

**Column Details:**
- `date`: Trade date (YYYY-MM-DD format)
- `ticker`: Security ticker symbol
- `qty`: Quantity (positive = buy, negative = sell)
- `price`: Execution price
- `tag`: Optional tag for attribution (e.g., counterparty name)

See [sample_data/external_trades_example.csv](sample_data/external_trades_example.csv) for a complete example.

## Documentation

### Comprehensive Guides

1. **[LOADING_EXTERNAL_TRADES_CSV.md](docs/LOADING_EXTERNAL_TRADES_CSV.md)**
   - Complete CSV loading guide
   - Format specifications
   - Usage examples
   - Error handling
   - Best practices

2. **[EXTERNAL_TRADE_TAGS.md](docs/EXTERNAL_TRADE_TAGS.md)**
   - Tag system overview
   - API reference
   - Use cases and patterns
   - Dynamic tag generation

3. **[DYNAMIC_TAGS_EXAMPLE.md](notebooks/DYNAMIC_TAGS_EXAMPLE.md)**
   - Dynamic tag generation patterns
   - Context-based routing
   - Hierarchical tags

### Changelogs

1. **[CHANGELOG_TAGS.md](CHANGELOG_TAGS.md)**
   - Tag attribution feature implementation
   - Technical details
   - API changes

2. **[CHANGELOG_CSV_LOADER.md](CHANGELOG_CSV_LOADER.md)**
   - CSV loading feature implementation
   - Format specification
   - Integration details

## Notebook Examples

### 1. Use Case 3: Risk-Managed Portfolio
**File**: [notebooks/04_use_case_3_risk_managed_portfolio.ipynb](notebooks/04_use_case_3_risk_managed_portfolio.ipynb)

Demonstrates:
- Loading trades from CSV (Option A)
- Generating trades programmatically (Option B)
- PnL attribution by counterparty
- Cumulative PnL visualization by tag
- Summary statistics by tag

### 2. Dynamic Trade Generation
**File**: [notebooks/07_dynamic_trade_generation.ipynb](notebooks/07_dynamic_trade_generation.ipynb)

Demonstrates:
- Dynamic tag generation based on context
- Market regime-based routing
- Hierarchical tags (Regime / Counterparty / Direction)
- Conditional counterparty assignment

## API Reference

### DataManager Methods

```python
# Load external trades from CSV
trades_df = data_manager.load_external_trades(filename='external_trades.csv')
# Returns: pandas DataFrame with columns [date, ticker, qty, price, tag]

# Convert to Use Case 3 format
trades_by_date = data_manager.get_external_trades_by_date()
# Returns: {date: {ticker: [{'qty': X, 'price': Y, 'tag': Z}, ...]}}
```

### BacktestResults Methods

```python
# Get PnL time series by tag
pnl_df = results.get_pnl_by_tag(tag=None)  # None = all tags
# Returns: DataFrame with columns [date, tag, pnl]

# Get summary statistics by tag
summary_df = results.get_pnl_summary_by_tag()
# Returns: DataFrame with columns [tag, total_pnl, mean_pnl, std_pnl, sharpe, num_days, win_rate]

# Get trades filtered by tag
trades_df = results.get_trades_by_tag(tag='Goldman Sachs')
# Returns: DataFrame of trades for specified tag
```

## Use Cases

### 1. Counterparty Attribution

Track PnL by prime broker or execution venue:

```python
# Tag trades by counterparty
trades['AAPL'] = [{
    'qty': 1000,
    'price': 150.25,
    'tag': 'Goldman Sachs'
}]

# Analyze results
pnl_summary = results.get_pnl_summary_by_tag()
print(pnl_summary[['tag', 'total_pnl', 'sharpe']])
```

### 2. Strategy Attribution

Track PnL by trading strategy:

```python
# Tag trades by strategy
trades['AAPL'] = [{
    'qty': 1000,
    'price': 150.25,
    'tag': 'Momentum Strategy'
}]
```

### 3. Multi-Dimensional Attribution

Use hierarchical tags for multiple dimensions:

```python
# Hierarchical tag: Regime / Counterparty / Direction
tag = f"{regime} / {counterparty} / {direction}"

trades['AAPL'] = [{
    'qty': 1000,
    'price': 150.25,
    'tag': 'High Vol / Goldman Sachs / Buy'
}]

# Parse and aggregate by dimension
pnl_summary = results.get_pnl_summary_by_tag()
pnl_summary['regime'] = pnl_summary['tag'].str.split(' / ').str[0]
by_regime = pnl_summary.groupby('regime')['total_pnl'].sum()
```

### 4. CSV-Based Backtesting

Load historical trades from file:

```python
# Load trades from your data warehouse export
trades_df = data_manager.load_external_trades('historical_trades.csv')
trades_by_date = data_manager.get_external_trades_by_date()

# Run backtest to validate performance
results = backtester.run(
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_case=3,
    inputs={'external_trades': trades_by_date}
)
```

## Common Patterns

### Pattern 1: Load from CSV and Backtest

```python
data_manager = DataManager('../sample_data')
trades_by_date = data_manager.get_external_trades_by_date()

backtester = Backtester(config, data_manager)
results = backtester.run(use_case=3, inputs={'external_trades': trades_by_date})

pnl_summary = results.get_pnl_summary_by_tag()
```

### Pattern 2: Dynamic Tag Assignment

```python
def generate_trades_with_tags(context):
    regime = 'High Vol' if context['volatility'] > 0.25 else 'Low Vol'

    trades = {}
    for ticker, qty in raw_trades.items():
        counterparty = select_counterparty(qty, context)
        tag = f"{regime} / {counterparty}"

        trades[ticker] = [{'qty': qty, 'price': price, 'tag': tag}]

    return trades
```

### Pattern 3: Filter and Analyze

```python
# Get all trades
all_trades = results.get_trades_dataframe()

# Filter by tag
gs_trades = results.get_trades_by_tag('Goldman Sachs')

# Get PnL time series
pnl_ts = results.get_pnl_by_tag('Goldman Sachs')

# Get summary stats
summary = results.get_pnl_summary_by_tag()
```

## Best Practices

### CSV Files

1. **Use ISO 8601 date format**: `YYYY-MM-DD`
2. **Keep tags consistent**: Use the same spelling/capitalization
3. **Validate before loading**: Check for missing values, outliers
4. **Version control examples**: Keep example CSVs in git
5. **Document format**: Add README in data directory

### Tags

1. **Be consistent**: Use standardized tag names
2. **Keep it simple**: Start with single dimension, add hierarchy later
3. **Document meaning**: Explain what each tag represents
4. **Avoid special characters**: Stick to alphanumeric + space/slash
5. **Use hierarchical for multi-dim**: `Dim1 / Dim2 / Dim3`

### Performance

1. **Cache loaded data**: DataManager caches automatically
2. **Filter early**: Load only needed date range from CSV
3. **Batch analysis**: Analyze multiple tags together
4. **Use appropriate dtypes**: Use float32 if possible

## Troubleshooting

### Issue: "Missing required columns"

**Solution**: Ensure CSV has columns: date, ticker, qty, price

```python
import pandas as pd
df = pd.read_csv('trades.csv')
print(df.columns.tolist())  # Check columns
```

### Issue: "No trades loaded"

**Solution**: Check file path and format

```python
from pathlib import Path
print(Path('../sample_data/trades.csv').exists())
```

### Issue: "AttributeError: 'BacktestResults' object has no attribute..."

**Solution**: Restart Jupyter kernel to reload updated modules

```python
# In Jupyter, run:
# Kernel → Restart Kernel
```

### Issue: "Tags not appearing in results"

**Solution**: Verify tags are in CSV and not null

```python
trades_df = data_manager.load_external_trades('trades.csv')
print(trades_df['tag'].unique())
print(trades_df['tag'].notna().sum())
```

## Examples

### Complete End-to-End Example

```python
import sys
sys.path.append('..')

from backtesting import DataManager, Backtester, BacktestConfig
import matplotlib.pyplot as plt

# 1. Load data
data_manager = DataManager('../sample_data')
trades_df = data_manager.load_external_trades('external_trades_example.csv')
trades_by_date = data_manager.get_external_trades_by_date()

# 2. Configure backtest
config = BacktestConfig(
    initial_cash=1_000_000,
    tc_fixed=0.001,
    max_portfolio_variance=0.01
)

# 3. Run backtest
backtester = Backtester(config, data_manager)
results = backtester.run(
    start_date='2023-01-02',
    end_date='2023-01-06',
    use_case=3,
    inputs={'external_trades': trades_by_date}
)

# 4. Analyze overall performance
metrics = results.calculate_metrics()
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# 5. Analyze by counterparty
pnl_summary = results.get_pnl_summary_by_tag()
print("\nPnL by Counterparty:")
print(pnl_summary[['tag', 'total_pnl', 'sharpe', 'win_rate']])

# 6. Visualize
pnl_by_tag = results.get_pnl_by_tag()

fig, ax = plt.subplots(figsize=(12, 6))
for tag in pnl_summary['tag']:
    tag_data = pnl_by_tag[pnl_by_tag['tag'] == tag]
    cumulative = tag_data['pnl'].cumsum()
    ax.plot(tag_data['date'], cumulative, label=tag, linewidth=2)

ax.set_title('Cumulative PnL by Counterparty')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Additional Resources

- **Sample Data**: [sample_data/external_trades_example.csv](sample_data/external_trades_example.csv)
- **API Documentation**: See individual method docstrings
- **Jupyter Notebooks**: [notebooks/](notebooks/)
- **Issue Tracker**: Report bugs and request features

## Version History

- **2025-10-27**: Added CSV loading feature
- **2025-10-27**: Added tag-based attribution feature
- **Earlier**: External trades support in Use Case 3

## Summary

The external trades feature provides a complete solution for:
1. **Loading** trades from CSV files
2. **Tracking** PnL by custom dimensions using tags
3. **Analyzing** performance with built-in methods
4. **Visualizing** results by tag groups

Whether you're attributing PnL to counterparties, strategies, or market regimes, the tag system provides the flexibility you need while maintaining simplicity and performance.

For detailed information, see the comprehensive documentation linked above.

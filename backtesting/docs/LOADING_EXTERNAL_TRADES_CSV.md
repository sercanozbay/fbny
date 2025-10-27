# Loading External Trades from CSV

This guide explains how to load external trades with tags from CSV files for use in Use Case 3 backtesting.

## Overview

The `DataManager` class provides methods to:
1. Load external trades from CSV files with optional tags
2. Convert CSV data to the format expected by Use Case 3
3. Validate data integrity and handle missing values

## CSV Format Specification

### Required Columns

Your CSV file must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | Date/String | Trade date (YYYY-MM-DD format) | `2023-01-02` |
| `ticker` | String | Security ticker symbol | `STOCK0000` |
| `qty` | Float | Trade quantity (positive=buy, negative=sell) | `1000` or `-500` |
| `price` | Float | Execution price | `150.25` |

### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `tag` | String | Tag for attribution (e.g., counterparty) | `Goldman Sachs` |

### Example CSV File

```csv
date,ticker,qty,price,tag
2023-01-02,STOCK0000,1000,150.25,Goldman Sachs
2023-01-02,STOCK0001,-500,200.50,Morgan Stanley
2023-01-02,STOCK0002,750,95.75,JPMorgan
2023-01-03,STOCK0000,500,151.00,Goldman Sachs
2023-01-03,STOCK0003,-300,180.25,Citadel
2023-01-04,STOCK0001,800,201.50,Morgan Stanley
2023-01-04,STOCK0002,-400,94.80,Goldman Sachs
2023-01-05,STOCK0000,-1500,149.50,JPMorgan
2023-01-05,STOCK0004,1200,75.25,Citadel
2023-01-06,STOCK0003,600,182.00,Morgan Stanley
```

**Notes:**
- Header row is required
- Date format should be YYYY-MM-DD (ISO 8601)
- Positive `qty` = buy, negative `qty` = sell
- `tag` column is optional (backward compatible)
- Multiple trades for same ticker/date are supported

## Basic Usage

### Step 1: Load CSV File

```python
from backtesting import DataManager

# Initialize data manager
data_manager = DataManager('../sample_data')

# Load external trades from CSV
trades_df = data_manager.load_external_trades('external_trades.csv')

print(trades_df.head())
```

**Output:**
```
Loaded external trades: 10 trades, 5 dates, 5 tickers
  Tags found: 4 unique tags

        date     ticker     qty   price             tag
0 2023-01-02  STOCK0000  1000.0  150.25  Goldman Sachs
1 2023-01-02  STOCK0001  -500.0  200.50  Morgan Stanley
2 2023-01-02  STOCK0002   750.0   95.75      JPMorgan
3 2023-01-03  STOCK0000   500.0  151.00  Goldman Sachs
4 2023-01-03  STOCK0003  -300.0  180.25         Citadel
```

### Step 2: Convert to Use Case 3 Format

```python
# Convert to nested dict format for backtesting
trades_by_date = data_manager.get_external_trades_by_date()

# Inspect structure
import pandas as pd
sample_date = list(trades_by_date.keys())[0]
print(f"\nTrades on {sample_date}:")
for ticker, trade_list in trades_by_date[sample_date].items():
    print(f"  {ticker}: {trade_list}")
```

**Output:**
```
Trades on 2023-01-02:
  STOCK0000: [{'qty': 1000.0, 'price': 150.25, 'tag': 'Goldman Sachs'}]
  STOCK0001: [{'qty': -500.0, 'price': 200.5, 'tag': 'Morgan Stanley'}]
  STOCK0002: [{'qty': 750.0, 'price': 95.75, 'tag': 'JPMorgan'}]
```

### Step 3: Run Backtest with CSV Data

```python
from backtesting import Backtester, BacktestConfig

# Configure backtest
config = BacktestConfig(
    initial_cash=1000000,
    tc_fixed=0.001,  # 10 bps transaction cost
    max_portfolio_variance=0.01
)

# Initialize backtester
backtester = Backtester(config, data_manager)

# Run with external trades from CSV
results = backtester.run(
    start_date='2023-01-02',
    end_date='2023-01-06',
    use_case=3,
    inputs={'external_trades': trades_by_date}
)

# View results
metrics = results.calculate_metrics()
print(f"\nTotal Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## Tag-Based Attribution

### Analyze PnL by Tag

```python
# Get PnL summary by tag (counterparty)
pnl_summary = results.get_pnl_summary_by_tag()

print("\n=== PnL by Counterparty ===")
print(pnl_summary[['tag', 'total_pnl', 'sharpe', 'win_rate', 'num_days']])
```

**Output:**
```
=== PnL by Counterparty ===
              tag  total_pnl  sharpe  win_rate  num_days
0  Goldman Sachs    25000.0    1.85      0.67         3
1  Morgan Stanley   18000.0    1.45      0.60         3
2        JPMorgan   12000.0    1.20      0.50         2
3         Citadel    8000.0    0.95      0.50         2
```

### Visualize by Tag

```python
import matplotlib.pyplot as plt

# Plot cumulative PnL by counterparty
pnl_by_tag = results.get_pnl_by_tag()

fig, ax = plt.subplots(figsize=(12, 6))

for tag in pnl_by_tag['tag'].unique():
    tag_data = pnl_by_tag[pnl_by_tag['tag'] == tag]
    cumulative = tag_data['pnl'].cumsum()
    ax.plot(tag_data['date'], cumulative, label=tag, linewidth=2)

ax.set_title('Cumulative PnL by Counterparty', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Filter Trades by Tag

```python
# Get all trades for specific counterparty
gs_trades = results.get_trades_by_tag('Goldman Sachs')

print("\n=== Goldman Sachs Trades ===")
print(gs_trades[['date', 'ticker', 'quantity', 'price', 'pnl']])
```

## Complete Workflow Example

Here's a complete end-to-end example:

```python
import sys
sys.path.append('..')

import pandas as pd
from backtesting import (
    DataManager, Backtester, BacktestConfig
)

# Step 1: Load data
print("Loading data...")
data_manager = DataManager('../sample_data')

# Load price data
prices = data_manager.load_prices()
print(f"Loaded prices: {prices.shape[0]} dates, {prices.shape[1]} tickers")

# Load external trades from CSV
trades_df = data_manager.load_external_trades('external_trades.csv')
print(f"Loaded trades: {len(trades_df)} trades")

# Step 2: Convert to backtest format
trades_by_date = data_manager.get_external_trades_by_date()
print(f"Trades across {len(trades_by_date)} dates")

# Step 3: Configure and run backtest
config = BacktestConfig(
    initial_cash=1000000,
    tc_fixed=0.001,
    max_portfolio_variance=0.01
)

backtester = Backtester(config, data_manager)

print("\nRunning backtest...")
results = backtester.run(
    start_date=trades_df['date'].min(),
    end_date=trades_df['date'].max(),
    use_case=3,
    inputs={'external_trades': trades_by_date}
)

# Step 4: Analyze results
print("\n=== Performance Metrics ===")
metrics = results.calculate_metrics()
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# Step 5: Attribution by counterparty
print("\n=== Attribution by Counterparty ===")
pnl_summary = results.get_pnl_summary_by_tag()
print(pnl_summary.to_string())

# Step 6: Get trade details
print("\n=== Top 5 Trades by PnL ===")
all_trades = results.get_trades_dataframe()
top_trades = all_trades.nlargest(5, 'pnl')
print(top_trades[['date', 'ticker', 'quantity', 'price', 'tag', 'pnl']])
```

## Error Handling

### Missing Required Columns

If CSV is missing required columns, you'll get a helpful error:

```python
try:
    trades_df = data_manager.load_external_trades('bad_file.csv')
except ValueError as e:
    print(f"Error: {e}")
```

**Output:**
```
Error: External trades CSV missing required columns: ['price'].
Required: ['date', 'ticker', 'qty', 'price']. Optional: ['tag']
```

### Missing File

If the file doesn't exist, you'll get a warning and empty DataFrame:

```python
trades_df = data_manager.load_external_trades('nonexistent.csv')
# Warning: External trades file not found: ../sample_data/nonexistent.csv
print(trades_df.empty)  # True
```

### Handle Empty Data

Always check if data was loaded successfully:

```python
trades_df = data_manager.load_external_trades('external_trades.csv')

if trades_df.empty:
    print("No trades loaded - check file path and format")
else:
    trades_by_date = data_manager.get_external_trades_by_date()
    print(f"Loaded {len(trades_by_date)} dates of trades")
```

## Backward Compatibility

### CSV Without Tags

If your CSV doesn't have a `tag` column, trades will work without attribution:

**CSV without tags:**
```csv
date,ticker,qty,price
2023-01-02,STOCK0000,1000,150.25
2023-01-02,STOCK0001,-500,200.50
```

**Code:**
```python
# Still works - tags will be None
trades_df = data_manager.load_external_trades('trades_no_tags.csv')

# Convert to backtest format (no tags)
trades_by_date = data_manager.get_external_trades_by_date()

# Run backtest normally
results = backtester.run(
    start_date='2023-01-02',
    end_date='2023-01-06',
    use_case=3,
    inputs={'external_trades': trades_by_date}
)

# Tag methods will return empty or show 'untagged'
pnl_summary = results.get_pnl_summary_by_tag()
print(pnl_summary)  # Empty or single 'untagged' entry
```

## Advanced Usage

### Multiple Trades per Ticker

CSV supports multiple trades for the same ticker on the same date:

```csv
date,ticker,qty,price,tag
2023-01-02,STOCK0000,500,150.00,Goldman Sachs
2023-01-02,STOCK0000,500,150.50,Goldman Sachs
2023-01-02,STOCK0000,300,151.00,Morgan Stanley
```

These will be correctly grouped:

```python
trades_by_date = data_manager.get_external_trades_by_date()
date = pd.Timestamp('2023-01-02')

print(trades_by_date[date]['STOCK0000'])
# Output:
# [
#   {'qty': 500.0, 'price': 150.0, 'tag': 'Goldman Sachs'},
#   {'qty': 500.0, 'price': 150.5, 'tag': 'Goldman Sachs'},
#   {'qty': 300.0, 'price': 151.0, 'tag': 'Morgan Stanley'}
# ]
```

### Hierarchical Tags

Use hierarchical tags for multi-dimensional attribution:

```csv
date,ticker,qty,price,tag
2023-01-02,STOCK0000,1000,150.25,High Vol / Goldman Sachs / Buy
2023-01-02,STOCK0001,-500,200.50,Low Vol / Morgan Stanley / Sell
2023-01-03,STOCK0002,750,95.75,High Vol / JPMorgan / Buy
```

Then parse tags in analysis:

```python
pnl_summary = results.get_pnl_summary_by_tag()

# Parse hierarchical tags
pnl_summary['regime'] = pnl_summary['tag'].str.split(' / ').str[0]
pnl_summary['counterparty'] = pnl_summary['tag'].str.split(' / ').str[1]
pnl_summary['direction'] = pnl_summary['tag'].str.split(' / ').str[2]

# Aggregate by dimension
by_regime = pnl_summary.groupby('regime')['total_pnl'].sum()
by_counterparty = pnl_summary.groupby('counterparty')['total_pnl'].sum()

print("PnL by Regime:")
print(by_regime)
print("\nPnL by Counterparty:")
print(by_counterparty)
```

### Filtering Data Before Loading

Filter CSV data before backtesting:

```python
# Load raw DataFrame
trades_df = data_manager.load_external_trades('external_trades.csv')

# Filter by date range
filtered = trades_df[
    (trades_df['date'] >= '2023-01-03') &
    (trades_df['date'] <= '2023-01-05')
]

# Filter by tag
gs_only = trades_df[trades_df['tag'] == 'Goldman Sachs']

# Filter by trade size
large_trades = trades_df[trades_df['qty'].abs() > 500]

# Convert filtered data
from backtesting.data_loader import DataManager

# Manually create trades_by_date from filtered DataFrame
trades_by_date = {}
for date in filtered['date'].unique():
    date_trades = filtered[filtered['date'] == date]
    ticker_trades = {}

    for ticker in date_trades['ticker'].unique():
        ticker_rows = date_trades[date_trades['ticker'] == ticker]
        trade_list = []

        for _, row in ticker_rows.iterrows():
            trade_dict = {'qty': float(row['qty']), 'price': float(row['price'])}
            if pd.notna(row['tag']):
                trade_dict['tag'] = str(row['tag'])
            trade_list.append(trade_dict)

        ticker_trades[ticker] = trade_list

    trades_by_date[pd.Timestamp(date)] = ticker_trades

# Run backtest with filtered data
results = backtester.run(
    start_date=filtered['date'].min(),
    end_date=filtered['date'].max(),
    use_case=3,
    inputs={'external_trades': trades_by_date}
)
```

## Best Practices

### 1. Use Consistent Date Format

Always use ISO 8601 date format (YYYY-MM-DD):

```csv
✓ Good: 2023-01-02
✗ Bad:  01/02/2023
✗ Bad:  2023-1-2
✗ Bad:  Jan 2, 2023
```

### 2. Validate Data Before Backtesting

Check your CSV data before running backtest:

```python
trades_df = data_manager.load_external_trades('external_trades.csv')

# Check for missing values
print("\nMissing values:")
print(trades_df.isnull().sum())

# Check date range
print(f"\nDate range: {trades_df['date'].min()} to {trades_df['date'].max()}")

# Check tickers
print(f"\nUnique tickers: {trades_df['ticker'].nunique()}")
print(trades_df['ticker'].unique())

# Check tags
if 'tag' in trades_df.columns:
    print(f"\nUnique tags: {trades_df['tag'].nunique()}")
    print(trades_df['tag'].unique())

# Check for outliers
print("\nQuantity statistics:")
print(trades_df['qty'].describe())

print("\nPrice statistics:")
print(trades_df['price'].describe())
```

### 3. Keep Tags Consistent

Use consistent tag naming:

```csv
✓ Good: Goldman Sachs, Morgan Stanley, JPMorgan
✗ Bad:  Goldman Sachs, goldman sachs, GS, Goldman
```

### 4. Store CSV in Version Control

Keep your trade data versioned:

```bash
# In .gitignore, exclude large CSV files but keep examples
sample_data/external_trades_example.csv
!sample_data/external_trades_example.csv
```

### 5. Document Your CSV Format

Add a README in your data directory:

```markdown
# External Trades Data

## Format
- `date`: Trade date (YYYY-MM-DD)
- `ticker`: Security ticker
- `qty`: Quantity (positive=buy, negative=sell)
- `price`: Execution price
- `tag`: Counterparty name

## Tags
- Goldman Sachs: Prime broker A
- Morgan Stanley: Prime broker B
- JPMorgan: Prime broker C
- Citadel: Liquidity provider
```

## Troubleshooting

### Issue: Dates Not Parsing

**Symptom:** Dates appear as strings instead of datetime

**Solution:** Ensure date format is YYYY-MM-DD or explicitly parse:

```python
import pandas as pd

# If auto-parse fails, manually convert
trades_df = pd.read_csv('external_trades.csv')
trades_df['date'] = pd.to_datetime(trades_df['date'], format='%Y-%m-%d')
```

### Issue: No Trades Loaded

**Symptom:** `get_external_trades_by_date()` returns empty dict

**Solution:** Check file path and format:

```python
from pathlib import Path

# Verify file exists
filepath = Path('../sample_data/external_trades.csv')
print(f"File exists: {filepath.exists()}")

# Verify CSV structure
import pandas as pd
df = pd.read_csv(filepath)
print(df.head())
print(df.columns.tolist())
```

### Issue: Tags Not Appearing

**Symptom:** `get_pnl_summary_by_tag()` returns empty or only 'untagged'

**Solution:** Verify tags are present and not null:

```python
trades_df = data_manager.load_external_trades('external_trades.csv')

# Check if tag column exists
print(f"Tag column exists: {'tag' in trades_df.columns}")

# Check for non-null tags
if 'tag' in trades_df.columns:
    print(f"Non-null tags: {trades_df['tag'].notna().sum()}")
    print(f"Unique tags: {trades_df['tag'].unique()}")
```

## See Also

- [External Trade Tags Documentation](EXTERNAL_TRADE_TAGS.md)
- [Use Case 3 Documentation](USE_CASE_3_EXTERNAL_TRADES.md)
- [Dynamic Tag Generation Guide](../notebooks/DYNAMIC_TAGS_EXAMPLE.md)
- [Notebook Examples](../notebooks/)
  - [04_use_case_3_risk_managed_portfolio.ipynb](../notebooks/04_use_case_3_risk_managed_portfolio.ipynb)
  - [06_external_trade_generation.ipynb](../notebooks/06_external_trade_generation.ipynb)
  - [07_dynamic_trade_generation.ipynb](../notebooks/07_dynamic_trade_generation.ipynb)

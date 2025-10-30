# Large Data Loading Guide

## Overview

The `LargeDataLoader` class (available in `backtesting.data_loader`) provides efficient loading and processing of institutional-grade datasets with 5000+ securities. It handles:

- **Ticker + Date indexed files** (Parquet/CSV)
- **Price adjustments** for corporate actions
- **Universe and date range filtering**
- **Date-based sector mappings** (sectors change over time)
- **Date-based factor exposures and covariances**
- **Conversion to backtester format**

## Why Use LargeDataLoader?

### Standard DataManager vs LargeDataLoader

| Feature | DataManager | LargeDataLoader |
|---------|-------------|-----------------|
| Dataset Size | <1000 securities | 5000+ securities |
| Format | Date × Ticker CSVs | Ticker+Date indexed Parquet |
| Memory Usage | Loads all data | Loads subsets only |
| Corporate Actions | Manual | Automatic application |
| Time-Varying Sectors | No | Yes |
| Use Case | Small backtests | Production/Research |

### Benefits

1. **Memory Efficient**: Load only needed securities and dates
2. **Fast**: Parquet format ~10x faster than CSV
3. **Flexible**: Handle time-varying classifications and covariances
4. **Production-Ready**: Handles corporate actions automatically
5. **Scalable**: Works with datasets of any size

## Quick Start

### Basic Workflow

```python
from backtesting.large_data_loader import LargeDataLoader, convert_to_backtester_format

# 1. Initialize loader
loader = LargeDataLoader(data_dir='/path/to/large/data')

# 2. Define universe and date range
universe = ['AAPL', 'MSFT', 'GOOGL', ...]  # Your 5000 tickers
start_date = '2023-01-01'
end_date = '2023-12-31'

# 3. Load data
prices = loader.load_prices_with_adjustments(universe, start_date, end_date)
adv = loader.load_adv(universe, start_date, end_date)
betas = loader.load_betas(universe, start_date, end_date)

# 4. Load factor data
factor_exposures = loader.load_factor_exposures_with_dates(
    universe, start_date, end_date
)

# 5. Load sector mapping (as of start date)
sector_mapping = loader.load_sector_mapping_with_dates(
    universe, date=start_date
)

# 6. Convert to backtester format
saved_files = convert_to_backtester_format(
    prices=prices,
    adv=adv,
    betas=betas,
    sector_mapping=sector_mapping,
    factor_exposures=factor_exposures,
    output_dir='./my_backtest_data'
)

# 7. Run backtest
from backtesting import DataManager, Backtester, BacktestConfig

data_manager = DataManager('./my_backtest_data')
config = BacktestConfig(initial_cash=10_000_000)
backtester = Backtester(config, data_manager)

results = backtester.run(...)
```

## Data File Formats

### Expected File Structures

#### 1. Prices File (`prices_large.parquet`)

**Format A: MultiIndex (Recommended)**
```
Index: MultiIndex(date, ticker)
Columns: [price] or [close]

Example:
                         price
date       ticker
2023-01-01 AAPL       150.25
           MSFT       300.50
           GOOGL      95.75
2023-01-02 AAPL       151.00
           MSFT       301.25
```

**Format B: Long Format**
```
Columns: [date, ticker, price]

Example:
date       ticker  price
2023-01-01 AAPL    150.25
2023-01-01 MSFT    300.50
2023-01-02 AAPL    151.00
```

#### 2. Price Adjustments File (`price_adjustments.parquet`)

```
Columns: [date, ticker, adjustment_factor]

Example:
date       ticker  adjustment_factor  # Reason
2023-06-15 AAPL    0.5                # 2-for-1 split
2023-09-01 MSFT    1.02               # 2% dividend
2023-11-20 GOOGL   0.05               # 20-for-1 split

Notes:
- date: Effective date of adjustment
- adjustment_factor: Multiplier for prices BEFORE this date
- For splits: factor = 1/split_ratio (e.g., 0.5 for 2-for-1)
- For dividends: factor = 1 + (dividend / price)
```

#### 3. ADV File (`adv_large.parquet`)

```
Format: Same as prices (MultiIndex or long format)
Column: [adv] or [volume]

Example:
date       ticker  adv
2023-01-01 AAPL    50000000
2023-01-01 MSFT    25000000
```

#### 4. Betas File (`betas_large.parquet`)

```
Format: Same as prices
Column: [beta]

Example:
date       ticker  beta
2023-01-01 AAPL    1.2
2023-01-01 MSFT    0.9
```

#### 5. Sector Mapping with Dates (`sector_mapping_dated.parquet`)

```
Columns: [date, ticker, sector] or [effective_date, ticker, sector]

Example:
date       ticker  sector
2023-01-01 AAPL    Technology
2023-01-01 MSFT    Technology
2023-06-01 AAPL    Communication Services  # Sector change
2023-01-01 GOOGL   Communication Services

Notes:
- Multiple rows per ticker for sector changes
- Use most recent effective_date <= query_date
```

#### 6. Factor Exposures (`factor_exposures_large.parquet`)

```
Index: MultiIndex(date, ticker)
Columns: [Factor1, Factor2, Factor3, ...]

Example:
                         Factor1  Factor2  Factor3
date       ticker
2023-01-01 AAPL         1.5      -0.3      0.8
           MSFT         1.2       0.1     -0.5
2023-01-02 AAPL         1.6      -0.2      0.7
```

#### 7. Factor Covariance (`factor_covariance_dated.parquet`)

**Format A: Time-Varying**
```
Index: MultiIndex(date, factor)
Columns: [Factor1, Factor2, Factor3, ...]

Example:
                      Factor1  Factor2  Factor3
date       factor
2023-01-01 Factor1    0.0025   0.0010   0.0005
           Factor2    0.0010   0.0030   0.0008
           Factor3    0.0005   0.0008   0.0020
2023-01-02 Factor1    0.0026   0.0011   0.0006
```

**Format B: Static**
```
Index: [Factor1, Factor2, Factor3, ...]
Columns: [Factor1, Factor2, Factor3, ...]

Example:
         Factor1  Factor2  Factor3
Factor1  0.0025   0.0010   0.0005
Factor2  0.0010   0.0030   0.0008
Factor3  0.0005   0.0008   0.0020
```

#### 8. Specific Variance (`specific_variance_large.parquet`)

```
Format: Same as prices
Column: [variance] or [specific_variance]

Example:
date       ticker  variance
2023-01-01 AAPL    0.0015
2023-01-01 MSFT    0.0020
```

## Detailed Usage

### 1. Loading Prices with Adjustments

```python
# Load raw prices and apply corporate actions
prices = loader.load_prices_with_adjustments(
    universe=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    prices_file='prices_large.parquet',
    adjustments_file='price_adjustments.parquet',
    apply_adjustments=True  # Apply splits/dividends
)

# Result: DataFrame with adjusted prices
print(prices.head())
#             AAPL    MSFT    GOOGL
# 2023-01-01  75.125  300.50  4.7875  # Adjusted for future splits
# 2023-01-02  75.500  301.25  4.8000
```

### How Price Adjustments Work

**Example: 2-for-1 Split on 2023-06-15**

```python
# Raw prices:
# 2023-06-14: $150
# 2023-06-15: $75  (after split)
# 2023-06-16: $76

# With adjustment_factor = 0.5:
# 2023-06-14: $150 * 0.5 = $75  (adjusted)
# 2023-06-15: $75             (no adjustment)
# 2023-06-16: $76             (no adjustment)

# Result: Continuous price series
```

### 2. Loading ADV and Betas

```python
# Load ADV
adv = loader.load_adv(
    universe=['AAPL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    adv_file='adv_large.parquet'
)

# Load Betas
betas = loader.load_betas(
    universe=['AAPL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    beta_file='betas_large.parquet'
)
```

### 3. Time-Varying Sector Mapping

```python
# Get sector classification as of specific date
sector_mapping = loader.load_sector_mapping_with_dates(
    universe=['AAPL', 'MSFT', 'GOOGL'],
    date='2023-06-01',  # As of this date
    sector_file='sector_mapping_dated.parquet'
)

# Result:
#   ticker  sector
# 0 AAPL    Communication Services  # Changed from Technology
# 1 MSFT    Technology
# 2 GOOGL   Communication Services
```

### 4. Factor Exposures

```python
# Load factor exposures
exposures = loader.load_factor_exposures_with_dates(
    universe=['AAPL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    exposures_file='factor_exposures_large.parquet'
)

# Result: MultiIndex DataFrame
#                         Factor1  Factor2  Factor3
# date       ticker
# 2023-01-01 AAPL         1.5      -0.3     0.8
#            MSFT         1.2       0.1    -0.5
```

### 5. Time-Varying Factor Covariance

```python
# Load factor covariance
cov = loader.load_factor_covariance_with_dates(
    start_date='2023-01-01',
    end_date='2023-12-31',
    covariance_file='factor_covariance_dated.parquet'
)

# Result: Dict[date -> DataFrame] or single DataFrame
if isinstance(cov, dict):
    # Time-varying
    print(f"Loaded {len(cov)} covariance matrices")
    print(cov[pd.Timestamp('2023-01-01')])
else:
    # Static
    print("Static covariance matrix")
    print(cov)
```

### 6. Converting to Backtester Format

```python
# Convert all data to backtester-compatible CSVs
saved_files = convert_to_backtester_format(
    prices=prices,
    adv=adv,
    betas=betas,
    sector_mapping=sector_mapping,
    factor_exposures=exposures,
    factor_returns=factor_returns,      # Optional
    factor_covariance=cov,               # Optional
    specific_variance=specific_var,      # Optional
    output_dir='./my_backtest_data'
)

# Prints:
# ✓ Saved prices to ./my_backtest_data/prices.csv
# ✓ Saved ADV to ./my_backtest_data/adv.csv
# ✓ Saved betas to ./my_backtest_data/betas.csv
# ...
```

## Complete Example

### Step-by-Step Walkthrough

```python
from backtesting.large_data_loader import LargeDataLoader, convert_to_backtester_format
from backtesting import DataManager, Backtester, BacktestConfig
import pandas as pd

# Step 1: Define universe (your 5000 tickers)
# Load from file or define manually
universe_df = pd.read_csv('my_universe.csv')
universe = universe_df['ticker'].tolist()

print(f"Universe size: {len(universe)} tickers")

# Step 2: Define backtest period
start_date = '2023-01-01'
end_date = '2023-12-31'

# Step 3: Initialize loader
loader = LargeDataLoader(
    data_dir='/institutional/data/path',
    use_float32=True  # Recommended for large datasets
)

# Step 4: Load all required data
print("Loading prices...")
prices = loader.load_prices_with_adjustments(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    apply_adjustments=True
)

print("Loading ADV...")
adv = loader.load_adv(
    universe=universe,
    start_date=start_date,
    end_date=end_date
)

print("Loading betas...")
betas = loader.load_betas(
    universe=universe,
    start_date=start_date,
    end_date=end_date
)

print("Loading sector mapping...")
sector_mapping = loader.load_sector_mapping_with_dates(
    universe=universe,
    date=start_date
)

print("Loading factor exposures...")
factor_exposures = loader.load_factor_exposures_with_dates(
    universe=universe,
    start_date=start_date,
    end_date=end_date
)

print("Loading factor covariance...")
factor_cov = loader.load_factor_covariance_with_dates(
    start_date=start_date,
    end_date=end_date
)

print("Loading specific variance...")
specific_var = loader.load_specific_variance(
    universe=universe,
    start_date=start_date,
    end_date=end_date
)

# Step 5: Convert to backtester format
print("\nConverting to backtester format...")
saved_files = convert_to_backtester_format(
    prices=prices,
    adv=adv,
    betas=betas,
    sector_mapping=sector_mapping,
    factor_exposures=factor_exposures,
    factor_covariance=factor_cov,
    specific_variance=specific_var,
    output_dir='./large_backtest_data'
)

# Step 6: Run backtest
print("\nRunning backtest...")
data_manager = DataManager('./large_backtest_data')

config = BacktestConfig(
    initial_cash=100_000_000,  # $100M for large universe
    max_adv_participation=0.05,
    enable_beta_hedge=True,
    enable_sector_hedge=True
)

backtester = Backtester(config, data_manager)

# Load signals or target positions
signals = pd.read_csv('my_signals.csv', index_col=0, parse_dates=True)
signals_by_date = {date: signals.loc[date].to_dict() for date in signals.index}

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=2,
    inputs={'signals': signals_by_date},
    show_progress=True
)

# Step 7: Analyze results
results.print_summary()
metrics = results.calculate_metrics()
print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Total Return: {metrics['total_return']:.2%}")
```

## Performance Considerations

### Memory Usage

For 5000 securities × 252 days:
- **CSV**: ~500 MB per dataset
- **Parquet**: ~50 MB per dataset (10x compression)
- **Memory**: ~200 MB loaded (float32)

### Loading Speed

| Format | Size | Load Time |
|--------|------|-----------|
| CSV | 500 MB | ~30 seconds |
| Parquet | 50 MB | ~3 seconds |

**Recommendation**: Always use Parquet for large datasets.

### Optimization Tips

1. **Use float32**: Reduces memory by 50%
2. **Filter early**: Load only needed universe/dates
3. **Parquet compression**: Use snappy or gzip
4. **Chunk processing**: For >10K securities, process in batches

## File Format Conversion

### Converting CSV to Parquet

```python
import pandas as pd

# Convert prices CSV to Parquet
df = pd.read_csv('prices_large.csv', parse_dates=['date'])
df.to_parquet('prices_large.parquet', compression='snappy')

# Verify
df_loaded = pd.read_parquet('prices_large.parquet')
print(f"Rows: {len(df_loaded)}, Size: {df_loaded.memory_usage().sum() / 1e6:.1f} MB")
```

### Creating MultiIndex Format

```python
# Convert long format to MultiIndex
df = pd.read_csv('prices_long.csv', parse_dates=['date'])
df_multi = df.set_index(['date', 'ticker'])
df_multi.to_parquet('prices_large.parquet')
```

## Troubleshooting

### Issue: Out of Memory

**Solution:** Load in smaller batches or use smaller universe

```python
# Split universe into chunks
chunk_size = 1000
for i in range(0, len(universe), chunk_size):
    chunk = universe[i:i+chunk_size]
    prices_chunk = loader.load_prices_with_adjustments(chunk, ...)
    # Process chunk
```

### Issue: Missing Data for Some Tickers

**Solution:** Filter universe to available tickers

```python
# Check which tickers have data
available_tickers = prices.columns.tolist()
missing = set(universe) - set(available_tickers)
print(f"Missing data for {len(missing)} tickers: {list(missing)[:10]}")

# Use only available tickers
universe_filtered = [t for t in universe if t in available_tickers]
```

### Issue: Corporate Actions Not Applied Correctly

**Solution:** Verify adjustment factors

```python
# Check adjustments for a ticker
adjs = pd.read_parquet('price_adjustments.parquet')
aapl_adjs = adjs[adjs['ticker'] == 'AAPL']
print(aapl_adjs)

# Verify prices before/after adjustment date
prices = loader.load_prices_with_adjustments(
    ['AAPL'],
    '2023-01-01',
    '2023-12-31',
    apply_adjustments=False  # Load raw first
)
print(prices.head(20))  # Check around adjustment date
```

## Best Practices

1. **Always use Parquet** for large files (10x faster)
2. **Filter aggressively** - only load what you need
3. **Verify adjustments** - check split-adjusted prices make sense
4. **Use float32** - saves 50% memory with negligible precision loss
5. **Save intermediate results** - don't reload large data repeatedly
6. **Document your data** - keep README with file formats and update dates

## API Reference

### LargeDataLoader

```python
loader = LargeDataLoader(data_dir, use_float32=True)

# Load methods
prices = loader.load_prices_with_adjustments(universe, start, end, ...)
adv = loader.load_adv(universe, start, end, ...)
betas = loader.load_betas(universe, start, end, ...)
sector = loader.load_sector_mapping_with_dates(universe, date, ...)
exposures = loader.load_factor_exposures_with_dates(universe, start, end, ...)
cov = loader.load_factor_covariance_with_dates(start, end, ...)
var = loader.load_specific_variance(universe, start, end, ...)

# Save subset
loader.save_subset(data, output_file, format='parquet')
```

### Conversion Function

```python
from backtesting.large_data_loader import convert_to_backtester_format

saved_files = convert_to_backtester_format(
    prices=prices,
    adv=adv,
    betas=betas,
    sector_mapping=sector_mapping,
    factor_exposures=exposures,
    factor_returns=None,       # Optional
    factor_covariance=None,    # Optional
    specific_variance=None,    # Optional
    output_dir='./output'
)
```

## Summary

The `LargeDataLoader` enables efficient backtesting with institutional-scale datasets:

✅ Handles 5000+ securities efficiently
✅ Automatic corporate action adjustments
✅ Time-varying sector classifications
✅ Parquet support (10x faster than CSV)
✅ Memory-efficient (float32, subset loading)
✅ Converts to backtester format automatically

Perfect for production strategies and large-scale research!

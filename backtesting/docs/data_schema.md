# Data Schema Documentation

This document provides detailed specifications for all data files required by the backtesting framework.

## Overview

All data files should be in CSV format with specific column and index structures. The framework uses pandas to read these files, so standard pandas-compatible CSV formats are expected.

## Required Files

### 1. prices.csv

**Description**: Daily close prices for all securities.

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Prices (float, > 0)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,150.25,250.50,100.75
2023-01-02,151.00,251.25,101.00
2023-01-03,150.75,250.00,100.50
```

**Requirements**:
- No missing values
- All prices must be positive
- Date index must be sorted
- All tickers must be unique
- All trading days must have the same set of tickers

**Loading**:
```python
prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
```

---

### 2. adv.csv

**Description**: Average daily volume for all securities.

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Share volume (float, > 0)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,50000000,30000000,20000000
2023-01-02,51000000,29000000,21000000
2023-01-03,49000000,31000000,19000000
```

**Requirements**:
- Must match the dates and tickers in prices.csv
- All volumes must be positive
- No missing values

**Notes**:
- ADV is used for transaction cost calculations
- ADV is used for participation rate constraints
- Typical calculation: 20-day rolling average of daily volume

**Loading**:
```python
adv = pd.read_csv('adv.csv', index_col=0, parse_dates=True)
```

---

## Optional Files (Feature-Dependent)

### 3. betas.csv

**Description**: Market beta for each security on each date.

**Required For**: Beta hedging (`enable_beta_hedge=True`)

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Beta (float, typically 0.5 to 2.0)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,1.2,1.1,1.3
2023-01-02,1.2,1.0,1.3
2023-01-03,1.1,1.1,1.4
```

**Requirements**:
- Must match dates in prices.csv
- Beta = 1.0 means moves with market
- Beta = 0.0 means uncorrelated with market

**Notes**:
- Typically calculated using 252-day rolling regression
- Can use different lookback windows for different securities

**Loading**:
```python
betas = pd.read_csv('betas.csv', index_col=0, parse_dates=True)
```

---

### 4. factor_exposures.csv

**Description**: Factor loadings for each security on each date.

**Required For**: Factor risk model, risk attribution

**Format**:
- Columns: date, ticker, Factor1, Factor2, ..., FactorN
- Values: Factor exposures (float, typically z-scored)

**Example**:
```csv
date,ticker,Momentum,Value,Quality,Size,Volatility
2023-01-01,AAPL,0.5,-0.3,1.2,0.8,-0.5
2023-01-01,MSFT,-0.2,0.8,0.9,0.5,-0.3
2023-01-01,GOOGL,1.1,-0.5,0.7,0.6,-0.8
2023-01-02,AAPL,0.6,-0.2,1.1,0.8,-0.6
```

**Requirements**:
- Multi-indexed by (date, ticker)
- Factor names must be consistent
- Exposures are typically standardized (mean=0, std=1)

**Notes**:
- Common factors: Market, Size, Value, Momentum, Quality, Volatility
- Exposures represent sensitivity to factor returns
- Can be generated using factor models like Barra, Axioma, or custom

**Loading**:
```python
factor_exposures = pd.read_csv(
    'factor_exposures.csv',
    index_col=[0, 1],
    parse_dates=[0]
)
```

---

### 5. factor_returns.csv

**Description**: Daily returns for each factor.

**Required For**: Factor attribution

**Format**:
- Index: Date (datetime)
- Columns: Factor names (string)
- Values: Returns (float, typically -0.05 to 0.05 for daily)

**Example**:
```csv
date,Momentum,Value,Quality,Size,Volatility
2023-01-01,0.0012,-0.0005,0.0008,0.0003,-0.0010
2023-01-02,-0.0008,0.0015,0.0003,0.0005,0.0012
2023-01-03,0.0020,-0.0010,0.0012,-0.0002,-0.0008
```

**Requirements**:
- Factor names must match factor_exposures.csv
- Returns are in decimal form (0.01 = 1%)
- Dates should match prices.csv

**Notes**:
- Factor returns represent the return of a portfolio with unit exposure to the factor
- Used for attribution: PnL = sum(exposure_i * factor_return_i)

**Loading**:
```python
factor_returns = pd.read_csv('factor_returns.csv', index_col=0, parse_dates=True)
```

---

### 6. factor_covariance.csv

**Description**: Covariance matrix of factor returns.

**Required For**: Portfolio variance calculation

**Format**:
- Index: Factor names (string)
- Columns: Factor names (string)
- Values: Covariance (float)

**Example**:
```csv
,Momentum,Value,Quality,Size,Volatility
Momentum,0.0004,0.0001,0.0002,0.0000,-0.0001
Value,0.0001,0.0003,0.0001,0.0001,0.0000
Quality,0.0002,0.0001,0.0005,0.0001,-0.0001
Size,0.0000,0.0001,0.0001,0.0002,0.0000
Volatility,-0.0001,0.0000,-0.0001,0.0000,0.0006
```

**Requirements**:
- Must be symmetric positive semi-definite
- Diagonal elements (variances) must be positive
- Factor names must match factor_exposures.csv

**Notes**:
- Can be time-varying (one matrix per date) or constant
- Current implementation uses constant covariance
- Typically estimated using exponentially weighted moving average

**Loading**:
```python
factor_cov = pd.read_csv('factor_covariance.csv', index_col=0)
```

---

### 7. specific_variance.csv

**Description**: Idiosyncratic (residual) variance for each security.

**Required For**: Portfolio variance calculation

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Variance (float, > 0)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,0.00040,0.00035,0.00050
2023-01-02,0.00041,0.00034,0.00051
2023-01-03,0.00039,0.00036,0.00049
```

**Requirements**:
- All variances must be positive
- Tickers must match prices.csv
- Dates should align with prices.csv

**Notes**:
- Represents variance not explained by factors
- Total variance = factor variance + specific variance
- Typically estimated from factor model residuals

**Loading**:
```python
specific_var = pd.read_csv('specific_variance.csv', index_col=0, parse_dates=True)
```

---

### 8. sector_mapping.csv

**Description**: Sector classification for each security.

**Required For**: Sector hedging, sector exposure constraints

**Format**:
- Columns: ticker, sector
- Values: ticker (string), sector (string)

**Example**:
```csv
ticker,sector
AAPL,Technology
MSFT,Technology
GOOGL,Technology
JPM,Financials
JNJ,Healthcare
XOM,Energy
```

**Requirements**:
- All tickers in prices.csv should have a sector
- Sector names should be consistent (no typos)
- One sector per ticker (no multi-classification)

**Notes**:
- Common sector classifications: GICS, ICB, or custom
- Can use sub-industries for finer granularity
- Static mapping (does not change over time)

**Loading**:
```python
sector_mapping = pd.read_csv('sector_mapping.csv')
```

---

### 9. trade_prices.csv

**Description**: Execution prices different from close prices.

**Required For**: Realistic execution modeling (`use_trade_prices=True`)

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Execution prices (float, > 0)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,150.30,250.55,100.80
2023-01-02,151.05,251.30,101.05
2023-01-03,150.80,250.05,100.55
```

**Requirements**:
- Format identical to prices.csv
- Trade prices typically slightly worse than close prices

**Notes**:
- Represents actual execution quality
- Can model slippage, spread, etc.
- Optional: if not provided, close prices are used

**Loading**:
```python
trade_prices = pd.read_csv('trade_prices.csv', index_col=0, parse_dates=True)
```

---

## Input Files (Use Case Specific)

### For Use Case 1: Target Positions

#### target_weights.csv (example)

**Description**: Target portfolio weights per date.

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Weights (float, sum to 1.0 for long-only)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,0.33,0.34,0.33
2023-01-02,0.35,0.33,0.32
2023-01-03,0.32,0.35,0.33
```

**Alternative Formats**:
- `target_shares.csv`: Target share counts
- `target_notional.csv`: Target dollar amounts

---

### For Use Case 2: Signals

#### signals.csv (example)

**Description**: Alpha signals per security per date.

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Signal strength (float, typically standardized)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,1.5,-0.8,0.3
2023-01-02,-0.5,1.2,0.8
2023-01-03,0.2,-0.3,1.5
```

**Notes**:
- Positive signals → long positions
- Negative signals → short positions
- Magnitude represents conviction

---

### For Use Case 3: External Trades

#### external_trades.csv (example)

**Description**: External trades to apply each day.

**Format**:
- Index: Date (datetime)
- Columns: Security tickers (string)
- Values: Share quantities (float, signed)

**Example**:
```csv
date,AAPL,MSFT,GOOGL
2023-01-01,1000,-500,0
2023-01-02,0,300,-200
2023-01-03,-800,0,400
```

**Notes**:
- Positive values = buys
- Negative values = sells
- Zero = no trade

---

## Data Validation

### Validation Checklist

Before running a backtest, verify:

1. **Date Alignment**:
   ```python
   assert prices.index.equals(adv.index)
   ```

2. **Ticker Alignment**:
   ```python
   assert set(prices.columns) == set(adv.columns)
   ```

3. **No Missing Values**:
   ```python
   assert not prices.isna().any().any()
   ```

4. **Positive Prices**:
   ```python
   assert (prices > 0).all().all()
   ```

5. **Positive Volumes**:
   ```python
   assert (adv > 0).all().all()
   ```

6. **Factor Names Consistent**:
   ```python
   assert set(factor_exposures.columns) == set(factor_returns.columns)
   ```

### Using Built-in Validation

```python
from backtesting import DataManager

data_manager = DataManager('./data')
issues = data_manager.validate_data()

if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("All data validated successfully!")
```

---

## Common Issues and Solutions

### Issue: Date Misalignment

**Problem**: Different files have different date ranges.

**Solution**:
```python
# Find common date range
prices_dates = set(prices.index)
adv_dates = set(adv.index)
common_dates = sorted(prices_dates & adv_dates)

# Subset to common dates
prices = prices.loc[common_dates]
adv = adv.loc[common_dates]
```

### Issue: Missing Tickers

**Problem**: Some files missing certain tickers.

**Solution**:
```python
# Find common tickers
prices_tickers = set(prices.columns)
adv_tickers = set(adv.columns)
common_tickers = sorted(prices_tickers & adv_tickers)

# Subset to common tickers
prices = prices[common_tickers]
adv = adv[common_tickers]
```

### Issue: Missing Values

**Problem**: NaN values in data.

**Solution**:
```python
# Forward fill missing values
prices = prices.fillna(method='ffill')

# Or interpolate
prices = prices.interpolate(method='linear')

# Or drop securities with missing data
prices = prices.dropna(axis=1)
```

### Issue: Wrong Data Types

**Problem**: Prices loaded as strings.

**Solution**:
```python
# Force numeric conversion
prices = prices.apply(pd.to_numeric, errors='coerce')
```

---

## Best Practices

1. **Consistent Naming**: Use consistent ticker symbols across all files
2. **Date Format**: Use ISO format (YYYY-MM-DD) for dates
3. **Missing Data**: Handle missing data before backtesting
4. **Data Quality**: Validate data using built-in checks
5. **Backups**: Keep original data files as backups
6. **Documentation**: Document any data transformations
7. **Version Control**: Use git to track data changes
8. **Compression**: For large datasets, consider compressed CSV (`.csv.gz`)

---

## Performance Tips

For large datasets (2000+ securities):

1. **Use float32**: Set `use_float32=True` in DataManager
2. **Chunked Loading**: Load data in date chunks
3. **Parquet Format**: Consider using Parquet instead of CSV
4. **Memory Mapping**: Use memory-mapped arrays for very large datasets

Example:
```python
# Save as Parquet for faster loading
prices.to_parquet('prices.parquet')

# Load with specific dtype
prices = pd.read_parquet('prices.parquet', dtype=np.float32)
```

---

## Questions?

If you have questions about data formats:
1. Check the example data in `sample_data/`
2. See the notebook examples
3. Run `generate_sample_data.py` to see correct formats
4. Open an issue on GitHub

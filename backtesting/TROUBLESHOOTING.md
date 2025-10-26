# Troubleshooting Guide

Common issues and solutions when using the backtesting framework.

## Error: KeyError with Timestamp

### Symptom

```python
KeyError: Timestamp('2023-01-01 00:00:00')
```

### Cause

The date you're trying to use doesn't exist in your data. This commonly happens because:
1. The date falls on a weekend or holiday
2. Your data starts/ends on a different date
3. Date formatting issues

### Solution

**Use the `get_date_range()` utility to automatically align dates:**

```python
import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager
from backtesting.utils import get_date_range

# Load data
data_manager = DataManager(data_dir='./sample_data')
prices = data_manager.load_prices()

# Check what dates are actually available
print(f"Data available from {prices.index[0].date()} to {prices.index[-1].date()}")

# Align your desired dates to actual trading days
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')
aligned_start, aligned_end = get_date_range(prices, start_date, end_date)

print(f"Using aligned dates: {aligned_start.date()} to {aligned_end.date()}")

# Use aligned dates in backtest
results = backtester.run(
    start_date=aligned_start,  # Not start_date!
    end_date=aligned_end,      # Not end_date!
    use_case=1,
    inputs=inputs
)
```

**Alternative: Use the data's actual date range:**

```python
# Simply use the first and last dates from your data
start_date = prices.index[0]
end_date = prices.index[-1]

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=1,
    inputs=inputs
)
```

---

## Error: Import Error

### Symptom

```python
ModuleNotFoundError: No module named 'backtesting'
```

### Solution

Make sure you're in the correct directory:

```bash
cd /path/to/backtesting

# Add to path or install
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install as package
pip install -e .
```

---

## Error: FileNotFoundError for CSV files

### Symptom

```python
FileNotFoundError: [Errno 2] No such file or directory: './sample_data/prices.csv'
```

### Solution

Generate the sample data first:

```bash
python generate_sample_data.py --securities 100 --days 252
```

Or check that you're in the right directory:

```bash
ls sample_data/
# Should show: prices.csv, adv.csv, betas.csv, etc.
```

---

## Error: Out of Memory

### Symptom

```python
MemoryError: Unable to allocate array
```

### Solution

**1. Use float32 instead of float64:**

```python
data_manager = DataManager(data_dir='./sample_data', use_float32=True)
```

**2. Reduce the universe size for testing:**

```python
prices = data_manager.load_prices()
# Use only first 500 securities
prices = prices.iloc[:, :500]
```

**3. Process shorter date ranges:**

```python
# Test with just one quarter
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-03-31')  # Just Q1
```

---

## Error: Optimization Failed

### Symptom

```python
Warning: Optimization failed on 2023-01-15: ...
```

### Solution

**1. Relax constraints:**

```python
config = BacktestConfig(
    max_portfolio_variance=0.001,  # Increased from 0.0004
    max_factor_exposure={
        'Factor1': 0.2,  # Increased from 0.1
    }
)
```

**2. Increase optimizer iterations:**

```python
config = BacktestConfig(
    optimizer_max_iter=2000,  # Increased from 1000
    optimizer_tolerance=1e-5  # Relaxed from 1e-6
)
```

**3. Check for infeasible constraints:**

```python
# Make sure your constraints are actually achievable
# For example, you can't have max_variance=0.0001 with real market data
```

---

## Error: Missing Factor Data

### Symptom

```python
Exception: Could not load factor model data
```

### Solution

Factor model data is optional. The backtest will run without it, but you won't get factor attribution.

**If you want factor attribution, make sure these files exist:**
- `factor_exposures.csv`
- `factor_returns.csv`
- `factor_covariance.csv`
- `specific_variance.csv`

**Or regenerate sample data:**

```bash
python generate_sample_data.py --factors 5
```

---

## Issue: Slow Performance

### Symptom

Backtest takes a long time to complete.

### Solution

**1. Disable progress bars for production:**

```python
results = backtester.run(..., show_progress=False)
```

**2. Use float32:**

```python
data_manager = DataManager(data_dir='./sample_data', use_float32=True)
```

**3. Profile your code:**

```python
import time

start = time.time()
results = backtester.run(...)
print(f"Backtest took {time.time() - start:.2f} seconds")
```

**4. Check for data issues:**

```python
# Make sure you don't have too many dates
prices = data_manager.load_prices()
print(f"Shape: {prices.shape}")  # Should be reasonable, e.g., (252, 100)
```

---

## Issue: Unexpected Results

### Symptom

Results don't match expectations (e.g., returns are too high/low).

### Solution

**1. Validate your input data:**

```python
issues = data_manager.validate_data()
if issues:
    for issue in issues:
        print(f"Warning: {issue}")
```

**2. Check your configuration:**

```python
print(f"Transaction cost power: {config.tc_power}")
print(f"ADV participation: {config.max_adv_participation}")
print(f"Beta hedge enabled: {config.enable_beta_hedge}")
```

**3. Inspect trades:**

```python
trades_df = results.get_trades_dataframe()
print(trades_df.head(20))

# Check transaction costs
print(f"Total costs: ${trades_df['cost'].sum():,.2f}")
```

**4. Review exposures:**

```python
results_df = results.to_dataframe()
print(results_df[['gross_exposure', 'net_exposure']].describe())
```

---

## Issue: Charts Not Displaying in Notebook

### Symptom

Charts don't show in Jupyter notebook.

### Solution

**Make sure you have the magic command:**

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

**Or use:**

```python
%matplotlib notebook  # For interactive plots
```

---

## Issue: Excel Report Won't Open

### Symptom

Excel file is corrupted or won't open.

### Solution

**Make sure openpyxl is installed:**

```bash
pip install openpyxl
```

**Try generating just HTML and CSV:**

```python
results.generate_full_report(
    output_dir='./output',
    formats=['html', 'csv']  # Skip excel
)
```

---

## Issue: Target Weights Don't Sum to 1

### Symptom

Warning about weights or unexpected portfolio size.

### Solution

**For long/short portfolios, weights don't need to sum to 1:**

```python
# This is fine for long/short:
weights = {
    'AAPL': 0.5,   # Long 50%
    'MSFT': -0.3,  # Short 30%
    'GOOGL': 0.2   # Long 20%
}
# Gross exposure: 100%, Net exposure: 40%
```

**For long-only, normalize weights:**

```python
weights = {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.2}
total = sum(weights.values())
normalized_weights = {k: v/total for k, v in weights.items()}
# Now sums to 1.0
```

---

## Getting More Help

If your issue isn't covered here:

1. **Check the examples:** Review the notebook examples in `notebooks/`
2. **Read the docs:** See `README.md` and `docs/data_schema.md`
3. **Run the quick fix:** Try `python quick_fix_example.py`
4. **Check data format:** Make sure your CSVs match the schema
5. **Enable logging:** Add print statements to debug
6. **Open an issue:** Report bugs on GitHub

## Quick Diagnostics

Run this diagnostic script to check your setup:

```python
import pandas as pd
import numpy as np
from backtesting import DataManager

print("=== Backtesting Framework Diagnostics ===\n")

# Check data directory
try:
    data_manager = DataManager('./sample_data')
    prices = data_manager.load_prices()
    print(f"✓ Data loaded successfully")
    print(f"  - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  - Securities: {len(prices.columns)}")
    print(f"  - Trading days: {len(prices.index)}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

# Validate data
try:
    issues = data_manager.validate_data()
    if issues:
        print(f"\n⚠ Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ Data validation passed")
except Exception as e:
    print(f"✗ Validation failed: {e}")

# Check dependencies
print("\n=== Dependencies ===")
try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except:
    print("✗ numpy not installed")

try:
    import pandas
    print(f"✓ pandas {pandas.__version__}")
except:
    print("✗ pandas not installed")

try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except:
    print("✗ scipy not installed")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except:
    print("✗ matplotlib not installed")

print("\n=== Setup Complete ===")
```

Save this as `diagnostics.py` and run it to check your setup.

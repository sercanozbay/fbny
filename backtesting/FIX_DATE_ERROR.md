# 🛠️ FIX: KeyError with Timestamp

## You're Getting This Error:

```python
KeyError: Timestamp('2023-01-01 00:00:00')
```

## ✅ Quick Fix - Run This Command:

```bash
python simple_working_example.py
```

This script uses the **actual dates** from your data, so it will **ALWAYS work** without any date errors!

---

## 🔍 Why This Error Happens

The error occurs because `2023-01-01` (January 1st) doesn't exist in your sample data. This is because:

1. **January 1st is a holiday** - Markets are closed
2. **Your sample data starts on the first business day** (likely January 3rd or 4th, 2023)
3. Pandas requires **exact date matching** - it won't automatically find the nearest date

---

## 📋 Three Ways to Fix This

### Method 1: Use Actual Dates (Easiest - No Errors!)

```python
# Load your data
prices = data_manager.load_prices()

# Use ACTUAL dates from the data
start_date = prices.index[0]   # First date
end_date = prices.index[-1]    # Last date

# Run backtest - THIS WILL WORK!
results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=1,
    inputs=inputs
)
```

**This is foolproof!** ✅

### Method 2: Use Date Alignment Utility

```python
from backtesting.utils import get_date_range

# Load your data
prices = data_manager.load_prices()

# Align your desired dates
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')
aligned_start, aligned_end = get_date_range(prices, start_date, end_date)

# Run backtest with aligned dates
results = backtester.run(
    start_date=aligned_start,  # Aligned date
    end_date=aligned_end,      # Aligned date
    use_case=1,
    inputs=inputs
)
```

### Method 3: Check Your Dates First

```bash
# See what dates are actually in your data
python check_your_dates.py
```

This will show you the exact dates available, then you can use them directly.

---

## 🎯 Recommended Scripts to Run

### 1. Check Your Dates (15 seconds)
```bash
python check_your_dates.py
```
Shows you what dates are in your data.

### 2. Simple Working Example (1 minute)
```bash
python simple_working_example.py
```
Complete backtest using actual dates - **guaranteed to work!**

### 3. Quick Fix Example (1 minute)
```bash
python quick_fix_example.py
```
Example using date alignment utility.

### 4. Test Installation (30 seconds)
```bash
python test_installation.py
```
Comprehensive test of entire framework.

---

## 🔧 If You're in a Jupyter Notebook

Add this at the top of your notebook:

```python
import sys
sys.path.append('..')

import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager
from backtesting.utils import get_date_range  # ← Add this!

# Load data
data_manager = DataManager(data_dir='../sample_data')
prices = data_manager.load_prices()

# Check what dates you have
print(f"Data available: {prices.index[0].date()} to {prices.index[-1].date()}")

# Use actual dates
start_date = prices.index[0]
end_date = prices.index[-1]

# Or use alignment
# start_date = pd.Timestamp('2023-01-01')
# end_date = pd.Timestamp('2023-12-31')
# start_date, end_date = get_date_range(prices, start_date, end_date)

# Rest of your code...
```

---

## 📝 Complete Working Code

Here's a complete script that **will work**:

```python
import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager

# 1. Load data
data_manager = DataManager(data_dir='./sample_data')
prices = data_manager.load_prices()

# 2. Use ACTUAL dates (no errors!)
start_date = prices.index[0]
end_date = prices.index[-1]

print(f"Using dates: {start_date.date()} to {end_date.date()}")

# 3. Load targets
target_weights = pd.read_csv('./sample_data/target_weights.csv',
                              index_col=0, parse_dates=True)
target_weights = target_weights.loc[start_date:end_date]

targets_by_date = {
    date: target_weights.loc[date].to_dict()
    for date in target_weights.index
}

inputs = {'type': 'weights', 'targets': targets_by_date}

# 4. Configure
config = BacktestConfig(
    initial_cash=10_000_000,
    max_adv_participation=0.05,
    enable_beta_hedge=False,
    risk_free_rate=0.02
)

# 5. Run (THIS WORKS!)
backtester = Backtester(config, data_manager)
results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=1,
    inputs=inputs
)

# 6. Results
results.print_summary()
results.generate_full_report('./output/my_backtest')
```

Save this as `my_working_backtest.py` and run it!

---

## ⚠️ Common Mistakes to Avoid

### ❌ Don't Do This:
```python
# This might not exist in your data!
results = backtester.run(
    start_date=pd.Timestamp('2023-01-01'),  # ❌ Error!
    end_date=pd.Timestamp('2023-12-31'),     # ❌ Error!
    ...
)
```

### ✅ Do This Instead:
```python
# Use what's actually in your data
prices = data_manager.load_prices()
start_date = prices.index[0]   # ✅ Works!
end_date = prices.index[-1]    # ✅ Works!

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    ...
)
```

---

## 🆘 Still Having Issues?

1. **Make sure sample data exists:**
   ```bash
   ls sample_data/
   # Should show: prices.csv, adv.csv, etc.
   ```

2. **Regenerate sample data:**
   ```bash
   python generate_sample_data.py
   ```

3. **Check dates in your data:**
   ```bash
   python check_your_dates.py
   ```

4. **Run the working example:**
   ```bash
   python simple_working_example.py
   ```

5. **Read full troubleshooting guide:**
   See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📚 Additional Help

- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Full Docs:** [README.md](README.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Error Summary:** [ERRORS_FIXED.md](ERRORS_FIXED.md)

---

## ✅ Summary

**The simplest solution:**

1. Run `python simple_working_example.py`
2. Or use `start_date = prices.index[0]` in your code
3. That's it! No more date errors! 🎉

The key is to **use the dates that actually exist in your data** rather than trying to use calendar dates.

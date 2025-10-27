# Troubleshooting Dynamic Trade Generation

## Issue: Total Return is 0%

**Symptom:** All backtests show 0% return, no trades executed

**Root Cause:** Ticker mismatch between signal functions and actual data

### Explanation

The notebook examples used placeholder tickers like:
- `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`

But the sample data in `../sample_data/` contains different tickers:
- `STOCK0000`, `STOCK0001`, `STOCK0002`, ..., `STOCK1499`

When signal functions generate trades for non-existent tickers, the backtester correctly ignores them (as it should), resulting in:
- No trades executed
- No PnL generated
- 0% returns

### Solution

**Always use tickers that exist in your data:**

```python
# BAD - Using fake tickers
def my_signals(context):
    return {
        'AAPL': 0.30,
        'MSFT': 0.25,
        'GOOGL': 0.20
    }

# GOOD - Using actual tickers from your data
def my_signals(context):
    return {
        'STOCK0000': 0.30,
        'STOCK0001': 0.25,
        'STOCK0002': 0.20
    }
```

### How to Find Available Tickers

```python
# Load your data
data_manager = DataManager('../sample_data')
prices_df = data_manager.load_prices()

# Get all available tickers
available_tickers = prices_df.columns.tolist()
print(f"Available tickers: {available_tickers[:10]}")  # Show first 10
```

### Debugging Tips

If your backtest shows 0% returns:

1. **Check if trades are being generated:**
   ```python
   trades_by_date = results.get_external_trades_by_date()
   print(f"Number of trading days: {len(trades_by_date)}")

   if trades_by_date.empty:
       print("No trades generated!")
   ```

2. **Add debug prints to your signal function:**
   ```python
   def my_signals(context):
       signals = {'STOCK0000': 0.3, 'STOCK0001': 0.2}
       print(f"Date: {context['date']}, Generated signals: {signals}")
       return signals
   ```

3. **Check if tickers exist in the data:**
   ```python
   def my_signals(context):
       signals = {'STOCK0000': 0.3, 'NONEXISTENT': 0.2}

       # Filter to only tickers that exist
       available = context['prices'].keys()
       filtered_signals = {k: v for k, v in signals.items() if k in available}

       if len(filtered_signals) < len(signals):
           print(f"Warning: Some tickers don't exist in data")

       return filtered_signals
   ```

4. **Verify date ranges:**
   - Make sure your backtest dates overlap with your data
   - Sample data is from 2023-01-02 to 2027-09-30

## Common Errors and Solutions

### Error: KeyError accessing DataFrame columns

**Symptom:** `KeyError: "None of [Index(['date', 'num_trades', ...`

**Cause:** DataFrame is empty (no trades generated)

**Solution:** Always check before accessing columns:
```python
trades_by_date = results.get_external_trades_by_date()
if not trades_by_date.empty:
    print(trades_by_date[['date', 'num_trades']])
else:
    print("No external trades found")
```

### Error: NameError for List/Dict types

**Symptom:** `NameError: name 'List' is not defined`

**Cause:** Missing typing imports

**Solution:** Add to imports:
```python
from typing import Dict, List, Optional
```

### Performance: Backtest is too slow

**Solutions:**
1. Set `max_portfolio_variance=None` (disables optimization)
2. Use shorter date ranges (1-3 months for testing)
3. Trade fewer tickers (3-5 instead of 10+)
4. Enable progress bar: `show_progress=True`

### Conditional trades never execute

**Symptom:** `ConditionalSignalGenerator` shows "Traded on 0 days"

**Possible causes:**
1. Condition function never returns True
2. No dates in range satisfy the condition
3. Ticker mismatch (same as main issue above)

**Debug:**
```python
def is_month_end(context):
    result = context['date'].is_month_end
    print(f"{context['date']}: is_month_end = {result}")
    return result
```

## Updated Notebooks

All example notebooks have been updated to use actual tickers from the sample data:
- `07_dynamic_trade_generation.ipynb` - Now uses `STOCK0000`, `STOCK0001`, etc.

Run the notebooks again to see actual trading activity and returns!

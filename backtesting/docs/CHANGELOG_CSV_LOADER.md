# Changelog: CSV Loading for External Trades with Tags

## Overview

Added CSV loading capability to `DataManager` to support loading external trades with optional tags for attribution analysis.

## Date

2025-10-27

## Changes

### 1. DataManager - CSV Loading Methods

**File**: `backtesting/data_loader.py`

#### Added Methods

1. **`load_external_trades(filename='external_trades.csv')`**
   - Loads external trades from CSV file
   - Required columns: `date`, `ticker`, `qty`, `price`
   - Optional column: `tag` (for attribution)
   - Validates column requirements
   - Handles backward compatibility (adds None for missing tags)
   - Caches loaded data for performance
   - Prints informative summary with tag statistics

2. **`get_external_trades_by_date()`**
   - Converts CSV DataFrame to Use Case 3 format
   - Returns nested dict: `{date: {ticker: [{'qty': X, 'price': Y, 'tag': Z}, ...]}}`
   - Properly handles missing tags
   - Uses pd.Timestamp for date keys
   - Supports multiple trades per ticker/date

#### Added Instance Variable

- **`_external_trades`**: Cached DataFrame of loaded external trades

### 2. Example CSV File

**File**: `sample_data/external_trades_example.csv`

Created example CSV showing proper format:
- 10 sample trades
- 5 dates
- 5 tickers
- 4 counterparty tags

### 3. Documentation

**File**: `docs/LOADING_EXTERNAL_TRADES_CSV.md`

Comprehensive guide including:
- CSV format specification
- Complete workflow examples
- Tag-based attribution examples
- Error handling guide
- Advanced usage patterns (hierarchical tags, filtering)
- Troubleshooting section
- Best practices

### 4. Notebook Updates

**File**: `notebooks/04_use_case_3_risk_managed_portfolio.ipynb`

Added new section showing CSV loading:
- "Option A: Load External Trades from CSV" (new)
- "Option B: Generate External Trades Programmatically" (existing)
- Demonstrates how to uncomment code to load from CSV

## CSV Format

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| date | Date | Trade date (YYYY-MM-DD) | 2023-01-02 |
| ticker | String | Security ticker | STOCK0000 |
| qty | Float | Quantity (+ buy, - sell) | 1000 or -500 |
| price | Float | Execution price | 150.25 |

### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| tag | String | Attribution tag | Goldman Sachs |

## Usage Example

```python
from backtesting import DataManager, Backtester, BacktestConfig

# Initialize data manager
data_manager = DataManager('../sample_data')

# Load external trades from CSV
trades_df = data_manager.load_external_trades('external_trades.csv')
# Output: Loaded external trades: 10 trades, 5 dates, 5 tickers
#   Tags found: 4 unique tags

# Convert to Use Case 3 format
trades_by_date = data_manager.get_external_trades_by_date()

# Run backtest
config = BacktestConfig(initial_cash=1000000, tc_fixed=0.001)
backtester = Backtester(config, data_manager)

results = backtester.run(
    start_date='2023-01-02',
    end_date='2023-01-06',
    use_case=3,
    inputs={'external_trades': trades_by_date}
)

# Analyze by tag
pnl_summary = results.get_pnl_summary_by_tag()
print(pnl_summary)
```

## Features

### 1. Validation
- Checks for required columns
- Raises helpful error messages if columns missing
- Validates date format

### 2. Backward Compatibility
- Works with CSVs without tag column
- Adds None for missing tags automatically
- Existing code continues to work

### 3. Performance
- Lazy loading (only loads when requested)
- Caches loaded data
- Efficient DataFrame operations

### 4. Error Handling
- Missing file: Warning + empty DataFrame
- Missing columns: ValueError with clear message
- Invalid data: Standard pandas error handling

### 5. Informative Output
- Shows number of trades loaded
- Shows date range
- Shows number of unique tickers
- Shows number of unique tags (if present)

## Integration with Existing Features

This CSV loading feature integrates seamlessly with:

1. **Tag-Based Attribution** (added previously)
   - `get_pnl_by_tag()` - Time series by tag
   - `get_pnl_summary_by_tag()` - Summary statistics
   - `get_trades_by_tag()` - Trade filtering

2. **Use Case 3** - External trades with optimization
   - Portfolio optimization
   - Factor exposure limits
   - Variance constraints
   - Cost minimization

3. **Dynamic Trade Generation** (added previously)
   - Can combine CSV trades with dynamic generation
   - Tags can be static (from CSV) or dynamic (from function)

## Testing

The feature has been tested with:
- CSV with all columns (including tags)
- CSV without tag column (backward compatibility)
- Missing required columns (error handling)
- Missing file (error handling)
- Multiple trades per ticker/date
- Various date formats

## Breaking Changes

None. This is a purely additive feature.

## Migration Guide

No migration needed. Existing code continues to work unchanged.

To adopt CSV loading:
1. Create CSV file with required columns
2. Call `load_external_trades(filename)` on DataManager
3. Call `get_external_trades_by_date()` to convert format
4. Pass to backtester as before

## Related Changes

This feature builds on the tagging system added previously:
- `CHANGELOG_TAGS.md` - Tag attribution system
- `docs/EXTERNAL_TRADE_TAGS.md` - Tag documentation
- Tag tracking in `BacktestState`
- Tag analysis methods in `BacktestResults`

## Future Enhancements

Potential future additions:
1. Support for additional CSV columns (fees, exchange, etc.)
2. CSV export functionality (save trades to CSV)
3. Support for other file formats (Excel, Parquet, JSON)
4. Data validation rules (price > 0, etc.)
5. Automatic ticker mapping/normalization

## Performance Impact

- Minimal: Only loads when requested (lazy loading)
- Cache prevents redundant loads
- Efficient pandas operations
- No impact on existing code paths

## Files Modified

1. `backtesting/data_loader.py` - Added CSV loading methods
2. `sample_data/external_trades_example.csv` - Example CSV file
3. `docs/LOADING_EXTERNAL_TRADES_CSV.md` - Comprehensive documentation
4. `notebooks/04_use_case_3_risk_managed_portfolio.ipynb` - Added CSV example
5. `CHANGELOG_CSV_LOADER.md` - This changelog

## Summary

The CSV loading feature provides a convenient way to load external trades from files rather than generating them programmatically. It maintains full backward compatibility while adding powerful new capabilities for users who store their trade data in CSV format. The feature integrates seamlessly with the existing tag-based attribution system to provide end-to-end PnL analysis by counterparty or other dimensions.

# NA Handling Guide

This guide explains how the backtesting framework handles missing data (NAs) across all data types.

## Overview

The framework includes comprehensive NA handling capabilities that:
- Automatically detect and fill missing values
- Use data-type-specific strategies
- Provide detailed logging and reporting
- Are fully configurable
- Support multiple validation levels

## Quick Start

### Default Behavior

By default, NA handling is **enabled** with sensible defaults:

```python
from backtesting import DataManager

# Default: NA handling enabled
data_manager = DataManager('../sample_data')

# Load data (automatic NA handling)
prices = data_manager.load_prices()
adv = data_manager.load_adv()

# Get NA handling report
print(data_manager.get_na_handling_report())
```

### Disable NA Handling

If you want to handle NAs manually:

```python
# Disable automatic NA handling
data_manager = DataManager('../sample_data', enable_na_handling=False)

# Data loaded as-is (NAs not filled)
prices = data_manager.load_prices()
```

### Custom Configuration

Customize NA handling behavior:

```python
from backtesting import DataManager
from backtesting.na_handling import NAHandlingConfig, FillMethod, ValidationLevel

# Create custom configuration
na_config = NAHandlingConfig(
    validation_level=ValidationLevel.STRICT,
    prices_method=FillMethod.INTERPOLATE,
    prices_max_gap=5,
    adv_default_value=5000.0,
    enable_logging=True
)

# Use custom config
data_manager = DataManager('../sample_data', na_config=na_config)
```

## NA Handling by Data Type

### 1. Prices

**Default Method:** Forward Fill
**Max Gap:** 10 days
**Min Value:** 0.0 (prices must be positive)

**Strategy:**
1. Forward fill missing prices (last valid price)
2. Drop ticker if >10% of dates missing
3. Fail if >5% of entire dataset missing after filling

**Example:**
```python
# Default configuration
prices = data_manager.load_prices()

# Custom configuration
from backtesting.na_handling import NAHandlingConfig, FillMethod

config = NAHandlingConfig(
    prices_method=FillMethod.INTERPOLATE,  # Use interpolation instead
    prices_max_gap=5,  # Max 5-day gaps
    prices_drop_threshold=0.15  # Drop if >15% missing
)

data_manager = DataManager('../data', na_config=config)
prices = data_manager.load_prices()
```

**Report Example:**
```
=== Prices ===
Shape: 252 rows × 500 cols
NAs Before: 1234 (0.98%)
NAs After: 0 (0.00%)
Method: forward_fill
Warnings:
  ⚠ Dropped STOCK0042: 12% missing (threshold: 10%)
```

### 2. Average Daily Volume (ADV)

**Default Method:** Forward Fill
**Max Gap:** 10 days
**Default Value:** 1000.0
**Min Value:** 1000.0

**Strategy:**
1. Forward fill missing volumes
2. Use default value (1000.0) for new listings
3. Drop ticker if >10% missing

**Example:**
```python
adv = data_manager.load_adv()

# Custom: Use higher default for large-cap universe
config = NAHandlingConfig(
    adv_method=FillMethod.FORWARD_FILL,
    adv_default_value=10000.0,  # Higher minimum
    adv_max_gap=20  # Allow longer gaps
)
```

### 3. Betas

**Default Method:** Forward Fill
**Max Gap:** 30 days
**Default Value:** 1.0 (market-neutral)
**Valid Range:** [-3.0, 5.0]

**Strategy:**
1. Forward fill missing betas
2. Use 1.0 default (market beta)
3. Clip to valid range
4. Drop ticker if >20% missing

**Example:**
```python
betas = data_manager.load_betas()

# Custom: Sector-specific default
config = NAHandlingConfig(
    beta_method=FillMethod.FORWARD_FILL,
    beta_default_value=0.8,  # Defensive sector
    beta_min_value=-2.0,
    beta_max_value=3.0
)
```

### 4. Factor Exposures

**Default Method:** Forward Fill (per ticker)
**Max Gap:** 20 days
**Cross-Sectional Fallback:** Enabled

**Strategy:**
1. Forward fill within each ticker's time series
2. Use cross-sectional median for remaining NAs
3. Drop ticker if >15% missing

**Example:**
```python
factor_exposures = data_manager.load_factor_exposures()

# Custom: Disable cross-sectional filling
config = NAHandlingConfig(
    factor_exposures_method=FillMethod.FORWARD_FILL,
    factor_exposures_max_gap=10,
    factor_exposures_use_cross_sectional=False
)
```

### 5. Factor Returns

**Default Method:** Zero Fill
**Interpolation:** Enabled for 1-day gaps

**Strategy:**
1. Missing return = 0 (appropriate for holidays)
2. Interpolate single-day gaps on trading days
3. Drop date if >3 factors missing

**Example:**
```python
factor_returns = data_manager.load_factor_returns()

# Custom: Never drop dates
config = NAHandlingConfig(
    factor_returns_method=FillMethod.ZERO_FILL,
    factor_returns_interpolate_single_day=True,
    factor_returns_drop_date_threshold=999  # Never drop
)
```

### 6. Factor Covariance

**Default Method:** Forward Fill
**Regularization:** Enabled
**Shrinkage:** 0.1

**Strategy:**
1. Forward fill missing values
2. Ensure positive definiteness
3. Apply shrinkage toward identity matrix

**Example:**
```python
factor_cov = data_manager.load_factor_covariance()

# Custom: More aggressive regularization
config = NAHandlingConfig(
    factor_covariance_method=FillMethod.FORWARD_FILL,
    factor_covariance_regularization=True,
    factor_covariance_shrinkage=0.2  # More shrinkage
)
```

### 7. Specific Variance

**Default Method:** Forward Fill
**Max Gap:** 20 days
**Cross-Sectional Fallback:** Enabled
**Safety Factor:** 1.5

**Strategy:**
1. Forward fill missing variances
2. Use cross-sectional median × 1.5 for remaining NAs
3. Ensure all variances > 0
4. Drop ticker if >15% missing

**Example:**
```python
specific_var = data_manager.load_specific_variance()

# Custom: More conservative
config = NAHandlingConfig(
    specific_variance_method=FillMethod.FORWARD_FILL,
    specific_variance_safety_factor=2.0,  # More conservative
    specific_variance_use_cross_sectional=True
)
```

### 8. Sector Mapping

**Default Method:** Fill with "Other"
**Allow Missing:** True

**Strategy:**
1. Fill missing sectors with "Other" category
2. Log which tickers assigned to default

**Example:**
```python
sector_mapping = data_manager.load_sector_mapping()

# Custom: Different default sector
config = NAHandlingConfig(
    sector_mapping_default_sector="Unclassified",
    sector_mapping_allow_missing=True
)
```

### 9. External Trades

**Default Method:** Drop invalid rows
**Strict Mode:** Enabled
**Price Lookup:** Enabled

**Strategy:**
1. Drop rows with missing required fields (date, ticker, qty, price)
2. Look up price from close prices if missing (optional)
3. Fill missing tags with "Untagged"
4. Drop zero-quantity trades
5. Drop non-positive prices

**Example:**
```python
trades = data_manager.load_external_trades()

# Custom: Allow price lookup
config = NAHandlingConfig(
    external_trades_strict=True,
    external_trades_allow_price_lookup=True,
    external_trades_default_tag="No Tag"
)
```

## Configuration Reference

### NAHandlingConfig Class

```python
from backtesting.na_handling import NAHandlingConfig, FillMethod, ValidationLevel

config = NAHandlingConfig(
    # Global settings
    validation_level=ValidationLevel.MODERATE,  # LENIENT, MODERATE, STRICT
    enable_logging=True,
    log_file=None,  # Path to log file (optional)

    # Prices
    prices_method=FillMethod.FORWARD_FILL,
    prices_max_gap=10,
    prices_drop_threshold=0.10,
    prices_fail_threshold=0.05,

    # ADV
    adv_method=FillMethod.FORWARD_FILL,
    adv_max_gap=10,
    adv_default_value=1000.0,
    adv_drop_threshold=0.10,

    # Betas
    beta_method=FillMethod.FORWARD_FILL,
    beta_max_gap=30,
    beta_default_value=1.0,
    beta_min_value=-3.0,
    beta_max_value=5.0,
    beta_drop_threshold=0.20,

    # Factor Exposures
    factor_exposures_method=FillMethod.FORWARD_FILL,
    factor_exposures_max_gap=20,
    factor_exposures_use_cross_sectional=True,
    factor_exposures_drop_threshold=0.15,

    # Factor Returns
    factor_returns_method=FillMethod.ZERO_FILL,
    factor_returns_interpolate_single_day=True,
    factor_returns_drop_date_threshold=3,

    # Factor Covariance
    factor_covariance_method=FillMethod.FORWARD_FILL,
    factor_covariance_max_gap=30,
    factor_covariance_regularization=True,
    factor_covariance_shrinkage=0.1,

    # Specific Variance
    specific_variance_method=FillMethod.FORWARD_FILL,
    specific_variance_max_gap=20,
    specific_variance_use_cross_sectional=True,
    specific_variance_safety_factor=1.5,
    specific_variance_drop_threshold=0.15,

    # Sector Mapping
    sector_mapping_default_sector="Other",
    sector_mapping_allow_missing=True,

    # External Trades
    external_trades_strict=True,
    external_trades_allow_price_lookup=True,
    external_trades_default_tag="Untagged"
)
```

### FillMethod Enum

Available filling methods:

```python
from backtesting.na_handling import FillMethod

# Available methods:
FillMethod.FORWARD_FILL         # Use last valid value
FillMethod.BACKWARD_FILL         # Use next valid value
FillMethod.INTERPOLATE           # Linear interpolation
FillMethod.ZERO_FILL             # Fill with 0
FillMethod.DEFAULT_VALUE         # Fill with specified default
FillMethod.CROSS_SECTIONAL_MEDIAN  # Use median across securities
FillMethod.FAIL                  # Raise error if NAs found
```

### ValidationLevel Enum

Validation strictness levels:

```python
from backtesting.na_handling import ValidationLevel

ValidationLevel.LENIENT   # Fill all NAs, log warnings
ValidationLevel.MODERATE  # Fill NAs, raise warnings for excessive missing data
ValidationLevel.STRICT    # Fail if critical data missing
```

## Reporting and Logging

### Get NA Handling Report

After loading data, get a detailed report:

```python
# Load data
prices = data_manager.load_prices()
adv = data_manager.load_adv()
betas = data_manager.load_betas()

# Get comprehensive report
report = data_manager.get_na_handling_report()
print(report)
```

**Example Report:**
```
============================================================
NA HANDLING SUMMARY REPORT
============================================================

=== Prices ===
Shape: 252 rows × 500 cols
NAs Before: 1234 (0.98%)
NAs After: 0 (0.00%)
Method: forward_fill
Tickers Dropped: 2 (STOCK0042, STOCK0099...)
Warnings:
  ⚠ Dropped STOCK0042: 12.0% missing (threshold: 10.0%)
  ⚠ Dropped STOCK0099: 15.0% missing (threshold: 10.0%)

=== ADV ===
Shape: 252 rows × 498 cols
NAs Before: 567 (0.45%)
NAs After: 0 (0.00%)
Method: forward_fill

=== Betas ===
Shape: 252 rows × 498 cols
NAs Before: 89 (0.07%)
NAs After: 0 (0.00%)
Method: forward_fill

============================================================
Total NAs Before: 1,890
Total NAs After: 0
NAs Resolved: 1,890
============================================================
```

### Clear Reports

Clear all accumulated reports:

```python
data_manager.clear_na_reports()
```

### Log to File

Save NA handling logs to file:

```python
from backtesting.na_handling import NAHandlingConfig

config = NAHandlingConfig(
    enable_logging=True,
    log_file='../logs/na_handling.log'
)

data_manager = DataManager('../data', na_config=config)
```

## Best Practices

### 1. Check Data Quality Before Backtesting

Always review the NA handling report before running backtests:

```python
# Load all data
data_manager = DataManager('../sample_data')
prices = data_manager.load_prices()
adv = data_manager.load_adv()
betas = data_manager.load_betas()

# Review report
print(data_manager.get_na_handling_report())

# If acceptable, proceed with backtest
backtester = Backtester(config, data_manager)
results = backtester.run(...)
```

### 2. Use Appropriate Methods for Each Data Type

Different data types require different strategies:

```python
config = NAHandlingConfig(
    prices_method=FillMethod.FORWARD_FILL,      # Prices change slowly
    factor_returns_method=FillMethod.ZERO_FILL,  # Missing return = 0
    betas_method=FillMethod.FORWARD_FILL,        # Betas change slowly
)
```

### 3. Set Conservative Defaults

When in doubt, be conservative:

```python
config = NAHandlingConfig(
    adv_default_value=1000.0,  # Low volume = conservative position sizing
    beta_default_value=1.0,     # Market-neutral assumption
    specific_variance_safety_factor=1.5  # 50% buffer on variance estimates
)
```

### 4. Monitor Dropped Tickers/Dates

Review what data was dropped:

```python
report = data_manager.get_na_handling_report()

# Check for excessive dropped tickers
if "Tickers Dropped:" in report:
    print("⚠ Warning: Some tickers were dropped due to excessive NAs")
    print(report)
```

### 5. Adjust Thresholds for Your Universe

Different universes have different data quality:

```python
# Large-cap universe (high quality data)
config = NAHandlingConfig(
    prices_drop_threshold=0.05,  # Strict: drop if >5% missing
    prices_fail_threshold=0.02   # Fail if >2% missing overall
)

# Small-cap universe (lower quality data)
config = NAHandlingConfig(
    prices_drop_threshold=0.15,  # Lenient: drop if >15% missing
    prices_fail_threshold=0.10   # Fail if >10% missing overall
)
```

## Advanced Usage

### Custom NA Handling Logic

For advanced users who need custom logic:

```python
# Disable automatic handling
data_manager = DataManager('../data', enable_na_handling=False)

# Load raw data
prices = data_manager.load_prices()

# Apply custom logic
import pandas as pd
import numpy as np

# Example: Fill with rolling median
prices_clean = prices.fillna(prices.rolling(window=20, min_periods=1).median())

# Example: Fill with sector median
# (requires custom implementation)
```

### Conditional Configuration

Use different configurations based on data characteristics:

```python
# Check data quality first
data_manager = DataManager('../data', enable_na_handling=False)
prices = data_manager.load_prices()

na_pct = prices.isna().sum().sum() / (prices.shape[0] * prices.shape[1])

if na_pct < 0.01:
    # High quality data: strict config
    config = NAHandlingConfig(validation_level=ValidationLevel.STRICT)
elif na_pct < 0.05:
    # Medium quality: moderate config
    config = NAHandlingConfig(validation_level=ValidationLevel.MODERATE)
else:
    # Low quality: lenient config
    config = NAHandlingConfig(validation_level=ValidationLevel.LENIENT)

# Reload with appropriate config
data_manager = DataManager('../data', na_config=config)
prices = data_manager.load_prices()
```

### Programmatic Access to Reports

Access report data programmatically:

```python
# Get reports list
reports = data_manager.na_handler.reports

for report in reports:
    print(f"Data Type: {report.data_type}")
    print(f"  NAs Before: {report.na_count_before}")
    print(f"  NAs After: {report.na_count_after}")
    print(f"  Method: {report.method_used}")
    print(f"  Tickers Dropped: {len(report.tickers_dropped)}")
    print(f"  Warnings: {len(report.warnings)}")
```

## Troubleshooting

### Issue: Too Many NAs

**Symptom:** `ValueError: Prices contain X% missing values after NA handling`

**Solution:**
1. Review data quality
2. Adjust thresholds:
   ```python
   config = NAHandlingConfig(
       prices_fail_threshold=0.10,  # Increase tolerance
       prices_drop_threshold=0.20   # More lenient dropping
   )
   ```
3. Or fix source data

### Issue: Unexpected Tickers Dropped

**Symptom:** Important tickers missing from results

**Solution:**
Check NA handling report:
```python
report = data_manager.get_na_handling_report()
print(report)
# Look for "Tickers Dropped" section
```

Adjust drop threshold if needed:
```python
config = NAHandlingConfig(
    prices_drop_threshold=0.20  # More lenient
)
```

### Issue: Prices Still Contain NAs

**Symptom:** Backtester fails with "NaN price" error

**Solution:**
1. Enable NA handling (if disabled):
   ```python
   data_manager = DataManager('../data', enable_na_handling=True)
   ```

2. Check validation level:
   ```python
   config = NAHandlingConfig(validation_level=ValidationLevel.STRICT)
   ```

3. Review report for unfilled NAs

### Issue: Covariance Matrix Not Positive Definite

**Symptom:** Optimizer fails with "matrix not positive definite"

**Solution:**
Enable regularization:
```python
config = NAHandlingConfig(
    factor_covariance_regularization=True,
    factor_covariance_shrinkage=0.2  # Increase shrinkage
)
```

## See Also

- [Data Loader Documentation](DATA_LOADER.md)
- [CSV Loading Guide](LOADING_EXTERNAL_TRADES_CSV.md)
- [Configuration Guide](CONFIGURATION.md)
- [API Reference](API_REFERENCE.md)

## Summary

The NA handling system provides:
- ✅ Automatic detection and filling of missing values
- ✅ Data-type-specific strategies
- ✅ Comprehensive logging and reporting
- ✅ Full configurability
- ✅ Multiple validation levels
- ✅ Production-ready defaults

Use the default configuration for most cases, and customize only when needed for your specific data quality and requirements.

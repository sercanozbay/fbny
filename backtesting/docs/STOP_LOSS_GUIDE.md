# Stop Loss Guide

## Overview

The backtesting framework includes a dollar-based stop loss system that automatically reduces gross exposure when portfolio drawdown exceeds specified thresholds. This risk management feature helps limit losses during adverse market conditions.

## Key Features

- **Dollar-based thresholds**: All stop loss levels use absolute dollar drawdowns from peak
- **Multiple levels**: Support for graduated risk reduction (e.g., 75% → 50% → 25% gross)
- **Immediate exit**: Exits stop loss when drawdown improves below threshold
- **Automatic recovery**: Clears all stop loss levels when portfolio reaches new peak
- **Configurable**: Easy to set up via `BacktestConfig`

## How It Works

### Basic Logic

The stop loss manager tracks:
1. **Peak portfolio value**: Running maximum of portfolio value
2. **Current drawdown**: `peak_value - current_value` (in dollars)
3. **Active level**: Which stop loss level (if any) is currently triggered

### Entry and Exit Rules

- **Entry**: When `drawdown >= threshold` → enter that level
- **Exit**: When `drawdown < threshold` → immediately clear stop loss (jump to 100% gross)
- **New Peak**: When portfolio value exceeds peak → clear all stop loss levels

### Important Behavior

**Immediate Exit (No Sticky Recovery)**:
- Once drawdown improves below a level's threshold, the system immediately exits ALL stop loss levels
- It does NOT gradually scale through levels
- Recovery jumps directly to 100% gross exposure

**Example**:
```
Levels: [(5000, 0.75), (10000, 0.50)]

$100k → $88k ($12k DD) → Enter L2, 50% gross (DD >= $10k)
$88k → $91k ($9k DD)   → Clear all, 100% gross (DD < $10k, jumped over L1)
$91k → $89k ($11k DD)  → Enter L2, 50% gross (DD >= $10k again)
```

## Configuration

### Basic Setup

```python
from backtesting import BacktestConfig

config = BacktestConfig(
    initial_cash=10_000_000,
    stop_loss_levels=[
        (5000, 0.75),   # At $5k DD, reduce to 75% gross
        (10000, 0.50),  # At $10k DD, reduce to 50% gross
    ]
)
```

### Stop Loss Level Format

Each level is a 2-tuple: `(drawdown_threshold, gross_reduction)`

**Parameters**:
- `drawdown_threshold` (float): Dollar drawdown from peak that triggers this level
  - Example: `5000` = activate when portfolio is $5,000 below peak

- `gross_reduction` (float): Target gross exposure as fraction
  - Range: 0.0 to 1.0
  - Example: `0.75` = reduce positions to 75% of normal size
  - Example: `0.50` = reduce positions to 50% of normal size

### Multiple Levels

You can define multiple levels for graduated risk reduction:

```python
config = BacktestConfig(
    stop_loss_levels=[
        (5000, 0.75),    # Level 1: Small DD → modest reduction
        (10000, 0.50),   # Level 2: Medium DD → larger reduction
        (15000, 0.25),   # Level 3: Large DD → severe reduction
    ]
)
```

**Requirements**:
- Thresholds should be increasing
- Gross reductions should be decreasing
- All values must be non-negative

### Validation

The config automatically validates:
- Drawdown thresholds >= 0
- Gross reductions in range [0, 1]
- Only 2-tuple format (legacy 3-tuple no longer supported)

## Use Cases

### Use Case 1: Target Positions

Stop loss applies to target positions AFTER hedging:

```python
# Config with stop loss
config = BacktestConfig(
    enable_beta_hedge=True,
    stop_loss_levels=[(10000, 0.50)]
)

# Run backtest
results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=1,
    inputs={'type': 'weights', 'targets': targets_by_date}
)
```

**Execution order**:
1. Process target positions
2. Apply hedging (beta/sector)
3. **Apply stop loss reduction** (if active)
4. Calculate trades

### Use Case 2: Signal-Based Trading

Stop loss applies to signal-derived positions:

```python
config = BacktestConfig(
    enable_beta_hedge=True,
    stop_loss_levels=[(5000, 0.75), (10000, 0.50)]
)

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=2,
    inputs={'signals': signals_by_date}
)
```

### Use Case 3: Not Supported

Stop loss is **NOT available** for Use Case 3 (risk-managed external trades), as that use case has its own risk management via optimization constraints.

## Examples

### Example 1: Conservative Stop Loss

Gentle reduction to protect against moderate drawdowns:

```python
config = BacktestConfig(
    initial_cash=10_000_000,
    stop_loss_levels=[
        (50000, 0.75),   # At 0.5% DD → 75% gross
        (100000, 0.50),  # At 1.0% DD → 50% gross
    ]
)
```

### Example 2: Aggressive Stop Loss

Quick reduction for tight risk control:

```python
config = BacktestConfig(
    initial_cash=10_000_000,
    stop_loss_levels=[
        (25000, 0.50),   # At 0.25% DD → 50% gross
        (50000, 0.25),   # At 0.50% DD → 25% gross
    ]
)
```

### Example 3: Single Level

Simple binary stop loss:

```python
config = BacktestConfig(
    initial_cash=10_000_000,
    stop_loss_levels=[
        (100000, 0.50),  # At 1% DD → halve exposure
    ]
)
```

## Behavior Example

### Detailed Walkthrough

```python
config = BacktestConfig(
    initial_cash=10_000_000,
    stop_loss_levels=[(50000, 0.75), (100000, 0.50)]
)
```

**Scenario**:

| Day | Portfolio Value | Drawdown | Peak | Level | Gross % | Explanation |
|-----|----------------|----------|------|-------|---------|-------------|
| 1 | $10,000,000 | $0 | $10,000,000 | None | 100% | Starting point |
| 2 | $10,050,000 | $0 | $10,050,000 | None | 100% | New peak |
| 3 | $9,980,000 | $70,000 | $10,050,000 | Level 1 | 75% | DD >= $50k → Enter L1 |
| 4 | $9,930,000 | $120,000 | $10,050,000 | Level 2 | 50% | DD >= $100k → Enter L2 |
| 5 | $9,960,000 | $90,000 | $10,050,000 | None | 100% | DD < $100k → Clear all |
| 6 | $10,020,000 | $30,000 | $10,050,000 | None | 100% | No stop loss |
| 7 | $10,100,000 | $0 | $10,100,000 | None | 100% | New peak → fully cleared |

**Key observations**:
- Day 4: Entered deeper level as DD worsened
- Day 5: **Jumped directly to 100%** when DD improved below $100k (didn't scale through 75%)
- Day 7: New peak automatically cleared everything

## Advanced Topics

### Choosing Thresholds

**Percentage-based approach**:
```python
initial_cash = 10_000_000

# 0.5%, 1.0%, 2.0% drawdowns
thresholds = [
    (initial_cash * 0.005, 0.75),
    (initial_cash * 0.010, 0.50),
    (initial_cash * 0.020, 0.25),
]

config = BacktestConfig(
    initial_cash=initial_cash,
    stop_loss_levels=thresholds
)
```

**Volatility-based approach**:
```python
# Set based on expected daily volatility
daily_vol_dollars = 50000  # Expected daily P&L volatility

config = BacktestConfig(
    stop_loss_levels=[
        (2 * daily_vol_dollars, 0.75),   # 2 sigma
        (3 * daily_vol_dollars, 0.50),   # 3 sigma
    ]
)
```

### Testing Stop Loss Effectiveness

Use the production functions to analyze historical behavior:

```python
from backtesting.stop_loss_production import (
    calculate_stop_loss_gross,
    calculate_stop_loss_metrics
)
import pandas as pd

# Load historical PnL
daily_pnl = pd.read_csv('historical_pnl.csv', index_col=0, parse_dates=True)

# Test different configurations
levels_1 = [(50000, 0.75), (100000, 0.50)]
levels_2 = [(25000, 0.50), (50000, 0.25)]

# Calculate gross multipliers
gross_1 = calculate_stop_loss_gross(daily_pnl, levels_1, initial_capital=10_000_000)
gross_2 = calculate_stop_loss_gross(daily_pnl, levels_2, initial_capital=10_000_000)

# Get detailed metrics
metrics_1 = calculate_stop_loss_metrics(daily_pnl, levels_1, initial_capital=10_000_000)
metrics_2 = calculate_stop_loss_metrics(daily_pnl, levels_2, initial_capital=10_000_000)

# Analyze: days at each level, max DD avoided, etc.
print(f"Config 1: Days with stop loss = {(gross_1 < 1.0).sum()}")
print(f"Config 2: Days with stop loss = {(gross_2 < 1.0).sum()}")
```

### Monitoring Stop Loss Activity

During backtesting, the system prints notifications when stop loss levels change:

```
============================================================
STOP LOSS TRIGGERED - Level 1
============================================================
Peak value: $10,050,000.00
Current value: $9,980,000.00
Dollar drawdown: $70,000.00
Threshold: $50,000.00
Reducing gross exposure to 75.0%
============================================================

============================================================
STOP LOSS CLEARED - Recovery
============================================================
Peak value: $10,050,000.00
Current value: $9,960,000.00
Dollar drawdown: $90,000.00
Restoring full gross exposure (100%)
============================================================
```

## Production Functions

### calculate_stop_loss_gross()

Calculate gross exposure multipliers for a PnL series:

```python
from backtesting.stop_loss_production import calculate_stop_loss_gross

gross_multipliers = calculate_stop_loss_gross(
    daily_pnl=daily_pnl_series,
    stop_loss_levels=[(50000, 0.75), (100000, 0.50)],
    initial_capital=10_000_000,
    dates=None  # Optional: provide if daily_pnl is numpy array
)

# Result: pandas Series with gross multipliers (1.0 = no reduction)
```

### calculate_stop_loss_metrics()

Get detailed stop loss metrics:

```python
from backtesting.stop_loss_production import calculate_stop_loss_metrics

metrics = calculate_stop_loss_metrics(
    daily_pnl=daily_pnl_series,
    stop_loss_levels=[(50000, 0.75), (100000, 0.50)],
    initial_capital=10_000_000
)

# Result: DataFrame with columns:
# - portfolio_value
# - peak_value
# - drawdown_dollar
# - triggered_level (None if no stop loss, 0/1/2/... for level index)
# - gross_multiplier
```

## Best Practices

1. **Backtest without stop loss first**: Understand baseline strategy behavior

2. **Set thresholds relative to volatility**: Avoid triggers from normal market noise

3. **Consider transaction costs**: Frequent entry/exit can erode returns

4. **Test multiple configurations**: Find optimal balance between protection and false positives

5. **Monitor in production**: Track how often stop loss activates

6. **Combine with other risk controls**: Stop loss complements (not replaces) position sizing, diversification, and hedging

## Limitations

1. **No gradual scaling**: Recovery jumps directly to 100% gross (no intermediate levels)

2. **Same thresholds all time**: Cannot adapt thresholds based on market regime

3. **No time-based logic**: Cannot distinguish quick vs. slow drawdowns

4. **Use Cases 1 & 2 only**: Not available for Use Case 3

5. **Position-level application**: Applies uniform multiplier to all positions (no selective reduction)

## Migration from Legacy System

**Old system (3-tuple with recovery_drawdown):**
```python
# No longer supported
stop_loss_levels=[(5000, 0.75, 2000)]  # ❌ 3-tuple format removed
```

**New system (2-tuple only):**
```python
# Current format
stop_loss_levels=[(5000, 0.75)]  # ✅ 2-tuple only
```

**Behavior change:**
- Old: Stayed at level until DD improved to recovery_drawdown (sticky)
- New: Exits immediately when DD improves below drawdown_threshold

## FAQs

**Q: What happens if I have overlapping thresholds?**
A: The system always uses the deepest (highest threshold) level that DD exceeds.

**Q: Can I disable stop loss for specific days?**
A: No, but you can set very high thresholds to effectively disable.

**Q: Does stop loss affect hedging?**
A: Stop loss is applied AFTER hedging, so hedge positions are also scaled.

**Q: Can I have different stop loss for longs vs. shorts?**
A: No, the gross reduction applies uniformly to all positions.

**Q: What if my strategy is short-biased?**
A: Stop loss still works - it tracks portfolio value drawdown regardless of long/short mix.

**Q: Does stop loss look ahead?**
A: No, it uses end-of-day portfolio values only. No intraday or look-ahead bias.

## See Also

- [BacktestConfig Documentation](../backtesting/config.py)
- [Risk Management Guide](RISK_MANAGEMENT.md) (if exists)
- [Transaction Costs](TRANSACTION_COSTS.md) (if exists)
- Example notebooks: `notebooks/stop_loss_examples.ipynb`

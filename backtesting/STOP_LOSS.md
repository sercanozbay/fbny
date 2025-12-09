# Stop Loss Functionality

## Overview

The stop loss system provides automatic risk management by reducing gross exposure when the portfolio experiences dollar-based drawdowns. It features **early recovery logic** that scales up positions before full recovery to recoup losses faster.

## Key Concepts

### Drawdown
- Measured in **dollars** from the peak portfolio value
- Both entry and exit thresholds use drawdown from peak for consistency
- Example: If peak is $100,000 and current value is $95,000, drawdown is $5,000

### Early Recovery Logic

**Philosophy: Scale up early during recovery to recoup losses faster**

Each level has two thresholds:
- **Entry threshold (drawdown_threshold)**: Dollar DD from peak that triggers entry
- **Recovery threshold (recovery_drawdown)**: Dollar DD from peak that triggers exit (scale up)

**Key insight**: recovery_drawdown > drawdown_threshold means you exit while still in a worse drawdown state, allowing you to scale up early and accelerate recovery.

### Simple, Stateless Logic

Unlike traditional "sticky" stop loss systems, this implementation is **stateless**: it simply looks at the current drawdown and determines the appropriate level. No complex bracket logic or state tracking needed.

**How it works:**
1. Start from the deepest level
2. Check if current DD is in the level's range: `drawdown_threshold <= DD < recovery_drawdown`
3. Use the first level that matches
4. If no level matches, use 100% gross

### Examples

**Immediate scale up (no recovery threshold):**
```python
levels = [(5000, 0.75), (10000, 0.50)]
# When recovery_drawdown is None, it defaults to drawdown_threshold
# This means immediate scale up when DD improves below entry threshold
```
- DD=$12k → 50% gross (DD ≥ $10k)
- DD=$9k → 75% gross (DD < $10k, DD ≥ $5k) - immediate scale up!
- DD=$4k → 100% gross (DD < $5k) - immediate scale up!

**Early recovery (recommended):**
```python
levels = [(5000, 0.75, 7500), (10000, 0.50, 15000)]
# recovery_drawdown > drawdown_threshold for early scale up
```
- DD=$15k → 50% gross (DD ≥ $10k, DD ≥ $15k)
- DD=$12k → 75% gross (DD < $15k → exit Level 2, DD ≥ $5k → enter Level 1)
- DD=$6k → 100% gross (DD < $7.5k → exit Level 1) - **scaled up at $6k DD even though entry was $5k!**

## Configuration Formats

All thresholds are dollar-based for simplicity.

### 2-Tuple Format (Immediate Scale Up)
```python
stop_loss_levels = [
    (5000, 0.75),   # Enter at $5k DD → 75% gross, exit when DD < $5k (immediate)
    (10000, 0.50),  # Enter at $10k DD → 50% gross, exit when DD < $10k (immediate)
]
```
When recovery_drawdown is not specified, it defaults to drawdown_threshold, providing immediate scale up.

### 3-Tuple Format (Early Recovery - RECOMMENDED)
```python
stop_loss_levels = [
    (5000, 0.75, 7500),   # Enter at $5k DD, exit at $7.5k DD → 75% gross
    (10000, 0.50, 15000), # Enter at $10k DD, exit at $15k DD → 50% gross
]
```

**Important**: recovery_drawdown > drawdown_threshold enables early recovery. This means you scale up while still in a worse drawdown state to accelerate recovery.

## Example Usage

```python
from backtesting import BacktestConfig

config = BacktestConfig(
    initial_cash=100000.0,
    stop_loss_levels=[
        (5000, 0.75, 7500),    # Level 1: Enter $5k DD, Exit $7.5k DD → 75% gross
        (10000, 0.50, 15000),  # Level 2: Enter $10k DD, Exit $15k DD → 50% gross
    ]
)
```

## Detailed Example Scenario (Early Recovery)

Using levels: [(5000, 0.75, 7500), (10000, 0.50, 15000)]

Starting with $100,000 portfolio:

| Portfolio | Drawdown | Level | Gross | Explanation |
|-----------|----------|-------|-------|-------------|
| $100,000  | $0       | None  | 100%  | Peak |
| $94,000   | $6,000   | 1     | 75%   | **Entered Level 1** (DD=$6k ≥ $5k, DD < $7.5k exit) |
| $95,000   | $5,000   | 1     | 75%   | **Stays Level 1** (DD ≥ $5k, DD < $7.5k) |
| $93,000   | $7,000   | 1     | 75%   | **Stays Level 1** (DD ≥ $5k, DD < $7.5k) |
| $85,000   | $15,000  | 2     | 50%   | **Entered Level 2** (DD=$15k ≥ $10k, DD ≥ $15k exit) |
| $88,000   | $12,000  | 1     | 75%   | **Early exit Level 2!** (DD < $15k), enters Level 1 |
| $86,000   | $14,000  | 1     | 75%   | **Stays Level 1** (DD < $15k, DD ≥ $7.5k) |
| $94,000   | $6,000   | None  | 100%  | **Early exit Level 1!** (DD < $7.5k), full recovery |
| $96,000   | $4,000   | None  | 100%  | **Stays at 100%** (DD < $5k) |
| $102,000  | $0       | None  | 100%  | **New peak**, stop loss cleared |

### Key Observations

1. **Entry**: When DD crosses entry threshold ($5k, $10k), you enter that level
2. **Early Recovery**: You exit levels before DD improves back to entry threshold
   - Exited Level 2 at DD=$12k (before DD improved to $10k entry)
   - Exited Level 1 at DD=$6k (before DD improved to $5k entry)
3. **Faster Recovery**: By scaling up early, you can recoup losses faster
4. **Simple Logic**: Just check current DD against all levels, no state tracking needed
5. **No Bouncing**: DD can fluctuate (row 6-7: $12k → $14k) without changing levels

## Important Rules

1. **Dollar-Based Drawdowns**: All thresholds (entry and exit) are specified as dollar drawdown from peak. This makes it easy to understand and configure based on your portfolio size.

2. **Early Recovery**: recovery_drawdown > drawdown_threshold enables early scale up. You exit levels while still in a worse drawdown state to accelerate recovery.

3. **Default Recovery**: When recovery_drawdown is None (2-tuple format), it defaults to drawdown_threshold, providing immediate scale up when DD improves.

4. **Simple, Stateless**: The system simply checks current DD against all levels from deepest to shallowest. No complex state tracking or bracket logic needed.

5. **Validation**: recovery_drawdown must be greater than drawdown_threshold. Example: Entry=$5k, Exit=$7.5k is valid. Entry=$5k, Exit=$2k is INVALID.

6. **New Peak Clears All**: Reaching a new portfolio peak clears all stop loss levels and resets to 100% gross.

## Use Case Support

- **Use Case 1 (Direct Targets)**: Stop loss applied ✓
- **Use Case 2 (Signals)**: Stop loss applied ✓
- **Use Case 3 (External Trades)**: Stop loss NOT applied (external trades are fixed)

## Benefits

1. **Simple Configuration**: Dollar-based thresholds are easy to understand and configure
2. **Early Recovery**: Scale up early to recoup losses faster
3. **Stateless Logic**: No complex state tracking or bracket calculations needed
4. **Consistent Semantics**: All thresholds use drawdown from peak
5. **Dynamic Risk Management**: Automatically reduces exposure during drawdowns
6. **Flexible**: Immediate scale up (2-tuple) or early recovery (3-tuple)
7. **Clear Notifications**: Prints detailed messages when levels trigger or clear

## Testing

Comprehensive tests are provided:

1. **test_early_recovery.py** - Tests demonstrating immediate and early recovery logic
2. **example_production_stop_loss.py** - Example usage scripts for production functions

Run the tests:
```bash
python test_early_recovery.py
python example_production_stop_loss.py
```

### What the Tests Demonstrate

- **Immediate scale up**: When recovery_drawdown is None, scale up immediately when DD improves
- **Early recovery**: When recovery_drawdown > drawdown_threshold, scale up early
- **No bouncing**: Prevents rapid level changes from DD fluctuations
- **New peak clearing**: Reaching new peak clears all stop loss
- **Multiple levels**: Correct behavior with 2+ cascading levels
- **Input validation**: Ensures recovery_drawdown > drawdown_threshold (or None)

## Implementation Details

### StopLossLevel Dataclass
```python
@dataclass
class StopLossLevel:
    drawdown_threshold: float        # Dollar drawdown from peak to ENTER this level
    gross_reduction: float           # Target gross as percentage (0-1)
    recovery_drawdown: Optional[float] = None  # Dollar drawdown from peak to EXIT this level
                                                # Must be > drawdown_threshold for early recovery
                                                # Defaults to drawdown_threshold if None
```

### StopLossManager State
- `peak_value`: Highest portfolio value reached
- `current_drawdown_dollar`: Current dollar drawdown from peak
- `triggered_level`: Currently active stop loss level (None if cleared)
- `current_gross_multiplier`: Current multiplier to apply to positions

### Update Logic (Simple & Stateless)

**Algorithm:**
1. If portfolio > peak → Update peak, clear all levels, return 100%
2. Calculate current DD from peak
3. For each level from deepest to shallowest:
   - Get recovery_dd (defaults to drawdown_threshold if None)
   - If DD >= drawdown_threshold AND DD >= recovery_dd → Use this level
4. If no level matches → Return 100%

**Key insight:** This is completely stateless - just check current DD against all levels. No complex bracket logic or state tracking needed!

**Example:**
- Level 1: Entry=$5k, Recovery=$7.5k
- DD=$6k: Checks Level 1: $6k >= $5k? YES. $6k >= $7.5k? NO → Don't use Level 1 → 100%
- DD=$8k: Checks Level 1: $8k >= $5k? YES. $8k >= $7.5k? YES → Use Level 1 (75%)

## Best Practices

1. **Set Levels Based on Portfolio Size**: For a $100,000 portfolio, levels like $5k, $10k, $15k are reasonable. Scale proportionally for larger portfolios.

2. **Use Decreasing Gross Reductions**: Each level should have a lower gross reduction than the previous level (e.g., 75%, 50%, 25%).

3. **Early Recovery for Faster Rebound**: Set recovery_drawdown > drawdown_threshold to scale up early. Example: Entry=$5k, Exit=$7.5k means you scale up at $7.5k DD (still worse than entry) to accelerate recovery.

4. **Tune Aggressiveness**:
   - Conservative: recovery_drawdown slightly > drawdown_threshold (e.g., Entry=$10k, Exit=$12k)
   - Aggressive: recovery_drawdown much > drawdown_threshold (e.g., Entry=$10k, Exit=$20k)
   - Immediate: Use 2-tuple format for instant scale up when DD improves

5. **Test Before Using**: Run backtests with your stop loss configuration to ensure it behaves as expected for your strategy.

6. **Monitor Notifications**: The system prints clear messages when stop loss levels trigger or clear, making it easy to understand what's happening during a backtest.

## Limitations

1. **Dollar-Based Only**: The system does not support percentage-based thresholds. This is by design for simplicity.

2. **Early Recovery Required for Speed**: To scale up before full recovery, you must use 3-tuple format with recovery_drawdown > drawdown_threshold. The 2-tuple format provides immediate (not early) scale up.

3. **Use Case 3 Exclusion**: External trades bypass stop loss (by design, since they are pre-determined).

4. **Daily Frequency**: Stop loss is evaluated once per day based on end-of-day portfolio values.

## Production Functions

For live trading or post-hoc analysis, you can use standalone production functions that calculate stop loss levels from a daily PnL time series.

### calculate_stop_loss_gross

Calculate gross exposure multipliers from daily PnL using early recovery logic.

```python
from backtesting import calculate_stop_loss_gross
import pandas as pd

# Your daily PnL series
daily_pnl = pd.Series([0, -15000, 3000, -2000, 8000, 4000],
                       index=pd.date_range('2023-01-01', periods=6))

# Define stop loss levels with early recovery
levels = [
    (5000, 0.75, 7500),   # Enter at $5k DD, Exit at $7.5k DD (early!)
    (10000, 0.50, 15000), # Enter at $10k DD, Exit at $15k DD (early!)
]

# Calculate gross multipliers
gross_multipliers = calculate_stop_loss_gross(
    daily_pnl=daily_pnl,
    stop_loss_levels=levels,
    initial_capital=100000
)

print(gross_multipliers)
# 2023-01-01    1.00  # $0 DD
# 2023-01-02    0.50  # $15k DD → Enter Level 2
# 2023-01-03    0.75  # $12k DD → Exit Level 2 (early!), Enter Level 1
# 2023-01-04    0.75  # $14k DD → Stay Level 1
# 2023-01-05    1.00  # $6k DD → Exit Level 1 (early!)
# 2023-01-06    1.00  # $2k DD → Stay at 100%
# dtype: float64
```

### calculate_stop_loss_metrics

Get detailed metrics including drawdowns, triggered levels, and gross multipliers.

```python
from backtesting import calculate_stop_loss_metrics

metrics = calculate_stop_loss_metrics(
    daily_pnl=daily_pnl,
    stop_loss_levels=levels,
    initial_capital=100000
)

print(metrics)
#             portfolio_value  peak_value  drawdown_dollar triggered_level  gross_multiplier
# 2023-01-01           100000      100000                0            None              1.00
# 2023-01-02            85000      100000            15000               1              0.50
# 2023-01-03            88000      100000            12000               0              0.75
# 2023-01-04            86000      100000            14000               0              0.75
# 2023-01-05            94000      100000             6000            None              1.00
# 2023-01-06            98000      100000             2000            None              1.00
```

### Input Formats

Both functions accept:
- **pd.Series**: Daily PnL with datetime index (recommended)
- **np.ndarray**: Daily PnL as array (requires `dates` parameter)

```python
import numpy as np

# Using numpy array
daily_pnl_array = np.array([0, -500, -800, 300, 400])
dates = pd.date_range('2023-01-01', periods=5)

gross = calculate_stop_loss_gross(
    daily_pnl=daily_pnl_array,
    stop_loss_levels=levels,
    initial_capital=100000,
    dates=dates  # Required for numpy arrays
)
```

### Use Cases

1. **Live Trading**: Calculate position sizes based on current day's gross multiplier
2. **Post-Trade Analysis**: Analyze how stop loss would have affected historical PnL
3. **Risk Monitoring**: Track when stop loss levels trigger in real-time
4. **Strategy Development**: Test different stop loss configurations on historical data

### Example: Live Trading Integration

```python
import pandas as pd
from backtesting import calculate_stop_loss_gross

class TradingSystem:
    def __init__(self, initial_capital, stop_loss_levels):
        self.initial_capital = initial_capital
        self.stop_loss_levels = stop_loss_levels
        self.daily_pnl = pd.Series(dtype=float)
    
    def record_daily_pnl(self, date, pnl):
        """Record today's PnL."""
        self.daily_pnl[date] = pnl
    
    def get_current_gross_multiplier(self):
        """Get today's gross exposure multiplier."""
        if len(self.daily_pnl) == 0:
            return 1.0
        
        gross_series = calculate_stop_loss_gross(
            daily_pnl=self.daily_pnl,
            stop_loss_levels=self.stop_loss_levels,
            initial_capital=self.initial_capital
        )
        return gross_series.iloc[-1]
    
    def calculate_position_sizes(self, target_positions):
        """Apply stop loss to target positions."""
        multiplier = self.get_current_gross_multiplier()
        return {ticker: qty * multiplier 
                for ticker, qty in target_positions.items()}

# Usage
system = TradingSystem(
    initial_capital=100000,
    stop_loss_levels=[
        (5000, 0.75, 2500),
        (10000, 0.50, 5000),
    ]
)

# Each day:
system.record_daily_pnl('2023-01-01', -800)
target_positions = {'AAPL': 100, 'GOOGL': 50}
actual_positions = system.calculate_position_sizes(target_positions)
print(f"Target: {target_positions}")
print(f"Actual: {actual_positions}")
print(f"Multiplier: {system.get_current_gross_multiplier():.2%}")
```

### Performance Considerations

- Both functions are optimized for vectorized pandas operations
- Suitable for daily frequency data (up to thousands of days)
- For intraday data, consider resampling to daily first
- Memory usage is O(n) where n is the number of days


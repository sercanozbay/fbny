# Stop Loss Functionality

## Overview

The stop loss system provides automatic risk management by reducing gross exposure when the portfolio experiences dollar-based drawdowns. It uses **sticky recovery logic** with drawdown-based thresholds to prevent unnecessary fluctuations in position sizes.

## Key Concepts

### Drawdown
- Measured in **dollars** from the peak portfolio value
- Both entry and exit thresholds use drawdown from peak for consistency
- Example: If peak is $100,000 and current value is $95,000, drawdown is $5,000

### Sticky Recovery Logic

**What is "Sticky"?**
Once you enter a stop loss level, you **STAY** at that level until:
1. Drawdown improves to ≤ the recovery threshold (bracket-based exit), OR
2. Drawdown worsens and triggers a deeper level, OR
3. Portfolio reaches a new peak (clears all stop loss)

**Key benefit**: The gross multiplier doesn't change just because drawdown fluctuates within the current bracket. This prevents rapid trading from unnecessary level changes.

### Bracket-Based Recovery

Each level defines a "bracket" based on its entry and exit thresholds:

**Example with two levels:**
- Level 1: Enter at $5k DD, Exit at $2k DD → 75% gross
- Level 2: Enter at $10k DD, Exit at $5k DD → 50% gross

**This creates three brackets:**
- Bracket 0 (Clear): [0, $2k) → 100% gross
- Bracket 1: [$2k, $5k) → 75% gross
- Bracket 2: [$5k, $10k) → 50% gross
- Level 3: [$10k, ∞) → 50% gross

**When recovering:**
If at Level 2 with DD=$12k, then DD improves to $4.5k:
- Exit Level 2 because DD=$4.5k ≤ $5k recovery threshold
- Find bracket: DD=$4.5k is in Bracket 1 [$2k, $5k)
- Enter Level 1 (75% gross)

**Sticky behavior:**
If at Level 1 with DD=$4.5k, then DD fluctuates to $4k or $4.8k:
- STAY at Level 1 (75% gross)
- Only exit when DD ≤ $2k or DD ≥ $5k (enters Level 2)

## Configuration Formats

All thresholds are dollar-based for simplicity.

### 2-Tuple Format (No Recovery)
```python
stop_loss_levels = [
    (5000, 0.75),   # $5k loss → 75% gross (no automatic recovery)
    (10000, 0.50),  # $10k loss → 50% gross (no automatic recovery)
]
```

### 3-Tuple Format (With Recovery - RECOMMENDED)
```python
stop_loss_levels = [
    (5000, 0.75, 2000),   # Enter at $5k DD, Exit at $2k DD → 75% gross
    (10000, 0.50, 5000),  # Enter at $10k DD, Exit at $5k DD → 50% gross
]
```

**Important**: The third value is the recovery_drawdown (exit threshold), measured as drawdown from peak, NOT recovery from trough.

## Example Usage

```python
from backtesting import BacktestConfig

config = BacktestConfig(
    initial_cash=100000.0,
    stop_loss_levels=[
        (5000, 0.75, 2000),    # Level 1: Enter $5k DD, Exit $2k DD → 75% gross
        (10000, 0.50, 5000),   # Level 2: Enter $10k DD, Exit $5k DD → 50% gross
        (15000, 0.25, 10000),  # Level 3: Enter $15k DD, Exit $10k DD → 25% gross
    ]
)
```

## Detailed Example Scenario (Sticky Logic)

Using levels: [(5000, 0.75, 2000), (10000, 0.50, 5000)]

Starting with $100,000 portfolio:

| Portfolio | Drawdown | Level | Gross | Explanation |
|-----------|----------|-------|-------|-------------|
| $100,000  | $0       | None  | 100%  | Peak |
| $94,000   | $6,000   | 1     | 75%   | **Entered Level 1** (DD=$6k ≥ $5k entry) |
| $95,000   | $5,000   | 1     | 75%   | **Sticky** (DD in bracket [$2k, $5k)) |
| $96,000   | $4,000   | 1     | 75%   | **Sticky** (DD in bracket [$2k, $5k)) |
| $94,500   | $5,500   | 1     | 75%   | **Sticky** (DD in bracket [$2k, $5k)) |
| $89,000   | $11,000  | 2     | 50%   | **Entered Level 2** (DD=$11k ≥ $10k entry) |
| $90,000   | $10,000  | 2     | 50%   | **Sticky** (DD in bracket [$5k, $10k)) |
| $93,000   | $7,000   | 2     | 50%   | **Sticky** (DD in bracket [$5k, $10k)) |
| $95,500   | $4,500   | 1     | 75%   | **Exited Level 2** (DD≤$5k), **Bracket 1** |
| $96,000   | $4,000   | 1     | 75%   | **Sticky** at Level 1 |
| $98,500   | $1,500   | None  | 100%  | **Exited Level 1** (DD≤$2k), **Bracket 0** |
| $99,000   | $1,000   | None  | 100%  | **Sticky** at 100% (no level) |
| $102,000  | $0       | None  | 100%  | **New peak**, stop loss cleared |

### Key Observations

1. **Entry**: When DD crosses entry threshold ($5k, $10k), you enter that level
2. **Sticky**: Once at a level, you STAY there even as DD fluctuates within the bracket
3. **Exit**: When DD improves past exit threshold ($2k, $5k), you exit to the bracket below
4. **No bouncing**: On row 4-5, DD goes from $4k to $5.5k, but stays at 75% (sticky!)
5. **Bracket-based**: On row 9, DD=$4.5k enters Bracket 1 [$2k, $5k), so 75% gross

## Important Rules

1. **Dollar-Based Drawdowns**: All thresholds (entry and exit) are specified as dollar drawdown from peak. This makes it easy to understand and configure based on your portfolio size.

2. **Sticky Behavior**: Once at a level, you STAY there until explicitly crossing an exit or entry threshold. Fluctuations within a bracket don't change the level.

3. **Bracket-Based Recovery**: When exiting a level, the system finds which bracket the current drawdown falls into and enters that level. This ensures smooth transitions.

4. **Recovery Must Be Less Than Entry**: The recovery_drawdown (exit threshold) must be less than drawdown_threshold (entry threshold) for each level. Example: Entry=$5k, Exit=$2k is valid.

5. **Optional Recovery**: Recovery thresholds are optional (2-tuple format). If not specified, the stop loss remains at the triggered level until a new peak is reached.

6. **New Peak Clears All**: Reaching a new portfolio peak clears all stop loss levels and resets to 100% gross.

## Use Case Support

- **Use Case 1 (Direct Targets)**: Stop loss applied ✓
- **Use Case 2 (Signals)**: Stop loss applied ✓
- **Use Case 3 (External Trades)**: Stop loss NOT applied (external trades are fixed)

## Benefits

1. **Simple Configuration**: Dollar-based thresholds are easy to understand and configure
2. **Sticky Logic**: Prevents unnecessary trading from rapid level changes within a bracket
3. **Consistent Semantics**: All thresholds use drawdown from peak (not mixed with trough recovery)
4. **Bracket-Based Recovery**: Smooth transitions between levels based on which bracket you're in
5. **Dynamic Risk Management**: Automatically reduces exposure during drawdowns
6. **Clear Notifications**: Prints detailed messages when levels trigger or clear
7. **Flexible**: Supports multiple cascading levels with different thresholds

## Testing

Comprehensive tests are provided:

1. **test_sticky_recovery.py** - Backtester integration test demonstrating sticky logic
2. **test_production_sticky_recovery.py** - Production function tests with 7 comprehensive scenarios
3. **example_production_stop_loss.py** - Example usage scripts for production functions

Run the tests:
```bash
python test_sticky_recovery.py
python test_production_sticky_recovery.py
python example_production_stop_loss.py
```

### What the Tests Demonstrate

- **Sticky behavior**: Levels don't change when DD fluctuates within a bracket
- **Bracket-based recovery**: Proper transitions when exiting levels
- **No bouncing**: Prevents rapid level changes from small DD fluctuations
- **New peak clearing**: Reaching new peak clears all stop loss
- **Multiple levels**: Correct behavior with 2+ cascading levels
- **Input validation**: Ensures recovery_drawdown < drawdown_threshold

## Implementation Details

### StopLossLevel Dataclass
```python
@dataclass
class StopLossLevel:
    drawdown_threshold: float        # Dollar drawdown from peak to ENTER this level
    gross_reduction: float           # Target gross as percentage (0-1)
    recovery_drawdown: Optional[float] = None  # Dollar drawdown from peak to EXIT this level
```

### StopLossManager State
- `peak_value`: Highest portfolio value reached
- `current_drawdown_dollar`: Current dollar drawdown from peak
- `triggered_level`: Currently active stop loss level (None if cleared)
- `current_gross_multiplier`: Current multiplier to apply to positions

### Update Logic (Sticky)

**When NOT at any level:**
1. Check if current DD exceeds any entry threshold → Enter that level

**When at a level (STICKY):**
1. Check if DD ≤ recovery_drawdown → Exit current level
   - Find which bracket current DD falls into
   - Enter that bracket's level (bracket-based recovery)
2. Check if DD exceeds deeper level's entry threshold → Enter deeper level
3. Otherwise → STAY at current level (sticky!)

**Special case:**
- New peak (portfolio > peak_value) → Clear all levels, reset to 100%

This sticky logic prevents unnecessary level changes from DD fluctuations within a bracket.

## Best Practices

1. **Set Levels Based on Portfolio Size**: For a $100,000 portfolio, levels like $5k, $10k, $15k are reasonable. Scale proportionally for larger portfolios.

2. **Use Decreasing Gross Reductions**: Each level should have a lower gross reduction than the previous level (e.g., 75%, 50%, 25%).

3. **Set Recovery Below Entry**: Recovery_drawdown (exit threshold) should be meaningfully below drawdown_threshold (entry threshold) to create clear brackets. Example: Entry=$10k, Exit=$5k creates a $5k-$10k bracket.

4. **Non-Overlapping Brackets**: Set recovery thresholds to match the next level's entry threshold for clean transitions. Example:
   - Level 1: Entry=$5k, Exit=$2k
   - Level 2: Entry=$10k, Exit=$5k (← matches Level 1 exit)

5. **Test Before Using**: Run backtests with your stop loss configuration to ensure it behaves as expected for your strategy.

6. **Monitor Notifications**: The system prints clear messages when stop loss levels trigger or clear, making it easy to understand what's happening during a backtest.

## Limitations

1. **Dollar-Based Only**: The system does not support percentage-based thresholds. This is by design for simplicity.

2. **Bracket-Based Recovery**: When exiting a level, you enter the bracket your DD falls into. You cannot skip brackets.

3. **Use Case 3 Exclusion**: External trades bypass stop loss (by design, since they are pre-determined).

4. **Daily Frequency**: Stop loss is evaluated once per day based on end-of-day portfolio values.

## Production Functions

For live trading or post-hoc analysis, you can use standalone production functions that calculate stop loss levels from a daily PnL time series.

### calculate_stop_loss_gross

Calculate gross exposure multipliers from daily PnL using sticky recovery logic.

```python
from backtesting import calculate_stop_loss_gross
import pandas as pd

# Your daily PnL series
daily_pnl = pd.Series([0, -6000, 1000, 1000, 2500],
                       index=pd.date_range('2023-01-01', periods=5))

# Define stop loss levels with sticky recovery
levels = [
    (5000, 0.75, 2000),   # Enter at $5k DD, Exit at $2k DD
    (10000, 0.50, 5000),  # Enter at $10k DD, Exit at $5k DD
]

# Calculate gross multipliers
gross_multipliers = calculate_stop_loss_gross(
    daily_pnl=daily_pnl,
    stop_loss_levels=levels,
    initial_capital=100000
)

print(gross_multipliers)
# 2023-01-01    1.00  # $0 DD
# 2023-01-02    0.75  # $6k DD → Enter Level 1
# 2023-01-03    0.75  # $5k DD → STICKY (in bracket)
# 2023-01-04    0.75  # $4k DD → STICKY (in bracket)
# 2023-01-05    1.00  # $1.5k DD → Exit Level 1
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
# 2023-01-02            94000      100000             6000               0              0.75
# 2023-01-03            95000      100000             5000               0              0.75
# 2023-01-04            96000      100000             4000               0              0.75
# 2023-01-05            98500      100000             1500            None              1.00
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


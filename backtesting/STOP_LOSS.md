# Stop Loss Functionality

## Overview

The stop loss system provides automatic risk management by reducing gross exposure when the portfolio experiences dollar-based drawdowns. It also supports **recovery thresholds** that automatically restore exposure as the portfolio recovers from drawdowns.

## Key Concepts

### Drawdown
- Measured in **dollars** from the peak portfolio value
- Triggers stop loss levels when specified dollar thresholds are exceeded
- Example: If peak is $100,000 and current value is $95,000, drawdown is $5,000

### Recovery
- Measured in **dollars** from the trough (lowest point during the current drawdown)
- Moves back to less restrictive stop loss levels when recovery thresholds are met
- Full recovery to a new peak clears the stop loss entirely
- Example: If trough is $85,000 and current value is $92,500, recovery is $7,500

### Trough Tracking
- The trough is the lowest portfolio value reached during a drawdown period
- It resets when a new peak is achieved
- Recovery is measured from this trough value

## Configuration Formats

All thresholds are dollar-based for simplicity.

### 2-Tuple Format (No Recovery)
```python
stop_loss_levels = [
    (5000, 0.75),   # $5k loss → 75% gross (no automatic recovery)
    (10000, 0.50),  # $10k loss → 50% gross (no automatic recovery)
]
```

### 3-Tuple Format (With Recovery)
```python
stop_loss_levels = [
    (5000, 0.75, 2500),   # $5k loss → 75%, recover at $2.5k from bottom
    (10000, 0.50, 5000),  # $10k loss → 50%, recover at $5k from bottom
]
```

## Example Usage

```python
from backtesting import BacktestConfig

config = BacktestConfig(
    initial_cash=100000.0,
    stop_loss_levels=[
        (5000, 0.75, 2500),    # Level 1: $5k loss → 75% gross
        (10000, 0.50, 5000),   # Level 2: $10k loss → 50% gross
        (15000, 0.25, 7500),   # Level 3: $15k loss → 25% gross
    ]
)
```

## Detailed Example Scenario

Starting with $100,000 portfolio:

1. **Portfolio at $100,000** (peak)
2. **Drops to $95,000** ($5k loss) → **Triggers Level 1: 75% gross**
   - System reduces target positions to 75% of normal size
3. **Drops to $90,000** ($10k loss) → **Triggers Level 2: 50% gross**
   - System further reduces to 50% of normal size
4. **Drops to $85,000** ($15k loss) → **Triggers Level 3: 25% gross** (trough)
   - System reduces to 25% of normal size
5. **Recovers to $92,500** ($7,500 recovery from $85k) → **Back to Level 2: 50% gross**
   - Recovery threshold of $7,500 met, moves to less restrictive level
6. **Recovers to $97,500** ($5,000 recovery from $92.5k) → **Back to Level 1: 75% gross**
   - Recovery threshold of $5,000 met, moves to next less restrictive level
7. **Recovers to $100,000** ($2,500 recovery from $97.5k) → **Cleared: 100% gross**
   - Recovery threshold of $2,500 met, stop loss fully cleared
8. **Reaches $102,000** → **New peak established, stop loss remains cleared**

## Important Rules

1. **Dollar-Based Only**: All thresholds are specified in dollars for simplicity. This makes it easy to understand and configure based on your portfolio size.

2. **Recovery Direction**: Recovery always moves to the **previous** (less restrictive) level, or clears the stop loss entirely if recovering from the first level.

3. **Sequential Application**: The system checks levels in order and applies the deepest level that is triggered.

4. **Optional Recovery**: Recovery thresholds are optional. If not specified, the stop loss will remain at the triggered level until a new peak is reached.

## Use Case Support

- **Use Case 1 (Direct Targets)**: Stop loss applied ✓
- **Use Case 2 (Signals)**: Stop loss applied ✓
- **Use Case 3 (External Trades)**: Stop loss NOT applied (external trades are fixed)

## Benefits

1. **Simple Configuration**: Dollar-based thresholds are easy to understand and configure
2. **Dynamic Risk Management**: Automatically reduces exposure during drawdowns
3. **Gradual Recovery**: Incrementally restores exposure as portfolio recovers
4. **Clear Notifications**: Prints detailed messages when levels trigger or clear
5. **Flexible**: Supports multiple cascading levels with different thresholds

## Testing

Comprehensive tests are provided:

1. **test_stop_loss.py** - Basic stop loss functionality tests
2. **test_stop_loss_dollar.py** - Dollar threshold tests (no recovery)
3. **test_stop_loss_trigger.py** - Verification that stop loss actually reduces positions
4. **test_stop_loss_recovery_demo.py** - Demonstrates recovery through all levels

Run all tests:
```bash
python test_stop_loss.py
python test_stop_loss_dollar.py
python test_stop_loss_trigger.py
python test_stop_loss_recovery_demo.py
```

## Implementation Details

### StopLossLevel Dataclass
```python
@dataclass
class StopLossLevel:
    drawdown_threshold: float        # Dollar loss from peak
    gross_reduction: float           # Target gross as percentage (0-1)
    recovery_threshold: Optional[float] = None  # Dollar recovery from trough
```

### StopLossManager State
- `peak_value`: Highest portfolio value reached
- `trough_value`: Lowest value during current drawdown
- `current_drawdown_dollar`: Current dollar loss from peak
- `current_recovery_dollar`: Current dollar recovery from trough
- `triggered_level`: Currently active stop loss level (None if cleared)
- `current_gross_multiplier`: Current multiplier to apply to positions

### Update Logic
1. Track peak and trough values
2. Calculate current drawdown and recovery in dollars
3. Check if deeper level should be triggered (going down)
4. Check if recovery threshold is met to move to shallower level (going up)
5. Update gross multiplier based on active level
6. Print notifications when levels change

## Best Practices

1. **Set Levels Based on Portfolio Size**: For a $100,000 portfolio, levels like $5k, $10k, $15k are reasonable. Scale proportionally for larger portfolios.

2. **Use Decreasing Gross Reductions**: Each level should have a lower gross reduction than the previous level (e.g., 75%, 50%, 25%).

3. **Set Recovery at 50%**: A common approach is to set recovery thresholds at about 50% of the drawdown amount for each level.

4. **Test Before Using**: Run backtests with your stop loss configuration to ensure it behaves as expected for your strategy.

5. **Monitor Notifications**: The system prints clear messages when stop loss levels trigger or clear, making it easy to understand what's happening during a backtest.

## Limitations

1. **Dollar-Based Only**: The system does not support percentage-based thresholds. This is by design for simplicity.

2. **Sequential Recovery**: Recovery moves through levels one at a time, not directly from level 3 to level 1.

3. **No Partial Recovery**: Recovery requires reaching the full threshold to trigger level changes.

4. **Use Case 3 Exclusion**: External trades bypass stop loss (by design, since they are pre-determined).

5. **Daily Frequency**: Stop loss is evaluated once per day based on end-of-day portfolio values.

## Production Functions

For live trading or post-hoc analysis, you can use standalone production functions that calculate stop loss levels from a daily PnL time series.

### calculate_stop_loss_gross

Calculate gross exposure multipliers from daily PnL.

```python
from backtesting import calculate_stop_loss_gross
import pandas as pd

# Your daily PnL series
daily_pnl = pd.Series([0, -500, -800, 300, 400],
                       index=pd.date_range('2023-01-01', periods=5))

# Define stop loss levels
levels = [
    (5000, 0.75, 2500),   # $5k loss → 75%, recover at $2.5k
    (10000, 0.50, 5000),  # $10k loss → 50%, recover at $5k
]

# Calculate gross multipliers
gross_multipliers = calculate_stop_loss_gross(
    daily_pnl=daily_pnl,
    stop_loss_levels=levels,
    initial_capital=100000
)

print(gross_multipliers)
# 2023-01-01    1.00
# 2023-01-02    1.00
# 2023-01-03    1.00
# 2023-01-04    1.00
# 2023-01-05    1.00
# dtype: float64
```

### calculate_stop_loss_metrics

Get detailed metrics including drawdowns, recovery, and triggered levels.

```python
from backtesting import calculate_stop_loss_metrics

metrics = calculate_stop_loss_metrics(
    daily_pnl=daily_pnl,
    stop_loss_levels=levels,
    initial_capital=100000
)

print(metrics)
#             portfolio_value  peak_value  trough_value  drawdown_dollar  recovery_dollar triggered_level  gross_multiplier
# 2023-01-01           100000      100000        100000                0                0            None              1.00
# 2023-01-02            99500      100000         99500              500                0            None              1.00
# 2023-01-03            98700      100000         98700             1300                0            None              1.00
# 2023-01-04            99000      100000         98700             1000              300            None              1.00
# 2023-01-05            99400      100000         98700              600              700            None              1.00
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


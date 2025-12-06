# Stop Loss Recovery Functionality

## Overview

The stop loss system now supports **recovery thresholds** that automatically increase gross exposure as the portfolio recovers from drawdowns. This allows for dynamic risk management that reduces exposure during drawdowns and gradually restores it during recoveries.

## Key Concepts

### Drawdown
- Measured from the **peak** portfolio value
- Triggers stop loss levels when thresholds are exceeded

### Recovery
- Measured from the **trough** (lowest point during the current drawdown)
- Moves back to less restrictive stop loss levels when recovery thresholds are met
- Full recovery to a new peak clears the stop loss entirely

### Trough Tracking
- The trough is the lowest portfolio value reached during a drawdown period
- It resets when a new peak is achieved
- Recovery is calculated as: `(current_value - trough) / (peak - trough)`

## Configuration Formats

### 2-Tuple Format (No Recovery)
```python
stop_loss_levels = [
    (0.05, 0.75),  # 5% DD → 75% gross (no automatic recovery)
    (0.10, 0.50),  # 10% DD → 50% gross (no automatic recovery)
]
```

### 3-Tuple Format (Recovery, Percent-based)
```python
stop_loss_levels = [
    (0.05, 0.75, 0.50),  # 5% DD → 75% gross, recover at 50% from bottom
    (0.10, 0.50, 0.50),  # 10% DD → 50% gross, recover at 50% from bottom
]
```

### 4-Tuple Format (Recovery, Dollar-based)
```python
stop_loss_levels = [
    (5000, 0.75, 2500, 'dollar'),   # $5k loss → 75%, recover at $2.5k from bottom
    (10000, 0.50, 5000, 'dollar'),  # $10k loss → 50%, recover at $5k from bottom
]
```

### 3-Tuple Format (Legacy, Dollar-based, No Recovery)
```python
stop_loss_levels = [
    (5000, 0.75, 'dollar'),   # $5k loss → 75% gross (no recovery)
    (10000, 0.50, 'dollar'),  # $10k loss → 50% gross (no recovery)
]
```

## Important Rules

1. **Consistent Threshold Types**: All levels must use the same threshold type (all `'percent'` or all `'dollar'`). Mixed threshold types are not supported.

2. **Recovery Direction**: Recovery always moves to the **previous** (less restrictive) level, or clears the stop loss entirely if recovering from the first level.

3. **Threshold Type Matching**: Recovery thresholds use the same type as drawdown thresholds:
   - Percent drawdown → Percent recovery (e.g., 50% = halfway back from trough to peak)
   - Dollar drawdown → Dollar recovery (e.g., $5,000 = $5,000 recovered from trough)

## Example Scenarios

### Percent-Based Recovery

```python
config = BacktestConfig(
    initial_cash=100000.0,
    stop_loss_levels=[
        (0.05, 0.75, 0.50),  # Level 1
        (0.10, 0.50, 0.50),  # Level 2
        (0.15, 0.25, 0.50),  # Level 3
    ]
)
```

**Scenario:**
1. Portfolio starts at $100,000 (peak)
2. Drops to $95,000 (5% DD) → **Triggers Level 1: 75% gross**
3. Drops to $90,000 (10% DD) → **Triggers Level 2: 50% gross**
4. Drops to $85,000 (15% DD) → **Triggers Level 3: 25% gross** (trough = $85,000)
5. Recovers to $92,500 (50% recovery from $85k) → **Back to Level 2: 50% gross**
6. Recovers to $96,250 (50% recovery from $92.5k) → **Back to Level 1: 75% gross**
7. Recovers to $98,125 (50% recovery from $96.25k) → **Cleared: 100% gross**
8. Reaches $101,000 → **New peak, stop loss fully cleared**

### Dollar-Based Recovery

```python
config = BacktestConfig(
    initial_cash=100000.0,
    stop_loss_levels=[
        (5000, 0.75, 2500, 'dollar'),   # Level 1
        (10000, 0.50, 5000, 'dollar'),  # Level 2
        (15000, 0.25, 7500, 'dollar'),  # Level 3
    ]
)
```

**Scenario:**
1. Portfolio starts at $100,000 (peak)
2. Drops to $95,000 ($5k loss) → **Triggers Level 1: 75% gross**
3. Drops to $90,000 ($10k loss) → **Triggers Level 2: 50% gross**
4. Drops to $85,000 ($15k loss) → **Triggers Level 3: 25% gross** (trough = $85,000)
5. Recovers to $92,500 ($7,500 recovery from $85k) → **Back to Level 2: 50% gross**
6. Recovers to $97,500 ($5,000 recovery from $92.5k) → **Back to Level 1: 75% gross**
7. Recovers to $100,000 ($2,500 recovery from $97.5k) → **Cleared: 100% gross**
8. Reaches $102,000 → **New peak, stop loss fully cleared**

## Backward Compatibility

The system maintains backward compatibility with the old format:
- 2-tuple format works as before (no recovery)
- 3-tuple format with string third element is recognized as the old dollar format without recovery
- All existing tests continue to work

## Testing

Comprehensive tests are provided:

1. **test_stop_loss_recovery.py** - Tests percent-based, dollar-based, and no-recovery scenarios
2. **test_stop_loss_recovery_demo.py** - Demonstrates recovery through all levels with clear output
3. **test_stop_loss.py** - Basic stop loss functionality tests
4. **test_stop_loss_dollar.py** - Dollar threshold tests
5. **test_stop_loss_trigger.py** - Verification that stop loss actually reduces positions

Run all tests:
```bash
python test_stop_loss.py
python test_stop_loss_dollar.py
python test_stop_loss_recovery.py
python test_stop_loss_recovery_demo.py
```

## Implementation Details

### StopLossLevel Dataclass
```python
@dataclass
class StopLossLevel:
    drawdown_threshold: float
    gross_reduction: float
    recovery_threshold: Optional[float] = None
    threshold_type: Literal['percent', 'dollar'] = 'percent'
```

### StopLossManager State
- `peak_value`: Highest portfolio value reached
- `trough_value`: Lowest value during current drawdown
- `current_drawdown_pct`: Current drawdown as percentage
- `current_drawdown_dollar`: Current drawdown in dollars
- `current_recovery_pct`: Current recovery from trough as percentage
- `current_recovery_dollar`: Current recovery from trough in dollars
- `triggered_level`: Currently active stop loss level (None if cleared)

### Update Logic
1. Track peak and trough values
2. Calculate current drawdown and recovery metrics
3. Check if deeper level should be triggered (going down)
4. Check if recovery threshold is met to move to shallower level (going up)
5. Update gross multiplier based on active level
6. Print notifications when levels change

## Use Cases

- **Use Case 1 (Direct Targets)**: Stop loss applied ✓
- **Use Case 2 (Signals)**: Stop loss applied ✓
- **Use Case 3 (External Trades)**: Stop loss NOT applied (external trades are fixed)

## Benefits

1. **Dynamic Risk Management**: Automatically reduces exposure during drawdowns
2. **Gradual Recovery**: Incrementally increases exposure as portfolio recovers
3. **Flexible Configuration**: Supports both percent and dollar-based thresholds
4. **Clear Notifications**: Prints detailed messages when levels trigger or clear
5. **Backward Compatible**: Works with existing configurations

## Limitations

1. **Same Threshold Type**: All levels must use consistent threshold types (percent or dollar)
2. **Sequential Recovery**: Recovery moves through levels one at a time
3. **No Partial Recovery**: Recovery requires reaching the full threshold to trigger
4. **Use Case 3 Exclusion**: External trades bypass stop loss (by design)

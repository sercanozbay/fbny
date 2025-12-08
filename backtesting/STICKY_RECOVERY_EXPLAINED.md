# Sticky Recovery Logic - Stop Loss Redesign

## Problem with Old Implementation

The previous recovery logic had several issues:

1. **Recovery measured from trough**: Recovery was calculated as dollar amount recovered from the lowest point, which kept changing
2. **Inconsistent thresholds**: Entry was based on drawdown from peak, but exit was based on recovery from trough
3. **Non-sticky behavior**: Portfolio could bounce between levels daily even without breaching recovery thresholds
4. **Confusing semantics**: "Recover $2,500 from trough" is less intuitive than "drawdown improves to $2,500 from peak"

## New Implementation: Sticky Recovery

### Core Concept

**All thresholds are now drawdown levels from peak:**
- **Entry threshold**: Drawdown from peak that triggers entering a level
- **Exit threshold**: Drawdown from peak that triggers exiting a level (recovery_drawdown)

### Sticky Behavior

Once you enter a level, you **STAY** at that level until:
1. Drawdown improves to ≤ the recovery_drawdown threshold, OR
2. Drawdown worsens and triggers a deeper level, OR
3. Portfolio reaches a new peak (clears all stop loss)

**Key**: The gross multiplier doesn't change just because drawdown fluctuates within the current bracket.

#### Bracket-Based Recovery

When you exit a level (drawdown ≤ recovery_drawdown), the system determines which bracket you're in:

- **Each level defines a bracket**: [recovery_drawdown, drawdown_threshold)
- **When exiting**, find which bracket the current DD falls into
- **Enter that bracket's level**

Example with Level 1 (Entry=$5k, Exit=$2k) and Level 2 (Entry=$10k, Exit=$5k):
- Bracket 0 (Clear): [0, $2k) → 100% gross
- Bracket 1: [$2k, $5k) → 75% gross
- Bracket 2: [$5k, $10k) → 50% gross
- Level 3: [$10k, ∞) → 50% gross

If at Level 2 with DD=$12k, then DD improves to $4.5k:
- Exit Level 2 (DD=$4.5k ≤ $5k recovery threshold)
- Find bracket: DD=$4.5k is in Bracket 1 [$2k, $5k)
- Enter Level 1 (75% gross)

### Example Configuration

```python
stop_loss_levels = [
    (5000, 0.75, 2000),    # Level 1: Enter at $5k DD, Exit at $2k DD
    (10000, 0.50, 5000),   # Level 2: Enter at $10k DD, Exit at $5k DD
    (15000, 0.25, 10000),  # Level 3: Enter at $15k DD, Exit at $10k DD
]
```

### Detailed Example

Starting portfolio: $100,000

| Portfolio | Drawdown | Level | Gross | Reason |
|-----------|----------|-------|-------|--------|
| $100,000  | $0       | None  | 100%  | Peak |
| $94,000   | $6,000   | 1     | 75%   | **Entered Level 1** ($6k ≥ $5k entry) |
| $95,000   | $5,000   | 1     | 75%   | **Sticky** (DD between $2k exit and $5k entry) |
| $96,000   | $4,000   | 1     | 75%   | **Sticky** (DD between $2k exit and $5k entry) |
| $94,500   | $5,500   | 1     | 75%   | **Sticky** (DD between $2k exit and $5k entry) |
| $89,000   | $11,000  | 2     | 50%   | **Entered Level 2** ($11k ≥ $10k entry) |
| $90,000   | $10,000  | 2     | 50%   | **Sticky** (DD between $5k exit and $10k entry) |
| $93,000   | $7,000   | 2     | 50%   | **Sticky** (DD between $5k exit and $10k entry) |
| $95,500   | $4,500   | 1     | 75%   | **Exited Level 2** ($4.5k ≤ $5k exit), **Entered Level 1** |
| $96,000   | $4,000   | 1     | 75%   | **Sticky** at Level 1 |
| $98,500   | $1,500   | None  | 100%  | **Exited Level 1** ($1.5k ≤ $2k exit) |
| $99,000   | $1,000   | None  | 100%  | **Sticky** at 100% (no level) |
| $102,000  | $0       | None  | 100%  | **New peak**, stop loss cleared |

### Key Benefits

1. **Consistent semantics**: All thresholds are drawdown from peak
2. **Sticky by design**: Gross doesn't fluctuate unnecessarily
3. **Clear brackets**: Each level has entry and exit points
4. **Intuitive**: "Exit when DD improves to $5k" is clearer than "recover $5k from trough"
5. **Hysteresis built-in**: Natural resistance to bouncing between levels

### Code Changes

#### StopLossLevel
```python
@dataclass
class StopLossLevel:
    drawdown_threshold: float      # Entry: DD from peak to enter this level
    gross_reduction: float          # Gross multiplier at this level
    recovery_drawdown: Optional[float]  # Exit: DD from peak to exit this level
```

#### StopLossManager Logic
```python
if currently_at_level:
    if drawdown <= current_level.recovery_drawdown:
        # Exit current level, check if we enter a shallower level
        exit_and_check_shallower_levels()
    elif drawdown >= deeper_level.drawdown_threshold:
        # Enter deeper level
        enter_deeper_level()
    else:
        # STAY at current level (sticky!)
        stay_at_current_level()
```

### Comparison: Old vs New

#### Old Logic (Problematic)
- Entry: $10k drawdown from peak → Enter Level 2 (50%)
- Recovery: $5k recovered from trough → Exit Level 2
- **Issue**: Trough keeps updating, recovery threshold is moving target
- **Result**: Can exit level prematurely or stay too long

#### New Logic (Sticky)
- Entry: $10k drawdown from peak → Enter Level 2 (50%)
- Exit: Drawdown improves to ≤ $5k from peak → Exit Level 2
- **Benefit**: Fixed thresholds, predictable behavior
- **Result**: Stays at level until clear improvement

### Migration Guide

**Old format** (still works for backward compatibility):
```python
[(10000, 0.50, 5000)]  # $10k loss → 50%, recover $5k from trough
```

**New recommended format**:
```python
[(10000, 0.50, 5000)]  # $10k DD → 50%, exit at $5k DD
```

The tuple format is the same, but the **semantics changed**:
- Old: third parameter was "recover $X from trough"
- New: third parameter is "exit when DD ≤ $X from peak"

### Testing

See `test_sticky_recovery.py` for a comprehensive test demonstrating the sticky behavior.

The test shows a portfolio moving through multiple levels and staying at each level appropriately until recovery thresholds are met.

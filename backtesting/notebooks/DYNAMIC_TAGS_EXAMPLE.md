# Dynamic Tag Generation Example

This document explains the dynamic tag generation feature demonstrated in Example 5b of notebook [07_dynamic_trade_generation.ipynb](07_dynamic_trade_generation.ipynb).

## Overview

Dynamic tag generation allows you to assign tags to external trades **during the simulation loop** based on:
- Current market conditions (volatility regime, momentum, etc.)
- Portfolio state (performance, drawdown, exposure)
- Trade characteristics (size, direction, ticker)
- Time-based factors (time of day, day of week, etc.)

This enables sophisticated multi-dimensional attribution analysis.

## Key Concept

Instead of pre-assigning tags to trades, you generate them dynamically:

```python
def generate_trades_with_dynamic_tags(context):
    # Access current state
    recent_vol = calculate_volatility(context['daily_returns'])

    # Determine regime
    regime = 'High Vol' if recent_vol > 0.25 else 'Low Vol'

    # Generate trades with regime-aware tags
    for ticker, qty in calculate_trades(context):
        counterparty = select_counterparty(ticker, qty)
        direction = 'Buy' if qty > 0 else 'Sell'

        # Create hierarchical tag
        tag = f"{regime} / {counterparty} / {direction}"

        trades[ticker] = [{
            'qty': qty,
            'price': context['prices'][ticker],
            'tag': tag  # Dynamic tag
        }]

    return trades
```

## Example: Multi-Dimensional Attribution

The notebook example demonstrates tagging by three dimensions:

1. **Market Regime** - High Vol / Normal Vol / Low Vol
2. **Counterparty** - Goldman Sachs / Morgan Stanley / JPMorgan / Citadel
3. **Trade Direction** - Buy / Sell

Tags are formatted hierarchically: `"High Vol / Goldman Sachs / Buy"`

## Benefits

### 1. Regime-Based Attribution

See which market conditions produce best results:

```python
pnl_summary = results.get_pnl_summary_by_tag()
pnl_summary['regime'] = pnl_summary['tag'].str.split(' / ').str[0]

regime_pnl = pnl_summary.groupby('regime')['total_pnl'].sum()
```

**Insight:** "We make 60% of our PnL in Low Vol regimes with only 30% of trading days"

### 2. Counterparty Performance by Condition

Compare broker performance in different market conditions:

```python
high_vol_tags = pnl_summary[pnl_summary['tag'].str.contains('High Vol')]
counterparty_perf = high_vol_tags.groupby('counterparty')['sharpe'].mean()
```

**Insight:** "Goldman Sachs provides best execution in High Vol, but Morgan Stanley is better in Low Vol"

### 3. Directional Bias Detection

Identify if strategy has directional bias:

```python
direction_pnl = pnl_summary.groupby('direction')['total_pnl'].sum()
```

**Insight:** "Buy trades generate 70% of PnL despite being only 50% of volume"

### 4. Smart Order Routing

Use tags to validate routing logic:

```python
# Tag shows where trade was routed
tag = "High Vol / Goldman Sachs / Buy"  # Large buy routed to GS

# Analyze if routing was optimal
routing_analysis = analyze_execution_quality_by_tag(results)
```

## Use Cases

### Use Case 1: Market Making

Tag by client type and market condition:

```python
def mm_strategy(context):
    regime = get_market_regime(context)

    for client_order in context['orders']:
        client_type = 'Institutional' if order.size > 1000 else 'Retail'

        # Dynamic counterparty selection based on order type
        if client_type == 'Institutional' and regime == 'High Vol':
            counterparty = 'Prime Broker A'
        else:
            counterparty = 'Prime Broker B'

        tag = f"{regime} / {client_type} / {counterparty}"

        trades[ticker] = [{'qty': qty, 'price': px, 'tag': tag}]
```

**Analysis:** Compare PnL by client type across market regimes

### Use Case 2: Flow Trading

Tag by flow direction and urgency:

```python
def flow_strategy(context):
    for order in get_client_orders(context):
        flow = 'Aggressive' if order.is_urgent else 'Passive'
        direction = 'Buy' if order.qty > 0 else 'Sell'
        desk = order.originating_desk

        tag = f"{desk} / {flow} / {direction}"

        trades[ticker] = [{'qty': qty, 'price': px, 'tag': tag}]
```

**Analysis:** Which desks provide most profitable flow?

### Use Case 3: Arbitrage

Tag by strategy type and market conditions:

```python
def arb_strategy(context):
    regime = get_volatility_regime(context)
    spread = calculate_spread(context)

    if spread > threshold:
        strategy = 'Stat Arb' if regime == 'Normal' else 'Vol Arb'
        confidence = 'High' if spread > 2*threshold else 'Medium'

        tag = f"{strategy} / {confidence} / {regime}"

        trades[ticker] = [{'qty': qty, 'price': px, 'tag': tag}]
```

**Analysis:** Compare Sharpe ratios across strategies and regimes

## Advanced Patterns

### Pattern 1: Time-Based Tags

Include time components:

```python
import datetime

def generate_with_time_tags(context):
    hour = context['date'].hour

    if hour < 10:
        session = 'Open'
    elif hour < 15:
        session = 'Midday'
    else:
        session = 'Close'

    day_of_week = context['date'].strftime('%A')

    tag = f"{day_of_week} / {session} / {counterparty}"

    trades[ticker] = [{'qty': qty, 'price': px, 'tag': tag}]
```

### Pattern 2: Performance-Based Tags

Tag based on recent performance:

```python
def performance_aware_tags(context):
    recent_perf = calculate_sharpe(context['daily_returns'][-20:])

    if recent_perf > 2.0:
        performance_tier = 'Hot'
    elif recent_perf < 0:
        performance_tier = 'Cold'
    else:
        performance_tier = 'Neutral'

    tag = f"{performance_tier} / {counterparty} / {strategy}"

    trades[ticker] = [{'qty': qty, 'price': px, 'tag': tag}]
```

### Pattern 3: Risk-Adjusted Tags

Tag by risk metrics:

```python
def risk_adjusted_tags(context):
    portfolio_var = calculate_var(context['portfolio'])
    exposure_pct = context['net_exposure'] / context['portfolio_value']

    if portfolio_var > 0.02:
        risk_level = 'High Risk'
    elif portfolio_var < 0.01:
        risk_level = 'Low Risk'
    else:
        risk_level = 'Medium Risk'

    if abs(exposure_pct) > 0.8:
        exposure_level = 'High Exposure'
    else:
        exposure_level = 'Normal Exposure'

    tag = f"{risk_level} / {exposure_level} / {counterparty}"

    trades[ticker] = [{'qty': qty, 'price': px, 'tag': tag}]
```

## Analysis Techniques

### Technique 1: Parse Hierarchical Tags

Split hierarchical tags for multi-dimensional analysis:

```python
pnl_summary = results.get_pnl_summary_by_tag()

# Parse tag components
pnl_summary['regime'] = pnl_summary['tag'].str.split(' / ').str[0]
pnl_summary['counterparty'] = pnl_summary['tag'].str.split(' / ').str[1]
pnl_summary['direction'] = pnl_summary['tag'].str.split(' / ').str[2]

# Aggregate by each dimension
by_regime = pnl_summary.groupby('regime')['total_pnl'].sum()
by_counterparty = pnl_summary.groupby('counterparty')['total_pnl'].sum()
by_direction = pnl_summary.groupby('direction')['total_pnl'].sum()
```

### Technique 2: Cross-Tabulation

Create pivot tables for interaction effects:

```python
import pandas as pd

# Pivot: Regime vs Counterparty
pivot = pnl_summary.pivot_table(
    values='total_pnl',
    index='regime',
    columns='counterparty',
    aggfunc='sum'
)

print(pivot)
```

### Technique 3: Conditional Analysis

Compare specific combinations:

```python
# Compare High Vol performance by counterparty
high_vol = pnl_summary[pnl_summary['regime'] == 'High Vol']
high_vol_by_cp = high_vol.groupby('counterparty').agg({
    'total_pnl': 'sum',
    'sharpe': 'mean',
    'win_rate': 'mean'
})
```

### Technique 4: Time Series by Tag

Plot cumulative PnL for each tag combination:

```python
pnl_by_tag = results.get_pnl_by_tag()

for tag in pnl_by_tag['tag'].unique()[:5]:  # Top 5 tags
    tag_data = pnl_by_tag[pnl_by_tag['tag'] == tag]
    cumulative = tag_data['pnl'].cumsum()
    plt.plot(tag_data['date'], cumulative, label=tag)

plt.legend()
plt.show()
```

## Best Practices

### 1. Keep Tags Consistent

Use consistent formatting for easier parsing:

```python
# Good - consistent format
tag = f"{regime} / {counterparty} / {direction}"

# Bad - inconsistent
tag = f"{regime}-{counterparty} / {direction}"  # Mixed delimiters
```

### 2. Limit Tag Cardinality

Don't create too many unique tag combinations:

```python
# Bad - too many combinations
tag = f"{date} / {hour} / {minute} / {ticker} / {regime} / {counterparty}"
# Could create 1000s of unique tags

# Good - reasonable cardinality
tag = f"{regime} / {counterparty} / {direction}"
# ~50 combinations max
```

### 3. Make Tags Meaningful

Use descriptive names:

```python
# Good - clear meaning
tag = "High Vol / Goldman Sachs / Buy"

# Bad - cryptic
tag = "HV / GS / B"
```

### 4. Test Tag Logic

Verify tags are assigned correctly:

```python
def test_tag_assignment():
    # Create test context with known conditions
    context = create_test_context(volatility=0.30)  # High vol

    trades = generate_trades_with_dynamic_tags(context)

    # Verify all tags contain "High Vol"
    for ticker, trade_list in trades.items():
        for trade in trade_list:
            assert 'High Vol' in trade['tag'], f"Expected High Vol tag, got {trade['tag']}"
```

## Performance Considerations

Dynamic tag generation is lightweight:

- **Overhead**: < 1% of execution time
- **Memory**: ~50 bytes per trade
- **Computation**: String concatenation only

The real value is in the insights gained from attribution analysis.

## Troubleshooting

### Issue: Too Many Unique Tags

**Symptom:** `get_pnl_summary_by_tag()` returns 1000s of rows

**Solution:** Reduce tag dimensions or use coarser categories

```python
# Instead of continuous values
tag = f"{volatility:.4f} / {counterparty}"  # Bad: unique for each vol level

# Use discrete buckets
vol_bucket = 'High' if volatility > 0.25 else 'Low'
tag = f"{vol_bucket} / {counterparty}"  # Good: only 2 x N combinations
```

### Issue: Empty PnL Summary

**Symptom:** `get_pnl_summary_by_tag()` returns empty DataFrame

**Solution:** Check that trades are being generated with tags:

```python
# Add debugging
def generate_trades_with_dynamic_tags(context):
    trades = calculate_trades(context)
    print(f"Generated {len(trades)} trades on {context['date']}")

    for ticker in trades:
        print(f"  {ticker}: tag = {trades[ticker][0]['tag']}")

    return trades
```

### Issue: Tag Not Found

**Symptom:** `get_pnl_by_tag('My Tag')` returns empty

**Solution:** Check exact tag string (case-sensitive):

```python
# See all available tags
all_tags = results.get_pnl_summary_by_tag()['tag'].unique()
print(all_tags)

# Use exact match
pnl = results.get_pnl_by_tag('High Vol / Goldman Sachs / Buy')  # Exact case
```

## See Also

- [External Trade Tags Documentation](../docs/EXTERNAL_TRADE_TAGS.md)
- [Dynamic Trade Generation Notebook](07_dynamic_trade_generation.ipynb)
- [Use Case 3 Documentation](../docs/USE_CASE_3_EXTERNAL_TRADES.md)

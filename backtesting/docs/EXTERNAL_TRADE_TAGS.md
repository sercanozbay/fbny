# External Trade Tags for Counterparty Attribution

## Overview

External trades in Use Case 3 can be tagged with identifiers (e.g., counterparty names, strategy IDs, desk names) to enable PnL attribution by group. This allows you to track performance by:

- **Counterparties** (e.g., Goldman Sachs, Morgan Stanley, JPMorgan)
- **Trading desks** (e.g., Equity Desk, Fixed Income Desk)
- **Strategies** (e.g., Market Making, Arbitrage, Flow Trading)
- **Client types** (e.g., Institutional, Retail, Prop)
- **Any custom grouping** you need

## Quick Start

### Adding Tags to External Trades

Tags are added as an optional `'tag'` field in each external trade:

```python
external_trades_by_date = {
    date: {
        'AAPL': [{
            'qty': 100,
            'price': 150.25,
            'tag': 'Goldman Sachs'  # Add tag here
        }],
        'MSFT': [{
            'qty': -50,
            'price': 350.75,
            'tag': 'Morgan Stanley'  # Each trade can have different tag
        }]
    }
}
```

### Retrieving PnL by Tag

After running the backtest, use these methods to analyze PnL by tag:

```python
# Run backtest
results = backtester.run(
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_case=3,
    inputs={'external_trades': external_trades_by_date}
)

# Get PnL summary by tag
summary = results.get_pnl_summary_by_tag()
print(summary)
```

**Output:**
```
              tag  total_pnl  mean_pnl  std_pnl  sharpe  num_days  win_rate
0   Goldman Sachs  125000.00   5000.00  8000.00    0.62        25      0.60
1  Morgan Stanley   98000.00   4900.00  7200.00    0.68        20      0.65
2        JPMorgan   45000.00   3000.00  6500.00    0.46        15      0.53
3         Citadel   67000.00   4200.00  7000.00    0.60        16      0.56
```

## API Reference

### BacktestResults Methods

#### `get_pnl_by_tag(tag=None)`

Get daily PnL time series by tag.

**Parameters:**
- `tag` (str, optional): Specific tag to retrieve. If None, returns all tags.

**Returns:**
- `pd.DataFrame` with columns: `date`, `tag`, `pnl`

**Examples:**

```python
# Get PnL for all counterparties
all_pnl = results.get_pnl_by_tag()

# Get PnL for specific counterparty
gs_pnl = results.get_pnl_by_tag('Goldman Sachs')

# Plot cumulative PnL by counterparty
import matplotlib.pyplot as plt

for tag in all_pnl['tag'].unique():
    tag_data = all_pnl[all_pnl['tag'] == tag]
    cumulative = tag_data['pnl'].cumsum()
    plt.plot(tag_data['date'], cumulative, label=tag)

plt.legend()
plt.xlabel('Date')
plt.ylabel('Cumulative PnL ($)')
plt.title('Cumulative PnL by Counterparty')
plt.show()
```

#### `get_pnl_summary_by_tag()`

Get summary statistics of PnL by tag.

**Returns:**
- `pd.DataFrame` with columns:
  - `tag`: Tag identifier
  - `total_pnl`: Total PnL for this tag
  - `mean_pnl`: Mean daily PnL (excluding zero days)
  - `std_pnl`: Standard deviation of daily PnL
  - `sharpe`: Annualized Sharpe ratio
  - `num_days`: Number of trading days with activity
  - `win_rate`: Fraction of profitable days

**Example:**

```python
summary = results.get_pnl_summary_by_tag()

# Find best performing counterparty
best = summary.iloc[0]
print(f"Best counterparty: {best['tag']}")
print(f"Total PnL: ${best['total_pnl']:,.0f}")
print(f"Sharpe: {best['sharpe']:.2f}")
```

#### `get_trades_by_tag(tag=None)`

Get external trades filtered by tag.

**Parameters:**
- `tag` (str, optional): Specific tag to retrieve. If None, returns all external trades.

**Returns:**
- `pd.DataFrame` with trade details including tag column

**Example:**

```python
# Get all trades from Goldman Sachs
gs_trades = results.get_trades_by_tag('Goldman Sachs')

# Analyze trade sizes
print(f"Total trades: {len(gs_trades)}")
print(f"Average size: {gs_trades['quantity'].abs().mean():.0f} shares")
print(f"Total notional: ${(gs_trades['quantity'] * gs_trades['price']).abs().sum():,.0f}")
```

## Use Cases

### 1. Counterparty Performance Analysis

Track which counterparties provide the best execution and PnL:

```python
# Generate trades with counterparty tags
counterparties = ['Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'Citadel']

for date in dates:
    trades = {}
    for ticker, qty in get_trades_for_date(date):
        counterparty = select_counterparty(ticker, qty)  # Your logic

        trades[ticker] = [{
            'qty': qty,
            'price': get_price(ticker, date),
            'tag': counterparty
        }]

    external_trades_by_date[date] = trades

# Run backtest
results = backtester.run(...)

# Analyze
summary = results.get_pnl_summary_by_tag()
best_counterparty = summary.loc[summary['sharpe'].idxmax()]
print(f"Best counterparty by Sharpe: {best_counterparty['tag']}")
```

### 2. Multi-Desk Attribution

Attribute PnL to different trading desks:

```python
# Tag by desk
external_trades = {
    date: {
        'AAPL': [{'qty': 1000, 'price': 150.0, 'tag': 'Equity Desk'}],
        'TLT': [{'qty': 500, 'price': 95.0, 'tag': 'Fixed Income Desk'}],
        'GC': [{'qty': 10, 'price': 1900.0, 'tag': 'Commodities Desk'}]
    }
}

# Analyze desk performance
summary = results.get_pnl_summary_by_tag()
print("\nDesk Performance:")
print(summary[['tag', 'total_pnl', 'sharpe', 'win_rate']])
```

### 3. Strategy Attribution

Track PnL by trading strategy:

```python
def generate_trades_with_strategy_tags(date, signals):
    """Generate trades with strategy tags."""
    trades = {}

    # Market making trades
    for ticker in market_making_universe:
        trades[ticker] = [{
            'qty': calculate_mm_position(ticker),
            'price': get_price(ticker, date),
            'tag': 'Market Making'
        }]

    # Arbitrage trades
    for ticker in arb_signals:
        trades[ticker] = [{
            'qty': arb_signals[ticker],
            'price': get_price(ticker, date),
            'tag': 'Arbitrage'
        }]

    # Flow trades
    for ticker in client_flows:
        trades[ticker] = [{
            'qty': client_flows[ticker],
            'price': get_price(ticker, date),
            'tag': 'Flow Trading'
        }]

    return trades

# Compare strategy performance
summary = results.get_pnl_summary_by_tag()
print("\nStrategy Performance:")
for _, row in summary.iterrows():
    print(f"{row['tag']:20s}: PnL=${row['total_pnl']:>12,.0f}  Sharpe={row['sharpe']:>5.2f}")
```

### 4. Client Type Attribution

Differentiate between institutional, retail, and proprietary trading:

```python
external_trades = {
    date: {
        'AAPL': [
            {'qty': 10000, 'price': 150.0, 'tag': 'Institutional'},
            {'qty': 100, 'price': 150.1, 'tag': 'Retail'},
            {'qty': -5000, 'price': 149.9, 'tag': 'Prop'}
        ]
    }
}

# Analyze by client type
summary = results.get_pnl_summary_by_tag()
```

## Advanced Example: Dynamic Tagging

Generate tags dynamically based on market conditions:

```python
def generate_trades_with_dynamic_tags(context):
    """Generate trades with context-aware tags."""
    date = context['date']
    portfolio_value = context['portfolio_value']
    daily_returns = context['daily_returns']

    # Determine market regime
    if len(daily_returns) >= 20:
        recent_vol = np.std(daily_returns[-20:]) * np.sqrt(252)
        regime = 'High Vol' if recent_vol > 0.25 else 'Low Vol'
    else:
        regime = 'Warmup'

    # Generate trades with regime tag
    trades = {}
    for ticker, signal in calculate_signals(context).items():
        qty = signal_to_quantity(signal, portfolio_value)

        trades[ticker] = [{
            'qty': qty,
            'price': context['prices'][ticker],
            'tag': f'{regime} - {get_counterparty(ticker)}'
        }]

    return trades

# Use callable for dynamic generation
results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=3,
    inputs={'external_trades': generate_trades_with_dynamic_tags}
)

# Compare performance across regimes
summary = results.get_pnl_summary_by_tag()
high_vol = summary[summary['tag'].str.contains('High Vol')]
low_vol = summary[summary['tag'].str.contains('Low Vol')]

print("High Vol Regime:")
print(f"  Total PnL: ${high_vol['total_pnl'].sum():,.0f}")
print(f"  Avg Sharpe: {high_vol['sharpe'].mean():.2f}")

print("\nLow Vol Regime:")
print(f"  Total PnL: ${low_vol['total_pnl'].sum():,.0f}")
print(f"  Avg Sharpe: {low_vol['sharpe'].mean():.2f}")
```

## Best Practices

### 1. Consistent Tag Naming

Use consistent naming conventions for tags:

```python
# Good - consistent naming
tags = ['Goldman Sachs', 'Morgan Stanley', 'JPMorgan']

# Bad - inconsistent
tags = ['goldman sachs', 'Morgan Stanley', 'JP Morgan']  # Different casing/abbreviations
```

### 2. Hierarchical Tags

For complex attribution, use hierarchical tags:

```python
# Format: Desk / Strategy / Counterparty
tag = f"{desk} / {strategy} / {counterparty}"

# Example
trade['tag'] = 'Equity / Market Making / Goldman Sachs'

# Later, can aggregate by parsing
summary = results.get_pnl_summary_by_tag()
summary['desk'] = summary['tag'].str.split(' / ').str[0]
desk_pnl = summary.groupby('desk')['total_pnl'].sum()
```

### 3. Handle Untagged Trades

Trades without tags are automatically tagged as 'untagged':

```python
# Trade without tag
trades = {
    'AAPL': [{'qty': 100, 'price': 150.0}]  # No 'tag' field
}

# Will appear as 'untagged' in attribution
summary = results.get_pnl_summary_by_tag()
```

### 4. Performance Considerations

Tags add minimal overhead:
- Storage: ~50 bytes per trade
- Computation: Negligible (simple dictionary lookups)
- No impact on backtest execution speed

## Visualization Examples

### Cumulative PnL by Counterparty

```python
import matplotlib.pyplot as plt

pnl_by_tag = results.get_pnl_by_tag()

fig, ax = plt.subplots(figsize=(12, 6))

for tag in pnl_by_tag['tag'].unique():
    tag_data = pnl_by_tag[pnl_by_tag['tag'] == tag]
    cumulative = tag_data['pnl'].cumsum() / 1e6  # In millions
    ax.plot(tag_data['date'], cumulative, label=tag, linewidth=2)

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL ($M)')
ax.set_title('Cumulative PnL by Counterparty')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Heatmap of Daily PnL by Tag

```python
import seaborn as sns

pnl_by_tag = results.get_pnl_by_tag()

# Pivot to wide format
pnl_pivot = pnl_by_tag.pivot(index='date', columns='tag', values='pnl')
pnl_pivot = pnl_pivot.fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pnl_pivot.T, cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Daily PnL ($)'})
plt.title('Daily PnL Heatmap by Counterparty')
plt.xlabel('Date')
plt.ylabel('Counterparty')
plt.show()
```

### Bar Chart Comparison

```python
summary = results.get_pnl_summary_by_tag()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Total PnL
axes[0].bar(summary['tag'], summary['total_pnl'] / 1e6)
axes[0].set_title('Total PnL by Counterparty')
axes[0].set_ylabel('Total PnL ($M)')
axes[0].tick_params(axis='x', rotation=45)

# Sharpe Ratio
axes[1].bar(summary['tag'], summary['sharpe'])
axes[1].set_title('Sharpe Ratio by Counterparty')
axes[1].set_ylabel('Sharpe Ratio')
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Win Rate
axes[2].bar(summary['tag'], summary['win_rate'] * 100)
axes[2].set_title('Win Rate by Counterparty')
axes[2].set_ylabel('Win Rate (%)')
axes[2].tick_params(axis='x', rotation=45)
axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

## FAQ

**Q: Can I have multiple tags per trade?**

A: Not directly. Use hierarchical tags (e.g., "Desk / Strategy / Counterparty") and parse later.

**Q: What happens if I don't specify a tag?**

A: The trade is automatically tagged as 'untagged'.

**Q: Can tags be changed after the backtest?**

A: No, tags are immutable once the backtest runs. Rerun the backtest with updated tags.

**Q: Do tags affect backtest execution?**

A: No, tags are purely for attribution and have no impact on portfolio optimization or trade execution.

**Q: Can I export tag attribution to Excel/CSV?**

A: Yes:
```python
summary = results.get_pnl_summary_by_tag()
summary.to_csv('counterparty_attribution.csv', index=False)
```

## See Also

- [Use Case 3 Documentation](USE_CASE_3_EXTERNAL_TRADES.md)
- [External Trade Generation Guide](../notebooks/06_external_trade_generation.ipynb)
- [Dynamic Trade Generation Guide](../notebooks/07_dynamic_trade_generation.ipynb)
- [Risk-Managed Portfolio Example](../notebooks/04_use_case_3_risk_managed_portfolio.ipynb)

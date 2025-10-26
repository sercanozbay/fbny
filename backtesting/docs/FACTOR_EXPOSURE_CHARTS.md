# Factor Exposure Charts

The backtesting framework now includes comprehensive factor exposure tracking and visualization. These charts help you understand how your portfolio's factor exposures evolve over time.

## Features

### Two Visualization Types

**1. Time Series Charts**
- Individual line plots for each factor
- Shows long/short exposure with color-coded areas (green for long, red for short)
- Zero line for reference
- Up to 6 factors displayed in a grid layout (2 columns x 3 rows)

**2. Heatmap**
- Color-coded visualization of all factors across time
- Red-Yellow-Green color scheme (red = negative, green = positive)
- Compact view showing patterns and trends
- Ideal for identifying factor rotation

### Automatic Tracking

Factor exposures are automatically calculated and tracked during the backtest:
- Calculated at each rebalance using the portfolio's factor loadings
- Stored as a time series for each factor
- Integrated into all reporting formats (PDF, HTML, charts directory)

## Usage

### Basic Usage

Factor exposure charts are generated automatically when:
1. Factor exposure data is available (`factor_exposures.csv` is loaded)
2. You run a backtest with factor model data
3. You generate reports

```python
# Run backtest with factor data
results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=2,
    inputs=inputs
)

# Generate reports with factor exposure charts
results.generate_full_report(
    output_dir='./output/my_backtest',
    formats=['pdf', 'html', 'excel', 'csv']
)

# Charts will be in: ./output/my_backtest/charts/
# - factor_exposures_timeseries.png
# - factor_exposures_heatmap.png
```

### Generate Factor Exposure Charts Only

```python
# Generate just the factor exposure charts
results.visualizer.plot_factor_exposures_timeseries(
    dates=results.dates,
    factor_exposures=results.factor_exposures,
    save_path='./factor_timeseries.png'
)

results.visualizer.plot_factor_exposures_heatmap(
    dates=results.dates,
    factor_exposures=results.factor_exposures,
    save_path='./factor_heatmap.png'
)
```

### Display in Jupyter Notebook

```python
# Display factor exposure charts interactively
results.visualizer.plot_factor_exposures_timeseries(
    dates=results.dates,
    factor_exposures=results.factor_exposures
    # No save_path = display inline
)

results.visualizer.plot_factor_exposures_heatmap(
    dates=results.dates,
    factor_exposures=results.factor_exposures
)
```

## Data Format

Factor exposures are stored as a list of dictionaries, one per date:

```python
[
    {'Factor_0': 0.15, 'Factor_1': -0.08, 'Factor_2': 0.22},  # Date 1
    {'Factor_0': 0.12, 'Factor_1': -0.10, 'Factor_2': 0.25},  # Date 2
    # ...
]
```

Access factor exposure data:

```python
# Get all factor exposures
factor_exp = results.factor_exposures

# Convert to DataFrame for analysis
import pandas as pd
factor_exp_df = pd.DataFrame(factor_exp, index=results.dates)

# Analyze specific factor
momentum_exp = factor_exp_df['Momentum']
print(f"Average Momentum exposure: {momentum_exp.mean():.4f}")
print(f"Max Momentum exposure: {momentum_exp.max():.4f}")
print(f"Min Momentum exposure: {momentum_exp.min():.4f}")
```

## Chart Interpretation

### Time Series Chart

```
Factor_0 Exposure          Factor_1 Exposure
     │                          │
 0.3 ├──────/\─────         0.2 ├─────────────
     │     /  \               │      /────\
 0.0 ├────/────\────        0.0 ├────/      \───
     │         \  /            │              \
-0.2 ├──────────\/           -0.1├──────────────\─
     └─────────────            └─────────────────
     Time →                    Time →
```

**What to look for:**
- **Drift**: Exposure moving away from target over time
- **Volatility**: Rapid changes indicate unstable factor bets
- **Reversals**: Sign changes show factor rotation
- **Magnitude**: Large exposures indicate concentrated factor bets

### Heatmap

```
          Jan  Feb  Mar  Apr  May
Factor_0  ███  ███  ░░░  ▒▒▒  ███  (Strong → Weak → Medium → Strong)
Factor_1  ▒▒▒  ▒▒▒  ▒▒▒  ▒▒▒  ▒▒▒  (Stable neutral)
Factor_2  ░░░  ░░░  ███  ███  ███  (Weak → Strong)

█ = Strong Positive (green)
▒ = Neutral (yellow)
░ = Negative (red)
```

**What to look for:**
- **Patterns**: Consistent colors indicate stable exposures
- **Transitions**: Color changes show rebalancing
- **Correlation**: Similar patterns across factors suggest factor clustering
- **Extremes**: Very dark/light areas show concentrated positions

## Examples

### Example 1: Momentum Strategy

```python
# Run momentum strategy
results = backtester.run(...)

# Generate factor charts
results.generate_charts('./output/momentum')

# Analyze momentum exposure
import pandas as pd
factor_df = pd.DataFrame(results.factor_exposures, index=results.dates)

print(f"Average Momentum exposure: {factor_df['Momentum'].mean():.3f}")
print(f"Momentum exposure std dev: {factor_df['Momentum'].std():.3f}")
```

**Expected Output:**
- Momentum factor shows positive bias (e.g., mean = 0.15)
- Other factors near zero (factor-neutral)
- Time series shows stable momentum exposure with small variations

### Example 2: Factor-Neutral Strategy

```python
# Configure with factor constraints
config = BacktestConfig(
    max_factor_exposure={
        'Value': 0.05,
        'Momentum': 0.05,
        'Quality': 0.05
    }
)

results = backtester.run(...)
factor_df = pd.DataFrame(results.factor_exposures, index=results.dates)

# Verify factor neutrality
for factor in factor_df.columns:
    print(f"{factor}: mean={factor_df[factor].mean():.4f}, "
          f"max={factor_df[factor].abs().max():.4f}")
```

**Expected Output:**
- All factors within ±0.05 range
- Heatmap shows mostly yellow (neutral)
- Time series oscillates around zero

### Example 3: Factor Rotation Strategy

```python
# Run strategy that rotates between factors
results = backtester.run(...)

# Generate charts to visualize rotation
results.visualizer.plot_factor_exposures_timeseries(
    results.dates,
    results.factor_exposures,
    save_path='./factor_rotation.png'
)

# Quantify rotation
factor_df = pd.DataFrame(results.factor_exposures, index=results.dates)

# Calculate turnover
factor_changes = factor_df.diff().abs().sum(axis=1)
avg_daily_turnover = factor_changes.mean()
print(f"Average daily factor turnover: {avg_daily_turnover:.3f}")
```

**Expected Output:**
- Time series shows factors moving in opposite directions
- High turnover indicates frequent rebalancing
- Heatmap shows distinct bands of color

## Integration with PDF Reports

Factor exposure charts are automatically included in PDF reports:

**PDF Report Structure:**
1. Executive Summary
2. Performance Metrics
3. **Performance Charts** (includes factor exposures)
   - Cumulative Returns
   - Drawdown
   - Rolling Sharpe
   - Return Distribution
   - Portfolio Exposures
   - Transaction Costs
   - Factor Attribution
   - **Factor Exposures Time Series** ← NEW
   - **Factor Exposures Heatmap** ← NEW
4. Detailed Statistics
5. Monthly Returns

The PDF file size increases by ~60-100KB with factor charts included.

## Customization

### Adjust Chart Size

```python
from backtesting.visualization import BacktestVisualizer

# Create visualizer with custom settings
visualizer = BacktestVisualizer(figsize=(16, 10), dpi=150)

# Generate larger charts
visualizer.plot_factor_exposures_timeseries(
    results.dates,
    results.factor_exposures,
    save_path='./large_factor_chart.png'
)
```

### Filter Factors

```python
# Show only specific factors
import pandas as pd

factor_df = pd.DataFrame(results.factor_exposures, index=results.dates)

# Select only momentum and value
selected_factors = factor_df[['Momentum', 'Value']]

# Convert back to list of dicts
filtered_exposures = selected_factors.to_dict('records')

# Plot
visualizer.plot_factor_exposures_timeseries(
    results.dates,
    filtered_exposures,
    save_path='./momentum_value_only.png'
)
```

### Custom Color Scheme (Heatmap)

To customize heatmap colors, edit `backtesting/visualization.py`:

```python
# In plot_factor_exposures_heatmap method, change:
im = ax.imshow(factor_exp_df.T.values, aspect='auto', cmap='RdYlGn', ...)

# To:
im = ax.imshow(factor_exp_df.T.values, aspect='auto', cmap='coolwarm', ...)
# Or: cmap='RdBu', 'seismic', 'bwr', etc.
```

## Troubleshooting

### No Factor Exposure Charts Generated

**Problem**: Charts directory doesn't contain `factor_exposures_timeseries.png` or `factor_exposures_heatmap.png`

**Solutions:**
1. Verify factor data is loaded:
   ```python
   print(f"Factor exposures tracked: {len(results.factor_exposures)}")
   print(f"Sample: {results.factor_exposures[0] if results.factor_exposures else 'None'}")
   ```

2. Check if factor model files exist:
   ```bash
   ls data/factor_exposures.csv
   ls data/factor_returns.csv
   ```

3. Ensure correct data format for `factor_exposures.csv`:
   - Must have MultiIndex: (date, ticker)
   - Columns are factor names
   - Example:
     ```csv
     date,ticker,Factor_0,Factor_1,Factor_2
     2023-01-01,AAPL,0.5,-0.2,0.3
     2023-01-01,MSFT,0.3,0.1,-0.1
     ...
     ```

### Empty or All-Zero Exposures

**Problem**: Charts show all zeros or empty data

**Causes:**
- Factor loadings file is malformed
- Risk calculator not being called
- Factor model not enabled

**Solution:**
```python
# Verify in backtester initialization
print(f"Risk model initialized: {backtester.risk_model is not None}")
print(f"Factor data loaded: {backtester.data_manager._factor_exposures is not None}")
```

### Charts Look Cluttered

**Problem**: Too many factors make the time series chart hard to read

**Solution:**
The visualization automatically limits to 6 factors (2x3 grid). If you have more:
1. Filter to important factors (see "Filter Factors" above)
2. Use the heatmap instead (handles many factors better)
3. Create separate charts for factor groups:
   ```python
   # Split into groups
   style_factors = ['Value', 'Momentum', 'Quality']
   macro_factors = ['Interest_Rate', 'Inflation', 'Credit']

   # Plot separately
   for group_name, factors in [('Style', style_factors), ('Macro', macro_factors)]:
       filtered = factor_df[factors].to_dict('records')
       visualizer.plot_factor_exposures_timeseries(
           results.dates, filtered,
           save_path=f'./factors_{group_name.lower()}.png'
       )
   ```

## Best Practices

1. **Monitor Factor Drift**
   - Set factor exposure limits in config
   - Review charts regularly to catch unintended factor bets
   - Use heatmap to spot gradual drift over long periods

2. **Compare Actual vs Target**
   ```python
   # Plot target exposures alongside actual
   fig, axes = plt.subplots(2, 1, figsize=(12, 8))

   # Actual
   factor_df['Momentum'].plot(ax=axes[0], label='Actual')
   axes[0].axhline(y=0.1, color='r', linestyle='--', label='Target')
   axes[0].legend()
   axes[0].set_title('Momentum Exposure: Actual vs Target')

   # Tracking error
   tracking_error = (factor_df['Momentum'] - 0.1).abs()
   tracking_error.plot(ax=axes[1])
   axes[1].set_title('Momentum Tracking Error')
   ```

3. **Analyze Factor Turnover**
   ```python
   # Calculate factor turnover
   factor_changes = factor_df.diff().abs().sum(axis=1)

   # Plot cumulative turnover
   plt.figure(figsize=(12, 6))
   factor_changes.cumsum().plot()
   plt.title('Cumulative Factor Turnover')
   plt.ylabel('Turnover')
   plt.show()
   ```

4. **Correlate with Performance**
   ```python
   # See which factors drove returns
   returns_series = pd.Series(results.daily_returns, index=results.dates)

   for factor in factor_df.columns:
       corr = returns_series.corr(factor_df[factor])
       print(f"{factor} correlation with returns: {corr:.3f}")
   ```

## Next Steps

1. Review the generated charts in your output directory
2. Try adjusting factor exposure limits in `BacktestConfig`
3. Compare factor exposures across different strategies
4. Integrate factor monitoring into your production workflow

## Related Documentation

- [README.md](README.md) - Main framework documentation
- [PDF_REPORTS.md](PDF_REPORTS.md) - PDF report generation guide
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [notebooks/](notebooks/) - Example notebooks with factor analysis

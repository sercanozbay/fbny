# Sector ETF Hedging Guide

## Overview

The backtesting framework supports two methods for sector hedging:
1. **Proportional Method**: Adjusts individual stock positions to achieve sector neutrality
2. **ETF Method**: Uses sector ETFs to hedge sector exposures (NEW)

This guide focuses on the ETF hedging method, which is more capital-efficient and easier to implement in practice.

## Why ETF Hedging?

### Advantages

1. **Capital Efficiency**: Don't need to adjust every individual position
2. **Simplicity**: Add/remove a few ETF positions instead of rebalancing hundreds of stocks
3. **Liquidity**: Sector ETFs are highly liquid with tight spreads
4. **Practical**: Mirrors how sector hedging is done in real trading
5. **Lower Transaction Costs**: Fewer trades required

### Use Cases

- **Long/Short Equity**: Maintain sector neutrality while expressing stock-specific alpha
- **Market Neutral Strategies**: Remove sector tilts to isolate alpha
- **Risk Management**: Reduce unintended sector bets
- **Sector Rotation**: Implement tactical sector views via ETF overlays

## Quick Start

### Basic ETF Hedging

```python
from backtesting import Backtester, BacktestConfig, DataManager

# Configure with ETF sector hedging
config = BacktestConfig(
    initial_cash=10_000_000,
    enable_sector_hedge=True,
    sector_hedge_method='etf',  # Use ETF hedging
    sector_target_exposures={    # Target exposures per sector
        'Technology': 0.0,        # Sector neutral
        'Healthcare': 0.0,
        'Financials': 0.0
        # ... other sectors
    }
)

# DataManager and backtester setup
data_manager = DataManager('../sample_data')
backtester = Backtester(config, data_manager)

# Run backtest
results = backtester.run(
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_case=2,  # Signal-based trading
    inputs={'signals': signals_by_date}
)
```

## Default Sector ETF Mapping

The framework uses standard US sector ETF mapping by default:

| Sector | ETF Ticker | ETF Name |
|--------|------------|----------|
| Technology | XLK | Technology Select Sector SPDR |
| Healthcare | XLV | Health Care Select Sector SPDR |
| Financials | XLF | Financial Select Sector SPDR |
| Energy | XLE | Energy Select Sector SPDR |
| Consumer Discretionary | XLY | Consumer Discretionary Select Sector SPDR |
| Consumer Staples | XLP | Consumer Staples Select Sector SPDR |
| Industrials | XLI | Industrial Select Sector SPDR |
| Materials | XLB | Materials Select Sector SPDR |
| Real Estate | XLRE | Real Estate Select Sector SPDR |
| Utilities | XLU | Utilities Select Sector SPDR |
| Communication Services | XLC | Communication Services Select Sector SPDR |

## Configuration Options

### BacktestConfig Parameters

```python
config = BacktestConfig(
    # Enable sector hedging
    enable_sector_hedge=True,

    # Choose hedging method
    sector_hedge_method='etf',  # 'proportional' or 'etf'

    # Target sector exposures (default: 0.0 for all sectors)
    sector_target_exposures={
        'Technology': 0.0,      # Neutral
        'Healthcare': 0.05,     # 5% overweight
        'Financials': -0.03     # 3% underweight
    },

    # Custom ETF mapping (optional, uses defaults if not provided)
    sector_etf_mapping={
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        # ... other sectors
    }
)
```

### Target Exposures

Target exposures are specified as fractions of gross notional:

- **0.0**: Sector neutral (no net exposure)
- **Positive**: Overweight sector (e.g., 0.10 = 10% of gross notional)
- **Negative**: Underweight/short sector (e.g., -0.05 = -5% of gross notional)
- **None/Omitted**: Defaults to 0.0 (neutral)

## How ETF Hedging Works

### Calculation Logic

1. **Calculate Current Sector Exposures**
   ```
   For each sector:
     sector_value = sum(position_value for all stocks in sector)
     sector_exposure = sector_value / total_portfolio_value
   ```

2. **Determine Required Adjustment**
   ```
   For each sector:
     adjustment_needed = (target_exposure - current_exposure) * portfolio_value
   ```

3. **Calculate ETF Hedge**
   ```
   For each sector:
     etf_shares = -adjustment_needed / etf_price
   ```

   Note: Negative sign because we're hedging (opposite direction)

### Example

**Portfolio:**
- Technology stocks: $500,000 (long)
- Healthcare stocks: $300,000 (long)
- Total portfolio: $1,000,000
- Target: All sectors neutral (0%)

**Current Exposures:**
- Technology: 50% ($500k / $1M)
- Healthcare: 30% ($300k / $1M)

**Required Hedges:**
- Technology: Need to offset 50% = $500,000
  - Short XLK: $500,000 / $150 (XLK price) = -3,333 shares
- Healthcare: Need to offset 30% = $300,000
  - Short XLV: $300,000 / $120 (XLV price) = -2,500 shares

**Hedged Portfolio:**
- Long: Technology stocks ($500k) + Healthcare stocks ($300k)
- Short: XLK ($500k) + XLV ($300k)
- Net sector exposure: 0% (neutral)

## Custom ETF Mapping

### Using Different ETFs

You can specify custom ETF mappings for:
- International markets
- Alternative ETF providers
- Industry-specific ETFs

```python
# Example: Using Vanguard sector ETFs
custom_etf_mapping = {
    'Technology': 'VGT',      # Vanguard Information Technology
    'Healthcare': 'VHT',      # Vanguard Health Care
    'Financials': 'VFH',      # Vanguard Financials
    'Energy': 'VDE',          # Vanguard Energy
    'Consumer Discretionary': 'VCR',
    'Consumer Staples': 'VDC',
    'Industrials': 'VIS',
    'Materials': 'VAW',
    'Real Estate': 'VNQ',
    'Utilities': 'VPU',
    'Communication Services': 'VOX'
}

config = BacktestConfig(
    enable_sector_hedge=True,
    sector_hedge_method='etf',
    sector_etf_mapping=custom_etf_mapping
)
```

### Industry-Level Hedging

Use more granular industry ETFs:

```python
# Example: Tech sub-sector ETFs
tech_industry_etfs = {
    'Software': 'IGV',           # iShares Software
    'Semiconductors': 'SOXX',    # iShares Semiconductor
    'Internet': 'FDN',           # First Trust Dow Jones Internet
    'Cybersecurity': 'HACK',     # ETFMG Cybersecurity
}

config = BacktestConfig(
    enable_sector_hedge=True,
    sector_hedge_method='etf',
    sector_etf_mapping=tech_industry_etfs
)
```

## Advanced Usage

### Tactical Sector Tilts

Express tactical views while hedging unwanted exposures:

```python
# Overweight technology, neutral everything else
config = BacktestConfig(
    enable_sector_hedge=True,
    sector_hedge_method='etf',
    sector_target_exposures={
        'Technology': 0.15,   # 15% overweight
        'Healthcare': 0.0,    # Neutral
        'Financials': 0.0,
        # ... all other sectors neutral
    }
)
```

### Dynamic Hedging

Adjust target exposures based on market conditions:

```python
# In your trade generation function
def generate_trades_with_dynamic_hedging(context):
    # Calculate market volatility
    recent_vol = context['volatility']

    # Reduce exposures in high volatility
    if recent_vol > 0.25:
        target_exposures = {sector: 0.0 for sector in all_sectors}  # Full neutral
    else:
        target_exposures = {sector: 0.05 for sector in favored_sectors}  # Mild tilts

    # Update config dynamically (if supported)
    # Or implement in custom logic

    return trades
```

### Combining with Beta Hedging

Use both beta and sector hedging:

```python
config = BacktestConfig(
    # Beta hedging
    enable_beta_hedge=True,
    beta_hedge_instrument='SPY',
    target_beta=0.0,

    # Sector hedging
    enable_sector_hedge=True,
    sector_hedge_method='etf',
    sector_target_exposures={...}
)
```

This creates a **dollar-neutral, beta-neutral, sector-neutral** portfolio.

## Comparing Methods

### Proportional vs ETF

| Aspect | Proportional | ETF |
|--------|-------------|-----|
| **Implementation** | Adjusts each stock position | Adds ETF hedges |
| **Capital Efficiency** | Less efficient (adjusts all stocks) | More efficient (few ETFs) |
| **Transaction Costs** | Higher (many small trades) | Lower (few large trades) |
| **Liquidity** | Varies by stock | High (ETFs very liquid) |
| **Tracking Error** | Lower (exact sector exposure) | Slightly higher (ETF tracking) |
| **Practical** | Hard to implement in live trading | Easy (standard practice) |
| **Best For** | Academic studies, theoretical purity | Real trading, production systems |

### When to Use Each

**Use Proportional Method When:**
- Need exact sector exposures
- Academic/research purposes
- Small portfolios with few stocks
- Transaction costs are negligible

**Use ETF Method When:**
- Real trading implementation
- Large portfolios (100+ stocks)
- Transaction costs matter
- Need operational simplicity
- Following industry best practices

## Data Requirements

### ETF Prices

Ensure your price data includes sector ETF prices:

```python
# prices.csv should include:
# - All individual stock prices
# - All sector ETF prices (XLK, XLV, XLF, etc.)

data_manager = DataManager('../sample_data')
prices = data_manager.load_prices()

# Verify ETF prices are available
required_etfs = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY',
                 'XLP', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC']

missing_etfs = [etf for etf in required_etfs if etf not in prices.columns]
if missing_etfs:
    print(f"Warning: Missing ETF prices: {missing_etfs}")
```

### Sector Mapping

Provide sector classification for all stocks:

```python
# sector_mapping.csv
# ticker,sector
# AAPL,Technology
# MSFT,Technology
# JNJ,Healthcare
# ...

sector_mapping = data_manager.load_sector_mapping()
```

## Monitoring and Analysis

### Hedge Information

Access hedge details from results:

```python
# Run backtest
results = backtester.run(...)

# Get hedge information
trades_df = results.get_trades_dataframe()

# Filter for ETF trades (hedges)
etf_tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY',
               'XLP', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC']

etf_trades = trades_df[trades_df['ticker'].isin(etf_tickers)]

print("\nETF Hedge Trades:")
print(etf_trades[['date', 'ticker', 'quantity', 'price']])
```

### Sector Exposure Over Time

Track sector exposures throughout the backtest:

```python
# Get daily positions
positions_history = results.get_position_history()

# Calculate sector exposures per day
# (This would require custom analysis code)

# Plot sector exposures over time
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
for sector in sectors:
    ax.plot(dates, sector_exposures[sector], label=sector)

ax.set_title('Sector Exposures Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Exposure (% of Gross)')
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.show()
```

## Best Practices

### 1. Choose Liquid ETFs

Use highly liquid ETFs with:
- High average daily volume (>10M shares)
- Tight bid-ask spreads (<0.05%)
- Large AUM (>$1B)

The default SPDRs (XLK, XLV, etc.) meet these criteria.

### 2. Rebalance Frequency

ETF hedges should be adjusted when:
- Sector exposures drift beyond tolerance (e.g., >2% from target)
- Major portfolio rebalancing occurs
- Market conditions change significantly

### 3. Cost Considerations

ETF hedging costs include:
- Bid-ask spread
- Borrow costs (for shorting)
- Transaction fees

These are typically lower than proportional method costs.

### 4. Tracking Error

ETF hedges have small tracking error because:
- Sector ETFs don't perfectly match your stock basket
- ETF composition differs from your portfolio

This is usually acceptable (<1% annually).

### 5. Test Both Methods

Compare proportional vs ETF hedging:

```python
# Test proportional
config_prop = BacktestConfig(
    enable_sector_hedge=True,
    sector_hedge_method='proportional'
)

results_prop = backtester.run(...)

# Test ETF
config_etf = BacktestConfig(
    enable_sector_hedge=True,
    sector_hedge_method='etf'
)

results_etf = backtester.run(...)

# Compare results
print(f"Proportional Sharpe: {results_prop.calculate_metrics()['sharpe_ratio']:.2f}")
print(f"ETF Sharpe: {results_etf.calculate_metrics()['sharpe_ratio']:.2f}")
```

## Troubleshooting

### Issue: Missing ETF Prices

**Symptom:** Hedges not applied, warning about missing prices

**Solution:**
```python
# Ensure price data includes ETFs
prices = data_manager.load_prices()
etf_tickers = ['XLK', 'XLV', ...]

for etf in etf_tickers:
    if etf not in prices.columns:
        print(f"Missing ETF price data: {etf}")
        # Add ETF price data or remove from mapping
```

### Issue: Incomplete Sector Mapping

**Symptom:** Some stocks not hedged

**Solution:**
```python
# Verify all stocks have sector classification
sector_mapping = data_manager.load_sector_mapping()
all_tickers = prices.columns

unmapped = [t for t in all_tickers if t not in sector_mapping.index]
if unmapped:
    print(f"Stocks without sector: {unmapped}")
    # Update sector_mapping.csv
```

### Issue: Large Hedge Positions

**Symptom:** ETF positions very large relative to portfolio

**Solution:**
- Check target exposures are reasonable (typically -0.2 to 0.2)
- Verify sector classification is correct
- Consider using industry-level hedging for finer control

## Example Notebook

See `notebooks/05_sector_hedging_etf.ipynb` for a complete example demonstrating:
- ETF hedging setup
- Comparison with proportional method
- Performance analysis
- Cost comparison
- Hedge monitoring

## API Reference

### SectorHedger Class

```python
from backtesting.hedging import SectorHedger

hedger = SectorHedger(
    target_exposures={'Technology': 0.0, 'Healthcare': 0.0},
    hedge_method='etf',
    sector_etf_mapping={'Technology': 'XLK', 'Healthcare': 'XLV'}
)

# Calculate ETF hedges
etf_hedges = hedger.calculate_hedge_etf(
    positions={'AAPL': 1000, 'MSFT': 500},
    prices={'AAPL': 150, 'MSFT': 300, 'XLK': 145},
    sector_mapping={'AAPL': 'Technology', 'MSFT': 'Technology'}
)

# Returns: {'XLK': -3448}  # Short XLK to hedge long tech exposure
```

### BacktestConfig

```python
from backtesting import BacktestConfig

config = BacktestConfig(
    enable_sector_hedge=True,           # Enable hedging
    sector_hedge_method='etf',          # Use ETF method
    sector_target_exposures={...},      # Target exposures
    sector_etf_mapping={...}            # Custom ETF mapping (optional)
)
```

## Summary

ETF sector hedging provides a practical, capital-efficient way to achieve sector neutrality in your backtests. Key benefits:

- ✅ Matches real-world trading practices
- ✅ Lower transaction costs
- ✅ Simpler implementation
- ✅ Highly liquid instruments
- ✅ Flexible configuration

Use the ETF method for production strategies and the proportional method for academic research requiring perfect sector matching.

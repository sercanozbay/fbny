# Portfolio Backtesting Framework

A comprehensive, production-ready backtesting framework for portfolio strategies with factor risk models, transaction costs, and sophisticated risk management. Designed to handle 2000-3000 securities at daily frequency with memory-efficient operations.

## Features

### Core Capabilities

- **Three Use Cases:**
  1. **Target Positions**: Input target shares/notional/weights with optional hedging
  2. **Signal-Based**: Convert alpha signals to positions with multiple scaling methods
  3. **Risk-Managed**: External trades with optimization to satisfy risk constraints

- **Factor Risk Model:**
  - Factor exposure calculation
  - Portfolio variance decomposition (factor + specific risk)
  - Factor PnL attribution
  - Risk constraint checking

- **Transaction Costs:**
  - Power-law cost model: `cost = a * (qty/adv)^power + fixed_cost`
  - ADV participation constraints
  - Configurable cost parameters

- **Hedging:**
  - Beta hedging via futures/ETF
  - Sector exposure hedging
  - Configurable target exposures

- **Optimization:**
  - Minimize transaction costs subject to constraints
  - Factor exposure limits
  - Portfolio variance limits
  - ADV constraints

- **Performance Analytics:**
  - Sharpe, Sortino, Calmar ratios
  - Maximum drawdown and underwater periods
  - VaR and CVaR
  - Factor attribution
  - Benchmark comparison (alpha, beta, information ratio)

- **Reporting:**
  - PDF reports with comprehensive metrics and embedded charts
  - HTML reports with embedded charts
  - Excel reports with multiple sheets
  - CSV exports
  - Publication-quality charts

### Performance Optimizations

- **Memory Efficient:**
  - Float32 option for reduced memory footprint
  - Lazy loading of data
  - Efficient pandas/numpy operations

- **Speed Optimized:**
  - Vectorized operations across securities
  - Minimal Python loops
  - Optimized for 2000-3000 securities

## Installation

### Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- openpyxl >= 3.0.0 (for Excel reports)
- tqdm >= 4.62.0 (for progress bars)

### Install

```bash
# Clone repository
git clone <repository-url>
cd backtesting

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py --securities 100 --days 252
```

## Quick Start

### 1. Generate Sample Data

```python
from generate_sample_data import generate_sample_data

generate_sample_data(
    n_securities=100,
    n_days=252,
    n_factors=5,
    output_dir='./sample_data',
    seed=42
)
```

### 2. Run a Basic Backtest

```python
import pandas as pd
from backtesting import Backtester, BacktestConfig, DataManager

# Load data
data_manager = DataManager(data_dir='./sample_data')

# Configure backtest
config = BacktestConfig(
    initial_cash=10_000_000,
    max_adv_participation=0.05,
    enable_beta_hedge=True,
    risk_free_rate=0.02
)

# Prepare inputs (Use Case 1: target weights)
target_weights = pd.read_csv('./sample_data/target_weights.csv',
                             index_col=0, parse_dates=True)
targets_by_date = {
    date: target_weights.loc[date].to_dict()
    for date in target_weights.index
}

inputs = {
    'type': 'weights',
    'targets': targets_by_date
}

# Run backtest
backtester = Backtester(config, data_manager)
results = backtester.run(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31'),
    use_case=1,
    inputs=inputs
)

# Analyze results
results.print_summary()
results.generate_full_report(output_dir='./output/my_backtest')
```

## Use Cases

### Use Case 1: Target Positions with Hedging

An external process generates target positions (shares, notional, or weights). The backtester applies hedging (beta and/or sector).

```python
# Input format
inputs = {
    'type': 'weights',  # or 'shares' or 'notional'
    'targets': {
        pd.Timestamp('2023-01-01'): {
            'STOCK0001': 0.01,
            'STOCK0002': 0.015,
            # ... more tickers
        },
        # ... more dates
    }
}

# Run with beta hedging
config = BacktestConfig(
    enable_beta_hedge=True,
    beta_hedge_instrument='SPY',
    target_beta=0.0  # Market neutral
)

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=1,
    inputs=inputs
)
```

### Use Case 2: Signal-Based Trading

Convert alpha signals to positions with automatic scaling and hedging.

```python
from backtesting.input_processor import SignalProcessor

# Configure signal processor
config = BacktestConfig(
    enable_beta_hedge=True,
    enable_sector_hedge=True
)

# Input format
inputs = {
    'signals': {
        pd.Timestamp('2023-01-01'): {
            'STOCK0001': 1.5,   # Positive signal = long
            'STOCK0002': -0.8,  # Negative signal = short
            # ... more tickers
        },
        # ... more dates
    }
}

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=2,
    inputs=inputs
)
```

### Use Case 3: Risk-Managed Portfolio

External trades with optimization to satisfy risk limits.

```python
# Configure with risk constraints
config = BacktestConfig(
    max_portfolio_variance=0.0004,  # Variance limit
    max_factor_exposure={
        'Factor1': 0.1,
        'Factor2': 0.15
    },
    max_sector_exposure={
        'Technology': 0.2,
        'Financials': 0.15
    }
)

# Input format
inputs = {
    'external_trades': {
        pd.Timestamp('2023-01-01'): {
            'STOCK0001': 1000,   # Buy 1000 shares
            'STOCK0002': -500,   # Sell 500 shares
            # ... more tickers
        },
        # ... more dates
    }
}

results = backtester.run(
    start_date=start_date,
    end_date=end_date,
    use_case=3,
    inputs=inputs
)
```

## Data Format

All data should be provided as CSV files in a single directory.

### Required Files

#### 1. prices.csv
Daily close prices.
```
Format: dates (rows) × tickers (columns)
Example:
date,STOCK0001,STOCK0002,STOCK0003,...
2023-01-01,100.5,25.3,150.2,...
2023-01-02,101.2,25.1,151.5,...
```

#### 2. adv.csv
Average daily volume.
```
Format: dates (rows) × tickers (columns)
Example:
date,STOCK0001,STOCK0002,STOCK0003,...
2023-01-01,1000000,500000,2000000,...
```

### Optional Files

#### 3. betas.csv
Market beta per security (required for beta hedging).
```
Format: dates (rows) × tickers (columns)
```

#### 4. factor_exposures.csv
Factor loadings per security.
```
Format: date, ticker, Factor1, Factor2, ...
Example:
date,ticker,Factor1,Factor2,Factor3
2023-01-01,STOCK0001,0.5,-0.3,1.2
2023-01-01,STOCK0002,-0.2,0.8,0.1
```

#### 5. factor_returns.csv
Daily factor returns.
```
Format: dates (rows) × factors (columns)
```

#### 6. factor_covariance.csv
Factor covariance matrix.
```
Format: factors (rows) × factors (columns)
```

#### 7. specific_variance.csv
Idiosyncratic variance per security.
```
Format: dates (rows) × tickers (columns)
```

#### 8. sector_mapping.csv
Sector classification.
```
Format: ticker, sector
Example:
ticker,sector
STOCK0001,Technology
STOCK0002,Healthcare
```

#### 9. trade_prices.csv (optional)
Execution prices if different from close prices.
```
Format: dates (rows) × tickers (columns)
```

See [docs/data_schema.md](docs/data_schema.md) for detailed specifications.

## Configuration

### BacktestConfig Parameters

```python
BacktestConfig(
    # Risk constraints
    max_factor_exposure=None,      # Dict[str, float]: max per factor
    max_sector_exposure=None,      # Dict[str, float]: max per sector
    max_gross_exposure=None,       # Max gross notional
    max_net_exposure=None,         # Max net notional
    max_portfolio_variance=None,   # Max portfolio variance

    # Trading constraints
    max_adv_participation=0.05,    # Max trade as % of ADV
    min_trade_size=1.0,            # Min shares to trade

    # Hedging
    enable_beta_hedge=False,       # Enable beta hedging
    enable_sector_hedge=False,     # Enable sector hedging
    beta_hedge_instrument='SPY',   # Hedge instrument ticker
    target_beta=0.0,               # Target portfolio beta

    # Transaction costs
    tc_power=1.5,                  # Power in cost function
    tc_coefficient=0.01,           # Cost coefficient
    tc_fixed=0.0001,               # Fixed cost (bps)

    # Execution
    use_trade_prices=False,        # Use separate execution prices

    # Optimization
    optimizer_method='SLSQP',      # Scipy optimizer method
    optimizer_max_iter=1000,       # Max iterations
    optimizer_tolerance=1e-6,      # Convergence tolerance

    # Performance
    use_float32=True,              # Use float32 for memory
    risk_free_rate=0.0,            # Annual risk-free rate

    # Initial state
    initial_cash=10_000_000,       # Starting cash
    initial_positions=None         # Dict[str, float]: initial holdings
)
```

## Architecture

### Module Structure

```
backtesting/
├── backtester.py           # Main engine
├── config.py               # Configuration classes
├── data_loader.py          # Data management
├── risk_calculator.py      # Factor risk model
├── transaction_costs.py    # Cost modeling
├── hedging.py              # Beta/sector hedging
├── optimizer.py            # Portfolio optimization
├── input_processor.py      # Input handling
├── execution.py            # Trade execution
├── attribution.py          # Performance attribution
├── metrics.py              # Performance metrics
├── visualization.py        # Charts
├── benchmarking.py         # Benchmark comparison
├── report_generator.py     # Report creation
├── results.py              # Results storage
└── utils.py                # Helper functions
```

### Daily Simulation Loop

For each trading day:
1. Load data (prices, ADV, betas, factors, etc.)
2. Process inputs based on use case
3. Calculate current portfolio exposures and risk
4. Apply hedging (beta/sector) if configured
5. For use case 3: run optimizer to meet risk constraints
6. Calculate required trades
7. Apply ADV constraints
8. Calculate transaction costs
9. Execute trades
10. Calculate end-of-day portfolio value
11. Perform factor attribution
12. Store results

## Performance Metrics

The framework calculates:

### Return Metrics
- Total return
- Annualized return
- Daily returns series
- Cumulative returns

### Risk Metrics
- Volatility (annualized)
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Drawdown series
- VaR (95%)
- CVaR (95%)
- Skewness
- Kurtosis

### Trading Metrics
- Win rate
- Profit factor
- Total transaction costs
- Turnover
- ADV utilization

### Factor Metrics
- Factor PnL attribution
- Factor Sharpe ratios
- Factor contributions to total PnL
- Specific vs. factor return decomposition

### Benchmark Comparison
- Alpha
- Beta
- R-squared
- Tracking error
- Information ratio
- Up/down capture ratios

## Examples

### Example 1: Equal-Weight Portfolio
See [notebooks/01_basic_setup_and_data_loading.ipynb](notebooks/01_basic_setup_and_data_loading.ipynb)

### Example 2: Signal-Based Long/Short
See [notebooks/02_signal_based_trading.ipynb](notebooks/02_signal_based_trading.ipynb)

### Example 3: Risk-Constrained Portfolio
See [notebooks/03_risk_managed_portfolio.ipynb](notebooks/03_risk_managed_portfolio.ipynb) (to be created)

## Advanced Usage

### Report Generation

The framework supports multiple report formats: PDF, HTML, Excel, and CSV.

#### Generate All Reports

```python
# Generate comprehensive reports in all formats
results.generate_full_report(
    output_dir='./output/my_backtest',
    formats=['pdf', 'html', 'excel', 'csv']
)
```

This creates:
- `backtest_report.pdf` - Professional PDF report with metrics tables and embedded charts
- `backtest_report.html` - Interactive HTML report
- `backtest_report.xlsx` - Excel workbook with multiple sheets (Summary, Performance, Trades)
- `backtest_results.csv` - Daily performance data
- `trades.csv` - Complete trade history
- `factor_attribution.csv` - Factor PnL breakdown
- `charts/` - Individual chart images (PNG format)

#### Generate Individual Reports

```python
# PDF only (includes all charts)
results.generate_pdf_report(
    output_path='./output/report.pdf',
    chart_dir='./output/charts'  # Optional: include charts
)

# HTML only
results.generate_html_report(
    output_path='./output/report.html',
    chart_dir='./output/charts'
)

# Excel only
results.generate_excel_report(
    output_path='./output/report.xlsx'
)

# CSV only
results.save_to_csv('./output/results.csv')
```

#### PDF Report Contents

The PDF report includes:
1. **Executive Summary** - Period, trading days, returns
2. **Performance Metrics** - Returns, risk metrics, additional statistics
3. **Performance Charts** - All visualization charts embedded
4. **Detailed Statistics** - Daily/monthly statistics, win rates, costs
5. **Monthly Returns Table** - Month-by-month performance

The PDF is professionally formatted with color-coded sections and is ideal for:
- Client presentations
- Investment committee reports
- Strategy documentation
- Archival purposes

### Custom Transaction Cost Models

```python
from backtesting.transaction_costs import TransactionCostModel

# Create custom cost model
custom_cost_model = TransactionCostModel(
    power=2.0,          # Quadratic impact
    coefficient=0.02,   # Higher impact
    fixed_cost=0.0002   # 2 bps fixed
)

# Use in backtester
backtester.cost_model = custom_cost_model
```

### Custom Risk Constraints

```python
# Set detailed factor constraints
config = BacktestConfig(
    max_factor_exposure={
        'Momentum': 0.2,
        'Value': 0.15,
        'Quality': 0.1,
        'Size': 0.05,
        'Volatility': 0.1
    },
    max_portfolio_variance=0.0005
)
```

### Parallel Backtests

```python
# Run multiple configurations in parallel
configs = [
    BacktestConfig(max_adv_participation=0.03, ...),
    BacktestConfig(max_adv_participation=0.05, ...),
    BacktestConfig(max_adv_participation=0.10, ...)
]

results_list = []
for config in configs:
    backtester = Backtester(config, data_manager)
    results = backtester.run(...)
    results_list.append(results)

# Compare results
for i, results in enumerate(results_list):
    metrics = results.calculate_metrics()
    print(f"Config {i}: Sharpe = {metrics['sharpe_ratio']:.2f}")
```

## Best Practices

1. **Data Quality**: Always validate data using `data_manager.validate_data()`
2. **Memory Management**: Use `use_float32=True` for large universes (2000+ securities)
3. **Transaction Costs**: Calibrate cost parameters to your specific market/strategy
4. **ADV Constraints**: Conservative constraints (3-5% ADV) reduce market impact
5. **Factor Models**: Use updated factor exposures and covariances
6. **Hedging**: Enable beta hedging for market-neutral strategies
7. **Risk Limits**: Set realistic variance and exposure limits
8. **Testing**: Test on sample data before running on full universe

## Troubleshooting

### Out of Memory
- Set `use_float32=True`
- Process data in chunks
- Reduce universe size for testing

### Slow Performance
- Ensure vectorized operations (no explicit loops over securities)
- Use smaller date ranges for testing
- Disable progress bars for production runs

### Optimization Failures
- Relax constraints
- Increase `optimizer_max_iter`
- Check for infeasible constraints

### Missing Data
- Check CSV file formats
- Ensure date alignment across files
- Use `data_manager.validate_data()`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions, issues, or feature requests, please open an issue on GitHub.

## Acknowledgments

This framework implements best practices from:
- Quantitative portfolio management literature
- Industry-standard risk models (Barra, Axioma)
- Transaction cost modeling research
- Portfolio optimization theory

## Citation

If you use this framework in academic research, please cite:

```bibtex
@software{portfolio_backtester,
  title = {Portfolio Backtesting Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/backtesting}
}
```

---

**Happy Backtesting!** 🚀

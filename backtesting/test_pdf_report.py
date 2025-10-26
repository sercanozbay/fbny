"""
Test PDF report generation.

This script runs a quick backtest and generates a PDF report.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add backtesting to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting import Backtester, BacktestConfig
from backtesting.utils import get_date_range

# Generate simple test data
print("Generating test data...")
np.random.seed(42)
n_securities = 50
n_days = 60

dates = pd.date_range('2023-01-03', periods=n_days, freq='B')  # Business days

# Prices
tickers = [f'STOCK_{i:03d}' for i in range(n_securities)]
prices = pd.DataFrame(
    np.random.randn(n_days, n_securities).cumsum(axis=0) + 100,
    index=dates,
    columns=tickers
)

# ADV
adv = pd.DataFrame(
    np.random.uniform(100000, 1000000, (n_days, n_securities)),
    index=dates,
    columns=tickers
)

# Betas
betas = pd.DataFrame(
    np.random.uniform(0.8, 1.2, (n_days, n_securities)),
    index=dates,
    columns=tickers
)

# Factor model
n_factors = 3
factor_names = [f'Factor_{i}' for i in range(n_factors)]

factor_exposures = pd.DataFrame(
    np.random.randn(n_securities, n_factors) * 0.5,
    index=tickers,
    columns=factor_names
)

factor_returns = pd.DataFrame(
    np.random.randn(n_days, n_factors) * 0.01,
    index=dates,
    columns=factor_names
)

factor_cov = pd.DataFrame(
    np.eye(n_factors) * 0.0001,
    index=factor_names,
    columns=factor_names
)

specific_var = pd.DataFrame(
    np.random.uniform(0.0001, 0.0005, (n_days, n_securities)),
    index=dates,
    columns=tickers
)

# Sector mapping
sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
sector_mapping = pd.DataFrame({
    'ticker': tickers,
    'sector': np.random.choice(sectors, n_securities)
})

# Save data
print("Saving data...")
data_dir = Path('data/pdf_test')
data_dir.mkdir(parents=True, exist_ok=True)

prices.to_csv(data_dir / 'prices.csv')
adv.to_csv(data_dir / 'adv.csv')
betas.to_csv(data_dir / 'betas.csv')
factor_exposures.to_csv(data_dir / 'factor_exposures.csv')
factor_returns.to_csv(data_dir / 'factor_returns.csv')
factor_cov.to_csv(data_dir / 'factor_covariance.csv')
specific_var.to_csv(data_dir / 'specific_variance.csv')
sector_mapping.to_csv(data_dir / 'sector_mapping.csv', index=False)

# Create simple signals
print("Creating signals...")
signals = pd.DataFrame(
    np.random.randn(n_days, n_securities),
    index=dates,
    columns=tickers
)

# Configure backtest
print("Configuring backtest...")
from backtesting.data_loader import DataManager

config = BacktestConfig(
    enable_beta_hedge=True,
    enable_sector_hedge=True,
    max_adv_participation=0.05,
    tc_coefficient=0.001,
    tc_power=1.5,
    max_factor_exposure={'Factor_0': 0.1, 'Factor_1': 0.1, 'Factor_2': 0.1}
)

# Initialize data manager
print("Loading data...")
data_manager = DataManager(str(data_dir))

# Initialize backtester
print("Initializing backtester...")
backtester = Backtester(config, data_manager)

# Get aligned dates
aligned_start, aligned_end = get_date_range(prices)

# Run backtest
print("Running backtest...")
results = backtester.run(
    start_date=aligned_start,
    end_date=aligned_end,
    use_case=2,  # Signal-based
    inputs={
        'signals': {date: dict(signals.loc[date]) for date in signals.index}
    }
)

# Generate PDF report
print("\n" + "="*60)
print("GENERATING PDF REPORT")
print("="*60)

output_dir = Path('output/pdf_test')
output_dir.mkdir(parents=True, exist_ok=True)

# Generate full report with PDF
results.generate_full_report(
    output_dir=str(output_dir),
    formats=['pdf', 'html', 'excel']
)

print("\n" + "="*60)
print("PDF REPORT TEST COMPLETE!")
print("="*60)
print(f"\nCheck the following files:")
print(f"  PDF Report:   {output_dir / 'backtest_report.pdf'}")
print(f"  HTML Report:  {output_dir / 'backtest_report.html'}")
print(f"  Excel Report: {output_dir / 'backtest_report.xlsx'}")
print(f"  Charts:       {output_dir / 'charts/'}")
print("\nOpen the PDF to see the comprehensive backtest report with charts!\n")

"""
Sample data generator for backtesting framework.

This script generates realistic sample data for testing the backtester.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_sample_data(
    n_securities: int = 100,
    n_days: int = 252,
    n_factors: int = 5,
    output_dir: str = './sample_data',
    seed: int = 42
):
    """
    Generate sample data for backtesting.

    Parameters:
    -----------
    n_securities : int
        Number of securities
    n_days : int
        Number of trading days
    n_factors : int
        Number of factors
    output_dir : str
        Output directory
    seed : int
        Random seed
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample data...")
    print(f"Securities: {n_securities}")
    print(f"Days: {n_days}")
    print(f"Factors: {n_factors}")

    # Generate tickers
    tickers = [f"STOCK{i:04d}" for i in range(n_securities)]

    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')

    # === 1. Generate Prices ===
    print("\nGenerating prices...")
    initial_prices = np.random.uniform(10, 200, n_securities)

    # Generate correlated returns
    returns = np.random.normal(0.0005, 0.02, (n_days, n_securities))

    # Add market factor
    market_returns = np.random.normal(0.0006, 0.015, n_days)
    betas = np.random.uniform(0.5, 1.5, n_securities)
    for i in range(n_securities):
        returns[:, i] += betas[i] * market_returns

    # Calculate prices
    prices = np.zeros((n_days, n_securities))
    prices[0] = initial_prices
    for t in range(1, n_days):
        prices[t] = prices[t-1] * (1 + returns[t])

    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)
    prices_df.to_csv(output_path / 'prices.csv')
    print(f"  Saved prices.csv")

    # === 2. Generate ADV ===
    print("Generating ADV...")
    base_adv = np.random.lognormal(14, 1.5, n_securities)  # Mean ~1M shares
    adv_variation = np.random.normal(1.0, 0.1, (n_days, n_securities))
    adv_variation = np.clip(adv_variation, 0.5, 1.5)

    adv_data = base_adv * adv_variation
    adv_df = pd.DataFrame(adv_data, index=dates, columns=tickers)
    adv_df.to_csv(output_path / 'adv.csv')
    print(f"  Saved adv.csv")

    # === 3. Generate Betas ===
    print("Generating betas...")
    # Betas with some time variation
    beta_data = np.zeros((n_days, n_securities))
    for i in range(n_securities):
        base_beta = betas[i]
        beta_data[:, i] = base_beta + np.random.normal(0, 0.1, n_days)

    beta_df = pd.DataFrame(beta_data, index=dates, columns=tickers)
    beta_df.to_csv(output_path / 'betas.csv')
    print(f"  Saved betas.csv")

    # === 4. Generate Factor Model ===
    print("Generating factor model...")
    factor_names = [f"Factor{i+1}" for i in range(n_factors)]

    # Factor exposures (date, ticker) -> factors
    factor_exposures_list = []
    for date in dates:
        for ticker in tickers:
            exposures = np.random.normal(0, 1, n_factors)
            row = {'date': date, 'ticker': ticker}
            for i, factor in enumerate(factor_names):
                row[factor] = exposures[i]
            factor_exposures_list.append(row)

    factor_exposures_df = pd.DataFrame(factor_exposures_list)
    factor_exposures_df.to_csv(output_path / 'factor_exposures.csv', index=False)
    print(f"  Saved factor_exposures.csv")

    # Factor returns
    factor_returns = np.random.normal(0.0003, 0.01, (n_days, n_factors))
    factor_returns_df = pd.DataFrame(factor_returns, index=dates, columns=factor_names)
    factor_returns_df.to_csv(output_path / 'factor_returns.csv')
    print(f"  Saved factor_returns.csv")

    # Factor covariance (simplified - same for all dates)
    factor_corr = np.eye(n_factors)
    # Add some correlation
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            corr = np.random.uniform(-0.3, 0.3)
            factor_corr[i, j] = corr
            factor_corr[j, i] = corr

    factor_vol = np.random.uniform(0.08, 0.15, n_factors)
    factor_cov = np.outer(factor_vol, factor_vol) * factor_corr

    factor_cov_df = pd.DataFrame(factor_cov, index=factor_names, columns=factor_names)
    factor_cov_df.to_csv(output_path / 'factor_covariance.csv')
    print(f"  Saved factor_covariance.csv")

    # Specific variance
    specific_var = np.random.uniform(0.01, 0.04, (n_days, n_securities))
    specific_var_df = pd.DataFrame(specific_var, index=dates, columns=tickers)
    specific_var_df.to_csv(output_path / 'specific_variance.csv')
    print(f"  Saved specific_variance.csv")

    # === 5. Generate Sector Mapping ===
    print("Generating sector mapping...")
    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Industrial',
               'Energy', 'Materials', 'Utilities', 'RealEstate', 'Telecom']

    ticker_sectors = np.random.choice(sectors, n_securities)
    sector_df = pd.DataFrame({
        'ticker': tickers,
        'sector': ticker_sectors
    })
    sector_df.to_csv(output_path / 'sector_mapping.csv', index=False)
    print(f"  Saved sector_mapping.csv")

    # === 6. Generate Sample Signals ===
    print("Generating sample signals...")
    signals = np.random.normal(0, 1, (n_days, n_securities))
    signals_df = pd.DataFrame(signals, index=dates, columns=tickers)
    signals_df.to_csv(output_path / 'signals.csv')
    print(f"  Saved signals.csv")

    # === 7. Generate Sample Target Positions ===
    print("Generating sample target positions...")
    # Equal weight portfolio
    equal_weights = np.ones(n_securities) / n_securities
    equal_weights_df = pd.DataFrame(
        [equal_weights] * n_days,
        index=dates,
        columns=tickers
    )
    equal_weights_df.to_csv(output_path / 'target_weights.csv')
    print(f"  Saved target_weights.csv")

    print(f"\nSample data generated successfully in {output_dir}")

    # Create README
    readme_content = """# Sample Data Directory

This directory contains generated sample data for testing the backtester.

## Files:

- **prices.csv**: Daily close prices for all securities (dates × tickers)
- **adv.csv**: Average daily volume (dates × tickers)
- **betas.csv**: Market beta per security (dates × tickers)
- **factor_exposures.csv**: Factor loadings per security (date, ticker, factors)
- **factor_returns.csv**: Daily factor returns (dates × factors)
- **factor_covariance.csv**: Factor covariance matrix (factors × factors)
- **specific_variance.csv**: Idiosyncratic variance (dates × tickers)
- **sector_mapping.csv**: Sector classification (ticker, sector)
- **signals.csv**: Sample alpha signals (dates × tickers)
- **target_weights.csv**: Sample target weights (dates × tickers)

## Data Characteristics:

- Number of securities: {n_securities}
- Number of trading days: {n_days}
- Number of factors: {n_factors}
- Date range: {start_date} to {end_date}
- Sectors: 10 sectors

## Note:

This is synthetic data generated for testing purposes.
Do not use for actual trading decisions.
""".format(
        n_securities=n_securities,
        n_days=n_days,
        n_factors=n_factors,
        start_date=dates[0].date(),
        end_date=dates[-1].date()
    )

    with open(output_path / 'README.md', 'w') as f:
        f.write(readme_content)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample data for backtester')
    parser.add_argument('--securities', type=int, default=100, help='Number of securities')
    parser.add_argument('--days', type=int, default=252, help='Number of trading days')
    parser.add_argument('--factors', type=int, default=5, help='Number of factors')
    parser.add_argument('--output', type=str, default='./sample_data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    generate_sample_data(
        n_securities=args.securities,
        n_days=args.days,
        n_factors=args.factors,
        output_dir=args.output,
        seed=args.seed
    )

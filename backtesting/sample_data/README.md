# Sample Data Directory

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

- Number of securities: 1500
- Number of trading days: 1250
- Number of factors: 5
- Date range: 2023-01-02 to 2027-10-15
- Sectors: 10 sectors

## Note:

This is synthetic data generated for testing purposes.
Do not use for actual trading decisions.

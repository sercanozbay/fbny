"""
Data loading and management module.

This module handles loading all required data from CSV files with
memory-efficient operations and lazy loading where possible.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings

from .na_handling import (
    NAHandlingConfig, NAHandler, FillMethod,
    ensure_positive_definite, apply_shrinkage
)


class DataManager:
    """
    Manages loading and access to all backtest data.

    Uses lazy loading to minimize memory footprint and supports
    efficient slicing by date.
    """

    def __init__(
        self,
        data_dir: str,
        use_float32: bool = True,
        na_config: Optional[NAHandlingConfig] = None,
        enable_na_handling: bool = True
    ):
        """
        Initialize data manager.

        Parameters:
        -----------
        data_dir : str
            Directory containing CSV data files
        use_float32 : bool
            Use float32 instead of float64 for memory efficiency
        na_config : NAHandlingConfig, optional
            Configuration for NA handling. If None, uses defaults.
        enable_na_handling : bool
            Enable automatic NA handling on load
        """
        self.data_dir = Path(data_dir)
        self.use_float32 = use_float32
        self.dtype = np.float32 if use_float32 else np.float64

        # NA handling configuration
        self.enable_na_handling = enable_na_handling
        self.na_config = na_config if na_config is not None else NAHandlingConfig()
        self.na_handler = NAHandler(self.na_config)

        # Cached data
        self._prices: Optional[pd.DataFrame] = None
        self._trade_prices: Optional[pd.DataFrame] = None
        self._adv: Optional[pd.DataFrame] = None
        self._betas: Optional[pd.DataFrame] = None
        self._factor_exposures: Optional[pd.DataFrame] = None
        self._factor_returns: Optional[pd.DataFrame] = None
        self._factor_covariance: Optional[pd.DataFrame] = None
        self._specific_variance: Optional[pd.DataFrame] = None
        self._sector_mapping: Optional[pd.DataFrame] = None
        self._external_trades: Optional[pd.DataFrame] = None

        self._date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._tickers: Optional[List[str]] = None

    def load_prices(self, filename: str = 'prices.csv') -> pd.DataFrame:
        """
        Load price data with automatic NA handling.

        Expected format: CSV with 'date' index and ticker columns.

        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index, tickers as columns
        """
        if self._prices is None:
            filepath = self.data_dir / filename
            self._prices = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True,
            ).astype(self.dtype)
            self._prices.index = pd.to_datetime(self._prices.index)
            print(f"Loaded prices: {self._prices.shape[0]} dates, {self._prices.shape[1]} securities")

            # Apply NA handling
            if self.enable_na_handling:
                self._prices = self.na_handler.handle_timeseries_data(
                    df=self._prices,
                    data_type="Prices",
                    method=self.na_config.prices_method,
                    max_gap=self.na_config.prices_max_gap,
                    drop_threshold=self.na_config.prices_drop_threshold,
                    min_value=0.0  # Prices must be positive
                )

                # Check if too many NAs remain
                na_pct = self._prices.isna().sum().sum() / (self._prices.shape[0] * self._prices.shape[1])
                if na_pct > self.na_config.prices_fail_threshold:
                    raise ValueError(
                        f"Prices contain {na_pct:.2%} missing values after NA handling "
                        f"(threshold: {self.na_config.prices_fail_threshold:.2%}). "
                        f"Please check data quality or adjust NA handling configuration."
                    )

        return self._prices

    def load_trade_prices(self, filename: str = 'trade_prices.csv') -> Optional[pd.DataFrame]:
        """
        Load execution prices (if different from close prices).

        Returns None if file doesn't exist.
        """
        if self._trade_prices is None:
            filepath = self.data_dir / filename
            if filepath.exists():
                self._trade_prices = pd.read_csv(
                    filepath,
                    index_col=0,
                    parse_dates=True,
                ).astype(self.dtype)
                self._trade_prices.index = pd.to_datetime(self._trade_prices.index)
                print(f"Loaded trade prices: {self._trade_prices.shape[0]} dates, {self._trade_prices.shape[1]} securities")
            else:
                warnings.warn(f"Trade prices file not found: {filepath}. Will use close prices.")

        return self._trade_prices

    def load_adv(self, filename: str = 'adv.csv') -> pd.DataFrame:
        """
        Load average daily volume data with automatic NA handling.

        Expected format: CSV with 'date' index and ticker columns.
        """
        if self._adv is None:
            filepath = self.data_dir / filename
            self._adv = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True,
            ).astype(self.dtype)
            self._adv.index = pd.to_datetime(self._adv.index)
            print(f"Loaded ADV: {self._adv.shape[0]} dates, {self._adv.shape[1]} securities")

            # Apply NA handling
            if self.enable_na_handling:
                self._adv = self.na_handler.handle_timeseries_data(
                    df=self._adv,
                    data_type="ADV",
                    method=self.na_config.adv_method,
                    max_gap=self.na_config.adv_max_gap,
                    default_value=self.na_config.adv_default_value,
                    drop_threshold=self.na_config.adv_drop_threshold,
                    min_value=self.na_config.adv_default_value  # ADV must be positive
                )

        return self._adv

    def load_betas(self, filename: str = 'betas.csv') -> pd.DataFrame:
        """
        Load beta data per security per date with automatic NA handling.

        Expected format: CSV with 'date' index and ticker columns.
        """
        if self._betas is None:
            filepath = self.data_dir / filename
            self._betas = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True,
            ).astype(self.dtype)
            self._betas.index = pd.to_datetime(self._betas.index)
            print(f"Loaded betas: {self._betas.shape[0]} dates, {self._betas.shape[1]} securities")

            # Apply NA handling
            if self.enable_na_handling:
                self._betas = self.na_handler.handle_timeseries_data(
                    df=self._betas,
                    data_type="Betas",
                    method=self.na_config.beta_method,
                    max_gap=self.na_config.beta_max_gap,
                    default_value=self.na_config.beta_default_value,
                    drop_threshold=self.na_config.beta_drop_threshold,
                    min_value=self.na_config.beta_min_value,
                    max_value=self.na_config.beta_max_value
                )

        return self._betas

    def load_factor_exposures(self, filename: str = 'factor_exposures.csv') -> pd.DataFrame:
        """
        Load factor exposures with automatic NA handling.

        Expected format: CSV with MultiIndex (date, ticker) and factor columns.
        """
        if self._factor_exposures is None:
            filepath = self.data_dir / filename
            self._factor_exposures = pd.read_csv(
                filepath,
                index_col=[0, 1],
                parse_dates=[0]
            ).astype(self.dtype)
            self._factor_exposures.index = pd.MultiIndex.from_tuples([
                (pd.to_datetime(date), ticker)
                for date, ticker in self._factor_exposures.index
            ], names=['date', 'ticker'])
            print(f"Loaded factor exposures: {len(self._factor_exposures.index.unique(level=0))} dates, "
                  f"{len(self._factor_exposures.index.unique(level=1))} securities, "
                  f"{self._factor_exposures.shape[1]} factors")

            # Apply NA handling
            if self.enable_na_handling:
                self._factor_exposures = self.na_handler.handle_multiindex_data(
                    df=self._factor_exposures,
                    data_type="Factor Exposures",
                    method=self.na_config.factor_exposures_method,
                    max_gap=self.na_config.factor_exposures_max_gap,
                    use_cross_sectional=self.na_config.factor_exposures_use_cross_sectional,
                    drop_threshold=self.na_config.factor_exposures_drop_threshold
                )

        return self._factor_exposures

    def load_factor_returns(self, filename: str = 'factor_returns.csv') -> pd.DataFrame:
        """
        Load factor returns with automatic NA handling.

        Expected format: CSV with 'date' index and factor columns.
        """
        if self._factor_returns is None:
            filepath = self.data_dir / filename
            self._factor_returns = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True,
            ).astype(self.dtype)
            self._factor_returns.index = pd.to_datetime(self._factor_returns.index)
            print(f"Loaded factor returns: {self._factor_returns.shape[0]} dates, {self._factor_returns.shape[1]} factors")

            # Apply NA handling
            if self.enable_na_handling:
                self._factor_returns = self.na_handler.handle_timeseries_data(
                    df=self._factor_returns,
                    data_type="Factor Returns",
                    method=self.na_config.factor_returns_method,
                    max_gap=None  # Use zero fill for missing returns
                )

        return self._factor_returns

    def load_factor_covariance(self, filename: str = 'factor_covariance.csv') -> pd.DataFrame:
        """
        Load factor covariance matrix with automatic NA handling and regularization.

        Expected format: CSV with 'date' index and factor×factor covariance.
        For simplicity, can also load a single covariance matrix (no date dimension).
        """
        if self._factor_covariance is None:
            filepath = self.data_dir / filename
            self._factor_covariance = pd.read_csv(
                filepath,
                index_col=0,
            ).astype(self.dtype)
            print(f"Loaded factor covariance: {self._factor_covariance.shape}")

            # Apply NA handling and regularization
            if self.enable_na_handling:
                # Fill NAs with forward fill
                self._factor_covariance = self._factor_covariance.ffill()

                # Ensure positive definiteness
                self._factor_covariance = ensure_positive_definite(self._factor_covariance)

                # Apply shrinkage if configured
                if self.na_config.factor_covariance_regularization:
                    self._factor_covariance = apply_shrinkage(
                        self._factor_covariance,
                        shrinkage=self.na_config.factor_covariance_shrinkage
                    )

        return self._factor_covariance

    def load_specific_variance(self, filename: str = 'specific_variance.csv') -> pd.DataFrame:
        """
        Load specific (idiosyncratic) variance per security per date with automatic NA handling.

        Expected format: CSV with 'date' index and ticker columns.
        """
        if self._specific_variance is None:
            filepath = self.data_dir / filename
            self._specific_variance = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True,
            ).astype(self.dtype)
            self._specific_variance.index = pd.to_datetime(self._specific_variance.index)
            print(f"Loaded specific variance: {self._specific_variance.shape[0]} dates, {self._specific_variance.shape[1]} securities")

            # Apply NA handling
            if self.enable_na_handling:
                self._specific_variance = self.na_handler.handle_timeseries_data(
                    df=self._specific_variance,
                    data_type="Specific Variance",
                    method=self.na_config.specific_variance_method,
                    max_gap=self.na_config.specific_variance_max_gap,
                    drop_threshold=self.na_config.specific_variance_drop_threshold,
                    min_value=1e-8  # Variance must be positive
                )

                # Use cross-sectional median as fallback if configured
                if self.na_config.specific_variance_use_cross_sectional and self._specific_variance.isna().any().any():
                    for date in self._specific_variance.index:
                        row_median = self._specific_variance.loc[date].median()
                        if pd.notna(row_median):
                            self._specific_variance.loc[date] = self._specific_variance.loc[date].fillna(
                                row_median * self.na_config.specific_variance_safety_factor
                            )

        return self._specific_variance

    def load_sector_mapping(self, filename: str = 'sector_mapping.csv') -> pd.DataFrame:
        """
        Load sector mapping for securities with automatic NA handling.

        Expected format: CSV with 'ticker' and 'sector' columns.
        """
        if self._sector_mapping is None:
            filepath = self.data_dir / filename
            self._sector_mapping = pd.read_csv(filepath)
            print(f"Loaded sector mapping: {len(self._sector_mapping)} securities, "
                  f"{self._sector_mapping['sector'].nunique()} sectors")

            # Apply NA handling
            if self.enable_na_handling:
                # Fill missing sectors with default
                if 'sector' in self._sector_mapping.columns:
                    na_count = self._sector_mapping['sector'].isna().sum()
                    if na_count > 0:
                        self._sector_mapping['sector'] = self._sector_mapping['sector'].fillna(
                            self.na_config.sector_mapping_default_sector
                        )
                        print(f"  Filled {na_count} missing sectors with '{self.na_config.sector_mapping_default_sector}'")

        return self._sector_mapping

    def load_external_trades(self, filename: str = 'external_trades.csv') -> pd.DataFrame:
        """
        Load external trades from CSV with optional tag support.

        Expected format: CSV with columns:
        - date: Trade date (will be parsed as datetime)
        - ticker: Security ticker
        - qty: Trade quantity (positive for buy, negative for sell)
        - price: Execution price
        - tag: (Optional) Tag for attribution (e.g., counterparty name)

        Returns:
        --------
        pd.DataFrame
            DataFrame with external trades

        Example CSV:
        ------------
        date,ticker,qty,price,tag
        2023-01-02,STOCK0000,1000,150.25,Goldman Sachs
        2023-01-02,STOCK0001,-500,200.50,Morgan Stanley
        2023-01-03,STOCK0000,500,151.00,JPMorgan
        """
        if self._external_trades is None:
            filepath = self.data_dir / filename
            if not filepath.exists():
                warnings.warn(f"External trades file not found: {filepath}")
                return pd.DataFrame()

            self._external_trades = pd.read_csv(
                filepath,
                parse_dates=['date']
            )
            self._external_trades['date'] = pd.to_datetime(self._external_trades['date'])

            # Validate required columns
            required_cols = ['date', 'ticker', 'qty', 'price']
            missing_cols = [col for col in required_cols if col not in self._external_trades.columns]
            if missing_cols:
                raise ValueError(
                    f"External trades CSV missing required columns: {missing_cols}. "
                    f"Required: {required_cols}. Optional: ['tag']"
                )

            # Add tag column if not present (for backward compatibility)
            if 'tag' not in self._external_trades.columns:
                self._external_trades['tag'] = None

            # Convert types
            self._external_trades['qty'] = self._external_trades['qty'].astype(float)
            self._external_trades['price'] = self._external_trades['price'].astype(self.dtype)

            print(f"Loaded external trades: {len(self._external_trades)} trades, "
                  f"{self._external_trades['date'].nunique()} dates, "
                  f"{self._external_trades['ticker'].nunique()} tickers")

            # Apply NA handling
            if self.enable_na_handling:
                self._external_trades = self.na_handler.handle_external_trades(
                    df=self._external_trades,
                    strict=self.na_config.external_trades_strict,
                    allow_price_lookup=self.na_config.external_trades_allow_price_lookup,
                    prices_df=self._prices  # May be None if prices not loaded yet
                )

            # Show tag summary if tags are present
            if self._external_trades['tag'].notna().any():
                n_tags = self._external_trades['tag'].nunique()
                print(f"  Tags found: {n_tags} unique tags")

        return self._external_trades

    def get_external_trades_by_date(self) -> Dict[pd.Timestamp, Dict[str, List[Dict]]]:
        """
        Convert external trades DataFrame to the format expected by Use Case 3.

        Returns:
        --------
        Dict[pd.Timestamp, Dict[str, List[Dict]]]
            Nested dict: {date: {ticker: [{'qty': X, 'price': Y, 'tag': Z}, ...]}}

        Example:
        --------
        >>> data_manager = DataManager('../data')
        >>> trades_by_date = data_manager.get_external_trades_by_date()
        >>> print(trades_by_date[pd.Timestamp('2023-01-02')])
        {
            'STOCK0000': [{'qty': 1000.0, 'price': 150.25, 'tag': 'Goldman Sachs'}],
            'STOCK0001': [{'qty': -500.0, 'price': 200.50, 'tag': 'Morgan Stanley'}]
        }
        """
        trades_df = self.load_external_trades()

        if trades_df.empty:
            return {}

        trades_by_date = {}

        for date in trades_df['date'].unique():
            date_trades = trades_df[trades_df['date'] == date]

            ticker_trades = {}
            for ticker in date_trades['ticker'].unique():
                ticker_rows = date_trades[date_trades['ticker'] == ticker]

                # Group into list of trade dicts
                trade_list = []
                for _, row in ticker_rows.iterrows():
                    trade_dict = {
                        'qty': float(row['qty']),
                        'price': float(row['price'])
                    }
                    # Add tag if present
                    if pd.notna(row['tag']):
                        trade_dict['tag'] = str(row['tag'])

                    trade_list.append(trade_dict)

                ticker_trades[ticker] = trade_list

            trades_by_date[pd.Timestamp(date)] = ticker_trades

        return trades_by_date

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get the available date range from price data."""
        if self._date_range is None:
            prices = self.load_prices()
            self._date_range = (prices.index.min(), prices.index.max())

        return self._date_range

    def get_tickers(self) -> List[str]:
        """Get list of all tickers."""
        if self._tickers is None:
            prices = self.load_prices()
            self._tickers = prices.columns.tolist()

        return self._tickers

    def get_na_handling_report(self) -> str:
        """
        Get summary report of all NA handling operations performed.

        Returns:
        --------
        str
            Formatted report showing NA handling statistics for all loaded data
        """
        return self.na_handler.get_summary_report()

    def clear_na_reports(self):
        """Clear all NA handling reports."""
        self.na_handler.clear_reports()

    def get_data_for_date(self, date: pd.Timestamp) -> Dict:
        """
        Get all data for a specific date.

        Returns:
        --------
        dict
            Dictionary containing all available data for the date
        """
        data = {}

        # Prices
        if self._prices is not None and date in self._prices.index:
            data['prices'] = self._prices.loc[date].to_dict()

        # Trade prices
        if self._trade_prices is not None and date in self._trade_prices.index:
            data['trade_prices'] = self._trade_prices.loc[date].to_dict()

        # ADV
        if self._adv is not None and date in self._adv.index:
            data['adv'] = self._adv.loc[date].to_dict()

        # Betas
        if self._betas is not None and date in self._betas.index:
            data['betas'] = self._betas.loc[date].to_dict()

        # Factor exposures
        if self._factor_exposures is not None and date in self._factor_exposures.index.get_level_values(0):
            data['factor_exposures'] = self._factor_exposures.loc[date]

        # Factor returns
        if self._factor_returns is not None and date in self._factor_returns.index:
            data['factor_returns'] = self._factor_returns.loc[date].to_dict()

        # Specific variance
        if self._specific_variance is not None and date in self._specific_variance.index:
            data['specific_variance'] = self._specific_variance.loc[date].to_dict()

        # Sector mapping (date-independent)
        if self._sector_mapping is not None:
            data['sector_mapping'] = self._sector_mapping.set_index('ticker')['sector'].to_dict()

        return data

    def validate_data(self) -> List[str]:
        """
        Validate loaded data for consistency.

        Returns:
        --------
        List[str]
            List of validation warnings/errors
        """
        issues = []

        # Load all data
        prices = self.load_prices()
        adv = self.load_adv()

        # Check date alignment
        if not prices.index.equals(adv.index):
            issues.append("Price and ADV dates do not match")

        # Check ticker alignment
        if set(prices.columns) != set(adv.columns):
            issues.append(f"Price and ADV tickers do not match. "
                        f"Prices: {len(prices.columns)}, ADV: {len(adv.columns)}")

        # Check for missing values
        price_missing = prices.isna().sum().sum()
        if price_missing > 0:
            issues.append(f"Prices contain {price_missing} missing values")

        adv_missing = adv.isna().sum().sum()
        if adv_missing > 0:
            issues.append(f"ADV contains {adv_missing} missing values")

        # Check for negative/zero prices
        if (prices <= 0).any().any():
            issues.append("Prices contain zero or negative values")

        # Check for negative/zero ADV
        if (adv <= 0).any().any():
            issues.append("ADV contains zero or negative values")

        return issues

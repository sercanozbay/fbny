"""
Data loading and management module.

This module handles loading all required data from CSV files with
memory-efficient operations and lazy loading where possible.

Also provides LargeDataLoader for institutional-grade datasets with
5000+ securities, supporting corporate action adjustments and time-varying
classifications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import warnings

from .na_handling import (
    NAHandlingConfig, NAHandler,
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


class LargeDataLoader:
    """
    Loader for large institutional datasets with ticker+date indexing.

    Handles:
    - Large price files (5000+ securities)
    - ADV (Average Daily Volume) data
    - Beta data
    - Price adjustments/corporate actions
    - Date-based sector mappings
    - Date-based factor exposures
    - Date-based factor covariances
    """

    def __init__(self, data_dir: str, use_float32: bool = True):
        """
        Initialize large data loader.

        Parameters:
        -----------
        data_dir : str
            Directory containing large data files
        use_float32 : bool
            Use float32 for memory efficiency (recommended for large datasets)
        """
        self.data_dir = Path(data_dir)
        self.dtype = np.float32 if use_float32 else np.float64

    def load_prices_with_adjustments(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        prices_file: str = 'prices_large.parquet',
        adjustments_file: Optional[str] = 'price_adjustments.parquet',
        apply_adjustments: bool = True
    ) -> pd.DataFrame:
        """
        Load raw prices and apply corporate action adjustments.

        Expected format for prices file:
        - Index: MultiIndex (date, ticker) or columns: [date, ticker, price]
        - Parquet format recommended for large files

        Expected format for adjustments file:
        - Columns: [date, ticker, adjustment_factor]
        - adjustment_factor: Multiplier to apply (e.g., 0.5 for 2-for-1 split)

        Parameters:
        -----------
        universe : List[str]
            List of ticker symbols to load
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        prices_file : str
            Name of prices file in data_dir
        adjustments_file : str, optional
            Name of adjustments file (None to skip adjustments)
        apply_adjustments : bool
            Whether to apply adjustments

        Returns:
        --------
        pd.DataFrame
            Prices DataFrame with dates as index, tickers as columns
        """
        filepath = self.data_dir / prices_file

        # Load prices file
        print(f"Loading prices from {filepath}...")

        if filepath.suffix == '.parquet':
            # Load parquet (efficient for large files)
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            # Load CSV with date parsing
            df = pd.read_csv(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Handle different input formats
        if isinstance(df.index, pd.MultiIndex):
            # Format: MultiIndex (date, ticker) with 'price' column
            df = df.reset_index()

        # Ensure we have required columns
        if 'date' not in df.columns or 'ticker' not in df.columns:
            raise ValueError("Prices file must have 'date' and 'ticker' columns")

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter by date range
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

        # Filter by universe
        df = df[df['ticker'].isin(universe)]

        print(f"  Filtered to {len(df['ticker'].unique())} tickers, "
              f"{len(df['date'].unique())} dates, "
              f"{len(df)} rows")

        # Get price column name
        price_col = 'price' if 'price' in df.columns else 'close'

        if price_col not in df.columns:
            # Try to find a suitable column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                warnings.warn(f"Using '{price_col}' as price column")
            else:
                raise ValueError("Could not find price column in data")

        # Apply adjustments if requested
        if apply_adjustments and adjustments_file is not None:
            adj_filepath = self.data_dir / adjustments_file

            if adj_filepath.exists():
                print(f"Loading price adjustments from {adj_filepath}...")
                df = self._apply_price_adjustments(df, adj_filepath, price_col,
                                                   start_date, end_date, universe)
            else:
                warnings.warn(f"Adjustments file not found: {adj_filepath}")

        # Pivot to date x ticker format
        prices = df.pivot(index='date', columns='ticker', values=price_col)

        # Sort and ensure correct dtype
        prices = prices.sort_index()
        prices = prices.astype(self.dtype)

        print(f"✓ Loaded prices: {prices.shape[0]} dates × {prices.shape[1]} tickers")

        return prices

    def _apply_price_adjustments(
        self,
        prices_df: pd.DataFrame,
        adj_filepath: Path,
        price_col: str,
        start_date: str,
        end_date: str,
        universe: List[str]
    ) -> pd.DataFrame:
        """
        Apply corporate action adjustments to prices.

        Adjustments are applied backward from the adjustment date.
        E.g., for a 2-for-1 split on 2023-06-15:
        - Prices on/after 2023-06-15: No adjustment
        - Prices before 2023-06-15: Multiply by 0.5
        """
        # Load adjustments
        if adj_filepath.suffix == '.parquet':
            adj_df = pd.read_parquet(adj_filepath)
        else:
            adj_df = pd.read_csv(adj_filepath, parse_dates=['date'])

        adj_df['date'] = pd.to_datetime(adj_df['date'])

        # Filter adjustments by date range and universe
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Include adjustments before start_date (they affect our prices)
        adj_df = adj_df[adj_df['date'] <= end_dt]
        adj_df = adj_df[adj_df['ticker'].isin(universe)]

        if len(adj_df) == 0:
            print("  No adjustments found for this period")
            return prices_df

        print(f"  Applying {len(adj_df)} price adjustments...")

        # Sort adjustments by date (oldest first)
        adj_df = adj_df.sort_values('date')

        # Create adjusted price column
        prices_df['adjusted_price'] = prices_df[price_col]

        # Apply adjustments backward in time
        for ticker in adj_df['ticker'].unique():
            ticker_adjs = adj_df[adj_df['ticker'] == ticker].sort_values('date')
            ticker_mask = prices_df['ticker'] == ticker

            # Calculate cumulative adjustment factor for each date
            for _, adj_row in ticker_adjs.iterrows():
                adj_date = adj_row['date']
                adj_factor = adj_row['adjustment_factor']

                # Apply to prices before adjustment date
                date_mask = prices_df['date'] < adj_date
                mask = ticker_mask & date_mask

                prices_df.loc[mask, 'adjusted_price'] *= adj_factor

        # Replace original price with adjusted price
        prices_df[price_col] = prices_df['adjusted_price']
        prices_df = prices_df.drop('adjusted_price', axis=1)

        return prices_df

    def load_adv(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        adv_file: str = 'adv_large.parquet'
    ) -> pd.DataFrame:
        """
        Load Average Daily Volume data for universe and date range.

        Expected format:
        - Index: MultiIndex (date, ticker) or columns: [date, ticker, adv]

        Parameters:
        -----------
        universe : List[str]
            List of tickers
        start_date : str
            Start date
        end_date : str
            End date
        adv_file : str
            Name of ADV file

        Returns:
        --------
        pd.DataFrame
            ADV DataFrame with dates as index, tickers as columns
        """
        filepath = self.data_dir / adv_file

        print(f"Loading ADV from {filepath}...")

        # Load file
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Handle MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Convert and filter
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        df = df[df['ticker'].isin(universe)]

        # Get ADV column
        adv_col = 'adv' if 'adv' in df.columns else 'volume'
        if adv_col not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                adv_col = numeric_cols[0]
                warnings.warn(f"Using '{adv_col}' as ADV column")
            else:
                raise ValueError("Could not find ADV column")

        # Pivot to date x ticker
        adv = df.pivot(index='date', columns='ticker', values=adv_col)
        adv = adv.sort_index()
        adv = adv.astype(self.dtype)

        print(f"✓ Loaded ADV: {adv.shape[0]} dates × {adv.shape[1]} tickers")

        return adv

    def load_betas(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        beta_file: str = 'betas_large.parquet'
    ) -> pd.DataFrame:
        """
        Load beta data for universe and date range.

        Expected format:
        - Index: MultiIndex (date, ticker) or columns: [date, ticker, beta]

        Parameters:
        -----------
        universe : List[str]
            List of tickers
        start_date : str
            Start date
        end_date : str
            End date
        beta_file : str
            Name of beta file

        Returns:
        --------
        pd.DataFrame
            Beta DataFrame with dates as index, tickers as columns
        """
        filepath = self.data_dir / beta_file

        print(f"Loading betas from {filepath}...")

        # Load file
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Handle MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Convert and filter
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        df = df[df['ticker'].isin(universe)]

        # Get beta column
        beta_col = 'beta' if 'beta' in df.columns else df.select_dtypes(include=[np.number]).columns[0]

        # Pivot to date x ticker
        betas = df.pivot(index='date', columns='ticker', values=beta_col)
        betas = betas.sort_index()
        betas = betas.astype(self.dtype)

        print(f"✓ Loaded betas: {betas.shape[0]} dates × {betas.shape[1]} tickers")

        return betas

    def load_sector_mapping_with_dates(
        self,
        universe: List[str],
        date: str,
        sector_file: str = 'sector_mapping_dated.parquet'
    ) -> pd.DataFrame:
        """
        Load sector mapping with date support (sectors can change over time).

        Expected format:
        - Columns: [date, ticker, sector] or [effective_date, ticker, sector]
        - Multiple rows per ticker for sector changes

        Parameters:
        -----------
        universe : List[str]
            List of tickers
        date : str
            Date for which to get sector mapping
        sector_file : str
            Name of sector mapping file

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns [ticker, sector]
        """
        filepath = self.data_dir / sector_file

        print(f"Loading sector mapping from {filepath}...")

        # Load file
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=[0])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Identify date column
        date_col = 'date' if 'date' in df.columns else 'effective_date'
        if date_col not in df.columns:
            # Assume static sector mapping (no dates)
            df = df[df['ticker'].isin(universe)]
            print(f"✓ Loaded sector mapping: {len(df)} tickers (static)")
            return df[['ticker', 'sector']]

        # Convert date
        df[date_col] = pd.to_datetime(df[date_col])
        target_date = pd.to_datetime(date)

        # Filter by universe
        df = df[df['ticker'].isin(universe)]

        # Get sector as of target date (use most recent effective date <= target)
        df = df[df[date_col] <= target_date]

        # Keep only most recent entry per ticker
        df = df.sort_values(date_col)
        df = df.groupby('ticker').tail(1)

        print(f"✓ Loaded sector mapping: {len(df)} tickers as of {date}")

        return df[['ticker', 'sector']]

    def load_factor_exposures_with_dates(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        exposures_file: str = 'factor_exposures_large.parquet'
    ) -> pd.DataFrame:
        """
        Load factor exposures for universe and date range.

        Expected format:
        - Index: MultiIndex (date, ticker)
        - Columns: Factor names (Factor1, Factor2, etc.)

        Parameters:
        -----------
        universe : List[str]
            List of tickers
        start_date : str
            Start date
        end_date : str
            End date
        exposures_file : str
            Name of exposures file

        Returns:
        --------
        pd.DataFrame
            Factor exposures with MultiIndex (date, ticker)
        """
        filepath = self.data_dir / exposures_file

        print(f"Loading factor exposures from {filepath}...")

        # Load file
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=[0])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Handle different formats
        if not isinstance(df.index, pd.MultiIndex):
            # Create MultiIndex from date and ticker columns
            if 'date' in df.columns and 'ticker' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index(['date', 'ticker'])
            else:
                raise ValueError("Expected MultiIndex or date/ticker columns")

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        dates = df.index.get_level_values(0)
        df = df[(dates >= start_dt) & (dates <= end_dt)]

        # Filter by universe
        tickers = df.index.get_level_values(1)
        df = df[tickers.isin(universe)]

        # Sort index
        df = df.sort_index()
        df = df.astype(self.dtype)

        n_dates = len(df.index.get_level_values(0).unique())
        n_tickers = len(df.index.get_level_values(1).unique())
        n_factors = df.shape[1]

        print(f"✓ Loaded factor exposures: {n_dates} dates × {n_tickers} tickers × {n_factors} factors")

        return df

    def load_factor_covariance_with_dates(
        self,
        start_date: str,
        end_date: str,
        covariance_file: str = 'factor_covariance_dated.parquet'
    ) -> Union[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
        """
        Load factor covariance matrices.

        Expected format:
        - Time-varying: MultiIndex (date, factor1) with factor columns
        - Static: factor × factor matrix

        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date
        covariance_file : str
            Name of covariance file

        Returns:
        --------
        Union[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]
            If time-varying: Dict mapping date -> covariance matrix
            If static: Single covariance DataFrame
        """
        filepath = self.data_dir / covariance_file

        print(f"Loading factor covariance from {filepath}...")

        # Load file
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=[0] if 'date' in pd.read_csv(filepath, nrows=1).columns else None)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Check if time-varying
        has_date = 'date' in df.columns or (isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names)

        if not has_date:
            # Static covariance matrix
            print(f"✓ Loaded static factor covariance: {df.shape}")
            return df.astype(self.dtype)

        # Time-varying covariance
        if not isinstance(df.index, pd.MultiIndex):
            # Create MultiIndex
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                # Assume first column after date is factor index
                factor_col = df.columns[1]
                df = df.set_index(['date', factor_col])
            else:
                raise ValueError("Could not determine covariance structure")

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        dates = df.index.get_level_values(0)
        df = df[(dates >= start_dt) & (dates <= end_dt)]

        # Convert to dict of covariance matrices by date
        cov_by_date = {}
        for date in df.index.get_level_values(0).unique():
            cov_matrix = df.loc[date]
            cov_by_date[date] = cov_matrix.astype(self.dtype)

        print(f"✓ Loaded time-varying factor covariance: {len(cov_by_date)} dates")

        return cov_by_date

    def load_specific_variance(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        variance_file: str = 'specific_variance_large.parquet'
    ) -> pd.DataFrame:
        """
        Load specific (idiosyncratic) variance data.

        Expected format:
        - Index: MultiIndex (date, ticker) or columns: [date, ticker, variance]

        Parameters:
        -----------
        universe : List[str]
            List of tickers
        start_date : str
            Start date
        end_date : str
            End date
        variance_file : str
            Name of variance file

        Returns:
        --------
        pd.DataFrame
            Variance DataFrame with dates as index, tickers as columns
        """
        filepath = self.data_dir / variance_file

        print(f"Loading specific variance from {filepath}...")

        # Load file
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Handle MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Convert and filter
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        df = df[df['ticker'].isin(universe)]

        # Get variance column
        var_col = 'variance' if 'variance' in df.columns else 'specific_variance'
        if var_col not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                var_col = numeric_cols[0]
                warnings.warn(f"Using '{var_col}' as variance column")

        # Pivot to date x ticker
        variance = df.pivot(index='date', columns='ticker', values=var_col)
        variance = variance.sort_index()
        variance = variance.astype(self.dtype)

        print(f"✓ Loaded specific variance: {variance.shape[0]} dates × {variance.shape[1]} tickers")

        return variance

    def save_subset(
        self,
        data: Union[pd.DataFrame, Dict],
        output_file: str,
        format: str = 'parquet'
    ):
        """
        Save extracted subset to file.

        Parameters:
        -----------
        data : Union[pd.DataFrame, Dict]
            Data to save
        output_file : str
            Output filename
        format : str
            'parquet' or 'csv'
        """
        output_path = self.data_dir / output_file

        if isinstance(data, dict):
            # Save dictionary (e.g., time-varying covariance)
            import pickle
            with open(output_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved to {output_path.with_suffix('.pkl')}")
        elif format == 'parquet':
            data.to_parquet(output_path)
            print(f"✓ Saved to {output_path}")
        elif format == 'csv':
            data.to_csv(output_path)
            print(f"✓ Saved to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")


def convert_to_backtester_format(
    prices: pd.DataFrame,
    adv: pd.DataFrame,
    betas: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    factor_exposures: Optional[pd.DataFrame] = None,
    factor_returns: Optional[pd.DataFrame] = None,
    factor_covariance: Optional[Union[pd.DataFrame, Dict]] = None,
    specific_variance: Optional[pd.DataFrame] = None,
    output_dir: str = './backtester_data'
) -> Dict[str, str]:
    """
    Convert loaded data to backtester-compatible format and save.

    Parameters:
    -----------
    prices : pd.DataFrame
        Prices (date × ticker)
    adv : pd.DataFrame
        ADV (date × ticker)
    betas : pd.DataFrame
        Betas (date × ticker)
    sector_mapping : pd.DataFrame
        Sector mapping (ticker, sector columns)
    factor_exposures : pd.DataFrame, optional
        Factor exposures (MultiIndex: date, ticker)
    factor_returns : pd.DataFrame, optional
        Factor returns (date × factor)
    factor_covariance : Union[pd.DataFrame, Dict], optional
        Factor covariance (factor × factor or time-varying)
    specific_variance : pd.DataFrame, optional
        Specific variance (date × ticker)
    output_dir : str
        Output directory

    Returns:
    --------
    Dict[str, str]
        Mapping of data type -> output file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save prices
    prices_file = output_path / 'prices.csv'
    prices.to_csv(prices_file)
    saved_files['prices'] = str(prices_file)
    print(f"✓ Saved prices to {prices_file}")

    # Save ADV
    adv_file = output_path / 'adv.csv'
    adv.to_csv(adv_file)
    saved_files['adv'] = str(adv_file)
    print(f"✓ Saved ADV to {adv_file}")

    # Save betas
    betas_file = output_path / 'betas.csv'
    betas.to_csv(betas_file)
    saved_files['betas'] = str(betas_file)
    print(f"✓ Saved betas to {betas_file}")

    # Save sector mapping
    sector_file = output_path / 'sector_mapping.csv'
    sector_mapping.to_csv(sector_file, index=False)
    saved_files['sector_mapping'] = str(sector_file)
    print(f"✓ Saved sector mapping to {sector_file}")

    # Save factor exposures if provided
    if factor_exposures is not None:
        exposures_file = output_path / 'factor_exposures.csv'
        factor_exposures.to_csv(exposures_file)
        saved_files['factor_exposures'] = str(exposures_file)
        print(f"✓ Saved factor exposures to {exposures_file}")

    # Save factor returns if provided
    if factor_returns is not None:
        returns_file = output_path / 'factor_returns.csv'
        factor_returns.to_csv(returns_file)
        saved_files['factor_returns'] = str(returns_file)
        print(f"✓ Saved factor returns to {returns_file}")

    # Save factor covariance if provided
    if factor_covariance is not None:
        if isinstance(factor_covariance, dict):
            # Time-varying: save as pickle
            cov_file = output_path / 'factor_covariance.pkl'
            import pickle
            with open(cov_file, 'wb') as f:
                pickle.dump(factor_covariance, f)
            saved_files['factor_covariance'] = str(cov_file)
            print(f"✓ Saved time-varying factor covariance to {cov_file}")
        else:
            # Static: save as CSV
            cov_file = output_path / 'factor_covariance.csv'
            factor_covariance.to_csv(cov_file)
            saved_files['factor_covariance'] = str(cov_file)
            print(f"✓ Saved factor covariance to {cov_file}")

    # Save specific variance if provided
    if specific_variance is not None:
        var_file = output_path / 'specific_variance.csv'
        specific_variance.to_csv(var_file)
        saved_files['specific_variance'] = str(var_file)
        print(f"✓ Saved specific variance to {var_file}")

    print(f"\n✓ All data saved to {output_path}")
    print(f"  Use: DataManager('{output_dir}') to load in backtester")

    return saved_files

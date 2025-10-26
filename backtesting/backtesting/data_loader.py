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


class DataManager:
    """
    Manages loading and access to all backtest data.

    Uses lazy loading to minimize memory footprint and supports
    efficient slicing by date.
    """

    def __init__(self, data_dir: str, use_float32: bool = True):
        """
        Initialize data manager.

        Parameters:
        -----------
        data_dir : str
            Directory containing CSV data files
        use_float32 : bool
            Use float32 instead of float64 for memory efficiency
        """
        self.data_dir = Path(data_dir)
        self.use_float32 = use_float32
        self.dtype = np.float32 if use_float32 else np.float64

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

        self._date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._tickers: Optional[List[str]] = None

    def load_prices(self, filename: str = 'prices.csv') -> pd.DataFrame:
        """
        Load price data.

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
        Load average daily volume data.

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

        return self._adv

    def load_betas(self, filename: str = 'betas.csv') -> pd.DataFrame:
        """
        Load beta data per security per date.

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

        return self._betas

    def load_factor_exposures(self, filename: str = 'factor_exposures.csv') -> pd.DataFrame:
        """
        Load factor exposures.

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

        return self._factor_exposures

    def load_factor_returns(self, filename: str = 'factor_returns.csv') -> pd.DataFrame:
        """
        Load factor returns.

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

        return self._factor_returns

    def load_factor_covariance(self, filename: str = 'factor_covariance.csv') -> pd.DataFrame:
        """
        Load factor covariance matrix.

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

        return self._factor_covariance

    def load_specific_variance(self, filename: str = 'specific_variance.csv') -> pd.DataFrame:
        """
        Load specific (idiosyncratic) variance per security per date.

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

        return self._specific_variance

    def load_sector_mapping(self, filename: str = 'sector_mapping.csv') -> pd.DataFrame:
        """
        Load sector mapping for securities.

        Expected format: CSV with 'ticker' and 'sector' columns.
        """
        if self._sector_mapping is None:
            filepath = self.data_dir / filename
            self._sector_mapping = pd.read_csv(filepath)
            print(f"Loaded sector mapping: {len(self._sector_mapping)} securities, "
                  f"{self._sector_mapping['sector'].nunique()} sectors")

        return self._sector_mapping

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

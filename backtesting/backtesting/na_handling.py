"""
NA handling configuration and utilities for data loading.

This module provides comprehensive missing data handling strategies
for all data types in the backtesting framework.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import warnings


class FillMethod(Enum):
    """Methods for handling missing values."""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    ZERO_FILL = "zero_fill"
    DROP = "drop"
    DEFAULT_VALUE = "default_value"
    CROSS_SECTIONAL_MEDIAN = "cross_sectional_median"
    ROLLING_ESTIMATE = "rolling_estimate"
    FAIL = "fail"


class ValidationLevel(Enum):
    """Data validation strictness levels."""
    LENIENT = "lenient"      # Fill all NAs, log warnings
    MODERATE = "moderate"    # Fill NAs, raise warnings for excessive missing data
    STRICT = "strict"        # Fail if critical data missing


@dataclass
class NAHandlingConfig:
    """
    Configuration for missing data handling across all data types.

    Each data type has:
    - primary_method: Main strategy for handling NAs
    - max_gap_days: Maximum consecutive NAs to fill (None = unlimited)
    - default_value: Default value for DEFAULT_VALUE method
    - drop_threshold: Drop ticker/column if % NAs exceeds this (0-1)
    - fail_threshold: Fail loading if % NAs exceeds this (0-1)
    """

    # Global settings
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    enable_logging: bool = True
    log_file: Optional[str] = None

    # Prices
    prices_method: FillMethod = FillMethod.FORWARD_FILL
    prices_max_gap: Optional[int] = 10
    prices_drop_threshold: float = 0.10  # Drop ticker if >10% missing
    prices_fail_threshold: float = 0.05  # Fail if >5% of entire dataset missing
    prices_interpolate_max_gap: int = 2  # Use interpolation for gaps <= 2 days

    # Trade Prices (optional data)
    trade_prices_method: FillMethod = FillMethod.FORWARD_FILL
    trade_prices_max_gap: Optional[int] = 10
    trade_prices_drop_threshold: float = 0.50  # More lenient since optional

    # Average Daily Volume
    adv_method: FillMethod = FillMethod.FORWARD_FILL
    adv_max_gap: Optional[int] = 10
    adv_default_value: float = 1000.0  # Conservative minimum ADV
    adv_decay_factor: float = 0.95  # Decay factor for stale data (per day)
    adv_decay_threshold_days: int = 20  # Start decay after this many days
    adv_drop_threshold: float = 0.10

    # Betas
    beta_method: FillMethod = FillMethod.FORWARD_FILL
    beta_max_gap: Optional[int] = 30
    beta_default_value: float = 1.0  # Market-neutral default
    beta_min_value: float = -3.0
    beta_max_value: float = 5.0
    beta_drop_threshold: float = 0.20

    # Factor Exposures
    factor_exposures_method: FillMethod = FillMethod.FORWARD_FILL
    factor_exposures_max_gap: Optional[int] = 20
    factor_exposures_use_cross_sectional: bool = True  # Fall back to cross-sectional median
    factor_exposures_drop_threshold: float = 0.15

    # Factor Returns
    factor_returns_method: FillMethod = FillMethod.ZERO_FILL  # Missing return = 0
    factor_returns_interpolate_single_day: bool = True  # Interpolate 1-day gaps
    factor_returns_drop_date_threshold: int = 3  # Drop date if >3 factors missing

    # Factor Covariance
    factor_covariance_method: FillMethod = FillMethod.FORWARD_FILL
    factor_covariance_max_gap: Optional[int] = 30
    factor_covariance_regularization: bool = True
    factor_covariance_shrinkage: float = 0.1  # Shrinkage toward identity

    # Specific Variance
    specific_variance_method: FillMethod = FillMethod.FORWARD_FILL
    specific_variance_max_gap: Optional[int] = 20
    specific_variance_use_cross_sectional: bool = True
    specific_variance_safety_factor: float = 1.5  # Multiply median by this for defaults
    specific_variance_drop_threshold: float = 0.15

    # Sector Mapping (static data)
    sector_mapping_default_sector: str = "Other"
    sector_mapping_allow_missing: bool = True  # Allow tickers without sector

    # External Trades
    external_trades_strict: bool = True  # Require all fields or drop row
    external_trades_allow_price_lookup: bool = True  # Look up price if missing
    external_trades_default_tag: str = "Untagged"


@dataclass
class NAHandlingReport:
    """Report of NA handling operations performed."""

    data_type: str
    rows_total: int
    cols_total: int
    na_count_before: int
    na_count_after: int
    na_percentage_before: float
    na_percentage_after: float
    method_used: str
    tickers_dropped: List[str] = field(default_factory=list)
    dates_dropped: List[pd.Timestamp] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format report as string."""
        lines = [
            f"=== {self.data_type} ===",
            f"Shape: {self.rows_total} rows × {self.cols_total} cols",
            f"NAs Before: {self.na_count_before} ({self.na_percentage_before:.2%})",
            f"NAs After: {self.na_count_after} ({self.na_percentage_after:.2%})",
            f"Method: {self.method_used}",
        ]

        if self.tickers_dropped:
            lines.append(f"Tickers Dropped: {len(self.tickers_dropped)} ({', '.join(self.tickers_dropped[:5])}...)")

        if self.dates_dropped:
            lines.append(f"Dates Dropped: {len(self.dates_dropped)}")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)


class NAHandler:
    """Handles missing data operations with logging and validation."""

    def __init__(self, config: NAHandlingConfig):
        self.config = config
        self.reports: List[NAHandlingReport] = []

    def handle_timeseries_data(
        self,
        df: pd.DataFrame,
        data_type: str,
        method: FillMethod,
        max_gap: Optional[int] = None,
        default_value: Optional[float] = None,
        drop_threshold: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle NAs in time series data (date index × ticker columns).

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with datetime index
        data_type : str
            Name of data type (for reporting)
        method : FillMethod
            Primary filling method
        max_gap : int, optional
            Maximum consecutive NAs to fill
        default_value : float, optional
            Default value for DEFAULT_VALUE method
        drop_threshold : float, optional
            Drop column if % NAs exceeds this
        min_value : float, optional
            Minimum valid value (for clipping)
        max_value : float, optional
            Maximum valid value (for clipping)

        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Initial statistics
        na_count_before = df_clean.isna().sum().sum()
        total_values = df_clean.shape[0] * df_clean.shape[1]
        na_pct_before = na_count_before / total_values if total_values > 0 else 0.0

        warnings_list = []
        tickers_dropped = []

        # Drop columns exceeding threshold
        if drop_threshold is not None:
            for col in df_clean.columns:
                col_na_pct = df_clean[col].isna().sum() / len(df_clean)
                if col_na_pct > drop_threshold:
                    tickers_dropped.append(col)
                    df_clean = df_clean.drop(columns=[col])
                    warnings_list.append(
                        f"Dropped {col}: {col_na_pct:.1%} missing (threshold: {drop_threshold:.1%})"
                    )

        # Apply filling method
        if method == FillMethod.FORWARD_FILL:
            if max_gap is not None:
                df_clean = df_clean.ffill(limit=max_gap)
            else:
                df_clean = df_clean.ffill()

        elif method == FillMethod.BACKWARD_FILL:
            if max_gap is not None:
                df_clean = df_clean.bfill(limit=max_gap)
            else:
                df_clean = df_clean.bfill()

        elif method == FillMethod.INTERPOLATE:
            df_clean = df_clean.interpolate(method='linear', limit=max_gap)

        elif method == FillMethod.ZERO_FILL:
            df_clean = df_clean.fillna(0.0)

        elif method == FillMethod.DEFAULT_VALUE:
            if default_value is not None:
                df_clean = df_clean.fillna(default_value)

        elif method == FillMethod.CROSS_SECTIONAL_MEDIAN:
            # Fill with median across columns for each date
            for date in df_clean.index:
                row_median = df_clean.loc[date].median()
                df_clean.loc[date] = df_clean.loc[date].fillna(row_median)

        # Clip to valid range
        if min_value is not None or max_value is not None:
            df_clean = df_clean.clip(lower=min_value, upper=max_value)

        # Final statistics
        na_count_after = df_clean.isna().sum().sum()
        total_values_after = df_clean.shape[0] * df_clean.shape[1]
        na_pct_after = na_count_after / total_values_after if total_values_after > 0 else 0.0

        # Check if still too many NAs
        if na_count_after > 0:
            warnings_list.append(f"{na_count_after} NAs remain after filling")

        # Create report
        report = NAHandlingReport(
            data_type=data_type,
            rows_total=df_clean.shape[0],
            cols_total=df_clean.shape[1],
            na_count_before=na_count_before,
            na_count_after=na_count_after,
            na_percentage_before=na_pct_before,
            na_percentage_after=na_pct_after,
            method_used=method.value,
            tickers_dropped=tickers_dropped,
            warnings=warnings_list
        )

        self.reports.append(report)

        # Log report
        if self.config.enable_logging:
            self._log_report(report)

        return df_clean

    def handle_multiindex_data(
        self,
        df: pd.DataFrame,
        data_type: str,
        method: FillMethod,
        max_gap: Optional[int] = None,
        use_cross_sectional: bool = False,
        drop_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle NAs in MultiIndex data (date, ticker) × factor columns.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with MultiIndex (date, ticker)
        data_type : str
            Name of data type
        method : FillMethod
            Primary filling method
        max_gap : int, optional
            Maximum consecutive NAs to fill per ticker
        use_cross_sectional : bool
            Use cross-sectional median as fallback
        drop_threshold : float, optional
            Drop ticker if % NAs exceeds this

        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Initial statistics
        na_count_before = df_clean.isna().sum().sum()
        total_values = df_clean.shape[0] * df_clean.shape[1]
        na_pct_before = na_count_before / total_values if total_values > 0 else 0.0

        warnings_list = []
        tickers_dropped = []

        # Get unique tickers
        tickers = df_clean.index.get_level_values(1).unique()

        # Drop tickers exceeding threshold
        if drop_threshold is not None:
            for ticker in tickers:
                ticker_data = df_clean.xs(ticker, level=1)
                ticker_na_pct = ticker_data.isna().sum().sum() / (ticker_data.shape[0] * ticker_data.shape[1])
                if ticker_na_pct > drop_threshold:
                    tickers_dropped.append(ticker)
                    df_clean = df_clean.drop(ticker, level=1)
                    warnings_list.append(
                        f"Dropped {ticker}: {ticker_na_pct:.1%} missing (threshold: {drop_threshold:.1%})"
                    )

        # Apply filling method per ticker
        if method == FillMethod.FORWARD_FILL:
            df_clean = df_clean.groupby(level=1).ffill(limit=max_gap)

        elif method == FillMethod.BACKWARD_FILL:
            df_clean = df_clean.groupby(level=1).bfill(limit=max_gap)

        elif method == FillMethod.INTERPOLATE:
            df_clean = df_clean.groupby(level=1).apply(
                lambda x: x.interpolate(method='linear', limit=max_gap)
            )

        # Cross-sectional fill for remaining NAs
        if use_cross_sectional and df_clean.isna().any().any():
            for date in df_clean.index.get_level_values(0).unique():
                date_data = df_clean.xs(date, level=0)
                for col in date_data.columns:
                    col_median = date_data[col].median()
                    if pd.notna(col_median):
                        df_clean.loc[(date, slice(None)), col] = df_clean.loc[(date, slice(None)), col].fillna(col_median)

        # Final statistics
        na_count_after = df_clean.isna().sum().sum()
        total_values_after = df_clean.shape[0] * df_clean.shape[1]
        na_pct_after = na_count_after / total_values_after if total_values_after > 0 else 0.0

        if na_count_after > 0:
            warnings_list.append(f"{na_count_after} NAs remain after filling")

        # Create report
        report = NAHandlingReport(
            data_type=data_type,
            rows_total=df_clean.shape[0],
            cols_total=df_clean.shape[1],
            na_count_before=na_count_before,
            na_count_after=na_count_after,
            na_percentage_before=na_pct_before,
            na_percentage_after=na_pct_after,
            method_used=method.value,
            tickers_dropped=tickers_dropped,
            warnings=warnings_list
        )

        self.reports.append(report)

        if self.config.enable_logging:
            self._log_report(report)

        return df_clean

    def handle_external_trades(
        self,
        df: pd.DataFrame,
        strict: bool = True,
        allow_price_lookup: bool = False,
        prices_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Handle NAs in external trades data.

        Parameters:
        -----------
        df : pd.DataFrame
            External trades DataFrame
        strict : bool
            If True, drop rows with any missing required fields
        allow_price_lookup : bool
            If True, look up price from prices_df if missing
        prices_df : pd.DataFrame, optional
            Price data for lookup

        Returns:
        --------
        pd.DataFrame
            Cleaned trades DataFrame
        """
        df_clean = df.copy()

        initial_rows = len(df_clean)
        warnings_list = []

        # Check required fields
        required_fields = ['date', 'ticker', 'qty', 'price']

        # Handle missing required fields
        if strict:
            # Drop rows with any missing required field
            na_mask = df_clean[required_fields].isna().any(axis=1)
            dropped_rows = na_mask.sum()

            if dropped_rows > 0:
                df_clean = df_clean[~na_mask]
                warnings_list.append(f"Dropped {dropped_rows} rows with missing required fields")

        else:
            # Try to fill missing prices
            if allow_price_lookup and prices_df is not None:
                price_na_mask = df_clean['price'].isna()
                if price_na_mask.any():
                    for idx in df_clean[price_na_mask].index:
                        date = df_clean.loc[idx, 'date']
                        ticker = df_clean.loc[idx, 'ticker']

                        if date in prices_df.index and ticker in prices_df.columns:
                            lookup_price = prices_df.loc[date, ticker]
                            df_clean.loc[idx, 'price'] = lookup_price
                            warnings_list.append(f"Filled price for {ticker} on {date} from close prices")

            # Drop rows still missing required fields
            na_mask = df_clean[required_fields].isna().any(axis=1)
            dropped_rows = na_mask.sum()

            if dropped_rows > 0:
                df_clean = df_clean[~na_mask]
                warnings_list.append(f"Dropped {dropped_rows} rows after price lookup attempt")

        # Validate values
        # Drop zero quantity trades
        zero_qty_mask = df_clean['qty'] == 0
        if zero_qty_mask.any():
            zero_count = zero_qty_mask.sum()
            df_clean = df_clean[~zero_qty_mask]
            warnings_list.append(f"Dropped {zero_count} trades with zero quantity")

        # Drop non-positive prices
        invalid_price_mask = df_clean['price'] <= 0
        if invalid_price_mask.any():
            invalid_count = invalid_price_mask.sum()
            df_clean = df_clean[~invalid_price_mask]
            warnings_list.append(f"Dropped {invalid_count} trades with non-positive prices")

        # Handle missing tags (optional field)
        if 'tag' not in df_clean.columns:
            df_clean['tag'] = None
        else:
            df_clean['tag'] = df_clean['tag'].fillna(self.config.external_trades_default_tag)

        final_rows = len(df_clean)

        # Create report
        report = NAHandlingReport(
            data_type="External Trades",
            rows_total=final_rows,
            cols_total=df_clean.shape[1],
            na_count_before=initial_rows - final_rows,
            na_count_after=0,
            na_percentage_before=(initial_rows - final_rows) / initial_rows if initial_rows > 0 else 0.0,
            na_percentage_after=0.0,
            method_used="drop_invalid" if strict else "fill_and_drop",
            warnings=warnings_list
        )

        self.reports.append(report)

        if self.config.enable_logging:
            self._log_report(report)

        return df_clean

    def get_summary_report(self) -> str:
        """Get summary of all NA handling operations."""
        if not self.reports:
            return "No NA handling operations performed yet."

        lines = ["=" * 60, "NA HANDLING SUMMARY REPORT", "=" * 60, ""]

        for report in self.reports:
            lines.append(str(report))
            lines.append("")

        # Overall statistics
        total_na_before = sum(r.na_count_before for r in self.reports)
        total_na_after = sum(r.na_count_after for r in self.reports)

        lines.append("=" * 60)
        lines.append(f"Total NAs Before: {total_na_before:,}")
        lines.append(f"Total NAs After: {total_na_after:,}")
        lines.append(f"NAs Resolved: {total_na_before - total_na_after:,}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _log_report(self, report: NAHandlingReport):
        """Log report to console and optionally to file."""
        # Console logging
        if report.na_count_before > 0 or report.warnings:
            print(f"\n{report}")

        # File logging
        if self.config.log_file:
            with open(self.config.log_file, 'a') as f:
                f.write(str(report) + "\n\n")

    def clear_reports(self):
        """Clear all stored reports."""
        self.reports = []


# Helper functions for specific operations

def ensure_positive_definite(cov_matrix: pd.DataFrame, min_eigenvalue: float = 1e-8) -> pd.DataFrame:
    """
    Ensure covariance matrix is positive definite.

    Parameters:
    -----------
    cov_matrix : pd.DataFrame
        Covariance matrix
    min_eigenvalue : float
        Minimum eigenvalue to ensure

    Returns:
    --------
    pd.DataFrame
        Positive definite covariance matrix
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)

    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    # Reconstruct matrix
    cov_clean = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return pd.DataFrame(cov_clean, index=cov_matrix.index, columns=cov_matrix.columns)


def apply_shrinkage(cov_matrix: pd.DataFrame, shrinkage: float = 0.1) -> pd.DataFrame:
    """
    Apply shrinkage toward identity matrix (regularization).

    Parameters:
    -----------
    cov_matrix : pd.DataFrame
        Input covariance matrix
    shrinkage : float
        Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage to identity)

    Returns:
    --------
    pd.DataFrame
        Regularized covariance matrix
    """
    n = cov_matrix.shape[0]
    identity = np.eye(n) * np.trace(cov_matrix.values) / n

    cov_shrunk = (1 - shrinkage) * cov_matrix.values + shrinkage * identity

    return pd.DataFrame(cov_shrunk, index=cov_matrix.index, columns=cov_matrix.columns)


def detect_outliers(series: pd.Series, n_std: float = 5.0) -> pd.Series:
    """
    Detect outliers using z-score method.

    Parameters:
    -----------
    series : pd.Series
        Input data
    n_std : float
        Number of standard deviations for outlier threshold

    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series(False, index=series.index)

    z_scores = np.abs((series - mean) / std)
    return z_scores > n_std

"""
Utility functions for the backtester.

This module contains helper functions for common operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def shares_to_weights(
    positions: Dict[str, float],
    prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Convert shares to portfolio weights.

    Parameters:
    -----------
    positions : Dict[str, float]
        Ticker -> shares
    prices : Dict[str, float]
        Ticker -> price

    Returns:
    --------
    Dict[str, float]
        Ticker -> weight
    """
    values = {
        ticker: shares * prices.get(ticker, 0.0)
        for ticker, shares in positions.items()
    }

    total_value = sum(abs(v) for v in values.values())

    if total_value == 0:
        return {ticker: 0.0 for ticker in positions}

    return {
        ticker: value / total_value
        for ticker, value in values.items()
    }


def weights_to_shares(
    weights: Dict[str, float],
    prices: Dict[str, float],
    total_value: float
) -> Dict[str, float]:
    """
    Convert weights to shares.

    Parameters:
    -----------
    weights : Dict[str, float]
        Ticker -> weight
    prices : Dict[str, float]
        Ticker -> price
    total_value : float
        Total portfolio value

    Returns:
    --------
    Dict[str, float]
        Ticker -> shares
    """
    shares = {}

    for ticker, weight in weights.items():
        price = prices.get(ticker, 0.0)
        if price > 0:
            notional = weight * total_value
            shares[ticker] = notional / price
        else:
            shares[ticker] = 0.0

    return shares


def notional_to_shares(
    notional: Dict[str, float],
    prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Convert notional amounts to shares.

    Parameters:
    -----------
    notional : Dict[str, float]
        Ticker -> dollar amount
    prices : Dict[str, float]
        Ticker -> price

    Returns:
    --------
    Dict[str, float]
        Ticker -> shares
    """
    shares = {}

    for ticker, amt in notional.items():
        price = prices.get(ticker, 0.0)
        if price > 0:
            shares[ticker] = amt / price
        else:
            shares[ticker] = 0.0

    return shares


def calculate_trades(
    current_positions: Dict[str, float],
    target_positions: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate trades needed to move from current to target positions.

    Parameters:
    -----------
    current_positions : Dict[str, float]
        Current holdings
    target_positions : Dict[str, float]
        Target holdings

    Returns:
    --------
    Dict[str, float]
        Required trades (ticker -> shares to trade)
    """
    all_tickers = set(current_positions.keys()) | set(target_positions.keys())

    trades = {}
    for ticker in all_tickers:
        current = current_positions.get(ticker, 0.0)
        target = target_positions.get(ticker, 0.0)
        trade = target - current

        if abs(trade) > 1e-6:  # Filter tiny trades
            trades[ticker] = trade

    return trades


def round_to_lot_size(
    shares: float,
    lot_size: int = 1
) -> float:
    """
    Round shares to nearest lot size.

    Parameters:
    -----------
    shares : float
        Number of shares
    lot_size : int
        Lot size (e.g., 100 for round lots)

    Returns:
    --------
    float
        Rounded shares
    """
    return np.round(shares / lot_size) * lot_size


def validate_positions(
    positions: Dict[str, float],
    prices: Dict[str, float]
) -> List[str]:
    """
    Validate positions for data quality issues.

    Parameters:
    -----------
    positions : Dict[str, float]
        Portfolio positions
    prices : Dict[str, float]
        Security prices

    Returns:
    --------
    List[str]
        List of validation warnings
    """
    warnings = []

    for ticker, shares in positions.items():
        if np.isnan(shares) or np.isinf(shares):
            warnings.append(f"Invalid shares for {ticker}: {shares}")

        price = prices.get(ticker)
        if price is None:
            warnings.append(f"Missing price for {ticker}")
        elif price <= 0:
            warnings.append(f"Invalid price for {ticker}: {price}")
        elif np.isnan(price) or np.isinf(price):
            warnings.append(f"Invalid price for {ticker}: {price}")

    return warnings


def format_currency(value: float) -> str:
    """Format value as currency string."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.2f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousand separators."""
    return f"{value:,.{decimals}f}"


def date_to_string(date: pd.Timestamp) -> str:
    """Convert timestamp to string."""
    return date.strftime('%Y-%m-%d')


def annualize_return(cumulative_return: float, n_days: int) -> float:
    """
    Annualize a cumulative return.

    Parameters:
    -----------
    cumulative_return : float
        Total return over period
    n_days : int
        Number of days in period

    Returns:
    --------
    float
        Annualized return
    """
    if n_days == 0:
        return 0.0

    years = n_days / 252.0  # Assuming 252 trading days per year
    if years == 0:
        return 0.0

    return (1 + cumulative_return) ** (1 / years) - 1


def annualize_volatility(daily_std: float) -> float:
    """
    Annualize daily volatility.

    Parameters:
    -----------
    daily_std : float
        Daily standard deviation

    Returns:
    --------
    float
        Annualized volatility
    """
    return daily_std * np.sqrt(252)


def calculate_turnover(
    trades: Dict[str, float],
    prices: Dict[str, float],
    portfolio_value: float
) -> float:
    """
    Calculate portfolio turnover.

    Turnover = sum(|trade_value|) / (2 * portfolio_value)

    Parameters:
    -----------
    trades : Dict[str, float]
        Trades (ticker -> shares)
    prices : Dict[str, float]
        Prices
    portfolio_value : float
        Total portfolio value

    Returns:
    --------
    float
        Turnover as fraction
    """
    if portfolio_value == 0:
        return 0.0

    trade_value = sum(
        abs(qty * prices.get(ticker, 0.0))
        for ticker, qty in trades.items()
    )

    return trade_value / (2 * portfolio_value)


def create_trade_record(
    date: pd.Timestamp,
    ticker: str,
    quantity: float,
    price: float,
    cost: float
) -> Dict:
    """
    Create a trade record dictionary.

    Parameters:
    -----------
    date : pd.Timestamp
        Trade date
    ticker : str
        Security ticker
    quantity : float
        Shares traded
    price : float
        Execution price
    cost : float
        Transaction cost

    Returns:
    --------
    Dict
        Trade record
    """
    return {
        'date': date,
        'ticker': ticker,
        'quantity': quantity,
        'price': price,
        'notional': quantity * price,
        'cost': cost,
        'side': 'BUY' if quantity > 0 else 'SELL'
    }


def merge_positions(
    *position_dicts: Dict[str, float]
) -> Dict[str, float]:
    """
    Merge multiple position dictionaries.

    Parameters:
    -----------
    *position_dicts : Dict[str, float]
        Multiple dictionaries of positions

    Returns:
    --------
    Dict[str, float]
        Merged positions
    """
    merged = {}

    for pos_dict in position_dicts:
        for ticker, shares in pos_dict.items():
            merged[ticker] = merged.get(ticker, 0.0) + shares

    return merged


def filter_small_positions(
    positions: Dict[str, float],
    min_shares: float = 1.0
) -> Dict[str, float]:
    """
    Filter out positions below minimum size.

    Parameters:
    -----------
    positions : Dict[str, float]
        Positions
    min_shares : float
        Minimum absolute share count

    Returns:
    --------
    Dict[str, float]
        Filtered positions
    """
    return {
        ticker: shares
        for ticker, shares in positions.items()
        if abs(shares) >= min_shares
    }


def align_date_to_data(
    target_date: pd.Timestamp,
    available_dates: pd.DatetimeIndex,
    method: str = 'nearest'
) -> pd.Timestamp:
    """
    Align a target date to the nearest available date in data.

    Parameters:
    -----------
    target_date : pd.Timestamp
        Target date
    available_dates : pd.DatetimeIndex
        Available dates in data
    method : str
        Alignment method: 'nearest', 'previous', or 'next'

    Returns:
    --------
    pd.Timestamp
        Aligned date
    """
    if target_date in available_dates:
        return target_date

    if method == 'nearest':
        idx = available_dates.searchsorted(target_date)
        if idx == 0:
            return available_dates[0]
        elif idx == len(available_dates):
            return available_dates[-1]
        else:
            # Compare distances
            prev_date = available_dates[idx - 1]
            next_date = available_dates[idx]
            if abs((target_date - prev_date).days) < abs((target_date - next_date).days):
                return prev_date
            else:
                return next_date
    elif method == 'previous':
        idx = available_dates.searchsorted(target_date)
        if idx == 0:
            return available_dates[0]
        else:
            return available_dates[idx - 1]
    elif method == 'next':
        idx = available_dates.searchsorted(target_date)
        if idx >= len(available_dates):
            return available_dates[-1]
        else:
            return available_dates[idx]
    else:
        raise ValueError(f"Unknown method: {method}")


def get_date_range(
    prices_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get valid date range for backtesting.

    If start/end dates are not provided or not in data,
    aligns them to available dates.

    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    start_date : pd.Timestamp, optional
        Desired start date
    end_date : pd.Timestamp, optional
        Desired end date

    Returns:
    --------
    Tuple[pd.Timestamp, pd.Timestamp]
        (aligned_start_date, aligned_end_date)
    """
    available_dates = prices_df.index

    if start_date is None:
        aligned_start = available_dates[0]
    else:
        aligned_start = align_date_to_data(start_date, available_dates, method='next')

    if end_date is None:
        aligned_end = available_dates[-1]
    else:
        aligned_end = align_date_to_data(end_date, available_dates, method='previous')

    return aligned_start, aligned_end

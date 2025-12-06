"""
Production Stop Loss Calculator

Standalone function for calculating stop loss gross reductions based on
daily PnL time series. Can be used for live trading or post-hoc analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union


def calculate_stop_loss_gross(
    daily_pnl: Union[pd.Series, np.ndarray],
    stop_loss_levels: List[Union[Tuple[float, float], Tuple[float, float, Optional[float]]]],
    initial_capital: float,
    dates: Optional[Union[pd.DatetimeIndex, List]] = None
) -> pd.Series:
    """
    Calculate gross exposure multipliers based on dollar-based stop loss levels.

    This is a production-ready function that processes a daily PnL time series
    and returns the gross exposure multiplier that should be applied each day.

    Parameters:
    -----------
    daily_pnl : pd.Series or np.ndarray
        Daily PnL values (in dollars). Can be a pandas Series with datetime index
        or a numpy array. If numpy array, dates parameter should be provided.

    stop_loss_levels : List[Tuple]
        List of stop loss levels, each specified as:
        - 2-tuple: (drawdown_threshold, gross_reduction)
        - 3-tuple: (drawdown_threshold, gross_reduction, recovery_threshold)

        Where:
        - drawdown_threshold: Dollar loss from peak that triggers this level
        - gross_reduction: Target gross exposure (0-1, e.g., 0.75 = 75%)
        - recovery_threshold: Optional dollar recovery from trough to move back

        Example: [(5000, 0.75, 2500), (10000, 0.50, 5000)]

    initial_capital : float
        Starting capital/portfolio value in dollars

    dates : pd.DatetimeIndex or List, optional
        Dates corresponding to daily_pnl if daily_pnl is a numpy array.
        If daily_pnl is a Series, this is ignored.

    Returns:
    --------
    pd.Series
        Time series of gross multipliers (1.0 = no reduction, 0.5 = 50% gross, etc.)
        Index is dates if provided, otherwise integer index.

    Examples:
    ---------
    >>> daily_pnl = pd.Series([0, 100, -200, -300, 50, 100],
    ...                        index=pd.date_range('2023-01-01', periods=6))
    >>> levels = [(500, 0.75, 250), (1000, 0.50, 500)]
    >>> gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=10000)
    >>> print(gross)
    2023-01-01    1.00
    2023-01-02    1.00
    2023-01-03    1.00
    2023-01-04    0.75
    2023-01-05    0.75
    2023-01-06    1.00
    dtype: float64
    """
    # Convert inputs to pandas Series if needed
    if isinstance(daily_pnl, np.ndarray):
        if dates is not None:
            daily_pnl = pd.Series(daily_pnl, index=dates)
        else:
            daily_pnl = pd.Series(daily_pnl)
    elif not isinstance(daily_pnl, pd.Series):
        raise ValueError("daily_pnl must be a pandas Series or numpy array")

    # Validate stop loss levels
    if not stop_loss_levels:
        raise ValueError("Must provide at least one stop loss level")

    # Parse stop loss levels
    levels = []
    for level_tuple in stop_loss_levels:
        if len(level_tuple) == 2:
            dd_threshold, gross_reduction = level_tuple
            recovery_threshold = None
        elif len(level_tuple) == 3:
            dd_threshold, gross_reduction, recovery_threshold = level_tuple
        else:
            raise ValueError(f"Stop loss level must be 2-tuple or 3-tuple, got {len(level_tuple)}-tuple")

        # Validate values
        if dd_threshold < 0:
            raise ValueError(f"Drawdown threshold must be non-negative, got {dd_threshold}")
        if gross_reduction < 0 or gross_reduction > 1:
            raise ValueError(f"Gross reduction must be in [0, 1], got {gross_reduction}")
        if recovery_threshold is not None and recovery_threshold < 0:
            raise ValueError(f"Recovery threshold must be non-negative, got {recovery_threshold}")

        levels.append({
            'drawdown_threshold': dd_threshold,
            'gross_reduction': gross_reduction,
            'recovery_threshold': recovery_threshold
        })

    # Sort levels by drawdown threshold
    levels = sorted(levels, key=lambda x: x['drawdown_threshold'])

    # Validate that gross reductions are decreasing
    for i in range(1, len(levels)):
        if levels[i]['gross_reduction'] > levels[i-1]['gross_reduction']:
            raise ValueError(
                f"Gross reductions must be decreasing. Level {i} has higher "
                f"reduction ({levels[i]['gross_reduction']}) than level {i-1} "
                f"({levels[i-1]['gross_reduction']})"
            )

    # Calculate cumulative portfolio value from PnL
    portfolio_values = initial_capital + daily_pnl.cumsum()

    # Initialize result series
    gross_multipliers = pd.Series(1.0, index=daily_pnl.index)

    # Track state
    peak_value = initial_capital
    trough_value = initial_capital
    current_level = None  # Index of currently triggered level (None = no stop loss)

    # Process each day
    for i, (date, pnl) in enumerate(daily_pnl.items()):
        portfolio_value = portfolio_values.iloc[i]

        # Update peak if new high
        if portfolio_value > peak_value:
            peak_value = portfolio_value
            trough_value = portfolio_value
            current_level = None  # Clear stop loss at new peak
            gross_multipliers.iloc[i] = 1.0
            continue

        # Update trough if new low
        if portfolio_value < trough_value:
            trough_value = portfolio_value

        # Calculate current drawdown and recovery
        current_drawdown = peak_value - portfolio_value
        current_recovery = portfolio_value - trough_value

        # Determine which level should be active
        new_level = None
        new_gross = 1.0

        # Check if we should trigger a deeper level (going down)
        for idx, level in enumerate(levels):
            if current_drawdown >= level['drawdown_threshold']:
                new_level = idx
                new_gross = level['gross_reduction']

        # Check if we should recover to a shallower level (going up)
        # Only check recovery if we're currently at a level and haven't found a deeper trigger
        if current_level is not None and new_level == current_level:
            level = levels[current_level]
            if level['recovery_threshold'] is not None:
                if current_recovery >= level['recovery_threshold']:
                    # Move to previous (less restrictive) level, or clear
                    if current_level > 0:
                        new_level = current_level - 1
                        new_gross = levels[new_level]['gross_reduction']
                    else:
                        new_level = None
                        new_gross = 1.0

        # Update state
        current_level = new_level
        gross_multipliers.iloc[i] = new_gross

    return gross_multipliers


def calculate_stop_loss_metrics(
    daily_pnl: Union[pd.Series, np.ndarray],
    stop_loss_levels: List[Union[Tuple[float, float], Tuple[float, float, Optional[float]]]],
    initial_capital: float,
    dates: Optional[Union[pd.DatetimeIndex, List]] = None
) -> pd.DataFrame:
    """
    Calculate detailed stop loss metrics including portfolio values, drawdowns, and gross multipliers.

    This function provides more detailed information than calculate_stop_loss_gross(),
    returning a DataFrame with all relevant metrics for analysis.

    Parameters:
    -----------
    daily_pnl : pd.Series or np.ndarray
        Daily PnL values (in dollars)

    stop_loss_levels : List[Tuple]
        List of stop loss levels (see calculate_stop_loss_gross for format)

    initial_capital : float
        Starting capital/portfolio value in dollars

    dates : pd.DatetimeIndex or List, optional
        Dates corresponding to daily_pnl if daily_pnl is a numpy array

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - portfolio_value: Current portfolio value
        - peak_value: Running peak value
        - trough_value: Running trough value (during current drawdown)
        - drawdown_dollar: Dollar drawdown from peak
        - recovery_dollar: Dollar recovery from trough
        - triggered_level: Currently active stop loss level (0-indexed, None if cleared)
        - gross_multiplier: Gross exposure multiplier

    Examples:
    ---------
    >>> daily_pnl = pd.Series([0, 100, -200, -300, 50, 100])
    >>> levels = [(500, 0.75, 250)]
    >>> metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=10000)
    >>> print(metrics[['portfolio_value', 'drawdown_dollar', 'gross_multiplier']])
    """
    # Convert inputs to pandas Series if needed
    if isinstance(daily_pnl, np.ndarray):
        if dates is not None:
            daily_pnl = pd.Series(daily_pnl, index=dates)
        else:
            daily_pnl = pd.Series(daily_pnl)
    elif not isinstance(daily_pnl, pd.Series):
        raise ValueError("daily_pnl must be a pandas Series or numpy array")

    # Parse and validate levels (reuse validation from calculate_stop_loss_gross)
    levels = []
    for level_tuple in stop_loss_levels:
        if len(level_tuple) == 2:
            dd_threshold, gross_reduction = level_tuple
            recovery_threshold = None
        elif len(level_tuple) == 3:
            dd_threshold, gross_reduction, recovery_threshold = level_tuple
        else:
            raise ValueError(f"Stop loss level must be 2-tuple or 3-tuple, got {len(level_tuple)}-tuple")

        levels.append({
            'drawdown_threshold': dd_threshold,
            'gross_reduction': gross_reduction,
            'recovery_threshold': recovery_threshold
        })

    levels = sorted(levels, key=lambda x: x['drawdown_threshold'])

    # Calculate cumulative portfolio value
    portfolio_values = initial_capital + daily_pnl.cumsum()

    # Initialize result DataFrame
    results = pd.DataFrame(index=daily_pnl.index)
    results['portfolio_value'] = portfolio_values
    results['peak_value'] = 0.0
    results['trough_value'] = 0.0
    results['drawdown_dollar'] = 0.0
    results['recovery_dollar'] = 0.0
    results['triggered_level'] = None
    results['gross_multiplier'] = 1.0

    # Track state
    peak_value = initial_capital
    trough_value = initial_capital
    current_level = None

    # Process each day
    for i, (date, pnl) in enumerate(daily_pnl.items()):
        portfolio_value = portfolio_values.iloc[i]

        # Update peak if new high
        if portfolio_value > peak_value:
            peak_value = portfolio_value
            trough_value = portfolio_value
            current_level = None

        # Update trough if new low
        if portfolio_value < trough_value:
            trough_value = portfolio_value

        # Calculate current drawdown and recovery
        current_drawdown = peak_value - portfolio_value
        current_recovery = portfolio_value - trough_value

        # Determine which level should be active
        new_level = None
        new_gross = 1.0

        # Check triggers
        for idx, level in enumerate(levels):
            if current_drawdown >= level['drawdown_threshold']:
                new_level = idx
                new_gross = level['gross_reduction']

        # Check recovery
        if current_level is not None and new_level == current_level:
            level = levels[current_level]
            if level['recovery_threshold'] is not None:
                if current_recovery >= level['recovery_threshold']:
                    if current_level > 0:
                        new_level = current_level - 1
                        new_gross = levels[new_level]['gross_reduction']
                    else:
                        new_level = None
                        new_gross = 1.0

        # Store results
        results.iloc[i, results.columns.get_loc('peak_value')] = peak_value
        results.iloc[i, results.columns.get_loc('trough_value')] = trough_value
        results.iloc[i, results.columns.get_loc('drawdown_dollar')] = current_drawdown
        results.iloc[i, results.columns.get_loc('recovery_dollar')] = current_recovery
        results.iloc[i, results.columns.get_loc('triggered_level')] = new_level
        results.iloc[i, results.columns.get_loc('gross_multiplier')] = new_gross

        # Update state
        current_level = new_level

    return results

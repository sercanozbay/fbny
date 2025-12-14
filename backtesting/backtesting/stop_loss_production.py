"""
Production Stop Loss Calculator

Standalone function for calculating stop loss gross reductions based on
daily PnL time series using simplified immediate exit logic.

All thresholds are drawdown levels from peak for consistency.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union


def calculate_stop_loss_gross(
    daily_pnl: Union[pd.Series, np.ndarray],
    stop_loss_levels: List[Tuple[float, float]],
    initial_capital: float,
    dates: Optional[Union[pd.DatetimeIndex, List]] = None
) -> pd.Series:
    """
    Calculate gross exposure multipliers using simplified immediate exit logic.

    Uses drawdown-based thresholds: enter when DD >= threshold, exit when DD < threshold.
    Exits jump directly to no stop loss (100% gross).

    Parameters:
    -----------
    daily_pnl : pd.Series or np.ndarray
        Daily PnL values (in dollars). Can be a pandas Series with datetime index
        or a numpy array. If numpy array, dates parameter should be provided.

    stop_loss_levels : List[Tuple[float, float]]
        List of stop loss levels, each specified as:
        - 2-tuple: (drawdown_threshold, gross_reduction)

        Where:
        - drawdown_threshold: Dollar drawdown from peak for both entry and exit
        - gross_reduction: Target gross exposure (0-1, e.g., 0.75 = 75%)

        Example: [(5000, 0.75), (10000, 0.50)]
        - Enter at $5k DD → 75% gross, exit when DD < $5k
        - Enter at $10k DD → 50% gross, exit when DD < $10k (jumps to 100%)

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
    >>> # Immediate exit example
    >>> daily_pnl = pd.Series([0, -12000, 3000, 2000],
    ...                        index=pd.date_range('2023-01-01', periods=4))
    >>> levels = [(5000, 0.75), (10000, 0.50)]
    >>> gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)
    >>> # Day 0: $0 DD → 100%
    >>> # Day 1: $12k DD → 50% (entered L2)
    >>> # Day 2: $9k DD → 100% (exited, DD < $10k, jumped to no stop loss)
    >>> # Day 3: $7k DD → 100% (no stop loss)
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
        if len(level_tuple) != 2:
            raise ValueError(f"Stop loss level must be 2-tuple (drawdown_threshold, gross_reduction), got {len(level_tuple)}-tuple")

        dd_threshold, gross_reduction = level_tuple

        # Validate values
        if dd_threshold < 0:
            raise ValueError(f"Drawdown threshold must be non-negative, got {dd_threshold}")
        if gross_reduction < 0 or gross_reduction > 1:
            raise ValueError(f"Gross reduction must be in [0, 1], got {gross_reduction}")

        levels.append({
            'drawdown_threshold': dd_threshold,
            'gross_reduction': gross_reduction
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
    current_level = None  # Index of currently triggered level (None = no stop loss)

    # Process each day using simplified logic
    for i, (date, pnl) in enumerate(daily_pnl.items()):
        portfolio_value = portfolio_values.iloc[i]

        # Update peak if new high (clears all stop loss)
        if portfolio_value > peak_value:
            peak_value = portfolio_value
            current_level = None
            gross_multipliers.iloc[i] = 1.0
            continue

        # Calculate current drawdown from peak
        current_drawdown = peak_value - portfolio_value

        # Determine level using simplified logic
        new_level = current_level
        new_gross = gross_multipliers.iloc[i-1] if i > 0 else 1.0

        if current_level is None:
            # Not at any level - check if we should enter one (deepest first)
            for idx in range(len(levels) - 1, -1, -1):
                if current_drawdown >= levels[idx]['drawdown_threshold']:
                    new_level = idx
                    new_gross = levels[idx]['gross_reduction']
                    break
        else:
            # At a level - check if we should exit completely or go deeper
            current_threshold = levels[current_level]['drawdown_threshold']

            if current_drawdown < current_threshold:
                # Exit completely (jump to no stop loss)
                new_level = None
                new_gross = 1.0
            else:
                # Check if we should enter a deeper level
                for idx in range(current_level + 1, len(levels)):
                    if current_drawdown >= levels[idx]['drawdown_threshold']:
                        new_level = idx
                        new_gross = levels[idx]['gross_reduction']
                        break

        # Update state
        current_level = new_level
        gross_multipliers.iloc[i] = new_gross

    return gross_multipliers


def calculate_stop_loss_metrics(
    daily_pnl: Union[pd.Series, np.ndarray],
    stop_loss_levels: List[Tuple[float, float]],
    initial_capital: float,
    dates: Optional[Union[pd.DatetimeIndex, List]] = None
) -> pd.DataFrame:
    """
    Calculate detailed stop loss metrics using simplified immediate exit logic.

    Returns a DataFrame with portfolio values, drawdowns, and gross multipliers.

    Parameters:
    -----------
    daily_pnl : pd.Series or np.ndarray
        Daily PnL values (in dollars)

    stop_loss_levels : List[Tuple[float, float]]
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
        - drawdown_dollar: Dollar drawdown from peak
        - triggered_level: Currently active stop loss level (0-indexed, None if cleared)
        - gross_multiplier: Gross exposure multiplier

    Examples:
    ---------
    >>> daily_pnl = pd.Series([0, -12000, 3000, 2000])
    >>> levels = [(5000, 0.75), (10000, 0.50)]
    >>> metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital=100000)
    """
    # Convert inputs to pandas Series if needed
    if isinstance(daily_pnl, np.ndarray):
        if dates is not None:
            daily_pnl = pd.Series(daily_pnl, index=dates)
        else:
            daily_pnl = pd.Series(daily_pnl)
    elif not isinstance(daily_pnl, pd.Series):
        raise ValueError("daily_pnl must be a pandas Series or numpy array")

    # Parse and validate levels (reuse logic from calculate_stop_loss_gross)
    levels = []
    for level_tuple in stop_loss_levels:
        if len(level_tuple) != 2:
            raise ValueError(f"Stop loss level must be 2-tuple (drawdown_threshold, gross_reduction), got {len(level_tuple)}-tuple")

        dd_threshold, gross_reduction = level_tuple

        levels.append({
            'drawdown_threshold': dd_threshold,
            'gross_reduction': gross_reduction
        })

    levels = sorted(levels, key=lambda x: x['drawdown_threshold'])

    # Calculate cumulative portfolio value
    portfolio_values = initial_capital + daily_pnl.cumsum()

    # Initialize result DataFrame
    results = pd.DataFrame(index=daily_pnl.index)
    results['portfolio_value'] = portfolio_values
    results['peak_value'] = 0.0
    results['drawdown_dollar'] = 0.0
    results['triggered_level'] = None
    results['gross_multiplier'] = 1.0

    # Track state
    peak_value = initial_capital
    current_level = None

    # Process each day using simplified logic
    for i, (date, pnl) in enumerate(daily_pnl.items()):
        portfolio_value = portfolio_values.iloc[i]

        # Update peak if new high
        if portfolio_value > peak_value:
            peak_value = portfolio_value
            current_level = None

        # Calculate current drawdown
        current_drawdown = peak_value - portfolio_value

        # Determine level using simplified logic
        new_level = current_level
        new_gross = results.iloc[i-1, results.columns.get_loc('gross_multiplier')] if i > 0 else 1.0

        if current_level is None:
            # Not at any level - check if we should enter one (deepest first)
            for idx in range(len(levels) - 1, -1, -1):
                if current_drawdown >= levels[idx]['drawdown_threshold']:
                    new_level = idx
                    new_gross = levels[idx]['gross_reduction']
                    break
        else:
            # At a level - check if we should exit completely or go deeper
            current_threshold = levels[current_level]['drawdown_threshold']

            if current_drawdown < current_threshold:
                # Exit completely (jump to no stop loss)
                new_level = None
                new_gross = 1.0
            else:
                # Check if we should enter a deeper level
                for idx in range(current_level + 1, len(levels)):
                    if current_drawdown >= levels[idx]['drawdown_threshold']:
                        new_level = idx
                        new_gross = levels[idx]['gross_reduction']
                        break

        # Store results
        results.iloc[i, results.columns.get_loc('peak_value')] = peak_value
        results.iloc[i, results.columns.get_loc('drawdown_dollar')] = current_drawdown
        results.iloc[i, results.columns.get_loc('triggered_level')] = new_level
        results.iloc[i, results.columns.get_loc('gross_multiplier')] = new_gross

        # Update state
        current_level = new_level

    return results

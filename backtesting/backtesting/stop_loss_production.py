"""
Production Stop Loss Calculator

Standalone function for calculating stop loss gross reductions based on
daily PnL time series using sticky recovery logic.

All thresholds are drawdown levels from peak for consistency.
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
    Calculate gross exposure multipliers using sticky recovery logic.

    Uses drawdown-based thresholds with sticky behavior: once at a level,
    stay there until drawdown improves past the recovery threshold.

    Parameters:
    -----------
    daily_pnl : pd.Series or np.ndarray
        Daily PnL values (in dollars). Can be a pandas Series with datetime index
        or a numpy array. If numpy array, dates parameter should be provided.

    stop_loss_levels : List[Tuple]
        List of stop loss levels, each specified as:
        - 2-tuple: (drawdown_threshold, gross_reduction)
        - 3-tuple: (drawdown_threshold, gross_reduction, recovery_drawdown)

        Where:
        - drawdown_threshold: Dollar drawdown from peak to ENTER this level
        - gross_reduction: Target gross exposure (0-1, e.g., 0.75 = 75%)
        - recovery_drawdown: Dollar drawdown from peak to EXIT this level (optional)

        STICKY LOGIC: Once at a level, stay there until drawdown ≤ recovery_drawdown

        Example: [(5000, 0.75, 2000), (10000, 0.50, 5000)]
        - Enter at $5k DD → 75% gross, exit when DD ≤ $2k
        - Enter at $10k DD → 50% gross, exit when DD ≤ $5k

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
    >>> # Sticky recovery example
    >>> daily_pnl = pd.Series([0, -6000, 1000, 1000, 2000],
    ...                        index=pd.date_range('2023-01-01', periods=5))
    >>> levels = [(5000, 0.75, 2000)]
    >>> gross = calculate_stop_loss_gross(daily_pnl, levels, initial_capital=100000)
    >>> # Day 0: $0 DD → 100%
    >>> # Day 1: $6k DD → 75% (entered level)
    >>> # Day 2: $5k DD → 75% (sticky, DD > $2k recovery)
    >>> # Day 3: $4k DD → 75% (sticky, DD > $2k recovery)
    >>> # Day 4: $2k DD → 75% (sticky, DD = $2k recovery, need DD ≤ $2k)
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
            recovery_drawdown = None
        elif len(level_tuple) == 3:
            dd_threshold, gross_reduction, recovery_drawdown = level_tuple
        else:
            raise ValueError(f"Stop loss level must be 2-tuple or 3-tuple, got {len(level_tuple)}-tuple")

        # Validate values
        if dd_threshold < 0:
            raise ValueError(f"Drawdown threshold must be non-negative, got {dd_threshold}")
        if gross_reduction < 0 or gross_reduction > 1:
            raise ValueError(f"Gross reduction must be in [0, 1], got {gross_reduction}")
        if recovery_drawdown is not None:
            if recovery_drawdown < 0:
                raise ValueError(f"Recovery drawdown must be non-negative, got {recovery_drawdown}")
            if recovery_drawdown <= dd_threshold:
                raise ValueError(
                    f"Recovery drawdown ({recovery_drawdown}) must be greater than "
                    f"drawdown threshold ({dd_threshold}). "
                    f"Use None to default to drawdown_threshold for immediate scale up."
                )

        levels.append({
            'drawdown_threshold': dd_threshold,
            'gross_reduction': gross_reduction,
            'recovery_drawdown': recovery_drawdown
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
    last_exited_level = None  # Track most recently exited level for hysteresis

    # Process each day using sticky recovery logic
    for i, (date, pnl) in enumerate(daily_pnl.items()):
        portfolio_value = portfolio_values.iloc[i]

        # Update peak if new high (clears all stop loss)
        if portfolio_value > peak_value:
            peak_value = portfolio_value
            current_level = None
            last_exited_level = None  # Clear last exited level on new peak
            gross_multipliers.iloc[i] = 1.0
            continue

        # Calculate current drawdown from peak
        current_drawdown = peak_value - portfolio_value

        # Clear last_exited_level if DD improves below its entry threshold
        if last_exited_level is not None:
            if current_drawdown < levels[last_exited_level]['drawdown_threshold']:
                last_exited_level = None

        # Determine level using proper hysteresis logic
        new_level = current_level
        new_gross = gross_multipliers.iloc[i-1] if i > 0 else 1.0

        if current_level is None:
            # Not at any level - check if we should enter one
            # For initial entry, use drawdown_threshold
            for idx in range(len(levels) - 1, -1, -1):
                if current_drawdown >= levels[idx]['drawdown_threshold']:
                    new_level = idx
                    new_gross = levels[idx]['gross_reduction']
                    break
        else:
            # Currently at a level - check for exit or deeper entry
            current_level_obj = levels[current_level]
            recovery_dd = current_level_obj['recovery_drawdown'] if current_level_obj['recovery_drawdown'] is not None else current_level_obj['drawdown_threshold']

            # Exit when DD < recovery_drawdown (improved below recovery threshold)
            if current_drawdown < recovery_dd:
                # Exiting current level - track it for hysteresis
                last_exited_level = current_level

                # Check if we enter a shallower level
                new_level = None
                new_gross = 1.0

                for idx in range(current_level - 1, -1, -1):
                    level_recovery = levels[idx]['recovery_drawdown'] if levels[idx]['recovery_drawdown'] is not None else levels[idx]['drawdown_threshold']
                    if current_drawdown >= level_recovery:
                        new_level = idx
                        new_gross = levels[idx]['gross_reduction']
                        break
            else:
                # Still at current level (DD >= recovery_drawdown)
                # Check if we should enter a deeper level
                for idx in range(current_level + 1, len(levels)):
                    # If this is the level we just exited, need DD >= recovery_drawdown (hysteresis)
                    # Otherwise, use drawdown_threshold for entry
                    if idx == last_exited_level:
                        threshold = levels[idx]['recovery_drawdown'] if levels[idx]['recovery_drawdown'] is not None else levels[idx]['drawdown_threshold']
                    else:
                        threshold = levels[idx]['drawdown_threshold']

                    if current_drawdown >= threshold:
                        new_level = idx
                        new_gross = levels[idx]['gross_reduction']
                        # Clear last_exited_level when we enter a deeper level
                        last_exited_level = None
                        break

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
    Calculate detailed stop loss metrics using sticky recovery logic.

    Returns a DataFrame with portfolio values, drawdowns, and gross multipliers.

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
        - drawdown_dollar: Dollar drawdown from peak
        - triggered_level: Currently active stop loss level (0-indexed, None if cleared)
        - gross_multiplier: Gross exposure multiplier

    Examples:
    ---------
    >>> daily_pnl = pd.Series([0, -5000, -3000, 2000])
    >>> levels = [(5000, 0.75, 2000), (10000, 0.50, 5000)]
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
        if len(level_tuple) == 2:
            dd_threshold, gross_reduction = level_tuple
            recovery_drawdown = None
        elif len(level_tuple) == 3:
            dd_threshold, gross_reduction, recovery_drawdown = level_tuple
        else:
            raise ValueError(f"Stop loss level must be 2-tuple or 3-tuple, got {len(level_tuple)}-tuple")

        levels.append({
            'drawdown_threshold': dd_threshold,
            'gross_reduction': gross_reduction,
            'recovery_drawdown': recovery_drawdown
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
    last_exited_level = None  # Track most recently exited level for hysteresis

    # Process each day using sticky logic
    for i, (date, pnl) in enumerate(daily_pnl.items()):
        portfolio_value = portfolio_values.iloc[i]

        # Update peak if new high
        if portfolio_value > peak_value:
            peak_value = portfolio_value
            current_level = None
            last_exited_level = None  # Clear last exited level on new peak

        # Calculate current drawdown
        current_drawdown = peak_value - portfolio_value

        # Clear last_exited_level if DD improves below its entry threshold
        if last_exited_level is not None:
            if current_drawdown < levels[last_exited_level]['drawdown_threshold']:
                last_exited_level = None

        # Determine level using proper hysteresis logic (same as calculate_stop_loss_gross)
        new_level = current_level
        new_gross = results.iloc[i-1, results.columns.get_loc('gross_multiplier')] if i > 0 else 1.0

        if current_level is None:
            # Not at any level - check if we should enter one
            # For initial entry, use drawdown_threshold
            for idx in range(len(levels) - 1, -1, -1):
                if current_drawdown >= levels[idx]['drawdown_threshold']:
                    new_level = idx
                    new_gross = levels[idx]['gross_reduction']
                    break
        else:
            # Currently at a level - check for exit or deeper entry
            current_level_obj = levels[current_level]
            recovery_dd = current_level_obj['recovery_drawdown'] if current_level_obj['recovery_drawdown'] is not None else current_level_obj['drawdown_threshold']

            # Exit when DD < recovery_drawdown (improved below recovery threshold)
            if current_drawdown < recovery_dd:
                # Exiting current level - track it for hysteresis
                last_exited_level = current_level

                # Check if we enter a shallower level
                new_level = None
                new_gross = 1.0

                for idx in range(current_level - 1, -1, -1):
                    level_recovery = levels[idx]['recovery_drawdown'] if levels[idx]['recovery_drawdown'] is not None else levels[idx]['drawdown_threshold']
                    if current_drawdown >= level_recovery:
                        new_level = idx
                        new_gross = levels[idx]['gross_reduction']
                        break
            else:
                # Still at current level (DD >= recovery_drawdown)
                # Check if we should enter a deeper level
                for idx in range(current_level + 1, len(levels)):
                    # If this is the level we just exited, need DD >= recovery_drawdown (hysteresis)
                    # Otherwise, use drawdown_threshold for entry
                    if idx == last_exited_level:
                        threshold = levels[idx]['recovery_drawdown'] if levels[idx]['recovery_drawdown'] is not None else levels[idx]['drawdown_threshold']
                    else:
                        threshold = levels[idx]['drawdown_threshold']

                    if current_drawdown >= threshold:
                        new_level = idx
                        new_gross = levels[idx]['gross_reduction']
                        # Clear last_exited_level when we enter a deeper level
                        last_exited_level = None
                        break

        # Store results
        results.iloc[i, results.columns.get_loc('peak_value')] = peak_value
        results.iloc[i, results.columns.get_loc('drawdown_dollar')] = current_drawdown
        results.iloc[i, results.columns.get_loc('triggered_level')] = new_level
        results.iloc[i, results.columns.get_loc('gross_multiplier')] = new_gross

        # Update state
        current_level = new_level

    return results

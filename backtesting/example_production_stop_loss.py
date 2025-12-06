"""
Example: Using Production Stop Loss Functions

This script demonstrates how to use the production stop loss functions
for calculating gross reductions from daily PnL.
"""

import pandas as pd
import numpy as np
from backtesting import calculate_stop_loss_gross, calculate_stop_loss_metrics


def example_basic_usage():
    """Basic example with simple daily PnL."""
    print("="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)

    # Create a simple daily PnL series
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    daily_pnl = pd.Series([
        0, 100, -300, -500, -200,  # Drawdown phase
        200, 300, 400, 200, 100     # Recovery phase
    ], index=dates)

    # Define stop loss levels
    levels = [
        (500, 0.75, 250),    # $500 loss → 75%, recover at $250
        (1000, 0.50, 500),   # $1000 loss → 50%, recover at $500
    ]

    # Calculate gross multipliers
    gross = calculate_stop_loss_gross(
        daily_pnl=daily_pnl,
        stop_loss_levels=levels,
        initial_capital=10000
    )

    # Calculate portfolio values
    portfolio_values = 10000 + daily_pnl.cumsum()

    print("\nDaily Results:")
    print(f"{'Date':<12} {'PnL':>8} {'Portfolio':>12} {'Gross':>8}")
    print("-" * 44)
    for date, pnl in daily_pnl.items():
        pv = portfolio_values[date]
        gr = gross[date]
        print(f"{date.strftime('%Y-%m-%d'):<12} {pnl:>8.0f} {pv:>12,.0f} {gr:>7.0%}")

    print(f"\n{'='*70}\n")


def example_detailed_metrics():
    """Example showing detailed metrics."""
    print("="*70)
    print("EXAMPLE 2: Detailed Metrics")
    print("="*70)

    # Create PnL with a significant drawdown
    dates = pd.date_range('2023-01-01', periods=8, freq='D')
    daily_pnl = pd.Series([
        0, -600, -800, 400, 600, 400, 200, 100
    ], index=dates)

    levels = [
        (500, 0.75, 250),
        (1000, 0.50, 500),
    ]

    # Get detailed metrics
    metrics = calculate_stop_loss_metrics(
        daily_pnl=daily_pnl,
        stop_loss_levels=levels,
        initial_capital=10000
    )

    print("\nDetailed Metrics:")
    print(metrics.to_string())

    print("\n\nSummary:")
    print(f"  Max Drawdown: ${metrics['drawdown_dollar'].max():,.0f}")
    print(f"  Days with Stop Loss Active: {metrics['triggered_level'].notna().sum()}")
    print(f"  Final Gross Multiplier: {metrics['gross_multiplier'].iloc[-1]:.0%}")

    print(f"\n{'='*70}\n")


def example_live_trading_simulation():
    """Example simulating live trading with daily updates."""
    print("="*70)
    print("EXAMPLE 3: Live Trading Simulation")
    print("="*70)

    # Simulate 20 days of trading
    np.random.seed(42)
    initial_capital = 100000

    levels = [
        (5000, 0.75, 2500),
        (10000, 0.50, 5000),
        (15000, 0.25, 7500),
    ]

    # Track daily PnL
    daily_pnl = pd.Series(dtype=float)

    print("\nSimulating 20 trading days...")
    print(f"{'Day':>3} {'Daily PnL':>10} {'Portfolio':>12} {'Gross':>8} {'Level':>6}")
    print("-" * 45)

    for day in range(20):
        date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=day)

        # Simulate daily PnL (with a crash event on day 10)
        if day == 10:
            pnl = -8000  # Crash event
        elif day == 11:
            pnl = -4000  # Continued selloff
        elif day > 11:
            pnl = np.random.normal(500, 800)  # Recovery
        else:
            pnl = np.random.normal(200, 1000)  # Normal trading

        daily_pnl[date] = pnl

        # Calculate current gross multiplier
        gross_series = calculate_stop_loss_gross(
            daily_pnl=daily_pnl,
            stop_loss_levels=levels,
            initial_capital=initial_capital
        )

        current_gross = gross_series.iloc[-1]
        portfolio_value = initial_capital + daily_pnl.sum()

        # Determine active level
        metrics = calculate_stop_loss_metrics(daily_pnl, levels, initial_capital)
        triggered_level = metrics['triggered_level'].iloc[-1]
        level_str = f"L{int(triggered_level)+1}" if pd.notna(triggered_level) else "None"

        print(f"{day+1:>3} {pnl:>10,.0f} {portfolio_value:>12,.0f} {current_gross:>7.0%} {level_str:>6}")

        # In live trading, you would use current_gross to scale your positions
        # target_positions = {...}
        # actual_positions = {ticker: qty * current_gross for ticker, qty in target_positions.items()}

    print(f"\nFinal Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Total Return: {(portfolio_value/initial_capital - 1):.2%}")

    print(f"\n{'='*70}\n")


def example_numpy_array_input():
    """Example using numpy arrays instead of pandas Series."""
    print("="*70)
    print("EXAMPLE 4: Numpy Array Input")
    print("="*70)

    # Create data as numpy arrays
    daily_pnl = np.array([0, -300, -400, 200, 300])
    dates = pd.date_range('2023-01-01', periods=5)

    levels = [(500, 0.75, 250)]

    # Calculate with numpy array
    gross = calculate_stop_loss_gross(
        daily_pnl=daily_pnl,
        stop_loss_levels=levels,
        initial_capital=10000,
        dates=dates  # Required for numpy arrays
    )

    print("\nInput type: numpy.ndarray")
    print("Output type: pandas.Series")
    print("\nGross Multipliers:")
    print(gross.to_string())

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    example_basic_usage()
    example_detailed_metrics()
    example_live_trading_simulation()
    example_numpy_array_input()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)

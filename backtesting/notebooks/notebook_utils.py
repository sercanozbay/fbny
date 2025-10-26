"""
Utility functions for Jupyter notebooks.

Helper functions to make notebooks cleaner and more concise.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append('..')

from backtesting import Backtester, BacktestConfig, DataManager


def setup_plotting_style(style='seaborn-v0_8-darkgrid'):
    """
    Set up matplotlib style for consistent plots.

    Parameters:
    -----------
    style : str
        Matplotlib style name
    """
    try:
        plt.style.use(style)
    except:
        pass  # Use default if style not available

    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10


def load_sample_data(data_dir='../sample_data'):
    """
    Quick helper to load all sample data.

    Parameters:
    -----------
    data_dir : str
        Data directory path

    Returns:
    --------
    DataManager
        Loaded data manager
    """
    data_manager = DataManager(data_dir=data_dir, use_float32=True)

    # Load all data
    data_manager.load_prices()
    data_manager.load_adv()
    data_manager.load_betas()
    data_manager.load_sector_mapping()

    try:
        data_manager.load_factor_exposures()
        data_manager.load_factor_returns()
        data_manager.load_factor_covariance()
        data_manager.load_specific_variance()
    except:
        print("Note: Factor model data not loaded")

    return data_manager


def quick_backtest(
    data_manager,
    use_case=1,
    inputs=None,
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_cash=10_000_000,
    enable_beta_hedge=False,
    enable_sector_hedge=False,
    show_progress=True
):
    """
    Run a quick backtest with default settings.

    Parameters:
    -----------
    data_manager : DataManager
        Data manager with loaded data
    use_case : int
        Use case (1, 2, or 3)
    inputs : dict
        Use case specific inputs
    start_date : str or pd.Timestamp
        Start date
    end_date : str or pd.Timestamp
        End date
    initial_cash : float
        Starting cash
    enable_beta_hedge : bool
        Enable beta hedging
    enable_sector_hedge : bool
        Enable sector hedging
    show_progress : bool
        Show progress bar

    Returns:
    --------
    BacktestResults
        Backtest results
    """
    config = BacktestConfig(
        initial_cash=initial_cash,
        max_adv_participation=0.05,
        tc_power=1.5,
        tc_coefficient=0.01,
        enable_beta_hedge=enable_beta_hedge,
        enable_sector_hedge=enable_sector_hedge,
        risk_free_rate=0.02
    )

    backtester = Backtester(config, data_manager)

    results = backtester.run(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        use_case=use_case,
        inputs=inputs,
        show_progress=show_progress
    )

    return results


def plot_results_summary(results, title='Backtest Results'):
    """
    Create a summary plot of backtest results.

    Parameters:
    -----------
    results : BacktestResults
        Backtest results
    title : str
        Plot title
    """
    results_df = results.to_dataframe()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Cumulative returns
    cum_returns = (results_df['portfolio_value'] / results_df['portfolio_value'].iloc[0] - 1) * 100
    axes[0, 0].plot(results_df['date'], cum_returns, linewidth=2)
    axes[0, 0].fill_between(results_df['date'], 0, cum_returns, alpha=0.3)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Drawdown
    values = results_df['portfolio_value'].values
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max * 100
    axes[0, 1].fill_between(results_df['date'], drawdowns, 0, alpha=0.7, color='red')
    axes[0, 1].plot(results_df['date'], drawdowns, color='darkred', linewidth=1)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)

    # Exposures
    axes[1, 0].plot(results_df['date'], results_df['gross_exposure'], label='Gross', linewidth=2)
    axes[1, 0].plot(results_df['date'], results_df['net_exposure'], label='Net', linewidth=2)
    axes[1, 0].set_title('Portfolio Exposures')
    axes[1, 0].set_ylabel('Exposure ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Daily PnL
    axes[1, 1].bar(results_df['date'], results_df['daily_pnl'], alpha=0.7)
    axes[1, 1].set_title('Daily PnL')
    axes[1, 1].set_ylabel('PnL ($)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def compare_strategies(results_list, labels, metric='sharpe_ratio'):
    """
    Compare multiple backtest results.

    Parameters:
    -----------
    results_list : list of BacktestResults
        List of backtest results
    labels : list of str
        Labels for each result
    metric : str
        Metric to compare
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot cumulative returns
    for results, label in zip(results_list, labels):
        results_df = results.to_dataframe()
        cum_returns = (results_df['portfolio_value'] / results_df['portfolio_value'].iloc[0] - 1) * 100
        axes[0].plot(results_df['date'], cum_returns, label=label, linewidth=2)

    axes[0].set_title('Cumulative Returns Comparison')
    axes[0].set_ylabel('Return (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Bar chart of selected metric
    metric_values = []
    for results in results_list:
        metrics = results.calculate_metrics()
        metric_values.append(metrics.get(metric, 0))

    axes[1].bar(range(len(labels)), metric_values)
    axes[1].set_title(f'{metric.replace("_", " ").title()} Comparison')
    axes[1].set_ylabel(metric.replace("_", " ").title())
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Print comparison table
    print("\nMetrics Comparison:")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Total Return':<15} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10}")
    print("-" * 80)

    for results, label in zip(results_list, labels):
        metrics = results.calculate_metrics()
        print(f"{label:<20} {metrics['total_return']:>13.2%} {metrics['sharpe_ratio']:>9.2f} "
              f"{metrics['max_drawdown']:>9.2%} {metrics['win_rate']:>9.2%}")

    print("=" * 80)


def print_metrics_table(metrics):
    """
    Print metrics in a formatted table.

    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    categories = {
        'Returns': ['total_return', 'annualized_return'],
        'Risk': ['volatility', 'max_drawdown'],
        'Risk-Adjusted': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
        'Distribution': ['skewness', 'kurtosis', 'var_95', 'cvar_95'],
        'Trading': ['win_rate', 'profit_factor']
    }

    for category, metric_names in categories.items():
        print(f"\n{category}:")
        for metric_name in metric_names:
            if metric_name in metrics:
                value = metrics[metric_name]
                formatted_name = metric_name.replace('_', ' ').title()

                if 'return' in metric_name.lower() or 'rate' in metric_name.lower():
                    print(f"  {formatted_name:.<40} {value:>15.2%}")
                elif 'ratio' in metric_name.lower() or 'factor' in metric_name.lower():
                    print(f"  {formatted_name:.<40} {value:>15.2f}")
                else:
                    print(f"  {formatted_name:.<40} {value:>15.4f}")

    print("=" * 60 + "\n")


def create_equal_weight_targets(prices, start_date=None, end_date=None):
    """
    Create equal-weight target positions.

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    start_date : pd.Timestamp, optional
        Start date
    end_date : pd.Timestamp, optional
        End date

    Returns:
    --------
    dict
        Targets by date
    """
    if start_date:
        prices = prices[prices.index >= start_date]
    if end_date:
        prices = prices[prices.index <= end_date]

    n_securities = len(prices.columns)
    equal_weight = 1.0 / n_securities

    targets_by_date = {}
    for date in prices.index:
        targets_by_date[date] = {ticker: equal_weight for ticker in prices.columns}

    return targets_by_date


def format_summary(results):
    """
    Get formatted summary string.

    Parameters:
    -----------
    results : BacktestResults
        Backtest results

    Returns:
    --------
    str
        Formatted summary
    """
    metrics = results.calculate_metrics()
    results_df = results.to_dataframe()

    summary = f"""
Backtest Summary
================
Period: {results_df['date'].iloc[0].date()} to {results_df['date'].iloc[-1].date()}
Initial Value: ${results_df['portfolio_value'].iloc[0]:,.2f}
Final Value: ${results_df['portfolio_value'].iloc[-1]:,.2f}

Returns:
  Total Return: {metrics['total_return']:.2%}
  Annualized Return: {metrics['annualized_return']:.2%}

Risk:
  Volatility: {metrics['volatility']:.2%}
  Max Drawdown: {metrics['max_drawdown']:.2%}

Risk-Adjusted:
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  Sortino Ratio: {metrics['sortino_ratio']:.2f}
  Calmar Ratio: {metrics['calmar_ratio']:.2f}

Trading:
  Win Rate: {metrics['win_rate']:.2%}
  Profit Factor: {metrics['profit_factor']:.2f}
  Trading Days: {metrics['n_days']:.0f}
"""
    return summary

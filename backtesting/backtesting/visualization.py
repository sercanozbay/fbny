"""
Visualization module for backtest results.

This module creates charts and plots for performance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class BacktestVisualizer:
    """
    Create visualizations of backtest results.

    Generates charts for returns, drawdowns, exposures, and attribution.
    """

    def __init__(
        self,
        style: str = 'seaborn-v0_8-darkgrid',
        figsize: tuple = (12, 6),
        dpi: int = 100
    ):
        """
        Initialize visualizer.

        Parameters:
        -----------
        style : str
            Matplotlib style
        figsize : tuple
            Default figure size
        dpi : int
            Dots per inch for figures
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available

        self.figsize = figsize
        self.dpi = dpi

    def plot_cumulative_returns(
        self,
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        benchmark_values: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot cumulative returns over time.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        benchmark_values : List[float], optional
            Benchmark values for comparison
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate returns
        port_returns = (np.array(portfolio_values) / portfolio_values[0] - 1) * 100

        ax.plot(dates, port_returns, label='Strategy', linewidth=2)

        if benchmark_values is not None:
            bench_returns = (np.array(benchmark_values) / benchmark_values[0] - 1) * 100
            ax.plot(dates, bench_returns, label='Benchmark', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Cumulative Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_drawdown_underwater(
        self,
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot underwater (drawdown) chart.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate drawdowns
        values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max * 100

        ax.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
        ax.plot(dates, drawdowns, color='red', linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Underwater Plot')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_rolling_sharpe(
        self,
        dates: List[pd.Timestamp],
        daily_returns: np.ndarray,
        window: int = 60,
        save_path: Optional[str] = None
    ):
        """
        Plot rolling Sharpe ratio.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        daily_returns : np.ndarray
            Daily returns
        window : int
            Rolling window size
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate rolling Sharpe
        returns_series = pd.Series(daily_returns, index=dates)
        rolling_sharpe = (
            returns_series.rolling(window).mean() /
            returns_series.rolling(window).std() *
            np.sqrt(252)
        )

        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_xlabel('Date')
        ax.set_ylabel(f'{window}-Day Rolling Sharpe Ratio')
        ax.set_title(f'Rolling Sharpe Ratio ({window} days)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_return_distribution(
        self,
        daily_returns: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot histogram of daily returns.

        Parameters:
        -----------
        daily_returns : np.ndarray
            Daily returns
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.hist(daily_returns * 100, bins=50, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Daily Returns')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean line
        mean_ret = np.mean(daily_returns) * 100
        ax.axvline(x=mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.2f}%')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_factor_attribution(
        self,
        factor_pnl_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot cumulative factor attribution.

        Parameters:
        -----------
        factor_pnl_df : pd.DataFrame
            DataFrame with factor PnL (dates × factors)
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate cumulative PnL
        cum_pnl = factor_pnl_df.cumsum()

        for column in cum_pnl.columns:
            ax.plot(cum_pnl.index, cum_pnl[column], label=column, linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Factor Attribution - Cumulative PnL')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_exposures(
        self,
        dates: List[pd.Timestamp],
        gross_exposures: List[float],
        net_exposures: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot gross and net exposures over time.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        gross_exposures : List[float]
            Gross exposure values
        net_exposures : List[float]
            Net exposure values
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(dates, gross_exposures, label='Gross Exposure', linewidth=2)
        ax.plot(dates, net_exposures, label='Net Exposure', linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Exposure ($)')
        ax.set_title('Portfolio Exposures')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_transaction_costs(
        self,
        dates: List[pd.Timestamp],
        transaction_costs: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot transaction costs over time.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        transaction_costs : List[float]
            Daily transaction costs
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.5), dpi=self.dpi)

        # Daily costs
        ax1.bar(dates, transaction_costs, alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Cost ($)')
        ax1.set_title('Daily Transaction Costs')
        ax1.grid(True, alpha=0.3, axis='y')

        # Cumulative costs
        cum_costs = np.cumsum(transaction_costs)
        ax2.plot(dates, cum_costs, linewidth=2, color='red')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Cost ($)')
        ax2.set_title('Cumulative Transaction Costs')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_all_charts(
        self,
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        daily_returns: np.ndarray,
        gross_exposures: List[float],
        net_exposures: List[float],
        transaction_costs: List[float],
        factor_pnl_df: Optional[pd.DataFrame],
        output_dir: str
    ):
        """
        Create all standard charts and save to directory.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        daily_returns : np.ndarray
            Daily returns
        gross_exposures : List[float]
            Gross exposures
        net_exposures : List[float]
            Net exposures
        transaction_costs : List[float]
            Transaction costs
        factor_pnl_df : pd.DataFrame, optional
            Factor PnL data
        output_dir : str
            Directory to save charts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating charts...")

        self.plot_cumulative_returns(
            dates, portfolio_values,
            save_path=str(output_path / 'cumulative_returns.png')
        )

        self.plot_drawdown_underwater(
            dates, portfolio_values,
            save_path=str(output_path / 'drawdown_underwater.png')
        )

        self.plot_rolling_sharpe(
            dates, daily_returns,
            save_path=str(output_path / 'rolling_sharpe.png')
        )

        self.plot_return_distribution(
            daily_returns,
            save_path=str(output_path / 'return_distribution.png')
        )

        self.plot_exposures(
            dates, gross_exposures, net_exposures,
            save_path=str(output_path / 'exposures.png')
        )

        self.plot_transaction_costs(
            dates, transaction_costs,
            save_path=str(output_path / 'transaction_costs.png')
        )

        if factor_pnl_df is not None and not factor_pnl_df.empty:
            self.plot_factor_attribution(
                factor_pnl_df,
                save_path=str(output_path / 'factor_attribution.png')
            )

        print(f"Charts saved to {output_dir}")

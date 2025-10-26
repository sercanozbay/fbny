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

    def plot_factor_exposures_timeseries(
        self,
        dates: List[pd.Timestamp],
        factor_exposures: List[Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot factor exposures over time.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        factor_exposures : List[Dict[str, float]]
            Factor exposures per date (list of dicts with factor names as keys)
        save_path : str, optional
            Path to save figure
        """
        if not factor_exposures or not any(factor_exposures):
            print("No factor exposure data available")
            return

        # Convert list of dicts to DataFrame
        factor_exp_df = pd.DataFrame(factor_exposures, index=dates)

        if factor_exp_df.empty or factor_exp_df.shape[1] == 0:
            print("No factor exposure data available")
            return

        # Get factor names
        factor_names = factor_exp_df.columns.tolist()
        n_factors = len(factor_names)

        if n_factors == 0:
            return

        # Create subplots - up to 6 factors per figure
        n_cols = min(2, n_factors)
        n_rows = min(3, int(np.ceil(n_factors / n_cols)))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows*3), dpi=self.dpi)

        if n_factors == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each factor
        for idx, factor in enumerate(factor_names[:n_rows*n_cols]):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Plot exposure line
            factor_values = factor_exp_df[factor].values
            ax.plot(dates, factor_values, linewidth=2, label=factor)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

            # Fill positive/negative areas
            ax.fill_between(dates, 0, factor_values, where=(factor_values >= 0),
                           alpha=0.3, color='green', label='Long')
            ax.fill_between(dates, 0, factor_values, where=(factor_values < 0),
                           alpha=0.3, color='red', label='Short')

            ax.set_xlabel('Date')
            ax.set_ylabel('Exposure')
            ax.set_title(f'{factor} Exposure')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

        # Hide unused subplots
        for idx in range(n_factors, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_factor_exposures_heatmap(
        self,
        dates: List[pd.Timestamp],
        factor_exposures: List[Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot factor exposures as a heatmap.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        factor_exposures : List[Dict[str, float]]
            Factor exposures per date
        save_path : str, optional
            Path to save figure
        """
        if not factor_exposures or not any(factor_exposures):
            print("No factor exposure data available")
            return

        # Convert to DataFrame
        factor_exp_df = pd.DataFrame(factor_exposures, index=dates)

        if factor_exp_df.empty or factor_exp_df.shape[1] == 0:
            print("No factor exposure data available")
            return

        fig, ax = plt.subplots(figsize=(self.figsize[0], max(4, len(factor_exp_df.columns)*0.5)), dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(factor_exp_df.T.values, aspect='auto', cmap='RdYlGn',
                       interpolation='nearest', vmin=-factor_exp_df.abs().max().max(),
                       vmax=factor_exp_df.abs().max().max())

        # Set ticks
        ax.set_yticks(range(len(factor_exp_df.columns)))
        ax.set_yticklabels(factor_exp_df.columns)

        # Set x-axis with fewer labels for readability
        n_ticks = min(10, len(dates))
        tick_indices = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in tick_indices], rotation=45, ha='right')

        ax.set_xlabel('Date')
        ax.set_ylabel('Factor')
        ax.set_title('Factor Exposures Heatmap')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Exposure', rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_pnl_breakdown(
        self,
        dates: List[pd.Timestamp],
        external_pnl: List[float],
        executed_pnl: List[float],
        overnight_pnl: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot stacked area chart of PnL breakdown.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        external_pnl : List[float]
            External trade PnL
        executed_pnl : List[float]
            Executed trade PnL
        overnight_pnl : List[float]
            Overnight PnL
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Cumulative PnL breakdown (stacked area)
        cum_external = np.cumsum(external_pnl)
        cum_executed = np.cumsum(executed_pnl)
        cum_overnight = np.cumsum(overnight_pnl)

        axes[0].plot(dates, cum_external, label='External Trades', linewidth=2, color='#2E86AB')
        axes[0].plot(dates, cum_executed, label='Executed Trades', linewidth=2, color='#A23B72')
        axes[0].plot(dates, cum_overnight, label='Overnight', linewidth=2, color='#F18F01')
        axes[0].plot(dates, cum_external + cum_executed + cum_overnight,
                     label='Total', linewidth=2.5, color='black', linestyle='--')

        axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        axes[0].set_title('Cumulative PnL Breakdown', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cumulative PnL ($)', fontsize=12)
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Daily PnL breakdown (stacked bar)
        width = 0.8
        x = np.arange(len(dates))

        # Create stacked bar chart
        bars1 = axes[1].bar(x, external_pnl, width, label='External Trades',
                           color='#2E86AB', alpha=0.8)
        bars2 = axes[1].bar(x, executed_pnl, width, label='Executed Trades',
                           bottom=external_pnl, color='#A23B72', alpha=0.8)

        bottom_vals = [e + x for e, x in zip(external_pnl, executed_pnl)]
        bars3 = axes[1].bar(x, overnight_pnl, width, label='Overnight',
                           bottom=bottom_vals, color='#F18F01', alpha=0.8)

        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        axes[1].set_title('Daily PnL Breakdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Daily PnL ($)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Format x-axis for lower plot only
        if len(dates) > 50:
            # Show fewer dates if too many
            step = len(dates) // 10
            axes[1].set_xticks(x[::step])
            axes[1].set_xticklabels([d.strftime('%Y-%m-%d') for d in dates[::step]],
                                   rotation=45, ha='right')
        else:
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([d.strftime('%Y-%m-%d') for d in dates],
                                   rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_external_trades_analysis(
        self,
        trades_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot external trades analysis (volume, count, execution).

        Parameters:
        -----------
        trades_df : pd.DataFrame
            DataFrame with external trades
        save_path : str, optional
            Path to save figure
        """
        if trades_df.empty or 'type' not in trades_df.columns:
            print("No external trades to plot")
            return

        external_trades = trades_df[trades_df['type'] == 'external'].copy()

        if external_trades.empty:
            print("No external trades to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Trade count by ticker
        ticker_counts = external_trades['ticker'].value_counts().head(15)
        axes[0, 0].barh(range(len(ticker_counts)), ticker_counts.values, color='#2E86AB')
        axes[0, 0].set_yticks(range(len(ticker_counts)))
        axes[0, 0].set_yticklabels(ticker_counts.index)
        axes[0, 0].set_xlabel('Number of Trades', fontsize=11)
        axes[0, 0].set_title('Trade Count by Ticker (Top 15)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 2. Total notional by ticker
        external_trades['notional'] = abs(external_trades['quantity'] * external_trades['price'])
        ticker_notional = external_trades.groupby('ticker')['notional'].sum().sort_values(ascending=False).head(15)
        axes[0, 1].barh(range(len(ticker_notional)), ticker_notional.values / 1e6, color='#A23B72')
        axes[0, 1].set_yticks(range(len(ticker_notional)))
        axes[0, 1].set_yticklabels(ticker_notional.index)
        axes[0, 1].set_xlabel('Total Notional ($M)', fontsize=11)
        axes[0, 1].set_title('Total Notional by Ticker (Top 15)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # 3. Daily trade volume
        external_trades['date_only'] = pd.to_datetime(external_trades['date']).dt.date
        daily_notional = external_trades.groupby('date_only')['notional'].sum()
        axes[1, 0].bar(range(len(daily_notional)), daily_notional.values / 1e6,
                      color='#F18F01', alpha=0.8)
        axes[1, 0].set_xlabel('Date', fontsize=11)
        axes[1, 0].set_ylabel('Notional ($M)', fontsize=11)
        axes[1, 0].set_title('Daily Trade Volume', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Format x-axis
        if len(daily_notional) > 20:
            step = len(daily_notional) // 10
            xticks = range(0, len(daily_notional), step)
            axes[1, 0].set_xticks(xticks)
            axes[1, 0].set_xticklabels([str(daily_notional.index[i]) for i in xticks],
                                       rotation=45, ha='right')
        else:
            axes[1, 0].set_xticks(range(len(daily_notional)))
            axes[1, 0].set_xticklabels([str(d) for d in daily_notional.index],
                                       rotation=45, ha='right')

        # 4. Trade costs
        ticker_costs = external_trades.groupby('ticker')['cost'].sum().sort_values(ascending=False).head(15)
        axes[1, 1].barh(range(len(ticker_costs)), ticker_costs.values, color='#C73E1D')
        axes[1, 1].set_yticks(range(len(ticker_costs)))
        axes[1, 1].set_yticklabels(ticker_costs.index)
        axes[1, 1].set_xlabel('Total Cost ($)', fontsize=11)
        axes[1, 1].set_title('Transaction Costs by Ticker (Top 15)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_execution_quality(
        self,
        execution_analysis: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot execution quality analysis.

        Parameters:
        -----------
        execution_analysis : pd.DataFrame
            DataFrame from get_execution_quality_analysis()
        save_path : str, optional
            Path to save figure
        """
        if execution_analysis.empty:
            print("No execution quality data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sort by total notional
        df = execution_analysis.copy()
        df['abs_notional'] = abs(df['total_qty'] * df['vwap'])
        df = df.sort_values('abs_notional', ascending=False).head(20)

        # 1. Slippage by ticker
        colors = ['green' if s > 0 else 'red' for s in df['slippage_pct']]
        axes[0, 0].barh(range(len(df)), df['slippage_pct'].values, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(df)))
        axes[0, 0].set_yticklabels(df['ticker'].values)
        axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[0, 0].set_xlabel('Slippage (%)', fontsize=11)
        axes[0, 0].set_title('Execution Slippage by Ticker (Top 20)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 2. VWAP vs Close Price
        x = np.arange(len(df))
        width = 0.35
        axes[0, 1].bar(x - width/2, df['vwap'].values, width, label='VWAP',
                      color='#2E86AB', alpha=0.8)
        axes[0, 1].bar(x + width/2, df['avg_close'].values, width, label='Avg Close',
                      color='#A23B72', alpha=0.8)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['ticker'].values, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Price ($)', fontsize=11)
        axes[0, 1].set_title('VWAP vs Average Close Price', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Execution PnL by ticker
        colors = ['green' if p > 0 else 'red' for p in df['execution_pnl']]
        axes[1, 0].barh(range(len(df)), df['execution_pnl'].values, color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(len(df)))
        axes[1, 0].set_yticklabels(df['ticker'].values)
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].set_xlabel('Execution PnL ($)', fontsize=11)
        axes[1, 0].set_title('Execution PnL by Ticker', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # 4. Slippage vs Trade Size
        abs_qty = abs(df['total_qty'])
        colors_scatter = ['green' if s > 0 else 'red' for s in df['slippage_pct']]
        axes[1, 1].scatter(abs_qty, df['slippage_pct'], c=colors_scatter,
                          alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_xlabel('Total Quantity (shares)', fontsize=11)
        axes[1, 1].set_ylabel('Slippage (%)', fontsize=11)
        axes[1, 1].set_title('Slippage vs Trade Size', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Add annotations for extreme slippage
        for idx, row in df.iterrows():
            if abs(row['slippage_pct']) > df['slippage_pct'].std() * 2:
                axes[1, 1].annotate(row['ticker'],
                                   (abs(row['total_qty']), row['slippage_pct']),
                                   fontsize=8, alpha=0.7)

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
        output_dir: str,
        factor_exposures: Optional[List[Dict[str, float]]] = None,
        external_trade_pnl: Optional[List[float]] = None,
        executed_trade_pnl: Optional[List[float]] = None,
        overnight_pnl: Optional[List[float]] = None,
        trades_df: Optional[pd.DataFrame] = None,
        close_prices: Optional[pd.DataFrame] = None
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
        factor_exposures : List[Dict[str, float]], optional
            Factor exposures over time
        external_trade_pnl : List[float], optional
            External trade PnL (use case 3)
        executed_trade_pnl : List[float], optional
            Executed trade PnL (use case 3)
        overnight_pnl : List[float], optional
            Overnight PnL (use case 3)
        trades_df : pd.DataFrame, optional
            Trades DataFrame for external trade analysis
        close_prices : pd.DataFrame, optional
            Close prices for execution quality analysis
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

        # Factor exposure charts
        if factor_exposures is not None and any(factor_exposures):
            self.plot_factor_exposures_timeseries(
                dates, factor_exposures,
                save_path=str(output_path / 'factor_exposures_timeseries.png')
            )
            self.plot_factor_exposures_heatmap(
                dates, factor_exposures,
                save_path=str(output_path / 'factor_exposures_heatmap.png')
            )

        # External trade charts (use case 3)
        has_external_pnl = (external_trade_pnl is not None and
                           any(abs(p) > 1e-6 for p in external_trade_pnl))

        if has_external_pnl and executed_trade_pnl is not None and overnight_pnl is not None:
            self.plot_pnl_breakdown(
                dates, external_trade_pnl, executed_trade_pnl, overnight_pnl,
                save_path=str(output_path / 'pnl_breakdown.png')
            )

        if trades_df is not None and not trades_df.empty and 'type' in trades_df.columns:
            external_trades = trades_df[trades_df['type'] == 'external']
            if not external_trades.empty:
                self.plot_external_trades_analysis(
                    trades_df,
                    save_path=str(output_path / 'external_trades_analysis.png')
                )

                # Execution quality analysis (requires close prices)
                if close_prices is not None:
                    from .results import BacktestResults
                    # Create a temporary results object to use analysis method
                    temp_results = type('obj', (object,), {
                        'trades': trades_df.to_dict('records'),
                        'get_trades_dataframe': lambda: trades_df,
                        'get_execution_quality_analysis': lambda cp: self._get_execution_quality_analysis(trades_df, cp)
                    })()

                    execution_analysis = self._get_execution_quality_analysis(trades_df, close_prices)
                    if not execution_analysis.empty:
                        self.plot_execution_quality(
                            execution_analysis,
                            save_path=str(output_path / 'execution_quality.png')
                        )

        print(f"Charts saved to {output_dir}")

    def _get_execution_quality_analysis(self, trades_df: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
        """Helper method for execution quality analysis."""
        if trades_df.empty or 'type' not in trades_df.columns:
            return pd.DataFrame()

        external_trades = trades_df[trades_df['type'] == 'external'].copy()

        if external_trades.empty:
            return pd.DataFrame()

        results = []

        for ticker in external_trades['ticker'].unique():
            ticker_trades = external_trades[external_trades['ticker'] == ticker]

            total_qty = ticker_trades['quantity'].sum()
            vwap = (ticker_trades['quantity'] * ticker_trades['price']).sum() / total_qty

            # Get weighted average close price for dates with trades
            weighted_close = 0.0
            for _, trade in ticker_trades.iterrows():
                date = trade['date']
                qty = trade['quantity']
                if date in close_prices.index and ticker in close_prices.columns:
                    close_px = close_prices.loc[date, ticker]
                    weighted_close += abs(qty) * close_px

            avg_close = weighted_close / abs(total_qty) if total_qty != 0 else 0.0

            # Calculate slippage (positive = favorable execution)
            if total_qty > 0:  # Net buyer
                slippage = avg_close - vwap
            else:  # Net seller
                slippage = vwap - avg_close

            slippage_pct = (slippage / avg_close * 100) if avg_close > 0 else 0.0

            # Execution PnL from this ticker
            ticker_pnl = ticker_trades.apply(
                lambda row: row['quantity'] * (
                    close_prices.loc[row['date'], ticker] - row['price']
                ) if row['date'] in close_prices.index and ticker in close_prices.columns else 0.0,
                axis=1
            ).sum()

            results.append({
                'ticker': ticker,
                'total_qty': total_qty,
                'vwap': vwap,
                'avg_close': avg_close,
                'slippage': slippage,
                'slippage_pct': slippage_pct,
                'execution_pnl': ticker_pnl
            })

        return pd.DataFrame(results)

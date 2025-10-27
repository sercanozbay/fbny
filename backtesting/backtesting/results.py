"""
Results storage and analysis module.

This module contains the BacktestResults class that stores all results
and provides methods for analysis and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from .metrics import PerformanceMetrics
from .visualization import BacktestVisualizer
from .report_generator import ReportGenerator
from .benchmarking import BenchmarkComparison
from .attribution import AttributionTracker


class BacktestResults:
    """
    Store and analyze backtest results.

    This class consolidates all backtest outputs and provides
    methods for generating reports, charts, and analysis.
    """

    def __init__(
        self,
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        daily_pnl: List[float],
        daily_returns: List[float],
        transaction_costs: List[float],
        gross_exposures: List[float],
        net_exposures: List[float],
        trades: List[Dict],
        attribution_tracker: Optional[AttributionTracker] = None,
        risk_free_rate: float = 0.0,
        factor_exposures: Optional[List[Dict[str, float]]] = None,
        external_trade_pnl: Optional[List[float]] = None,
        executed_trade_pnl: Optional[List[float]] = None,
        overnight_pnl: Optional[List[float]] = None,
        external_trades_by_tag: Optional[List[Dict[str, List[Dict]]]] = None,
        external_pnl_by_tag: Optional[Dict[str, List[float]]] = None
    ):
        """
        Initialize backtest results.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values over time
        daily_pnl : List[float]
            Daily PnL
        daily_returns : List[float]
            Daily returns
        transaction_costs : List[float]
            Daily transaction costs
        gross_exposures : List[float]
            Gross exposures
        net_exposures : List[float]
            Net exposures
        trades : List[Dict]
            Trade records
        attribution_tracker : AttributionTracker, optional
            Attribution data
        risk_free_rate : float
            Risk-free rate for calculations
        factor_exposures : List[Dict[str, float]], optional
            Factor exposures over time
        external_trade_pnl : List[float], optional
            PnL from external trades (use case 3)
        executed_trade_pnl : List[float], optional
            PnL from executed/optimized trades
        overnight_pnl : List[float], optional
            PnL from overnight price changes
        external_trades_by_tag : List[Dict[str, List[Dict]]], optional
            External trades grouped by tag for each date
        external_pnl_by_tag : Dict[str, List[float]], optional
            Daily PnL by tag/counterparty
        """
        self.dates = dates
        self.portfolio_values = portfolio_values
        self.daily_pnl = daily_pnl
        self.daily_returns = daily_returns
        self.transaction_costs = transaction_costs
        self.gross_exposures = gross_exposures
        self.net_exposures = net_exposures
        self.trades = trades
        self.attribution_tracker = attribution_tracker
        self.factor_exposures = factor_exposures if factor_exposures is not None else []

        # PnL breakdown
        self.external_trade_pnl = external_trade_pnl if external_trade_pnl is not None else [0.0] * len(dates)
        self.executed_trade_pnl = executed_trade_pnl if executed_trade_pnl is not None else [0.0] * len(dates)
        self.overnight_pnl = overnight_pnl if overnight_pnl is not None else [0.0] * len(dates)

        # Tag-based attribution
        self.external_trades_by_tag = external_trades_by_tag if external_trades_by_tag is not None else []
        self.external_pnl_by_tag = external_pnl_by_tag if external_pnl_by_tag is not None else {}

        # Initialize analysis objects
        self.metrics_calculator = PerformanceMetrics(risk_free_rate)
        self.visualizer = BacktestVisualizer()
        self.report_generator = ReportGenerator()
        self.benchmark_comparison = BenchmarkComparison()

        # Cached metrics
        self._metrics: Optional[Dict[str, float]] = None

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Returns:
        --------
        Dict[str, float]
            All performance metrics
        """
        if self._metrics is None:
            self._metrics = self.metrics_calculator.get_all_metrics(
                self.portfolio_values,
                self.daily_pnl,
                self.dates
            )

        return self._metrics

    def compare_to_benchmark(
        self,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare strategy to benchmark.

        Parameters:
        -----------
        benchmark_returns : np.ndarray
            Benchmark daily returns

        Returns:
        --------
        Dict[str, float]
            Comparison metrics
        """
        strategy_returns = np.array(self.daily_returns)
        return self.benchmark_comparison.get_all_metrics(
            strategy_returns, benchmark_returns
        )

    def generate_charts(self, output_dir: str, close_prices: Optional[pd.DataFrame] = None):
        """
        Generate all charts.

        Parameters:
        -----------
        output_dir : str
            Directory to save charts
        close_prices : pd.DataFrame, optional
            Close prices for execution quality analysis (use case 3)
        """
        factor_pnl_df = None
        if self.attribution_tracker:
            factor_pnl_df = self.attribution_tracker.get_factor_pnl_series()

        trades_df = self.get_trades_dataframe()

        self.visualizer.create_all_charts(
            self.dates,
            self.portfolio_values,
            np.array(self.daily_returns),
            self.gross_exposures,
            self.net_exposures,
            self.transaction_costs,
            factor_pnl_df,
            output_dir,
            factor_exposures=self.factor_exposures,
            external_trade_pnl=self.external_trade_pnl,
            executed_trade_pnl=self.executed_trade_pnl,
            overnight_pnl=self.overnight_pnl,
            trades_df=trades_df,
            close_prices=close_prices
        )

    def generate_html_report(
        self,
        output_path: str,
        chart_dir: Optional[str] = None
    ):
        """
        Generate HTML report.

        Parameters:
        -----------
        output_path : str
            Path to save report
        chart_dir : str, optional
            Directory containing charts
        """
        metrics = self.calculate_metrics()
        self.report_generator.generate_html_report(
            metrics,
            self.dates,
            self.portfolio_values,
            chart_dir,
            output_path
        )

    def generate_pdf_report(self, output_path: str, chart_dir: Optional[str] = None):
        """
        Generate PDF report.

        Parameters:
        -----------
        output_path : str
            Path to save PDF report
        chart_dir : str, optional
            Directory containing charts
        """
        metrics = self.calculate_metrics()
        self.report_generator.generate_pdf_report(
            metrics,
            self.dates,
            self.portfolio_values,
            self.daily_pnl,
            self.daily_returns,
            self.gross_exposures,
            self.net_exposures,
            self.transaction_costs,
            chart_dir,
            output_path
        )

    def generate_excel_report(self, output_path: str):
        """
        Generate Excel report.

        Parameters:
        -----------
        output_path : str
            Path to save Excel file
        """
        metrics = self.calculate_metrics()
        self.report_generator.generate_excel_report(
            metrics,
            self.dates,
            self.portfolio_values,
            self.daily_pnl,
            self.gross_exposures,
            self.net_exposures,
            self.transaction_costs,
            self.trades,
            output_path
        )

    def print_summary(self):
        """Print summary to console."""
        metrics = self.calculate_metrics()
        self.report_generator.print_summary(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Returns:
        --------
        pd.DataFrame
            Results as DataFrame
        """
        df = pd.DataFrame({
            'date': self.dates,
            'portfolio_value': self.portfolio_values,
            'daily_pnl': self.daily_pnl,
            'daily_return': self.daily_returns,
            'transaction_cost': self.transaction_costs,
            'gross_exposure': self.gross_exposures,
            'net_exposure': self.net_exposures
        })

        return df

    def save_to_csv(self, output_path: str):
        """
        Save results to CSV.

        Parameters:
        -----------
        output_path : str
            Path to save CSV
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.

        Returns:
        --------
        pd.DataFrame
            Trades
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_factor_attribution(self) -> Optional[pd.DataFrame]:
        """
        Get factor attribution data.

        Returns:
        --------
        pd.DataFrame, optional
            Factor PnL over time
        """
        if self.attribution_tracker:
            return self.attribution_tracker.get_factor_pnl_series()
        return None

    def get_pnl_breakdown_dataframe(self) -> pd.DataFrame:
        """
        Get PnL breakdown as DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: date, external_pnl, executed_pnl, overnight_pnl, total_pnl
        """
        return pd.DataFrame({
            'date': self.dates,
            'external_pnl': self.external_trade_pnl,
            'executed_pnl': self.executed_trade_pnl,
            'overnight_pnl': self.overnight_pnl,
            'total_pnl': self.daily_pnl
        })

    def get_external_trades_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for external trades.

        Returns:
        --------
        pd.DataFrame
            Summary by ticker with columns: ticker, num_trades, total_qty,
            avg_price, vwap, total_cost
        """
        trades_df = self.get_trades_dataframe()

        if trades_df.empty or 'type' not in trades_df.columns:
            return pd.DataFrame()

        external_trades = trades_df[trades_df['type'] == 'external'].copy()

        if external_trades.empty:
            return pd.DataFrame()

        # Calculate summary by ticker
        summary = external_trades.groupby('ticker').agg({
            'quantity': ['count', 'sum'],
            'price': 'mean',
            'cost': 'sum'
        }).reset_index()

        # Flatten column names
        summary.columns = ['ticker', 'num_trades', 'total_qty', 'avg_price', 'total_cost']

        # Calculate VWAP
        vwap = external_trades.groupby('ticker').apply(
            lambda x: (x['quantity'] * x['price']).sum() / x['quantity'].sum()
        ).reset_index(name='vwap')

        summary = summary.merge(vwap, on='ticker')

        # Reorder columns
        summary = summary[['ticker', 'num_trades', 'total_qty', 'vwap', 'avg_price', 'total_cost']]

        return summary

    def get_execution_quality_analysis(self, close_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze execution quality of external trades vs close prices.

        Parameters:
        -----------
        close_prices : pd.DataFrame
            DataFrame with dates as index and tickers as columns

        Returns:
        --------
        pd.DataFrame
            Analysis by ticker with columns: ticker, total_qty, vwap,
            avg_close, slippage, slippage_pct, execution_pnl
        """
        trades_df = self.get_trades_dataframe()

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
            # For buys: close - vwap (positive means bought below close)
            # For sells: vwap - close (positive means sold above close)
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

    def get_external_trades_by_date(self) -> pd.DataFrame:
        """
        Get external trades grouped by date.

        Returns:
        --------
        pd.DataFrame
            Daily summary with columns: date, num_trades, num_tickers,
            total_notional, total_cost
        """
        trades_df = self.get_trades_dataframe()

        if trades_df.empty or 'type' not in trades_df.columns:
            return pd.DataFrame()

        external_trades = trades_df[trades_df['type'] == 'external'].copy()

        if external_trades.empty:
            return pd.DataFrame()

        external_trades['notional'] = external_trades['quantity'] * external_trades['price']

        daily_summary = external_trades.groupby('date').agg({
            'quantity': 'count',  # num_trades
            'ticker': 'nunique',  # num_tickers
            'notional': lambda x: abs(x).sum(),  # total_notional
            'cost': 'sum'  # total_cost
        }).reset_index()

        daily_summary.columns = ['date', 'num_trades', 'num_tickers', 'total_notional', 'total_cost']

        return daily_summary

    def get_pnl_by_tag(self, tag: Optional[str] = None) -> pd.DataFrame:
        """
        Get PnL attribution by tag/counterparty.

        Parameters:
        -----------
        tag : str, optional
            Specific tag to retrieve. If None, returns all tags.

        Returns:
        --------
        pd.DataFrame
            Daily PnL by tag with columns: date, tag, pnl

        Example:
        --------
        >>> # Get PnL for specific counterparty
        >>> counterparty_pnl = results.get_pnl_by_tag('Goldman Sachs')
        >>>
        >>> # Get PnL for all counterparties
        >>> all_pnl = results.get_pnl_by_tag()
        """
        if not self.external_pnl_by_tag:
            return pd.DataFrame(columns=['date', 'tag', 'pnl'])

        records = []

        if tag is not None:
            # Get specific tag
            if tag in self.external_pnl_by_tag:
                pnl_series = self.external_pnl_by_tag[tag]
                for i, pnl in enumerate(pnl_series):
                    if i < len(self.dates):
                        records.append({
                            'date': self.dates[i],
                            'tag': tag,
                            'pnl': pnl
                        })
        else:
            # Get all tags
            for tag_name, pnl_series in self.external_pnl_by_tag.items():
                for i, pnl in enumerate(pnl_series):
                    if i < len(self.dates):
                        records.append({
                            'date': self.dates[i],
                            'tag': tag_name,
                            'pnl': pnl
                        })

        return pd.DataFrame(records)

    def get_pnl_summary_by_tag(self) -> pd.DataFrame:
        """
        Get summary statistics of PnL by tag/counterparty.

        Returns:
        --------
        pd.DataFrame
            Summary with columns: tag, total_pnl, mean_pnl, std_pnl, sharpe,
            num_days, win_rate

        Example:
        --------
        >>> summary = results.get_pnl_summary_by_tag()
        >>> print(summary)
                      tag  total_pnl  mean_pnl  std_pnl  sharpe  num_days  win_rate
        0  Goldman Sachs    125000.0   5000.0   8000.0    0.62        25      0.60
        1    Morgan Stanley  98000.0   4900.0   7200.0    0.68        20      0.65
        """
        if not self.external_pnl_by_tag:
            return pd.DataFrame(columns=[
                'tag', 'total_pnl', 'mean_pnl', 'std_pnl',
                'sharpe', 'num_days', 'win_rate'
            ])

        summaries = []

        for tag, pnl_series in self.external_pnl_by_tag.items():
            pnl_array = np.array(pnl_series)

            # Filter out zero days
            non_zero_pnl = pnl_array[pnl_array != 0]

            if len(non_zero_pnl) == 0:
                continue

            total_pnl = np.sum(pnl_array)
            mean_pnl = np.mean(non_zero_pnl)
            std_pnl = np.std(non_zero_pnl)
            sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
            num_days = len(non_zero_pnl)
            win_rate = np.sum(non_zero_pnl > 0) / len(non_zero_pnl) if len(non_zero_pnl) > 0 else 0.0

            summaries.append({
                'tag': tag,
                'total_pnl': total_pnl,
                'mean_pnl': mean_pnl,
                'std_pnl': std_pnl,
                'sharpe': sharpe,
                'num_days': num_days,
                'win_rate': win_rate
            })

        df = pd.DataFrame(summaries)

        # Sort by total PnL descending
        if not df.empty:
            df = df.sort_values('total_pnl', ascending=False).reset_index(drop=True)

        return df

    def get_trades_by_tag(self, tag: Optional[str] = None) -> pd.DataFrame:
        """
        Get external trades filtered by tag.

        Parameters:
        -----------
        tag : str, optional
            Specific tag to retrieve. If None, returns all external trades with tags.

        Returns:
        --------
        pd.DataFrame
            Trades with tag information

        Example:
        --------
        >>> gs_trades = results.get_trades_by_tag('Goldman Sachs')
        """
        trades_df = self.get_trades_dataframe()

        if trades_df.empty:
            return pd.DataFrame()

        # Filter to external trades only
        if 'type' in trades_df.columns:
            external_trades = trades_df[trades_df['type'] == 'external'].copy()
        else:
            external_trades = trades_df.copy()

        if external_trades.empty:
            return pd.DataFrame()

        # Filter by tag if specified
        if tag is not None:
            if 'tag' in external_trades.columns:
                external_trades = external_trades[external_trades['tag'] == tag]

        return external_trades

    def generate_full_report(
        self,
        output_dir: str,
        formats: List[str] = ['html', 'excel']
    ):
        """
        Generate complete report with all formats.

        Parameters:
        -----------
        output_dir : str
            Output directory
        formats : List[str]
            Formats to generate ('html', 'excel', 'csv', 'pdf')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating reports in {output_dir}...")

        # Generate charts
        charts_dir = output_path / 'charts'
        self.generate_charts(str(charts_dir))

        # Generate reports
        if 'html' in formats:
            self.generate_html_report(
                str(output_path / 'backtest_report.html'),
                str(charts_dir)
            )

        if 'excel' in formats:
            self.generate_excel_report(
                str(output_path / 'backtest_report.xlsx')
            )

        if 'csv' in formats:
            self.save_to_csv(str(output_path / 'backtest_results.csv'))

            # Save trades
            trades_df = self.get_trades_dataframe()
            if not trades_df.empty:
                trades_df.to_csv(
                    str(output_path / 'trades.csv'),
                    index=False
                )

            # Save factor attribution
            factor_pnl = self.get_factor_attribution()
            if factor_pnl is not None and not factor_pnl.empty:
                factor_pnl.to_csv(str(output_path / 'factor_attribution.csv'))

        if 'pdf' in formats:
            self.generate_pdf_report(
                str(output_path / 'backtest_report.pdf'),
                str(charts_dir)
            )

        print(f"Reports generated successfully in {output_dir}\n")

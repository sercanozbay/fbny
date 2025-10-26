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
        factor_exposures: Optional[List[Dict[str, float]]] = None
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

    def generate_charts(self, output_dir: str):
        """
        Generate all charts.

        Parameters:
        -----------
        output_dir : str
            Directory to save charts
        """
        factor_pnl_df = None
        if self.attribution_tracker:
            factor_pnl_df = self.attribution_tracker.get_factor_pnl_series()

        self.visualizer.create_all_charts(
            self.dates,
            self.portfolio_values,
            np.array(self.daily_returns),
            self.gross_exposures,
            self.net_exposures,
            self.transaction_costs,
            factor_pnl_df,
            output_dir,
            factor_exposures=self.factor_exposures
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

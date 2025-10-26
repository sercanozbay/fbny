"""
Benchmarking module for comparing strategy to benchmarks.

This module calculates alpha, beta, tracking error, and other
comparison metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy import stats


class BenchmarkComparison:
    """
    Compare strategy performance to a benchmark.

    Calculates alpha, beta, tracking error, information ratio, etc.
    """

    def __init__(self):
        """Initialize benchmark comparison."""
        pass

    def calculate_alpha_beta(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate alpha and beta via linear regression.

        strategy_return = alpha + beta * benchmark_return + epsilon

        Parameters:
        -----------
        strategy_returns : np.ndarray
            Strategy daily returns
        benchmark_returns : np.ndarray
            Benchmark daily returns

        Returns:
        --------
        Tuple[float, float, float]
            (alpha, beta, r_squared)
        """
        if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0, 0.0

        # Ensure same length
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strat_ret = strategy_returns[:min_len]
        bench_ret = benchmark_returns[:min_len]

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            bench_ret, strat_ret
        )

        # Annualize alpha
        alpha_daily = intercept
        alpha_annual = (1 + alpha_daily) ** 252 - 1

        beta = slope
        r_squared = r_value ** 2

        return float(alpha_annual), float(beta), float(r_squared)

    def calculate_tracking_error(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Calculate tracking error.

        TE = std(strategy_return - benchmark_return)

        Parameters:
        -----------
        strategy_returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray
            Benchmark returns
        annualize : bool
            Annualize the tracking error

        Returns:
        --------
        float
            Tracking error
        """
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        min_len = min(len(strategy_returns), len(benchmark_returns))
        excess_returns = strategy_returns[:min_len] - benchmark_returns[:min_len]

        te = np.std(excess_returns, ddof=1)

        if annualize:
            te *= np.sqrt(252)

        return float(te)

    def calculate_information_ratio(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate information ratio.

        IR = mean(excess_return) / std(excess_return) * sqrt(252)

        Parameters:
        -----------
        strategy_returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray
            Benchmark returns

        Returns:
        --------
        float
            Information ratio
        """
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        min_len = min(len(strategy_returns), len(benchmark_returns))
        excess_returns = strategy_returns[:min_len] - benchmark_returns[:min_len]

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            return 0.0

        ir = (mean_excess / std_excess) * np.sqrt(252)

        return float(ir)

    def calculate_up_down_capture(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate up and down capture ratios.

        Up capture = mean(strategy return when benchmark > 0) / mean(benchmark return when benchmark > 0)
        Down capture = mean(strategy return when benchmark < 0) / mean(benchmark return when benchmark < 0)

        Parameters:
        -----------
        strategy_returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray
            Benchmark returns

        Returns:
        --------
        Tuple[float, float]
            (up_capture, down_capture)
        """
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0, 0.0

        min_len = min(len(strategy_returns), len(benchmark_returns))
        strat_ret = strategy_returns[:min_len]
        bench_ret = benchmark_returns[:min_len]

        # Up markets
        up_mask = bench_ret > 0
        if np.sum(up_mask) > 0:
            up_capture = np.mean(strat_ret[up_mask]) / np.mean(bench_ret[up_mask])
        else:
            up_capture = 0.0

        # Down markets
        down_mask = bench_ret < 0
        if np.sum(down_mask) > 0:
            down_capture = np.mean(strat_ret[down_mask]) / np.mean(bench_ret[down_mask])
        else:
            down_capture = 0.0

        return float(up_capture), float(down_capture)

    def get_all_metrics(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all benchmark comparison metrics.

        Parameters:
        -----------
        strategy_returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray
            Benchmark returns

        Returns:
        --------
        Dict[str, float]
            All comparison metrics
        """
        alpha, beta, r_squared = self.calculate_alpha_beta(
            strategy_returns, benchmark_returns
        )
        te = self.calculate_tracking_error(strategy_returns, benchmark_returns)
        ir = self.calculate_information_ratio(strategy_returns, benchmark_returns)
        up_capture, down_capture = self.calculate_up_down_capture(
            strategy_returns, benchmark_returns
        )

        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': te,
            'information_ratio': ir,
            'up_capture': up_capture,
            'down_capture': down_capture
        }

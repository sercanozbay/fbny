"""
Performance metrics calculation module.

This module calculates standard performance metrics including
Sharpe ratio, drawdowns, and other risk-adjusted return measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.

    Handles returns, risk, drawdowns, and trading metrics.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance metrics calculator.

        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252.0

    def calculate_cumulative_returns(
        self,
        portfolio_values: List[float]
    ) -> np.ndarray:
        """
        Calculate cumulative returns series.

        Parameters:
        -----------
        portfolio_values : List[float]
            Time series of portfolio values

        Returns:
        --------
        np.ndarray
            Cumulative returns
        """
        if len(portfolio_values) < 2:
            return np.array([0.0])

        values = np.array(portfolio_values)
        initial_value = values[0]

        if initial_value == 0:
            return np.zeros(len(values))

        return (values / initial_value) - 1.0

    def calculate_total_return(self, portfolio_values: List[float]) -> float:
        """Calculate total return over period."""
        if len(portfolio_values) < 2 or portfolio_values[0] == 0:
            return 0.0

        return (portfolio_values[-1] / portfolio_values[0]) - 1.0

    def calculate_annualized_return(
        self,
        portfolio_values: List[float],
        n_days: int
    ) -> float:
        """
        Calculate annualized return.

        Parameters:
        -----------
        portfolio_values : List[float]
            Portfolio values
        n_days : int
            Number of trading days

        Returns:
        --------
        float
            Annualized return
        """
        total_return = self.calculate_total_return(portfolio_values)
        if n_days == 0:
            return 0.0

        years = n_days / 252.0
        if years == 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1.0

    def calculate_daily_returns(
        self,
        portfolio_values: List[float]
    ) -> np.ndarray:
        """Calculate daily returns."""
        if len(portfolio_values) < 2:
            return np.array([])

        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]

        return returns

    def calculate_volatility(
        self,
        daily_returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Calculate volatility.

        Parameters:
        -----------
        daily_returns : np.ndarray
            Daily returns
        annualize : bool
            If True, annualize the volatility

        Returns:
        --------
        float
            Volatility (annualized if requested)
        """
        if len(daily_returns) == 0:
            return 0.0

        vol = np.std(daily_returns, ddof=1)

        if annualize:
            vol *= np.sqrt(252)

        return float(vol)

    def calculate_sharpe_ratio(
        self,
        daily_returns: np.ndarray
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (mean_return - rf) / std_return * sqrt(252)

        Parameters:
        -----------
        daily_returns : np.ndarray
            Daily returns

        Returns:
        --------
        float
            Annualized Sharpe ratio
        """
        if len(daily_returns) == 0:
            return 0.0

        excess_returns = daily_returns - self.daily_rf
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(252)

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        daily_returns: np.ndarray
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).

        Parameters:
        -----------
        daily_returns : np.ndarray
            Daily returns

        Returns:
        --------
        float
            Annualized Sortino ratio
        """
        if len(daily_returns) == 0:
            return 0.0

        excess_returns = daily_returns - self.daily_rf
        mean_excess = np.mean(excess_returns)

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_excess > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(252)

        return float(sortino)

    def calculate_drawdown_series(
        self,
        portfolio_values: List[float]
    ) -> np.ndarray:
        """
        Calculate running drawdown series.

        Drawdown at time t = (portfolio_value[t] - running_max) / running_max

        Parameters:
        -----------
        portfolio_values : List[float]
            Portfolio values

        Returns:
        --------
        np.ndarray
            Drawdown at each point (negative values)
        """
        if len(portfolio_values) == 0:
            return np.array([])

        values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(values)

        drawdowns = (values - running_max) / running_max

        return drawdowns

    def calculate_max_drawdown(
        self,
        portfolio_values: List[float]
    ) -> float:
        """
        Calculate maximum drawdown.

        Returns:
        --------
        float
            Maximum drawdown (positive number, e.g., 0.15 for 15% drawdown)
        """
        drawdowns = self.calculate_drawdown_series(portfolio_values)

        if len(drawdowns) == 0:
            return 0.0

        return float(abs(np.min(drawdowns)))

    def calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio.

        Calmar = annualized_return / max_drawdown

        Parameters:
        -----------
        annualized_return : float
            Annualized return
        max_drawdown : float
            Maximum drawdown

        Returns:
        --------
        float
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        return annualized_return / max_drawdown

    def calculate_var(
        self,
        daily_returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Parameters:
        -----------
        daily_returns : np.ndarray
            Daily returns
        confidence : float
            Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
        --------
        float
            VaR (positive number representing loss)
        """
        if len(daily_returns) == 0:
            return 0.0

        var = np.percentile(daily_returns, (1 - confidence) * 100)

        return float(abs(var))

    def calculate_cvar(
        self,
        daily_returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        CVaR = expected loss given that we're in the worst (1-confidence)% of cases

        Parameters:
        -----------
        daily_returns : np.ndarray
            Daily returns
        confidence : float
            Confidence level

        Returns:
        --------
        float
            CVaR (positive number)
        """
        if len(daily_returns) == 0:
            return 0.0

        var = -self.calculate_var(daily_returns, confidence)
        cvar = np.mean(daily_returns[daily_returns <= var])

        return float(abs(cvar))

    def calculate_skewness(self, daily_returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(daily_returns) < 3:
            return 0.0

        return float(stats.skew(daily_returns))

    def calculate_kurtosis(self, daily_returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns."""
        if len(daily_returns) < 4:
            return 0.0

        return float(stats.kurtosis(daily_returns))

    def calculate_win_rate(self, daily_returns: np.ndarray) -> float:
        """Calculate percentage of positive return days."""
        if len(daily_returns) == 0:
            return 0.0

        win_rate = np.sum(daily_returns > 0) / len(daily_returns)

        return float(win_rate)

    def calculate_profit_factor(self, daily_pnl: np.ndarray) -> float:
        """
        Calculate profit factor.

        Profit factor = gross_profit / gross_loss

        Parameters:
        -----------
        daily_pnl : np.ndarray
            Daily PnL values

        Returns:
        --------
        float
            Profit factor
        """
        if len(daily_pnl) == 0:
            return 0.0

        gross_profit = np.sum(daily_pnl[daily_pnl > 0])
        gross_loss = abs(np.sum(daily_pnl[daily_pnl < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    def get_all_metrics(
        self,
        portfolio_values: List[float],
        daily_pnl: List[float],
        dates: List[pd.Timestamp]
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Parameters:
        -----------
        portfolio_values : List[float]
            Portfolio values over time
        daily_pnl : List[float]
            Daily PnL
        dates : List[pd.Timestamp]
            Dates

        Returns:
        --------
        Dict[str, float]
            Dictionary of all metrics
        """
        if len(portfolio_values) < 2:
            return {}

        daily_returns = self.calculate_daily_returns(portfolio_values)
        n_days = len(dates) - 1

        total_return = self.calculate_total_return(portfolio_values)
        ann_return = self.calculate_annualized_return(portfolio_values, n_days)
        volatility = self.calculate_volatility(daily_returns)
        sharpe = self.calculate_sharpe_ratio(daily_returns)
        sortino = self.calculate_sortino_ratio(daily_returns)
        max_dd = self.calculate_max_drawdown(portfolio_values)
        calmar = self.calculate_calmar_ratio(ann_return, max_dd)
        var_95 = self.calculate_var(daily_returns, 0.95)
        cvar_95 = self.calculate_cvar(daily_returns, 0.95)
        skew = self.calculate_skewness(daily_returns)
        kurt = self.calculate_kurtosis(daily_returns)
        win_rate = self.calculate_win_rate(daily_returns)
        profit_factor = self.calculate_profit_factor(np.array(daily_pnl))

        metrics = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skew,
            'kurtosis': kurt,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_days': n_days
        }

        return metrics

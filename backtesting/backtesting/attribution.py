"""
Performance attribution module.

This module calculates PnL attribution to factors and specific returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .risk_calculator import FactorRiskModel


class PerformanceAttributor:
    """
    Attribute portfolio returns to factors and specific components.

    Attribution methodology:
    - Factor PnL = factor_exposure * factor_return * portfolio_value
    - Specific PnL = position_weight * specific_return * portfolio_value
    """

    def __init__(self, risk_model: FactorRiskModel):
        """
        Initialize performance attributor.

        Parameters:
        -----------
        risk_model : FactorRiskModel
            Risk model for calculating exposures
        """
        self.risk_model = risk_model

    def calculate_factor_attribution(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate PnL attribution to factors.

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions at start of period
        prices : Dict[str, float]
            Prices at start of period
        factor_loadings : pd.DataFrame
            Factor exposures (index=ticker, columns=factors)
        factor_returns : Dict[str, float]
            Factor returns for the period

        Returns:
        --------
        Dict[str, float]
            PnL per factor
        """
        # Calculate portfolio factor exposures
        factor_exp = self.risk_model.calculate_factor_exposures(
            positions, prices, factor_loadings
        )

        # Calculate portfolio value
        portfolio_value = sum(
            abs(shares * prices.get(ticker, 0.0))
            for ticker, shares in positions.items()
        )

        # Attribution: factor_pnl = exposure * factor_return * portfolio_value
        factor_pnl = {}
        for i, factor_name in enumerate(factor_loadings.columns):
            factor_ret = factor_returns.get(factor_name, 0.0)
            factor_pnl[factor_name] = factor_exp[i] * factor_ret * portfolio_value

        return factor_pnl

    def calculate_specific_attribution(
        self,
        positions: Dict[str, float],
        start_prices: Dict[str, float],
        end_prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate specific (idiosyncratic) return attribution.

        Specific return = total return - factor returns

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions
        start_prices : Dict[str, float]
            Prices at start of period
        end_prices : Dict[str, float]
            Prices at end of period
        factor_loadings : pd.DataFrame
            Factor exposures
        factor_returns : Dict[str, float]
            Factor returns

        Returns:
        --------
        Dict[str, float]
            Specific PnL per security
        """
        specific_pnl = {}

        for ticker, shares in positions.items():
            start_px = start_prices.get(ticker, 0.0)
            end_px = end_prices.get(ticker, 0.0)

            if start_px == 0:
                continue

            # Total return
            total_return = (end_px - start_px) / start_px

            # Factor return contribution
            if ticker in factor_loadings.index:
                factor_contribution = 0.0
                for i, factor_name in enumerate(factor_loadings.columns):
                    factor_loading = factor_loadings.loc[ticker, factor_name]
                    factor_ret = factor_returns.get(factor_name, 0.0)
                    factor_contribution += factor_loading * factor_ret

                # Specific return = total - factor
                specific_return = total_return - factor_contribution
            else:
                # If no factor loadings, treat all as specific
                specific_return = total_return

            # Specific PnL
            position_value = shares * start_px
            specific_pnl[ticker] = specific_return * position_value

        return specific_pnl

    def calculate_total_attribution(
        self,
        positions: Dict[str, float],
        start_prices: Dict[str, float],
        end_prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_returns: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
        """
        Calculate complete attribution breakdown.

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions
        start_prices : Dict[str, float]
            Start prices
        end_prices : Dict[str, float]
            End prices
        factor_loadings : pd.DataFrame
            Factor exposures
        factor_returns : Dict[str, float]
            Factor returns

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float], float, float]
            (factor_pnl, specific_pnl, total_factor_pnl, total_specific_pnl)
        """
        factor_pnl = self.calculate_factor_attribution(
            positions, start_prices, factor_loadings, factor_returns
        )

        specific_pnl = self.calculate_specific_attribution(
            positions, start_prices, end_prices, factor_loadings, factor_returns
        )

        total_factor = sum(factor_pnl.values())
        total_specific = sum(specific_pnl.values())

        return factor_pnl, specific_pnl, total_factor, total_specific


class AttributionTracker:
    """
    Track attribution over time.

    Maintains time series of factor and specific PnL.
    """

    def __init__(self):
        """Initialize attribution tracker."""
        self.dates = []
        self.factor_pnl_history = []
        self.specific_pnl_history = []

    def add_period(
        self,
        date: pd.Timestamp,
        factor_pnl: Dict[str, float],
        specific_pnl: Dict[str, float]
    ):
        """
        Add attribution for a period.

        Parameters:
        -----------
        date : pd.Timestamp
            Period date
        factor_pnl : Dict[str, float]
            Factor PnL
        specific_pnl : Dict[str, float]
            Specific PnL
        """
        self.dates.append(date)
        self.factor_pnl_history.append(factor_pnl)
        self.specific_pnl_history.append(specific_pnl)

    def get_factor_pnl_series(self) -> pd.DataFrame:
        """
        Get factor PnL as time series.

        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index, factors as columns
        """
        if not self.dates:
            return pd.DataFrame()

        df = pd.DataFrame(self.factor_pnl_history, index=self.dates)
        return df.fillna(0.0)

    def get_specific_pnl_series(self) -> pd.DataFrame:
        """
        Get specific PnL as time series.

        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index, tickers as columns
        """
        if not self.dates:
            return pd.DataFrame()

        df = pd.DataFrame(self.specific_pnl_history, index=self.dates)
        return df.fillna(0.0)

    def get_cumulative_factor_pnl(self) -> pd.DataFrame:
        """Get cumulative factor PnL over time."""
        factor_series = self.get_factor_pnl_series()
        if factor_series.empty:
            return pd.DataFrame()

        return factor_series.cumsum()

    def get_factor_sharpe_ratios(self) -> Dict[str, float]:
        """
        Calculate Sharpe ratio for each factor's PnL contribution.

        Returns:
        --------
        Dict[str, float]
            Sharpe ratio per factor
        """
        factor_series = self.get_factor_pnl_series()
        if factor_series.empty:
            return {}

        sharpe_ratios = {}
        for factor in factor_series.columns:
            pnl = factor_series[factor]
            if pnl.std() > 0:
                sharpe_ratios[factor] = pnl.mean() / pnl.std() * np.sqrt(252)
            else:
                sharpe_ratios[factor] = 0.0

        return sharpe_ratios

    def get_factor_contributions(self) -> Dict[str, float]:
        """
        Calculate each factor's contribution to total PnL.

        Returns:
        --------
        Dict[str, float]
            Percentage contribution per factor
        """
        factor_series = self.get_factor_pnl_series()
        if factor_series.empty:
            return {}

        total_pnl = factor_series.sum().sum()
        if total_pnl == 0:
            return {factor: 0.0 for factor in factor_series.columns}

        contributions = {}
        for factor in factor_series.columns:
            factor_total = factor_series[factor].sum()
            contributions[factor] = (factor_total / total_pnl) * 100

        return contributions

"""
Risk calculation module using factor models.

This module handles portfolio risk calculations based on factor exposures,
factor covariance, and specific variance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class FactorRiskModel:
    """
    Factor-based risk model for portfolio risk calculation.

    Risk decomposition: Var(portfolio) = B' * F * B + sum(h_i^2 * sigma_i^2)
    where:
        B = factor exposures (factor loadings)
        F = factor covariance matrix
        h_i = holding in security i
        sigma_i^2 = specific variance of security i
    """

    def __init__(self, use_float32: bool = True):
        """
        Initialize risk model.

        Parameters:
        -----------
        use_float32 : bool
            Use float32 for calculations (memory efficient)
        """
        self.dtype = np.float32 if use_float32 else np.float64

    def calculate_factor_exposures(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        factor_loadings: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate portfolio factor exposures.

        Parameters:
        -----------
        positions : Dict[str, float]
            Ticker -> shares
        prices : Dict[str, float]
            Ticker -> price
        factor_loadings : pd.DataFrame
            Security factor exposures (index=ticker, columns=factors)

        Returns:
        --------
        np.ndarray
            Portfolio factor exposures (one value per factor)
        """
        # Calculate portfolio weights
        position_values = {
            ticker: shares * prices.get(ticker, 0.0)
            for ticker, shares in positions.items()
        }
        total_value = sum(abs(v) for v in position_values.values())

        if total_value == 0:
            return np.zeros(len(factor_loadings.columns), dtype=self.dtype)

        weights = {
            ticker: value / total_value
            for ticker, value in position_values.items()
        }

        # Calculate weighted factor exposures
        factor_exp = np.zeros(len(factor_loadings.columns), dtype=self.dtype)

        for ticker, weight in weights.items():
            if ticker in factor_loadings.index:
                factor_exp += weight * factor_loadings.loc[ticker].values.astype(self.dtype)

        return factor_exp

    def calculate_portfolio_variance(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_cov: np.ndarray,
        specific_var: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio variance decomposed into factor and specific risk.

        Parameters:
        -----------
        positions : Dict[str, float]
            Ticker -> shares
        prices : Dict[str, float]
            Ticker -> price
        factor_loadings : pd.DataFrame
            Security factor exposures
        factor_cov : np.ndarray
            Factor covariance matrix
        specific_var : Dict[str, float]
            Specific variance per security

        Returns:
        --------
        Tuple[float, float, float]
            (total_variance, factor_variance, specific_variance)
        """
        # Get portfolio factor exposures
        factor_exp = self.calculate_factor_exposures(positions, prices, factor_loadings)

        # Factor variance: B' * F * B
        factor_variance = float(factor_exp @ factor_cov @ factor_exp.T)

        # Specific variance: sum(h_i^2 * sigma_i^2)
        total_value = sum(abs(shares * prices.get(ticker, 0.0))
                         for ticker, shares in positions.items())

        if total_value == 0:
            return 0.0, 0.0, 0.0

        specific_variance = 0.0
        for ticker, shares in positions.items():
            if ticker in specific_var:
                value = shares * prices.get(ticker, 0.0)
                weight = value / total_value
                specific_variance += (weight ** 2) * specific_var[ticker]

        total_variance = factor_variance + specific_variance

        return float(total_variance), float(factor_variance), float(specific_variance)

    def calculate_marginal_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_cov: np.ndarray,
        specific_var: Dict[str, float],
        ticker: str
    ) -> float:
        """
        Calculate marginal contribution to risk for a security.

        This is useful for optimization to understand how trading
        a security affects portfolio risk.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions
        prices : Dict[str, float]
            Security prices
        factor_loadings : pd.DataFrame
            Factor exposures
        factor_cov : np.ndarray
            Factor covariance
        specific_var : Dict[str, float]
            Specific variances
        ticker : str
            Security to calculate marginal risk for

        Returns:
        --------
        float
            Marginal contribution to portfolio variance
        """
        if ticker not in factor_loadings.index:
            return 0.0

        # Portfolio factor exposures
        port_factor_exp = self.calculate_factor_exposures(positions, prices, factor_loadings)

        # Security factor exposures
        sec_factor_exp = factor_loadings.loc[ticker].values.astype(self.dtype)

        # Marginal factor risk: 2 * B_portfolio' * F * B_security
        marginal_factor = 2.0 * float(port_factor_exp @ factor_cov @ sec_factor_exp)

        # Marginal specific risk
        total_value = sum(abs(shares * prices.get(t, 0.0))
                         for t, shares in positions.items())

        if total_value == 0:
            return marginal_factor

        weight = positions.get(ticker, 0.0) * prices.get(ticker, 0.0) / total_value
        marginal_specific = 2.0 * weight * specific_var.get(ticker, 0.0)

        return marginal_factor + marginal_specific


class RiskConstraintChecker:
    """Check portfolio against risk constraints."""

    def __init__(self, risk_model: FactorRiskModel):
        """
        Initialize constraint checker.

        Parameters:
        -----------
        risk_model : FactorRiskModel
            Risk model for calculations
        """
        self.risk_model = risk_model

    def check_factor_exposures(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        max_exposures: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Check factor exposure constraints.

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions
        prices : Dict[str, float]
            Security prices
        factor_loadings : pd.DataFrame
            Factor exposures
        max_exposures : Dict[str, float]
            Maximum allowed absolute exposure per factor

        Returns:
        --------
        Dict[str, float]
            Violations per factor (0 if within limits)
        """
        factor_exp = self.risk_model.calculate_factor_exposures(
            positions, prices, factor_loadings
        )

        violations = {}
        for i, factor_name in enumerate(factor_loadings.columns):
            if factor_name in max_exposures:
                abs_exp = abs(factor_exp[i])
                max_exp = max_exposures[factor_name]
                if abs_exp > max_exp:
                    violations[factor_name] = abs_exp - max_exp
                else:
                    violations[factor_name] = 0.0

        return violations

    def check_sector_exposures(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        sector_mapping: Dict[str, str],
        max_exposures: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Check sector exposure constraints.

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions
        prices : Dict[str, float]
            Security prices
        sector_mapping : Dict[str, str]
            Ticker -> sector
        max_exposures : Dict[str, float]
            Maximum allowed absolute exposure per sector

        Returns:
        --------
        Dict[str, float]
            Violations per sector (0 if within limits)
        """
        # Calculate sector exposures
        sector_exposures = {}
        total_value = sum(abs(shares * prices.get(ticker, 0.0))
                         for ticker, shares in positions.items())

        if total_value == 0:
            return {sector: 0.0 for sector in max_exposures}

        for ticker, shares in positions.items():
            sector = sector_mapping.get(ticker)
            if sector:
                value = shares * prices.get(ticker, 0.0)
                weight = value / total_value
                sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight

        # Check violations
        violations = {}
        for sector, max_exp in max_exposures.items():
            abs_exp = abs(sector_exposures.get(sector, 0.0))
            if abs_exp > max_exp:
                violations[sector] = abs_exp - max_exp
            else:
                violations[sector] = 0.0

        return violations

    def check_portfolio_variance(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_cov: np.ndarray,
        specific_var: Dict[str, float],
        max_variance: float
    ) -> float:
        """
        Check portfolio variance constraint.

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions
        prices : Dict[str, float]
            Security prices
        factor_loadings : pd.DataFrame
            Factor exposures
        factor_cov : np.ndarray
            Factor covariance
        specific_var : Dict[str, float]
            Specific variances
        max_variance : float
            Maximum allowed portfolio variance

        Returns:
        --------
        float
            Violation amount (0 if within limit)
        """
        total_var, _, _ = self.risk_model.calculate_portfolio_variance(
            positions, prices, factor_loadings, factor_cov, specific_var
        )

        if total_var > max_variance:
            return total_var - max_variance
        return 0.0

    def get_sector_exposures(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        sector_mapping: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate current sector exposures.

        Returns:
        --------
        Dict[str, float]
            Sector -> weight
        """
        sector_exposures = {}
        total_value = sum(abs(shares * prices.get(ticker, 0.0))
                         for ticker, shares in positions.items())

        if total_value == 0:
            return {}

        for ticker, shares in positions.items():
            sector = sector_mapping.get(ticker)
            if sector:
                value = shares * prices.get(ticker, 0.0)
                weight = value / total_value
                sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight

        return sector_exposures

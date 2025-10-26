"""
Portfolio optimization module.

This module implements optimization for use case 3: minimizing transaction
costs while satisfying risk constraints.
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from typing import Dict, Optional, Tuple
import pandas as pd

from .transaction_costs import TransactionCostModel
from .risk_calculator import FactorRiskModel


class PortfolioOptimizer:
    """
    Optimize portfolio to minimize transaction costs subject to constraints.

    Used primarily for use case 3: external trades with risk management.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        risk_model: FactorRiskModel,
        method: str = 'SLSQP',
        max_iter: int = 1000,
        tolerance: float = 1e-6
    ):
        """
        Initialize optimizer.

        Parameters:
        -----------
        cost_model : TransactionCostModel
            Transaction cost model
        risk_model : FactorRiskModel
            Risk model
        method : str
            Optimization method for scipy.optimize.minimize
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
        """
        self.cost_model = cost_model
        self.risk_model = risk_model
        self.method = method
        self.max_iter = max_iter
        self.tolerance = tolerance

    def optimize_trades(
        self,
        current_positions: Dict[str, float],
        prices: Dict[str, float],
        adv: Dict[str, float],
        factor_loadings: pd.DataFrame,
        factor_cov: np.ndarray,
        specific_var: Dict[str, float],
        max_variance: Optional[float] = None,
        max_factor_exposure: Optional[Dict[str, float]] = None,
        max_adv_participation: float = 0.05,
        target_positions: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Optimize trades to minimize cost subject to constraints.

        Parameters:
        -----------
        current_positions : Dict[str, float]
            Current holdings
        prices : Dict[str, float]
            Security prices
        adv : Dict[str, float]
            Average daily volume
        factor_loadings : pd.DataFrame
            Factor exposures
        factor_cov : np.ndarray
            Factor covariance
        specific_var : Dict[str, float]
            Specific variances
        max_variance : float, optional
            Maximum portfolio variance constraint
        max_factor_exposure : Dict[str, float], optional
            Maximum factor exposures
        max_adv_participation : float
            Maximum ADV participation
        target_positions : Dict[str, float], optional
            Target positions (if provided, optimize towards them)

        Returns:
        --------
        Tuple[Dict[str, float], Dict]
            (optimal_trades, info_dict)
        """
        # Get universe of tickers
        tickers = sorted(set(current_positions.keys()) | set(prices.keys()))
        n = len(tickers)
        ticker_idx = {ticker: i for i, ticker in enumerate(tickers)}

        # Initial guess: no trades
        x0 = np.zeros(n)

        # Objective function: minimize transaction costs
        def objective(trades_vec):
            trades_dict = {
                ticker: trades_vec[ticker_idx[ticker]]
                for ticker in tickers
            }
            _, total_cost = self.cost_model.calculate_costs_vectorized(
                trades_dict, adv, prices
            )
            return total_cost

        # Objective gradient
        def objective_grad(trades_vec):
            grad = np.zeros(n)
            for ticker in tickers:
                idx = ticker_idx[ticker]
                trade_qty = trades_vec[idx]
                adv_val = adv.get(ticker, 1e-6)
                price = prices.get(ticker, 0.0)
                grad[idx] = self.cost_model.calculate_gradient(
                    trade_qty, adv_val, price
                )
            return grad

        # Constraints
        constraints = []

        # ADV constraints
        for ticker in tickers:
            idx = ticker_idx[ticker]
            max_qty = max_adv_participation * adv.get(ticker, 0.0)

            # -max_qty <= trade <= max_qty
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=idx, m=max_qty: m - abs(x[i])
            })

        # Variance constraint
        if max_variance is not None:
            def variance_constraint(trades_vec):
                # Calculate new positions
                new_positions = current_positions.copy()
                for ticker in tickers:
                    idx = ticker_idx[ticker]
                    new_positions[ticker] = (
                        new_positions.get(ticker, 0.0) + trades_vec[idx]
                    )

                total_var, _, _ = self.risk_model.calculate_portfolio_variance(
                    new_positions, prices, factor_loadings, factor_cov, specific_var
                )

                return max_variance - total_var

            constraints.append({
                'type': 'ineq',
                'fun': variance_constraint
            })

        # Factor exposure constraints
        if max_factor_exposure:
            for factor_name, max_exp in max_factor_exposure.items():
                if factor_name not in factor_loadings.columns:
                    continue

                def factor_constraint(trades_vec, fname=factor_name, mexp=max_exp):
                    # Calculate new positions
                    new_positions = current_positions.copy()
                    for ticker in tickers:
                        idx = ticker_idx[ticker]
                        new_positions[ticker] = (
                            new_positions.get(ticker, 0.0) + trades_vec[idx]
                        )

                    factor_exp = self.risk_model.calculate_factor_exposures(
                        new_positions, prices, factor_loadings
                    )

                    factor_idx = list(factor_loadings.columns).index(fname)
                    return mexp - abs(factor_exp[factor_idx])

                constraints.append({
                    'type': 'ineq',
                    'fun': factor_constraint
                })

        # Bounds: allow both long and short
        bounds = [(-1e8, 1e8) for _ in range(n)]

        # Run optimization
        result = minimize(
            objective,
            x0,
            method=self.method,
            jac=objective_grad,
            constraints=constraints,
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tolerance
            }
        )

        # Extract optimal trades
        optimal_trades = {
            ticker: result.x[ticker_idx[ticker]]
            for ticker in tickers
            if abs(result.x[ticker_idx[ticker]]) > 1e-6
        }

        info = {
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit if hasattr(result, 'nit') else None,
            'final_cost': result.fun
        }

        return optimal_trades, info


class SimpleTradeOptimizer:
    """
    Simple trade optimizer for use cases 1 and 2.

    Applies ADV constraints and hedging without full optimization.
    """

    def __init__(self, max_adv_participation: float = 0.05):
        """
        Initialize simple optimizer.

        Parameters:
        -----------
        max_adv_participation : float
            Maximum ADV participation
        """
        self.max_adv_participation = max_adv_participation

    def calculate_constrained_trades(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
        adv: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate trades with ADV constraints.

        Parameters:
        -----------
        target_positions : Dict[str, float]
            Target holdings
        current_positions : Dict[str, float]
            Current holdings
        adv : Dict[str, float]
            Average daily volume

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float]]
            (constrained_trades, unfilled_trades)
        """
        from .utils import calculate_trades

        # Calculate desired trades
        desired_trades = calculate_trades(current_positions, target_positions)

        # Apply ADV constraints
        constrained_trades = {}
        unfilled_trades = {}

        for ticker, trade_qty in desired_trades.items():
            adv_val = adv.get(ticker, 0.0)

            if adv_val == 0:
                # Can't trade without ADV
                constrained_trades[ticker] = 0.0
                unfilled_trades[ticker] = trade_qty
                continue

            max_qty = self.max_adv_participation * adv_val
            abs_trade = abs(trade_qty)

            if abs_trade <= max_qty:
                constrained_trades[ticker] = trade_qty
                unfilled_trades[ticker] = 0.0
            else:
                sign = np.sign(trade_qty)
                constrained_trades[ticker] = sign * max_qty
                unfilled_trades[ticker] = trade_qty - constrained_trades[ticker]

        return constrained_trades, unfilled_trades

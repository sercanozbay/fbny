"""
Transaction cost modeling.

This module implements transaction cost calculations based on
trade size relative to average daily volume (ADV).
"""

import numpy as np
from typing import Dict, Tuple


class TransactionCostModel:
    """
    Power-law transaction cost model.

    Cost function: TC = coefficient * (|qty| / adv) ^ power + fixed_cost * |qty| * price

    This captures market impact that grows non-linearly with trade size.
    """

    def __init__(
        self,
        power: float = 1.5,
        coefficient: float = 0.01,
        fixed_cost: float = 0.0001
    ):
        """
        Initialize transaction cost model.

        Parameters:
        -----------
        power : float
            Power in the cost function (typically 1.0 to 2.0)
        coefficient : float
            Coefficient multiplying the power term
        fixed_cost : float
            Fixed cost per dollar traded (in bps, e.g., 0.0001 = 1bp)
        """
        self.power = power
        self.coefficient = coefficient
        self.fixed_cost = fixed_cost

    def calculate_cost(
        self,
        trade_qty: float,
        adv: float,
        price: float
    ) -> float:
        """
        Calculate transaction cost for a single trade.

        Parameters:
        -----------
        trade_qty : float
            Number of shares to trade (signed)
        adv : float
            Average daily volume
        price : float
            Price per share

        Returns:
        --------
        float
            Transaction cost in dollars
        """
        if trade_qty == 0 or adv == 0 or price == 0:
            return 0.0

        abs_qty = abs(trade_qty)
        notional = abs_qty * price

        # Market impact component
        participation_rate = abs_qty / adv
        market_impact = self.coefficient * (participation_rate ** self.power) * notional

        # Fixed cost component
        fixed_component = self.fixed_cost * notional

        return market_impact + fixed_component

    def calculate_costs_vectorized(
        self,
        trades: Dict[str, float],
        adv_dict: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculate transaction costs for multiple trades efficiently.

        Parameters:
        -----------
        trades : Dict[str, float]
            Ticker -> trade quantity (shares)
        adv_dict : Dict[str, float]
            Ticker -> average daily volume
        prices : Dict[str, float]
            Ticker -> price

        Returns:
        --------
        Tuple[Dict[str, float], float]
            (per-security costs, total cost)
        """
        costs = {}
        total_cost = 0.0

        for ticker, qty in trades.items():
            if qty == 0:
                costs[ticker] = 0.0
                continue

            adv = adv_dict.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)

            cost = self.calculate_cost(qty, adv, price)
            costs[ticker] = cost
            total_cost += cost

        return costs, total_cost

    def calculate_gradient(
        self,
        trade_qty: float,
        adv: float,
        price: float
    ) -> float:
        """
        Calculate gradient of cost function with respect to trade quantity.

        This is useful for optimization.

        dCost/dQty = sign(qty) * coefficient * power * (|qty|/adv)^(power-1) * price / adv
                     + sign(qty) * fixed_cost * price

        Parameters:
        -----------
        trade_qty : float
            Trade quantity
        adv : float
            Average daily volume
        price : float
            Price per share

        Returns:
        --------
        float
            Gradient of cost with respect to quantity
        """
        if trade_qty == 0 or adv == 0 or price == 0:
            return 0.0

        sign = np.sign(trade_qty)
        abs_qty = abs(trade_qty)

        # Gradient of market impact
        if abs_qty > 0:
            impact_grad = (
                sign * self.coefficient * self.power *
                ((abs_qty / adv) ** (self.power - 1)) *
                price / adv
            )
        else:
            impact_grad = 0.0

        # Gradient of fixed cost
        fixed_grad = sign * self.fixed_cost * price

        return impact_grad + fixed_grad


class ADVConstraintCalculator:
    """
    Calculate and enforce ADV constraints on trades.

    Ensures that trades do not exceed a specified percentage of ADV.
    """

    def __init__(self, max_participation: float = 0.05):
        """
        Initialize ADV constraint calculator.

        Parameters:
        -----------
        max_participation : float
            Maximum trade size as fraction of ADV (e.g., 0.05 = 5%)
        """
        self.max_participation = max_participation

    def apply_constraints(
        self,
        target_trades: Dict[str, float],
        adv_dict: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Apply ADV constraints to target trades.

        Parameters:
        -----------
        target_trades : Dict[str, float]
            Desired trades (ticker -> shares)
        adv_dict : Dict[str, float]
            Average daily volume per ticker

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float]]
            (constrained trades, unfilled trades)
        """
        constrained = {}
        unfilled = {}

        for ticker, target_qty in target_trades.items():
            adv = adv_dict.get(ticker, 0.0)

            if adv == 0:
                # Can't trade if no ADV data
                constrained[ticker] = 0.0
                unfilled[ticker] = target_qty
                continue

            max_qty = self.max_participation * adv
            abs_target = abs(target_qty)

            if abs_target <= max_qty:
                # Within constraint
                constrained[ticker] = target_qty
                unfilled[ticker] = 0.0
            else:
                # Constrained
                sign = np.sign(target_qty)
                constrained[ticker] = sign * max_qty
                unfilled[ticker] = target_qty - constrained[ticker]

        return constrained, unfilled

    def get_max_trade_sizes(
        self,
        adv_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get maximum allowable trade sizes.

        Parameters:
        -----------
        adv_dict : Dict[str, float]
            Average daily volume per ticker

        Returns:
        --------
        Dict[str, float]
            Maximum trade size per ticker
        """
        return {
            ticker: self.max_participation * adv
            for ticker, adv in adv_dict.items()
        }

    def calculate_participation_rates(
        self,
        trades: Dict[str, float],
        adv_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate actual participation rates for trades.

        Parameters:
        -----------
        trades : Dict[str, float]
            Actual trades
        adv_dict : Dict[str, float]
            Average daily volume

        Returns:
        --------
        Dict[str, float]
            Participation rate per ticker
        """
        rates = {}

        for ticker, qty in trades.items():
            adv = adv_dict.get(ticker, 0.0)
            if adv > 0:
                rates[ticker] = abs(qty) / adv
            else:
                rates[ticker] = 0.0

        return rates

    def check_violations(
        self,
        trades: Dict[str, float],
        adv_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Check for ADV constraint violations.

        Parameters:
        -----------
        trades : Dict[str, float]
            Trades to check
        adv_dict : Dict[str, float]
            Average daily volume

        Returns:
        --------
        Dict[str, float]
            Excess participation per ticker (0 if no violation)
        """
        violations = {}

        for ticker, qty in trades.items():
            adv = adv_dict.get(ticker, 0.0)
            if adv > 0:
                participation = abs(qty) / adv
                if participation > self.max_participation:
                    violations[ticker] = participation - self.max_participation
                else:
                    violations[ticker] = 0.0
            else:
                violations[ticker] = 0.0

        return violations

"""
Trade execution module.

This module handles applying trades to the portfolio and
calculating execution prices and costs.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple

from .config import Portfolio
from .transaction_costs import TransactionCostModel
from .utils import create_trade_record


class TradeExecutor:
    """
    Execute trades and update portfolio state.

    Handles price determination (close vs trade prices),
    transaction cost calculation, and cash management.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        use_trade_prices: bool = False
    ):
        """
        Initialize trade executor.

        Parameters:
        -----------
        cost_model : TransactionCostModel
            Model for calculating transaction costs
        use_trade_prices : bool
            If True, use separate trade prices; otherwise use close prices
        """
        self.cost_model = cost_model
        self.use_trade_prices = use_trade_prices

    def execute_trades(
        self,
        portfolio: Portfolio,
        trades: Dict[str, float],
        close_prices: Dict[str, float],
        trade_prices: Optional[Dict[str, float]],
        adv: Dict[str, float],
        date: pd.Timestamp
    ) -> Tuple[Portfolio, float, List[Dict]]:
        """
        Execute trades and create new portfolio.

        Parameters:
        -----------
        portfolio : Portfolio
            Current portfolio
        trades : Dict[str, float]
            Trades to execute (ticker -> shares)
        close_prices : Dict[str, float]
            Close prices
        trade_prices : Dict[str, float], optional
            Execution prices (if different from close)
        adv : Dict[str, float]
            Average daily volume
        date : pd.Timestamp
            Trade date

        Returns:
        --------
        Tuple[Portfolio, float, List[Dict]]
            (new_portfolio, total_cost, trade_records)
        """
        # Determine execution prices
        if self.use_trade_prices and trade_prices is not None:
            exec_prices = trade_prices
        else:
            exec_prices = close_prices

        # Calculate transaction costs
        trade_costs, total_cost = self.cost_model.calculate_costs_vectorized(
            trades, adv, exec_prices
        )

        # Execute trades
        new_positions = portfolio.positions.copy()
        new_cash = portfolio.cash

        trade_records = []

        for ticker, qty in trades.items():
            if qty == 0:
                continue

            exec_price = exec_prices.get(ticker, 0.0)
            cost = trade_costs.get(ticker, 0.0)

            # Update positions
            new_positions[ticker] = new_positions.get(ticker, 0.0) + qty

            # Update cash (negative for buys, positive for sells, minus costs)
            trade_value = qty * exec_price
            new_cash -= trade_value
            new_cash -= cost

            # Record trade
            trade_records.append(create_trade_record(
                date, ticker, qty, exec_price, cost
            ))

        # Clean up zero positions
        new_positions = {
            ticker: shares
            for ticker, shares in new_positions.items()
            if abs(shares) > 1e-6
        }

        # Create new portfolio with end-of-day prices
        new_portfolio = Portfolio(
            date=date,
            positions=new_positions,
            cash=new_cash,
            prices=close_prices
        )

        return new_portfolio, total_cost, trade_records

    def calculate_execution_shortfall(
        self,
        trades: Dict[str, float],
        close_prices: Dict[str, float],
        execution_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate execution shortfall (slippage).

        Shortfall = (execution_price - close_price) * quantity * sign(quantity)

        Parameters:
        -----------
        trades : Dict[str, float]
            Executed trades
        close_prices : Dict[str, float]
            Close prices (benchmark)
        execution_prices : Dict[str, float]
            Actual execution prices

        Returns:
        --------
        Dict[str, float]
            Shortfall per ticker (negative means worse than benchmark)
        """
        shortfall = {}

        for ticker, qty in trades.items():
            if qty == 0:
                continue

            close_px = close_prices.get(ticker, 0.0)
            exec_px = execution_prices.get(ticker, 0.0)

            # For buys: positive shortfall means paid less than close (good)
            # For sells: positive shortfall means received more than close (good)
            if qty > 0:  # Buy
                shortfall[ticker] = (close_px - exec_px) * qty
            else:  # Sell
                shortfall[ticker] = (exec_px - close_px) * abs(qty)

        return shortfall

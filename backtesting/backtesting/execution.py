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

    def execute_trades_with_breakdown(
        self,
        portfolio: Portfolio,
        external_trades: Dict[str, float],
        internal_trades: Dict[str, float],
        close_prices: Dict[str, float],
        external_exec_prices: Optional[Dict[str, float]],
        internal_exec_prices: Optional[Dict[str, float]],
        adv: Dict[str, float],
        date: pd.Timestamp,
        prev_close_prices: Optional[Dict[str, float]] = None
    ) -> Tuple[Portfolio, float, List[Dict], Dict[str, float]]:
        """
        Execute trades with PnL breakdown.

        Separates external trades (use case 3 input) from internal trades
        (optimization/rebalancing) and calculates PnL components.

        Parameters:
        -----------
        portfolio : Portfolio
            Current portfolio
        external_trades : Dict[str, float] or Dict[str, List[Dict]]
            External trades to execute
            Format 1: ticker -> shares (simple)
            Format 2: ticker -> [{'qty': shares, 'price': price}, ...] (multiple trades)
        internal_trades : Dict[str, float]
            Internal/optimized trades (ticker -> shares)
        close_prices : Dict[str, float]
            End-of-day close prices
        external_exec_prices : Dict[str, float], optional
            Execution prices for external trades (ignored if Format 2 used)
        internal_exec_prices : Dict[str, float], optional
            Execution prices for internal trades
        adv : Dict[str, float]
            Average daily volume
        date : pd.Timestamp
            Trade date
        prev_close_prices : Dict[str, float], optional
            Previous close prices for overnight PnL calculation

        Returns:
        --------
        Tuple[Portfolio, float, List[Dict], Dict[str, float]]
            (new_portfolio, total_cost, trade_records, pnl_breakdown)
            pnl_breakdown: {'external': float, 'executed': float, 'overnight': float}
        """
        # Determine execution prices
        if external_exec_prices is None:
            external_exec_prices = close_prices
        if internal_exec_prices is None:
            internal_exec_prices = close_prices if not self.use_trade_prices else close_prices

        # Calculate overnight PnL (price changes on existing positions)
        overnight_pnl = 0.0
        if prev_close_prices is not None:
            for ticker, shares in portfolio.positions.items():
                prev_px = prev_close_prices.get(ticker, 0.0)
                curr_px = close_prices.get(ticker, 0.0)
                overnight_pnl += shares * (curr_px - prev_px)

        # Execute external trades first
        external_positions = portfolio.positions.copy()
        external_cash = portfolio.cash
        external_pnl = 0.0
        external_tc = 0.0
        trade_records = []

        for ticker, trade_list in external_trades.items():
            close_price = close_prices.get(ticker, 0.0)
            ticker_adv = adv.get(ticker, 1e9)

            # Process list of trades for this ticker
            if not isinstance(trade_list, list):
                raise ValueError(
                    f"External trades must be a list of dicts with 'qty' and 'price'. "
                    f"Got {type(trade_list)} for {ticker}. "
                    f"Format: {{'ticker': [{{'qty': 100, 'price': 150.25}}, ...]}}"
                )

            for trade in trade_list:
                qty = trade.get('qty', 0)
                exec_price = trade.get('price', close_price)

                if qty == 0:
                    continue

                # Calculate transaction cost
                cost = self.cost_model.calculate_cost(qty, ticker_adv, exec_price)
                external_tc += cost

                # Update positions
                old_shares = external_positions.get(ticker, 0.0)
                external_positions[ticker] = old_shares + qty

                # Cash impact
                trade_value = qty * exec_price
                external_cash -= trade_value
                external_cash -= cost

                # PnL from external trade (difference between exec and close)
                # Positive = bought below close or sold above close (good execution)
                external_pnl += qty * (close_price - exec_price)

                # Record trade
                trade_records.append({
                    'date': date,
                    'ticker': ticker,
                    'quantity': qty,
                    'price': exec_price,
                    'cost': cost,
                    'type': 'external'
                })

        # Execute internal trades
        new_positions = external_positions.copy()
        new_cash = external_cash
        executed_pnl = 0.0
        internal_tc = 0.0

        for ticker, qty in internal_trades.items():
            if qty == 0:
                continue

            exec_price = internal_exec_prices.get(ticker, 0.0)
            close_price = close_prices.get(ticker, 0.0)
            ticker_adv = adv.get(ticker, 1e9)

            # Calculate transaction cost
            cost = self.cost_model.calculate_cost(qty, ticker_adv, exec_price)
            internal_tc += cost

            # Update positions
            old_shares = new_positions.get(ticker, 0.0)
            new_positions[ticker] = old_shares + qty

            # Cash impact
            trade_value = qty * exec_price
            new_cash -= trade_value
            new_cash -= cost

            # PnL from executed trade
            executed_pnl += qty * (close_price - exec_price)

            # Record trade
            trade_records.append({
                'date': date,
                'ticker': ticker,
                'quantity': qty,
                'price': exec_price,
                'cost': cost,
                'type': 'internal'
            })

        # Clean up zero positions
        new_positions = {
            ticker: shares
            for ticker, shares in new_positions.items()
            if abs(shares) > 1e-6
        }

        # Create new portfolio
        new_portfolio = Portfolio(
            date=date,
            positions=new_positions,
            cash=new_cash,
            prices=close_prices
        )

        total_cost = external_tc + internal_tc

        pnl_breakdown = {
            'external': external_pnl,
            'executed': executed_pnl,
            'overnight': overnight_pnl
        }

        return new_portfolio, total_cost, trade_records, pnl_breakdown

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

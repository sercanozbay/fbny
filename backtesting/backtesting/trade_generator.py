"""
Trade generator module for creating external trades from signals.

This module provides utilities to generate external trade lists from various
signal types (target positions, target weights, deltas) while accounting for
current portfolio state.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class TradeGeneratorConfig:
    """Configuration for trade generation."""

    # Price determination
    price_impact_bps: float = 5.0  # Slippage in basis points
    use_random_fills: bool = False  # Generate random fill prices (for simulation)
    num_fills_per_ticker: int = 1  # Number of fills to simulate per ticker

    # Trade size constraints
    min_trade_size: float = 1.0  # Minimum shares to trade
    round_lots: bool = True  # Round to whole shares

    # ADV constraints
    max_adv_participation: Optional[float] = None  # Max fraction of ADV (e.g., 0.1 = 10%)


class ExternalTradeGenerator:
    """
    Generate external trades from signals and portfolio state.

    This class provides methods to convert various signal types into
    the external trade format required by Use Case 3.
    """

    def __init__(self, config: Optional[TradeGeneratorConfig] = None):
        """
        Initialize trade generator.

        Parameters:
        -----------
        config : TradeGeneratorConfig, optional
            Configuration for trade generation
        """
        self.config = config or TradeGeneratorConfig()

    def from_target_positions(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
        close_prices: Dict[str, float],
        adv: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate trades from target positions.

        Parameters:
        -----------
        target_positions : Dict[str, float]
            Desired positions (shares) by ticker
        current_positions : Dict[str, float]
            Current positions (shares) by ticker
        close_prices : Dict[str, float]
            Close prices by ticker
        adv : Dict[str, float], optional
            Average daily volume by ticker (for ADV constraints)

        Returns:
        --------
        Dict[str, List[Dict]]
            External trades in format: {ticker: [{'qty': shares, 'price': price}, ...]}
        """
        trades = {}

        # Get all tickers (union of target and current)
        all_tickers = set(target_positions.keys()) | set(current_positions.keys())

        for ticker in all_tickers:
            target = target_positions.get(ticker, 0.0)
            current = current_positions.get(ticker, 0.0)

            # Calculate required trade
            trade_qty = target - current

            if abs(trade_qty) < self.config.min_trade_size:
                continue

            # Round if needed
            if self.config.round_lots:
                trade_qty = round(trade_qty)
                if trade_qty == 0:
                    continue

            # Apply ADV constraints if provided
            if adv is not None and self.config.max_adv_participation is not None:
                ticker_adv = adv.get(ticker, float('inf'))
                max_qty = abs(ticker_adv * self.config.max_adv_participation)

                if abs(trade_qty) > max_qty:
                    trade_qty = np.sign(trade_qty) * max_qty
                    if self.config.round_lots:
                        trade_qty = round(trade_qty)

            if abs(trade_qty) < self.config.min_trade_size:
                continue

            # Generate execution prices
            close_price = close_prices.get(ticker, 0.0)
            if close_price == 0:
                continue

            trade_list = self._generate_fills(trade_qty, close_price, ticker)

            if trade_list:
                trades[ticker] = trade_list

        return trades

    def from_target_weights(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, float],
        close_prices: Dict[str, float],
        portfolio_value: float,
        adv: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate trades from target weights.

        Parameters:
        -----------
        target_weights : Dict[str, float]
            Desired weights (fraction of portfolio) by ticker
        current_positions : Dict[str, float]
            Current positions (shares) by ticker
        close_prices : Dict[str, float]
            Close prices by ticker
        portfolio_value : float
            Current portfolio value
        adv : Dict[str, float], optional
            Average daily volume by ticker

        Returns:
        --------
        Dict[str, List[Dict]]
            External trades
        """
        # Convert weights to target positions
        target_positions = {}
        for ticker, weight in target_weights.items():
            price = close_prices.get(ticker, 0.0)
            if price > 0:
                target_notional = portfolio_value * weight
                target_positions[ticker] = target_notional / price

        return self.from_target_positions(
            target_positions, current_positions, close_prices, adv
        )

    def from_signal_deltas(
        self,
        signal_deltas: Dict[str, float],
        close_prices: Dict[str, float],
        adv: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate trades directly from signal deltas (trade quantities).

        Parameters:
        -----------
        signal_deltas : Dict[str, float]
            Desired trade quantities by ticker
        close_prices : Dict[str, float]
            Close prices by ticker
        adv : Dict[str, float], optional
            Average daily volume by ticker

        Returns:
        --------
        Dict[str, List[Dict]]
            External trades
        """
        # Treat deltas as target positions with zero current
        return self.from_target_positions(
            signal_deltas,
            {},  # Empty current positions
            close_prices,
            adv
        )

    def from_signal_scores(
        self,
        signal_scores: Dict[str, float],
        current_positions: Dict[str, float],
        close_prices: Dict[str, float],
        portfolio_value: float,
        target_notional: float,
        adv: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate trades from signal scores (e.g., alpha, z-scores).

        Converts scores to target weights using rank-based allocation.

        Parameters:
        -----------
        signal_scores : Dict[str, float]
            Signal scores by ticker (higher = more bullish)
        current_positions : Dict[str, float]
            Current positions (shares) by ticker
        close_prices : Dict[str, float]
            Close prices by ticker
        portfolio_value : float
            Current portfolio value
        target_notional : float
            Total notional to allocate (can be > portfolio_value for leverage)
        adv : Dict[str, float], optional
            Average daily volume by ticker

        Returns:
        --------
        Dict[str, List[Dict]]
            External trades
        """
        if not signal_scores:
            return {}

        # Convert scores to weights (equal-weighted with sign from score)
        total_abs_score = sum(abs(score) for score in signal_scores.values())

        if total_abs_score == 0:
            return {}

        target_weights = {}
        for ticker, score in signal_scores.items():
            # Weight proportional to absolute score, direction from sign
            weight = (score / total_abs_score) * (target_notional / portfolio_value)
            target_weights[ticker] = weight

        return self.from_target_weights(
            target_weights, current_positions, close_prices, portfolio_value, adv
        )

    def _generate_fills(
        self,
        trade_qty: float,
        close_price: float,
        ticker: str
    ) -> List[Dict]:
        """
        Generate fill prices for a trade.

        Parameters:
        -----------
        trade_qty : float
            Total quantity to trade
        close_price : float
            Close price
        ticker : str
            Ticker symbol

        Returns:
        --------
        List[Dict]
            List of fills with qty and price
        """
        fills = []

        if self.config.num_fills_per_ticker == 1:
            # Single fill
            price = self._calculate_execution_price(close_price, trade_qty)
            fills.append({'qty': trade_qty, 'price': price})
        else:
            # Multiple fills - split quantity
            num_fills = min(self.config.num_fills_per_ticker, int(abs(trade_qty)))

            if num_fills <= 1:
                price = self._calculate_execution_price(close_price, trade_qty)
                fills.append({'qty': trade_qty, 'price': price})
            else:
                # Split into chunks
                qty_per_fill = trade_qty / num_fills

                for i in range(num_fills):
                    # Vary price slightly across fills if using random
                    if self.config.use_random_fills:
                        # Random walk around close price
                        price_offset = np.random.randn() * (close_price * self.config.price_impact_bps / 10000)
                    else:
                        # Linear interpolation from better to worse execution
                        impact_factor = (i / (num_fills - 1)) if num_fills > 1 else 0.5
                        price_offset = impact_factor * np.sign(trade_qty) * close_price * self.config.price_impact_bps / 10000

                    fill_price = close_price + price_offset

                    # Handle last fill (to avoid rounding issues)
                    if i == num_fills - 1:
                        fill_qty = trade_qty - sum(f['qty'] for f in fills)
                    else:
                        fill_qty = round(qty_per_fill) if self.config.round_lots else qty_per_fill

                    if abs(fill_qty) >= self.config.min_trade_size:
                        fills.append({'qty': fill_qty, 'price': fill_price})

        return fills

    def _calculate_execution_price(self, close_price: float, trade_qty: float) -> float:
        """
        Calculate execution price with slippage.

        Parameters:
        -----------
        close_price : float
            Close price
        trade_qty : float
            Trade quantity (positive = buy, negative = sell)

        Returns:
        --------
        float
            Execution price
        """
        if self.config.use_random_fills:
            # Random slippage
            slippage = np.random.randn() * (close_price * self.config.price_impact_bps / 10000)
        else:
            # Deterministic slippage (unfavorable for buys, favorable for sells)
            slippage = np.sign(trade_qty) * (close_price * self.config.price_impact_bps / 10000)

        return close_price + slippage

    def generate_multi_day_trades(
        self,
        dates: List[pd.Timestamp],
        target_positions_by_date: Dict[pd.Timestamp, Dict[str, float]],
        prices_df: pd.DataFrame,
        initial_positions: Optional[Dict[str, float]] = None,
        adv_df: Optional[pd.DataFrame] = None
    ) -> Dict[pd.Timestamp, Dict[str, List[Dict]]]:
        """
        Generate trades for multiple dates.

        Parameters:
        -----------
        dates : List[pd.Timestamp]
            Trading dates
        target_positions_by_date : Dict[pd.Timestamp, Dict[str, float]]
            Target positions for each date
        prices_df : pd.DataFrame
            Prices DataFrame (dates as index, tickers as columns)
        initial_positions : Dict[str, float], optional
            Starting positions
        adv_df : pd.DataFrame, optional
            ADV DataFrame (dates as index, tickers as columns)

        Returns:
        --------
        Dict[pd.Timestamp, Dict[str, List[Dict]]]
            External trades by date
        """
        all_trades = {}
        current_positions = initial_positions.copy() if initial_positions else {}

        for date in dates:
            if date not in target_positions_by_date:
                continue

            target_positions = target_positions_by_date[date]

            # Get prices for this date
            if date not in prices_df.index:
                continue

            close_prices = prices_df.loc[date].to_dict()

            # Get ADV if available
            adv = None
            if adv_df is not None and date in adv_df.index:
                adv = adv_df.loc[date].to_dict()

            # Generate trades for this date
            daily_trades = self.from_target_positions(
                target_positions,
                current_positions,
                close_prices,
                adv
            )

            if daily_trades:
                all_trades[date] = daily_trades

            # Update current positions based on generated trades
            for ticker, trade_list in daily_trades.items():
                total_qty = sum(trade['qty'] for trade in trade_list)
                current_positions[ticker] = current_positions.get(ticker, 0.0) + total_qty

        return all_trades


def generate_external_trades_from_signals(
    signals: Union[Dict[str, float], pd.Series],
    current_positions: Dict[str, float],
    close_prices: Union[Dict[str, float], pd.Series],
    portfolio_value: float,
    signal_type: str = 'weights',
    target_notional: Optional[float] = None,
    adv: Optional[Union[Dict[str, float], pd.Series]] = None,
    price_impact_bps: float = 5.0,
    num_fills: int = 1
) -> Dict[str, List[Dict]]:
    """
    Convenience function to generate external trades from signals.

    Parameters:
    -----------
    signals : Dict or Series
        Signal values by ticker
    current_positions : Dict[str, float]
        Current positions (shares)
    close_prices : Dict or Series
        Close prices
    portfolio_value : float
        Current portfolio value
    signal_type : str
        Type of signal: 'weights', 'positions', 'deltas', or 'scores'
    target_notional : float, optional
        Target notional for 'scores' type (defaults to portfolio_value)
    adv : Dict or Series, optional
        Average daily volume
    price_impact_bps : float
        Price impact in basis points (default: 5)
    num_fills : int
        Number of fills to generate per ticker (default: 1)

    Returns:
    --------
    Dict[str, List[Dict]]
        External trades

    Examples:
    ---------
    >>> # From target weights
    >>> signals = {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': -0.1}
    >>> trades = generate_external_trades_from_signals(
    ...     signals, current_positions, prices, portfolio_value,
    ...     signal_type='weights'
    ... )

    >>> # From target positions
    >>> signals = {'AAPL': 1000, 'MSFT': 500, 'GOOGL': -200}
    >>> trades = generate_external_trades_from_signals(
    ...     signals, current_positions, prices, portfolio_value,
    ...     signal_type='positions'
    ... )

    >>> # From trade deltas
    >>> signals = {'AAPL': 100, 'MSFT': -50}  # Buy 100 AAPL, sell 50 MSFT
    >>> trades = generate_external_trades_from_signals(
    ...     signals, current_positions, prices, portfolio_value,
    ...     signal_type='deltas'
    ... )
    """
    # Convert Series to dict if needed
    if isinstance(signals, pd.Series):
        signals = signals.to_dict()
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_dict()
    if isinstance(adv, pd.Series):
        adv = adv.to_dict()

    # Create config
    config = TradeGeneratorConfig(
        price_impact_bps=price_impact_bps,
        num_fills_per_ticker=num_fills,
        max_adv_participation=0.1 if adv is not None else None
    )

    generator = ExternalTradeGenerator(config)

    # Generate based on signal type
    if signal_type == 'weights':
        return generator.from_target_weights(
            signals, current_positions, close_prices, portfolio_value, adv
        )
    elif signal_type == 'positions':
        return generator.from_target_positions(
            signals, current_positions, close_prices, adv
        )
    elif signal_type == 'deltas':
        return generator.from_signal_deltas(signals, close_prices, adv)
    elif signal_type == 'scores':
        target_not = target_notional or portfolio_value
        return generator.from_signal_scores(
            signals, current_positions, close_prices, portfolio_value, target_not, adv
        )
    else:
        raise ValueError(
            f"Unknown signal_type: {signal_type}. "
            f"Must be one of: 'weights', 'positions', 'deltas', 'scores'"
        )

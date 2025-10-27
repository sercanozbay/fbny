"""
Signal generators for dynamic external trade generation.

This module provides base classes and utilities for generating trades
dynamically during backtesting based on current portfolio state.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod

from .trade_generator import ExternalTradeGenerator, TradeGeneratorConfig


class SignalGenerator(ABC):
    """
    Base class for signal generators.

    Signal generators produce trading signals dynamically during backtesting
    based on the current portfolio state and market data.
    """

    def __init__(self, trade_generator_config: Optional[TradeGeneratorConfig] = None):
        """
        Initialize signal generator.

        Parameters:
        -----------
        trade_generator_config : TradeGeneratorConfig, optional
            Configuration for trade generation
        """
        self.trade_generator = ExternalTradeGenerator(trade_generator_config)
        self.history = []  # Store history of signals/trades

    @abstractmethod
    def generate_signals(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate trading signals based on context.

        Parameters:
        -----------
        context : Dict
            Backtest context containing:
            - date: Current date
            - portfolio: Current portfolio state
            - prices: Current prices
            - adv: Average daily volume
            - portfolio_value: Current portfolio value
            - dates: Historical dates
            - daily_returns: Historical returns
            - daily_pnl: Historical PnL

        Returns:
        --------
        Dict[str, float]
            Trading signals (format depends on signal_type)
        """
        pass

    def __call__(self, context: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Generate external trades for the current date.

        This method is called by the backtester during simulation.

        Parameters:
        -----------
        context : Dict
            Backtest context

        Returns:
        --------
        Dict[str, List[Dict]]
            External trades in format: {ticker: [{'qty': shares, 'price': price}, ...]}
        """
        # Generate signals
        signals = self.generate_signals(context)

        if not signals:
            return {}

        # Convert signals to trades
        trades = self.signals_to_trades(signals, context)

        # Store history
        self.history.append({
            'date': context['date'],
            'signals': signals,
            'trades': trades
        })

        return trades

    @abstractmethod
    def signals_to_trades(
        self,
        signals: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """
        Convert signals to external trades.

        Parameters:
        -----------
        signals : Dict[str, float]
            Trading signals
        context : Dict
            Backtest context

        Returns:
        --------
        Dict[str, List[Dict]]
            External trades
        """
        pass

    def get_history(self) -> pd.DataFrame:
        """
        Get history of signals and trades.

        Returns:
        --------
        pd.DataFrame
            Historical signals and trades
        """
        if not self.history:
            return pd.DataFrame()

        records = []
        for entry in self.history:
            for ticker, signal in entry['signals'].items():
                records.append({
                    'date': entry['date'],
                    'ticker': ticker,
                    'signal': signal,
                    'has_trade': ticker in entry['trades']
                })

        return pd.DataFrame(records)


class TargetWeightSignalGenerator(SignalGenerator):
    """
    Generate trades from target weight signals.

    This is the most common use case - generate target portfolio weights
    and convert them to trades.
    """

    def __init__(
        self,
        signal_function: Callable[[Dict[str, Any]], Dict[str, float]],
        trade_generator_config: Optional[TradeGeneratorConfig] = None
    ):
        """
        Initialize target weight signal generator.

        Parameters:
        -----------
        signal_function : Callable
            Function that takes context and returns target weights
            Signature: f(context) -> Dict[ticker, weight]
        trade_generator_config : TradeGeneratorConfig, optional
            Configuration for trade generation
        """
        super().__init__(trade_generator_config)
        self.signal_function = signal_function

    def generate_signals(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate target weight signals."""
        return self.signal_function(context)

    def signals_to_trades(
        self,
        signals: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """Convert target weights to trades."""
        return self.trade_generator.from_target_weights(
            target_weights=signals,
            current_positions=context['portfolio'].positions,
            close_prices=context['prices'],
            portfolio_value=context['portfolio_value'],
            adv=context.get('adv')
        )


class TargetPositionSignalGenerator(SignalGenerator):
    """Generate trades from target position signals (share counts)."""

    def __init__(
        self,
        signal_function: Callable[[Dict[str, Any]], Dict[str, float]],
        trade_generator_config: Optional[TradeGeneratorConfig] = None
    ):
        """
        Initialize target position signal generator.

        Parameters:
        -----------
        signal_function : Callable
            Function that takes context and returns target positions
            Signature: f(context) -> Dict[ticker, shares]
        """
        super().__init__(trade_generator_config)
        self.signal_function = signal_function

    def generate_signals(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate target position signals."""
        return self.signal_function(context)

    def signals_to_trades(
        self,
        signals: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """Convert target positions to trades."""
        return self.trade_generator.from_target_positions(
            target_positions=signals,
            current_positions=context['portfolio'].positions,
            close_prices=context['prices'],
            adv=context.get('adv')
        )


class AlphaSignalGenerator(SignalGenerator):
    """Generate trades from alpha/score signals."""

    def __init__(
        self,
        signal_function: Callable[[Dict[str, Any]], Dict[str, float]],
        target_notional: Optional[float] = None,
        trade_generator_config: Optional[TradeGeneratorConfig] = None
    ):
        """
        Initialize alpha signal generator.

        Parameters:
        -----------
        signal_function : Callable
            Function that takes context and returns alpha scores
            Signature: f(context) -> Dict[ticker, score]
        target_notional : float, optional
            Target notional to allocate (defaults to portfolio_value)
        """
        super().__init__(trade_generator_config)
        self.signal_function = signal_function
        self.target_notional = target_notional

    def generate_signals(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate alpha signals."""
        return self.signal_function(context)

    def signals_to_trades(
        self,
        signals: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """Convert alpha scores to trades."""
        target_not = self.target_notional or context['portfolio_value']

        return self.trade_generator.from_signal_scores(
            signal_scores=signals,
            current_positions=context['portfolio'].positions,
            close_prices=context['prices'],
            portfolio_value=context['portfolio_value'],
            target_notional=target_not,
            adv=context.get('adv')
        )


class MomentumSignalGenerator(TargetWeightSignalGenerator):
    """
    Example: Momentum-based signal generator.

    Generates target weights based on recent price momentum.
    """

    def __init__(
        self,
        lookback_days: int = 10,
        long_threshold: float = 0.01,
        short_threshold: float = -0.01,
        long_weight: float = 0.2,
        short_weight: float = -0.1,
        universe: Optional[List[str]] = None,
        trade_generator_config: Optional[TradeGeneratorConfig] = None
    ):
        """
        Initialize momentum signal generator.

        Parameters:
        -----------
        lookback_days : int
            Number of days for momentum calculation
        long_threshold : float
            Momentum threshold for long positions
        short_threshold : float
            Momentum threshold for short positions
        long_weight : float
            Target weight for long positions
        short_weight : float
            Target weight for short positions
        universe : List[str], optional
            Stock universe to trade
        """
        self.lookback_days = lookback_days
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.long_weight = long_weight
        self.short_weight = short_weight
        self.universe = universe

        # Create signal function
        def momentum_signals(context):
            return self._calculate_momentum(context)

        super().__init__(momentum_signals, trade_generator_config)

    def _calculate_momentum(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate momentum signals."""
        signals = {}

        # Get historical returns
        if len(context['daily_returns']) < self.lookback_days:
            return signals

        recent_returns = context['daily_returns'][-self.lookback_days:]

        # Calculate momentum for each ticker
        tickers = self.universe if self.universe else context['prices'].keys()

        for ticker in tickers:
            if ticker not in context['prices']:
                continue

            # Calculate average return (momentum)
            # This is a simplified calculation - in practice you'd want
            # to track per-ticker returns
            # For now, we'll use a placeholder
            try:
                # Placeholder: use random for demonstration
                # In real usage, you'd calculate actual ticker momentum
                momentum = np.random.randn() * 0.02

                if momentum > self.long_threshold:
                    signals[ticker] = self.long_weight
                elif momentum < self.short_threshold:
                    signals[ticker] = self.short_weight
            except:
                continue

        return signals


class ConditionalSignalGenerator(SignalGenerator):
    """
    Signal generator with conditional logic.

    Generates trades only when certain conditions are met.
    """

    def __init__(
        self,
        signal_function: Callable[[Dict[str, Any]], Dict[str, float]],
        condition_function: Callable[[Dict[str, Any]], bool],
        signal_type: str = 'weights',
        target_notional: Optional[float] = None,
        trade_generator_config: Optional[TradeGeneratorConfig] = None
    ):
        """
        Initialize conditional signal generator.

        Parameters:
        -----------
        signal_function : Callable
            Function to generate signals
        condition_function : Callable
            Function that returns True if trades should be generated
            Signature: f(context) -> bool
        signal_type : str
            Type of signals: 'weights', 'positions', 'scores'
        """
        super().__init__(trade_generator_config)
        self.signal_function = signal_function
        self.condition_function = condition_function
        self.signal_type = signal_type
        self.target_notional = target_notional

    def generate_signals(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate signals if condition is met."""
        if not self.condition_function(context):
            return {}

        return self.signal_function(context)

    def signals_to_trades(
        self,
        signals: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """Convert signals to trades based on signal type."""
        if self.signal_type == 'weights':
            return self.trade_generator.from_target_weights(
                signals,
                context['portfolio'].positions,
                context['prices'],
                context['portfolio_value'],
                context.get('adv')
            )
        elif self.signal_type == 'positions':
            return self.trade_generator.from_target_positions(
                signals,
                context['portfolio'].positions,
                context['prices'],
                context.get('adv')
            )
        elif self.signal_type == 'scores':
            target_not = self.target_notional or context['portfolio_value']
            return self.trade_generator.from_signal_scores(
                signals,
                context['portfolio'].positions,
                context['prices'],
                context['portfolio_value'],
                target_not,
                context.get('adv')
            )
        else:
            raise ValueError(f"Unknown signal_type: {self.signal_type}")


def create_simple_signal_generator(
    signal_function: Callable[[Dict[str, Any]], Dict[str, float]],
    signal_type: str = 'weights',
    target_notional: Optional[float] = None,
    trade_generator_config: Optional[TradeGeneratorConfig] = None
) -> SignalGenerator:
    """
    Create a signal generator from a simple signal function.

    This is a convenience function for quick setup.

    Parameters:
    -----------
    signal_function : Callable
        Function that generates signals from context
    signal_type : str
        Type of signals: 'weights', 'positions', or 'scores'
    target_notional : float, optional
        Target notional for 'scores' type
    trade_generator_config : TradeGeneratorConfig, optional
        Configuration for trade generation

    Returns:
    --------
    SignalGenerator
        Configured signal generator

    Example:
    --------
    >>> def my_signals(context):
    ...     # Your signal logic
    ...     return {'AAPL': 0.3, 'MSFT': 0.2}
    ...
    >>> generator = create_simple_signal_generator(my_signals, signal_type='weights')
    >>> results = backtester.run(use_case=3, inputs={'external_trades': generator})
    """
    if signal_type == 'weights':
        return TargetWeightSignalGenerator(signal_function, trade_generator_config)
    elif signal_type == 'positions':
        return TargetPositionSignalGenerator(signal_function, trade_generator_config)
    elif signal_type == 'scores':
        return AlphaSignalGenerator(signal_function, target_notional, trade_generator_config)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}. Must be 'weights', 'positions', or 'scores'")

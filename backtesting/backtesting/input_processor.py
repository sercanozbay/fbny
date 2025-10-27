"""
Input processing module for different use cases.

This module handles processing of inputs for the three main use cases:
1. Target positions (shares/notional/weights)
2. Signals → positions
3. External trades with risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Literal, List

from .utils import weights_to_shares, notional_to_shares


class TargetPortfolioProcessor:
    """
    Process use case 1: target positions input.

    Handles conversion between shares, notional, and weights.
    """

    def __init__(self):
        """Initialize target portfolio processor."""
        pass

    def process_target_shares(
        self,
        target_shares: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Process target share counts.

        Parameters:
        -----------
        target_shares : Dict[str, float]
            Ticker -> target shares
        prices : Dict[str, float]
            Ticker -> price

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
            (shares, notional, weights)
        """
        shares = target_shares.copy()

        # Calculate notional
        notional = {
            ticker: qty * prices.get(ticker, 0.0)
            for ticker, qty in shares.items()
        }

        # Calculate weights
        total_notional = sum(abs(n) for n in notional.values())
        if total_notional > 0:
            weights = {
                ticker: n / total_notional
                for ticker, n in notional.items()
            }
        else:
            weights = {ticker: 0.0 for ticker in shares}

        return shares, notional, weights

    def process_target_notional(
        self,
        target_notional: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Process target dollar notional.

        Parameters:
        -----------
        target_notional : Dict[str, float]
            Ticker -> target dollar amount
        prices : Dict[str, float]
            Ticker -> price

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
            (shares, notional, weights)
        """
        notional = target_notional.copy()

        # Calculate shares
        shares = notional_to_shares(notional, prices)

        # Calculate weights
        total_notional = sum(abs(n) for n in notional.values())
        if total_notional > 0:
            weights = {
                ticker: n / total_notional
                for ticker, n in notional.items()
            }
        else:
            weights = {ticker: 0.0 for ticker in notional}

        return shares, notional, weights

    def process_target_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Process target portfolio weights.

        Parameters:
        -----------
        target_weights : Dict[str, float]
            Ticker -> target weight
        prices : Dict[str, float]
            Ticker -> price
        portfolio_value : float
            Total portfolio value to allocate

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
            (shares, notional, weights)
        """
        weights = target_weights.copy()

        # Calculate notional
        notional = {
            ticker: weight * portfolio_value
            for ticker, weight in weights.items()
        }

        # Calculate shares
        shares = weights_to_shares(weights, prices, portfolio_value)

        return shares, notional, weights


class SignalProcessor:
    """
    Process use case 2: signals → positions.

    Converts alpha signals to portfolio positions.
    """

    def __init__(
        self,
        scaling_method: Literal['linear', 'rank', 'zscore'] = 'zscore',
        target_gross_exposure: float = 1.0,
        long_short: bool = True
    ):
        """
        Initialize signal processor.

        Parameters:
        -----------
        scaling_method : str
            How to scale signals: 'linear', 'rank', or 'zscore'
        target_gross_exposure : float
            Target gross notional as fraction of portfolio value
        long_short : bool
            If True, create long/short portfolio; if False, long only
        """
        self.scaling_method = scaling_method
        self.target_gross_exposure = target_gross_exposure
        self.long_short = long_short

    def process_signals(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Convert signals to positions.

        Parameters:
        -----------
        signals : Dict[str, float]
            Ticker -> signal value
        prices : Dict[str, float]
            Ticker -> price
        portfolio_value : float
            Portfolio value

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
            (shares, notional, weights)
        """
        # Scale signals
        scaled_signals = self._scale_signals(signals)

        # Convert to weights
        if self.long_short:
            weights = self._signals_to_long_short_weights(scaled_signals)
        else:
            weights = self._signals_to_long_only_weights(scaled_signals)

        # Apply target gross exposure
        current_gross = sum(abs(w) for w in weights.values())
        if current_gross > 0:
            scale_factor = self.target_gross_exposure / current_gross
            weights = {
                ticker: w * scale_factor
                for ticker, w in weights.items()
            }

        # Convert to notional and shares
        notional = {
            ticker: weight * portfolio_value
            for ticker, weight in weights.items()
        }

        shares = weights_to_shares(weights, prices, portfolio_value)

        return shares, notional, weights

    def _scale_signals(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Scale signals according to method."""
        if not signals:
            return {}

        values = np.array(list(signals.values()))
        tickers = list(signals.keys())

        if self.scaling_method == 'linear':
            # Simple linear scaling to [-1, 1] or [0, 1]
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                scaled = (values - min_val) / (max_val - min_val)
                if self.long_short:
                    scaled = 2 * scaled - 1  # Scale to [-1, 1]
            else:
                scaled = np.zeros_like(values)

        elif self.scaling_method == 'rank':
            # Rank-based scaling
            ranks = np.argsort(np.argsort(values))
            scaled = ranks / (len(ranks) - 1) if len(ranks) > 1 else np.zeros(len(ranks))
            if self.long_short:
                scaled = 2 * scaled - 1  # Scale to [-1, 1]

        elif self.scaling_method == 'zscore':
            # Z-score normalization
            mean = values.mean()
            std = values.std()
            if std > 0:
                scaled = (values - mean) / std
            else:
                scaled = np.zeros_like(values)

        else:
            scaled = values

        return {ticker: float(scaled[i]) for i, ticker in enumerate(tickers)}

    def _signals_to_long_short_weights(
        self,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Convert signals to long/short weights."""
        # Separate positive and negative signals
        long_signals = {t: s for t, s in signals.items() if s > 0}
        short_signals = {t: s for t, s in signals.items() if s < 0}

        # Normalize each side
        long_sum = sum(long_signals.values()) if long_signals else 0
        short_sum = sum(abs(s) for s in short_signals.values()) if short_signals else 0

        weights = {}

        # Long side: scale to 0.5 (half of gross exposure)
        if long_sum > 0:
            for ticker, signal in long_signals.items():
                weights[ticker] = 0.5 * signal / long_sum

        # Short side: scale to -0.5
        if short_sum > 0:
            for ticker, signal in short_signals.items():
                weights[ticker] = -0.5 * abs(signal) / short_sum

        return weights

    def _signals_to_long_only_weights(
        self,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Convert signals to long-only weights."""
        # Only use positive signals
        positive_signals = {t: max(s, 0) for t, s in signals.items()}

        total = sum(positive_signals.values())
        if total > 0:
            return {
                ticker: signal / total
                for ticker, signal in positive_signals.items()
            }
        else:
            return {ticker: 0.0 for ticker in signals}


class ExternalTradesProcessor:
    """
    Process use case 3: external trades.

    Applies external trades to current portfolio.
    """

    def __init__(self):
        """Initialize external trades processor."""
        pass

    def apply_external_trades(
        self,
        current_positions: Dict[str, float],
        external_trades: Dict[str, List[Dict]]
    ) -> tuple[Dict[str, float], List[Dict]]:
        """
        Apply external trades to current positions.

        Parameters:
        -----------
        current_positions : Dict[str, float]
            Current holdings
        external_trades : Dict[str, List[Dict]]
            External trades to apply
            Format: {ticker: [{'qty': shares, 'price': price, 'tag': 'optional_tag'}, ...], ...}

        Returns:
        --------
        tuple[Dict[str, float], List[Dict]]
            - New positions after external trades
            - List of trade records with tags extracted
        """
        new_positions = current_positions.copy()
        trade_records = []

        for ticker, trade_list in external_trades.items():
            if not isinstance(trade_list, list):
                raise ValueError(
                    f"External trades must be a list of dicts with 'qty' and 'price'. "
                    f"Got {type(trade_list)} for {ticker}"
                )

            # Sum up all trades for this ticker and extract tags
            for trade in trade_list:
                qty = trade.get('qty', 0)
                price = trade.get('price', 0)
                tag = trade.get('tag', None)  # Optional tag for grouping

                # Record trade with tag
                trade_records.append({
                    'ticker': ticker,
                    'qty': qty,
                    'price': price,
                    'tag': tag
                })

                # Update position
                new_positions[ticker] = new_positions.get(ticker, 0.0) + qty

        # Remove zero positions
        new_positions = {
            ticker: qty
            for ticker, qty in new_positions.items()
            if abs(qty) > 1e-6
        }

        return new_positions, trade_records

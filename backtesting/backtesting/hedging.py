"""
Hedging module for beta and sector hedging.

This module implements logic to hedge portfolio beta exposure
and sector exposures.
"""

from typing import Dict, Tuple, Optional


class BetaHedger:
    """
    Beta hedging using a market proxy (e.g., futures or ETF).

    Calculates the required hedge quantity to achieve target beta.
    """

    def __init__(
        self,
        hedge_instrument: str = 'SPY',
        target_beta: float = 0.0
    ):
        """
        Initialize beta hedger.

        Parameters:
        -----------
        hedge_instrument : str
            Ticker for the hedge instrument (e.g., 'SPY', 'ES')
        target_beta : float
            Target net portfolio beta (typically 0.0 for market neutral)
        """
        self.hedge_instrument = hedge_instrument
        self.target_beta = target_beta

    def calculate_hedge(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        betas: Dict[str, float],
        hedge_beta: float = 1.0,
        hedge_price: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate hedge quantity needed.

        Portfolio beta = sum(weight_i * beta_i)
        Hedge quantity = (current_beta - target_beta) * portfolio_value / hedge_price

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions (ticker -> shares)
        prices : Dict[str, float]
            Security prices
        betas : Dict[str, float]
            Beta per security
        hedge_beta : float
            Beta of the hedge instrument (typically 1.0)
        hedge_price : float, optional
            Price of hedge instrument (if None, use from prices dict)

        Returns:
        --------
        Tuple[float, float, float]
            (hedge_shares, current_beta, hedged_beta)
        """
        # Calculate portfolio value and beta
        total_value = 0.0
        weighted_beta = 0.0

        for ticker, shares in positions.items():
            if ticker == self.hedge_instrument:
                continue  # Exclude existing hedge from calculation

            value = shares * prices.get(ticker, 0.0)
            beta = betas.get(ticker, 0.0)

            total_value += abs(value)
            weighted_beta += value * beta

        if total_value == 0:
            return 0.0, 0.0, 0.0

        current_beta = weighted_beta / total_value

        # Calculate required hedge
        if hedge_price is None:
            hedge_price = prices.get(self.hedge_instrument, 1.0)

        if hedge_price == 0:
            return 0.0, current_beta, current_beta

        # Hedge shares needed to achieve target beta
        # hedge_beta * hedge_notional = (current_beta - target_beta) * portfolio_value
        required_notional = (current_beta - self.target_beta) * total_value
        hedge_shares = -required_notional / (hedge_beta * hedge_price)

        # Calculate hedged beta
        hedge_value = hedge_shares * hedge_price
        hedged_beta = (weighted_beta + hedge_value * hedge_beta) / (total_value + abs(hedge_value))

        return hedge_shares, current_beta, hedged_beta

    def apply_hedge(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        betas: Dict[str, float],
        hedge_beta: float = 1.0,
        hedge_price: Optional[float] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Apply beta hedge to positions.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions
        prices : Dict[str, float]
            Prices
        betas : Dict[str, float]
            Betas
        hedge_beta : float
            Beta of hedge instrument
        hedge_price : float, optional
            Price of hedge instrument

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float]]
            (hedged_positions, hedge_info)
        """
        hedge_shares, current_beta, hedged_beta = self.calculate_hedge(
            positions, prices, betas, hedge_beta, hedge_price
        )

        hedged_positions = positions.copy()

        if hedge_shares != 0:
            # Add hedge position
            hedged_positions[self.hedge_instrument] = (
                hedged_positions.get(self.hedge_instrument, 0.0) + hedge_shares
            )

        hedge_info = {
            'hedge_shares': hedge_shares,
            'current_beta': current_beta,
            'hedged_beta': hedged_beta,
            'hedge_instrument': self.hedge_instrument
        }

        return hedged_positions, hedge_info


class SectorHedger:
    """
    Sector hedging to achieve sector neutrality.

    Adjusts positions to bring sector exposures within target ranges.
    """

    def __init__(
        self,
        target_exposures: Optional[Dict[str, float]] = None,
        hedge_method: str = 'proportional',
        sector_etf_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize sector hedger.

        Parameters:
        -----------
        target_exposures : Dict[str, float], optional
            Target sector exposure per sector (default: 0.0 for all)
        hedge_method : str
            Method for hedging: 'proportional' or 'etf'
        sector_etf_mapping : Dict[str, str], optional
            Mapping of sector to ETF ticker for ETF hedging method
            Example: {'Technology': 'XLK', 'Healthcare': 'XLV', ...}
        """
        self.target_exposures = target_exposures or {}
        self.hedge_method = hedge_method
        self.sector_etf_mapping = sector_etf_mapping or self._get_default_sector_etfs()

    @staticmethod
    def _get_default_sector_etfs() -> Dict[str, str]:
        """
        Get default sector ETF mapping (US market).

        Returns:
        --------
        Dict[str, str]
            Sector -> ETF ticker mapping
        """
        return {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communication Services': 'XLC'
        }

    def calculate_sector_exposures(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        sector_mapping: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate current sector exposures.

        Parameters:
        -----------
        positions : Dict[str, float]
            Portfolio positions
        prices : Dict[str, float]
            Prices
        sector_mapping : Dict[str, str]
            Ticker -> sector

        Returns:
        --------
        Dict[str, float]
            Sector -> exposure (as fraction of gross notional)
        """
        sector_values = {}
        total_value = 0.0

        for ticker, shares in positions.items():
            value = shares * prices.get(ticker, 0.0)
            sector = sector_mapping.get(ticker)

            if sector:
                sector_values[sector] = sector_values.get(sector, 0.0) + value

            total_value += abs(value)

        if total_value == 0:
            return {}

        # Convert to exposures (fractions)
        sector_exposures = {
            sector: value / total_value
            for sector, value in sector_values.items()
        }

        return sector_exposures

    def calculate_hedge_proportional(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        sector_mapping: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate proportional hedge adjustments.

        This method scales positions within each sector to achieve neutrality.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions
        prices : Dict[str, float]
            Prices
        sector_mapping : Dict[str, str]
            Ticker -> sector

        Returns:
        --------
        Dict[str, float]
            Adjustments to positions (ticker -> additional shares)
        """
        # Calculate current sector exposures
        sector_exposures = self.calculate_sector_exposures(
            positions, prices, sector_mapping
        )

        # Calculate required adjustments
        adjustments = {}
        total_value = sum(
            abs(shares * prices.get(ticker, 0.0))
            for ticker, shares in positions.items()
        )

        if total_value == 0:
            return adjustments

        for sector, current_exp in sector_exposures.items():
            target_exp = self.target_exposures.get(sector, 0.0)
            adjustment_value = (target_exp - current_exp) * total_value

            # Distribute adjustment proportionally across securities in sector
            sector_tickers = [
                t for t, s in sector_mapping.items()
                if s == sector and t in positions
            ]

            if not sector_tickers:
                continue

            sector_value = sum(
                positions[t] * prices.get(t, 0.0)
                for t in sector_tickers
            )

            if sector_value == 0:
                continue

            # Proportional adjustment
            for ticker in sector_tickers:
                ticker_value = positions[ticker] * prices.get(ticker, 0.0)
                ticker_fraction = ticker_value / sector_value
                ticker_adjustment_value = adjustment_value * ticker_fraction

                price = prices.get(ticker, 0.0)
                if price > 0:
                    adjustments[ticker] = ticker_adjustment_value / price

        return adjustments

    def calculate_hedge_etf(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        sector_mapping: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate ETF-based hedge adjustments.

        This method uses sector ETFs to hedge sector exposures instead of
        adjusting individual stock positions.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions
        prices : Dict[str, float]
            Prices (must include sector ETF prices)
        sector_mapping : Dict[str, str]
            Ticker -> sector

        Returns:
        --------
        Dict[str, float]
            ETF hedge positions (ETF ticker -> shares)
        """
        # Calculate current sector exposures
        sector_exposures = self.calculate_sector_exposures(
            positions, prices, sector_mapping
        )

        # Calculate total portfolio value
        total_value = sum(
            abs(shares * prices.get(ticker, 0.0))
            for ticker, shares in positions.items()
        )

        if total_value == 0:
            return {}

        # Calculate required hedge for each sector using sector ETFs
        etf_hedges = {}

        for sector, current_exp in sector_exposures.items():
            target_exp = self.target_exposures.get(sector, 0.0)

            # Calculate notional adjustment needed
            adjustment_value = (target_exp - current_exp) * total_value

            # Get sector ETF ticker
            etf_ticker = self.sector_etf_mapping.get(sector)

            if etf_ticker is None:
                # Skip if no ETF mapping for this sector
                continue

            # Get ETF price
            etf_price = prices.get(etf_ticker)

            if etf_price is None or etf_price == 0:
                # Skip if ETF price not available
                continue

            # Calculate ETF shares needed
            # Negative adjustment_value means we need to short the ETF to reduce exposure
            # Positive adjustment_value means we need to long the ETF to increase exposure
            etf_shares = -adjustment_value / etf_price

            if etf_shares != 0:
                etf_hedges[etf_ticker] = etf_hedges.get(etf_ticker, 0.0) + etf_shares

        return etf_hedges

    def apply_hedge(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        sector_mapping: Dict[str, str]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Apply sector hedge.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions
        prices : Dict[str, float]
            Prices (must include sector ETF prices if using 'etf' method)
        sector_mapping : Dict[str, str]
            Ticker -> sector

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float]]
            (hedged_positions, hedge_info with sector exposures)
        """
        current_exposures = self.calculate_sector_exposures(
            positions, prices, sector_mapping
        )

        hedged_positions = positions.copy()

        if self.hedge_method == 'proportional':
            # Proportional method: adjust individual stock positions
            adjustments = self.calculate_hedge_proportional(
                positions, prices, sector_mapping
            )

            for ticker, adj in adjustments.items():
                hedged_positions[ticker] = hedged_positions.get(ticker, 0.0) + adj

            hedge_info = {
                'method': 'proportional',
                'current_exposures': current_exposures,
                'adjustments': adjustments
            }

        elif self.hedge_method == 'etf':
            # ETF method: add sector ETF positions
            etf_hedges = self.calculate_hedge_etf(
                positions, prices, sector_mapping
            )

            for etf_ticker, shares in etf_hedges.items():
                hedged_positions[etf_ticker] = hedged_positions.get(etf_ticker, 0.0) + shares

            hedge_info = {
                'method': 'etf',
                'current_exposures': current_exposures,
                'etf_hedges': etf_hedges,
                'etf_mapping': self.sector_etf_mapping
            }

        else:
            raise ValueError(f"Unknown hedge method: {self.hedge_method}. "
                           f"Must be 'proportional' or 'etf'")

        # Calculate final exposures
        final_exposures = self.calculate_sector_exposures(
            hedged_positions, prices, sector_mapping
        )

        hedge_info['final_exposures'] = final_exposures

        return hedged_positions, hedge_info

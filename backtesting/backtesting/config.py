"""
Configuration classes for the backtester.

This module contains dataclass definitions for all configuration parameters
used throughout the backtesting system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
import numpy as np


@dataclass
class BacktestConfig:
    """Main configuration for backtest execution."""

    # Risk constraints
    max_factor_exposure: Optional[Dict[str, float]] = None  # Max absolute exposure per factor
    max_sector_exposure: Optional[Dict[str, float]] = None  # Max absolute exposure per sector
    max_gross_exposure: Optional[float] = None  # Max gross notional
    max_net_exposure: Optional[float] = None  # Max net notional
    max_portfolio_variance: Optional[float] = None  # Max portfolio variance

    # Trading constraints
    max_adv_participation: float = 0.05  # Maximum trade size as % of ADV
    min_trade_size: float = 1.0  # Minimum number of shares to trade

    # Hedging configuration
    enable_beta_hedge: bool = False
    enable_sector_hedge: bool = False
    beta_hedge_instrument: str = 'SPY'  # Ticker for beta hedge (future/ETF)
    target_beta: float = 0.0  # Target net portfolio beta
    sector_hedge_method: str = 'proportional'  # 'proportional' or 'etf'
    sector_target_exposures: Optional[Dict[str, float]] = None  # Target exposure per sector
    sector_etf_mapping: Optional[Dict[str, str]] = None  # Sector -> ETF ticker for ETF hedging

    # Transaction costs
    tc_power: float = 1.5  # Power in transaction cost function: cost = a * (qty/adv)^power
    tc_coefficient: float = 0.01  # Coefficient 'a' in cost function
    tc_fixed: float = 0.0001  # Fixed component (bps)

    # Execution
    use_trade_prices: bool = False  # If False, use close prices

    # Optimization (for use case 3)
    optimizer_method: str = 'SLSQP'  # scipy optimizer method
    optimizer_max_iter: int = 1000
    optimizer_tolerance: float = 1e-6

    # Performance
    use_float32: bool = True  # Use float32 for memory efficiency
    risk_free_rate: float = 0.0  # Annual risk-free rate for Sharpe calculation

    # Initial portfolio
    initial_cash: float = 10_000_000.0  # Starting cash
    initial_positions: Optional[Dict[str, float]] = None  # Initial holdings (ticker: shares)

    # Stop loss configuration (use cases 1 and 2 only)
    stop_loss_levels: Optional[List[Union[
        Tuple[float, float],
        Tuple[float, float, Optional[float]],
        Tuple[float, float, Optional[float], str]
    ]]] = None
    # List of stop loss level tuples:
    #   - 2-tuple: (drawdown_threshold, gross_reduction) - assumes 'percent' type, no recovery
    #   - 3-tuple: (drawdown_threshold, gross_reduction, recovery_threshold) - assumes 'percent' type
    #   - 4-tuple: (drawdown_threshold, gross_reduction, recovery_threshold, threshold_type)
    #
    # threshold_type can be 'percent' or 'dollar'
    # recovery_threshold is optional (can be None) - when portfolio recovers by this amount from
    # the drawdown trough, it moves back to the previous (less restrictive) level
    #
    # IMPORTANT: All levels must use the same threshold_type (all 'percent' or all 'dollar').
    # Mixed threshold types are not supported.
    #
    # Examples:
    #   Percent-based with recovery:
    #     [(0.05, 0.75, 0.50), (0.10, 0.50, 0.50)] means:
    #       - At 5% drawdown, reduce gross to 75%; recover to 100% when portfolio recovers 50% from bottom
    #       - At 10% drawdown, reduce gross to 50%; recover to 75% when portfolio recovers 50% from bottom
    #
    #   Dollar-based with recovery:
    #     [(5000, 0.75, 2500, 'dollar'), (10000, 0.50, 5000, 'dollar')] means:
    #       - At $5k loss, reduce gross to 75%; recover to 100% when portfolio recovers $2,500 from bottom
    #       - At $10k loss, reduce gross to 50%; recover to 75% when portfolio recovers $5,000 from bottom
    #
    #   Without recovery (backward compatible):
    #     [(0.05, 0.75), (0.10, 0.50)] means:
    #       - At 5% drawdown, reduce gross to 75% (no automatic recovery)
    #       - At 10% drawdown, reduce gross to 50% (no automatic recovery)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_adv_participation <= 0 or self.max_adv_participation > 1:
            raise ValueError("max_adv_participation must be in (0, 1]")
        if self.tc_power < 0:
            raise ValueError("tc_power must be non-negative")
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")

        # Validate stop loss levels if provided
        if self.stop_loss_levels is not None:
            # First pass: extract all threshold types to ensure consistency
            threshold_types = set()
            for level_tuple in self.stop_loss_levels:
                if len(level_tuple) == 2:
                    threshold_types.add('percent')
                elif len(level_tuple) == 3:
                    # Check if 3rd element is a string (old format) or recovery threshold (new format)
                    if isinstance(level_tuple[2], str):
                        threshold_types.add(level_tuple[2])  # Old format: (dd, gross, 'dollar')
                    else:
                        threshold_types.add('percent')  # New format: (dd, gross, recovery)
                elif len(level_tuple) == 4:
                    threshold_types.add(level_tuple[3])
                else:
                    raise ValueError(f"Stop loss level must be 2-tuple, 3-tuple, or 4-tuple, got {len(level_tuple)}-tuple")

            # Ensure all levels use the same threshold type
            if len(threshold_types) > 1:
                raise ValueError(
                    f"All stop loss levels must use the same threshold_type. "
                    f"Found mixed types: {threshold_types}. "
                    f"Use either all 'percent' or all 'dollar'."
                )

            # Second pass: validate each level
            for level_tuple in self.stop_loss_levels:
                if len(level_tuple) == 2:
                    dd_threshold, gross_reduction = level_tuple
                    recovery_threshold = None
                    threshold_type = 'percent'
                elif len(level_tuple) == 3:
                    # Check if 3rd element is a string (old format) or recovery threshold (new format)
                    if isinstance(level_tuple[2], str):
                        # Old format: (dd, gross, 'dollar')
                        dd_threshold, gross_reduction, threshold_type = level_tuple
                        recovery_threshold = None
                    else:
                        # New format: (dd, gross, recovery) - assumes percent
                        dd_threshold, gross_reduction, recovery_threshold = level_tuple
                        threshold_type = 'percent'
                elif len(level_tuple) == 4:
                    dd_threshold, gross_reduction, recovery_threshold, threshold_type = level_tuple
                else:
                    raise ValueError(f"Stop loss level must be 2-tuple, 3-tuple, or 4-tuple, got {len(level_tuple)}-tuple")

                # Validate threshold type
                if threshold_type not in ('percent', 'dollar'):
                    raise ValueError(f"threshold_type must be 'percent' or 'dollar', got {threshold_type}")

                # Validate drawdown threshold based on type
                if threshold_type == 'percent':
                    if dd_threshold < 0 or dd_threshold > 1:
                        raise ValueError(f"Percent drawdown threshold must be in [0, 1], got {dd_threshold}")
                elif threshold_type == 'dollar':
                    if dd_threshold < 0:
                        raise ValueError(f"Dollar drawdown threshold must be non-negative, got {dd_threshold}")

                # Validate gross reduction
                if gross_reduction < 0 or gross_reduction > 1:
                    raise ValueError(f"Gross reduction must be in [0, 1], got {gross_reduction}")

                # Validate recovery threshold if provided
                if recovery_threshold is not None:
                    if threshold_type == 'percent':
                        if recovery_threshold < 0 or recovery_threshold > 1:
                            raise ValueError(f"Percent recovery threshold must be in [0, 1], got {recovery_threshold}")
                    elif threshold_type == 'dollar':
                        if recovery_threshold < 0:
                            raise ValueError(f"Dollar recovery threshold must be non-negative, got {recovery_threshold}")


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Metrics to include
    include_sharpe: bool = True
    include_sortino: bool = True
    include_calmar: bool = True
    include_var: bool = True
    include_cvar: bool = True

    # Chart settings
    figure_width: float = 12.0
    figure_height: float = 6.0
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'  # matplotlib style
    color_scheme: List[str] = field(default_factory=lambda: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Report content
    include_charts: bool = True
    include_trade_log: bool = True
    include_position_history: bool = True
    max_trades_in_report: int = 1000  # Limit trade log size

    # Output formats
    save_html: bool = True
    save_excel: bool = True
    save_pdf: bool = False  # Requires additional dependencies

    # Risk-free rate for calculations
    risk_free_rate: float = 0.0

    # VaR/CVaR confidence level
    var_confidence: float = 0.95


@dataclass
class Portfolio:
    """Represents a portfolio state at a point in time."""

    date: np.datetime64
    positions: Dict[str, float]  # ticker -> shares
    cash: float
    prices: Dict[str, float]  # ticker -> price

    def get_market_value(self) -> float:
        """Calculate total market value including cash."""
        return self.cash + sum(
            shares * self.prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )

    @property
    def portfolio_value(self) -> float:
        """Convenience property for get_market_value()."""
        return self.get_market_value()

    def get_position_values(self) -> Dict[str, float]:
        """Get market value per position."""
        return {
            ticker: shares * self.prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        }

    def get_weights(self) -> Dict[str, float]:
        """Get position weights."""
        mv = self.get_market_value()
        if mv == 0:
            return {ticker: 0.0 for ticker in self.positions}
        return {
            ticker: (shares * self.prices.get(ticker, 0.0)) / mv
            for ticker, shares in self.positions.items()
        }

    def get_gross_exposure(self) -> float:
        """Calculate gross notional exposure."""
        return sum(
            abs(shares * self.prices.get(ticker, 0.0))
            for ticker, shares in self.positions.items()
        )

    def get_net_exposure(self) -> float:
        """Calculate net notional exposure."""
        return sum(
            shares * self.prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )


@dataclass
class BacktestState:
    """Maintains the state of the backtest simulation."""

    current_date: np.datetime64
    portfolio: Portfolio

    # History tracking
    portfolio_values: List[float] = field(default_factory=list)
    dates: List[np.datetime64] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)
    transaction_costs: List[float] = field(default_factory=list)

    # PnL breakdown
    external_trade_pnl: List[float] = field(default_factory=list)  # PnL from external trades
    executed_trade_pnl: List[float] = field(default_factory=list)  # PnL from executed/optimized trades
    overnight_pnl: List[float] = field(default_factory=list)       # PnL from price changes (holding returns)

    # Exposure tracking
    gross_exposures: List[float] = field(default_factory=list)
    net_exposures: List[float] = field(default_factory=list)
    factor_exposures: List[Dict[str, float]] = field(default_factory=list)  # Factor name -> exposure per date

    # Trade tracking
    trades: List[Dict] = field(default_factory=list)  # List of trade records

    # External trade tag tracking (for counterparty/group attribution)
    external_trades_by_tag: List[Dict[str, List[Dict]]] = field(default_factory=list)  # Date -> {tag -> [trades]}
    external_pnl_by_tag: Dict[str, List[float]] = field(default_factory=dict)  # tag -> [daily PnLs]

    def update(
        self,
        new_portfolio: Portfolio,
        tc: float = 0.0,
        external_pnl: float = 0.0,
        executed_pnl: float = 0.0,
        overnight_pnl: float = 0.0
    ):
        """
        Update state with new portfolio and record history.

        Parameters:
        -----------
        new_portfolio : Portfolio
            New portfolio state
        tc : float
            Transaction costs
        external_pnl : float
            PnL from external trades (use case 3)
        executed_pnl : float
            PnL from executed/optimized trades
        overnight_pnl : float
            PnL from overnight price changes
        """
        old_value = self.portfolio.get_market_value()
        new_value = new_portfolio.get_market_value()

        self.portfolio = new_portfolio
        self.current_date = new_portfolio.date

        self.dates.append(new_portfolio.date)
        self.portfolio_values.append(new_value)
        self.transaction_costs.append(tc)

        pnl = new_value - old_value - tc
        self.daily_pnl.append(pnl)

        # Store PnL components
        self.external_trade_pnl.append(external_pnl)
        self.executed_trade_pnl.append(executed_pnl)
        self.overnight_pnl.append(overnight_pnl)

        if old_value > 0:
            daily_return = pnl / old_value
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)

        self.gross_exposures.append(new_portfolio.get_gross_exposure())
        self.net_exposures.append(new_portfolio.get_net_exposure())

    def add_trades(self, trade_records: List[Dict]):
        """Add trade records to history."""
        self.trades.extend(trade_records)

    def add_external_trades_with_tags(self, date: np.datetime64, trade_records: List[Dict]):
        """
        Record external trades grouped by tag for attribution.

        Parameters:
        -----------
        date : np.datetime64
            Trade date
        trade_records : List[Dict]
            List of trade records with 'ticker', 'qty', 'price', and optional 'tag'
        """
        # Group trades by tag
        trades_by_tag = {}
        for trade in trade_records:
            tag = trade.get('tag', 'untagged')
            if tag not in trades_by_tag:
                trades_by_tag[tag] = []
            trades_by_tag[tag].append(trade)

        self.external_trades_by_tag.append(trades_by_tag)

    def record_external_pnl_by_tag(self, tag: str, pnl: float):
        """
        Record PnL for a specific tag/counterparty.

        Parameters:
        -----------
        tag : str
            Tag identifier (e.g., counterparty name)
        pnl : float
            PnL amount for this tag on this day
        """
        if tag not in self.external_pnl_by_tag:
            self.external_pnl_by_tag[tag] = []
        self.external_pnl_by_tag[tag].append(pnl)

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
    # All thresholds are dollar-based drawdown levels for consistency
    stop_loss_levels: Optional[List[Union[
        Tuple[float, float],
        Tuple[float, float, Optional[float]]
    ]]] = None
    # List of stop loss level tuples (dollar-based):
    #   - 2-tuple: (drawdown_threshold, gross_reduction) - no automatic recovery
    #   - 3-tuple: (drawdown_threshold, gross_reduction, recovery_drawdown)
    #
    # drawdown_threshold: Dollar drawdown from peak that ENTERS this level (e.g., 10000 = enter at $10k loss)
    # gross_reduction: Target gross exposure as percentage (e.g., 0.75 = 75% of normal gross)
    # recovery_drawdown: Dollar drawdown from peak that EXITS this level (optional, must be < drawdown_threshold)
    #
    # STICKY RECOVERY: Once at a level, stay there until drawdown improves past recovery_drawdown
    #
    # Examples:
    #   Without recovery:
    #     [(5000, 0.75), (10000, 0.50)] means:
    #       - At $5k drawdown, reduce gross to 75% (stay until new peak)
    #       - At $10k drawdown, reduce gross to 50% (stay until new peak)
    #
    #   With recovery (RECOMMENDED):
    #     [(5000, 0.75, 2500), (10000, 0.50, 5000), (15000, 0.25, 10000)] means:
    #       - Enter at $5k drawdown → 75% gross, exit when drawdown ≤ $2.5k
    #       - Enter at $10k drawdown → 50% gross, exit when drawdown ≤ $5k (back to 75%)
    #       - Enter at $15k drawdown → 25% gross, exit when drawdown ≤ $10k (back to 50%)
    #
    #     Recovery sequence example:
    #       Portfolio: $100k → $85k (15k DD) → triggers 25% gross
    #       Portfolio: $85k → $95k (5k DD) → stays at 25% (DD > $10k recovery threshold)
    #       Portfolio: $95k → $91k (9k DD) → recovers to 50% gross (DD ≤ $10k recovery)
    #       Portfolio: $91k → $96k (4k DD) → recovers to 75% gross (DD ≤ $5k recovery)
    #       Portfolio: $96k → $102k (0k DD) → fully cleared at new peak

    def __post_init__(self):
        """Validate configuration."""
        if self.max_adv_participation <= 0 or self.max_adv_participation > 1:
            raise ValueError("max_adv_participation must be in (0, 1]")
        if self.tc_power < 0:
            raise ValueError("tc_power must be non-negative")
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")

        # Validate stop loss levels if provided (all dollar-based drawdown levels)
        if self.stop_loss_levels is not None:
            for level_tuple in self.stop_loss_levels:
                if len(level_tuple) == 2:
                    dd_threshold, gross_reduction = level_tuple
                    recovery_drawdown = None
                elif len(level_tuple) == 3:
                    dd_threshold, gross_reduction, recovery_drawdown = level_tuple
                else:
                    raise ValueError(f"Stop loss level must be 2-tuple or 3-tuple, got {len(level_tuple)}-tuple")

                # Validate drawdown threshold
                if dd_threshold < 0:
                    raise ValueError(f"Drawdown threshold must be non-negative, got {dd_threshold}")

                # Validate gross reduction
                if gross_reduction < 0 or gross_reduction > 1:
                    raise ValueError(f"Gross reduction must be in [0, 1], got {gross_reduction}")

                # Validate recovery drawdown if provided
                if recovery_drawdown is not None:
                    if recovery_drawdown < 0:
                        raise ValueError(f"Recovery drawdown must be non-negative, got {recovery_drawdown}")
                    if recovery_drawdown >= dd_threshold:
                        raise ValueError(
                            f"Recovery drawdown ({recovery_drawdown}) must be less than "
                            f"drawdown threshold ({dd_threshold})"
                        )


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

"""
Configuration classes for the backtester.

This module contains dataclass definitions for all configuration parameters
used throughout the backtesting system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
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

    def __post_init__(self):
        """Validate configuration."""
        if self.max_adv_participation <= 0 or self.max_adv_participation > 1:
            raise ValueError("max_adv_participation must be in (0, 1]")
        if self.tc_power < 0:
            raise ValueError("tc_power must be non-negative")
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")


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

    # Exposure tracking
    gross_exposures: List[float] = field(default_factory=list)
    net_exposures: List[float] = field(default_factory=list)
    factor_exposures: List[Dict[str, float]] = field(default_factory=list)  # Factor name -> exposure per date

    # Trade tracking
    trades: List[Dict] = field(default_factory=list)  # List of trade records

    def update(self, new_portfolio: Portfolio, tc: float = 0.0):
        """Update state with new portfolio and record history."""
        old_value = self.portfolio.get_market_value()
        new_value = new_portfolio.get_market_value()

        self.portfolio = new_portfolio
        self.current_date = new_portfolio.date

        self.dates.append(new_portfolio.date)
        self.portfolio_values.append(new_value)
        self.transaction_costs.append(tc)

        pnl = new_value - old_value - tc
        self.daily_pnl.append(pnl)

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

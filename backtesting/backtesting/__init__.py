"""
Backtesting Framework

A comprehensive backtesting framework for portfolio strategies with
factor risk models, transaction costs, and multiple use cases.
"""

__version__ = "1.0.0"

# Core classes
from .backtester import Backtester
from .config import BacktestConfig, ReportConfig, Portfolio, BacktestState
from .data_loader import DataManager
from .results import BacktestResults

# Trade generation
from .trade_generator import (
    ExternalTradeGenerator,
    TradeGeneratorConfig,
    generate_external_trades_from_signals
)

# Signal generators
from .signal_generators import (
    SignalGenerator,
    TargetWeightSignalGenerator,
    TargetPositionSignalGenerator,
    AlphaSignalGenerator,
    MomentumSignalGenerator,
    ConditionalSignalGenerator,
    create_simple_signal_generator
)

# NA handling
from .na_handling import (
    NAHandlingConfig,
    FillMethod,
    ValidationLevel
)

__all__ = [
    # Core
    'Backtester',
    'BacktestConfig',
    'ReportConfig',
    'Portfolio',
    'BacktestState',
    'DataManager',
    'BacktestResults',
    # Trade generation
    'ExternalTradeGenerator',
    'TradeGeneratorConfig',
    'generate_external_trades_from_signals',
    # Signal generators
    'SignalGenerator',
    'TargetWeightSignalGenerator',
    'TargetPositionSignalGenerator',
    'AlphaSignalGenerator',
    'MomentumSignalGenerator',
    'ConditionalSignalGenerator',
    'create_simple_signal_generator',
    # NA handling
    'NAHandlingConfig',
    'FillMethod',
    'ValidationLevel'
]

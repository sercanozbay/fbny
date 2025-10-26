"""
Backtesting Framework

A comprehensive backtesting framework for portfolio strategies with
factor risk models, transaction costs, and multiple use cases.
"""

__version__ = "1.0.0"

from .backtester import Backtester
from .config import BacktestConfig, ReportConfig, Portfolio, BacktestState
from .data_loader import DataManager
from .results import BacktestResults
from .trade_generator import (
    ExternalTradeGenerator,
    TradeGeneratorConfig,
    generate_external_trades_from_signals
)

__all__ = [
    'Backtester',
    'BacktestConfig',
    'ReportConfig',
    'Portfolio',
    'BacktestState',
    'DataManager',
    'BacktestResults',
    'ExternalTradeGenerator',
    'TradeGeneratorConfig',
    'generate_external_trades_from_signals'
]

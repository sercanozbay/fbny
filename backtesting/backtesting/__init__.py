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

__all__ = [
    'Backtester',
    'BacktestConfig',
    'ReportConfig',
    'Portfolio',
    'BacktestState',
    'DataManager',
    'BacktestResults'
]

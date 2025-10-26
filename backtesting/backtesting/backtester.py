"""
Main backtester engine.

This module contains the Backtester class that orchestrates
the entire backtest simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal
from tqdm import tqdm

from .config import BacktestConfig, Portfolio, BacktestState
from .data_loader import DataManager
from .risk_calculator import FactorRiskModel, RiskConstraintChecker
from .transaction_costs import TransactionCostModel, ADVConstraintCalculator
from .hedging import BetaHedger, SectorHedger
from .optimizer import PortfolioOptimizer, SimpleTradeOptimizer
from .input_processor import TargetPortfolioProcessor, SignalProcessor, ExternalTradesProcessor
from .execution import TradeExecutor
from .attribution import PerformanceAttributor, AttributionTracker
from .results import BacktestResults


class Backtester:
    """
    Main backtester engine.

    Orchestrates the daily simulation loop and coordinates all components.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_manager: DataManager
    ):
        """
        Initialize backtester.

        Parameters:
        -----------
        config : BacktestConfig
            Backtest configuration
        data_manager : DataManager
            Data manager with loaded data
        """
        self.config = config
        self.data_manager = data_manager

        # Initialize components
        self.risk_model = FactorRiskModel(use_float32=config.use_float32)
        self.risk_checker = RiskConstraintChecker(self.risk_model)

        self.cost_model = TransactionCostModel(
            power=config.tc_power,
            coefficient=config.tc_coefficient,
            fixed_cost=config.tc_fixed
        )

        self.adv_calculator = ADVConstraintCalculator(config.max_adv_participation)

        self.beta_hedger = BetaHedger(
            hedge_instrument=config.beta_hedge_instrument,
            target_beta=config.target_beta
        ) if config.enable_beta_hedge else None

        self.sector_hedger = SectorHedger() if config.enable_sector_hedge else None

        self.executor = TradeExecutor(
            cost_model=self.cost_model,
            use_trade_prices=config.use_trade_prices
        )

        self.attributor = PerformanceAttributor(self.risk_model)
        self.attribution_tracker = AttributionTracker()

        # Processors
        self.target_processor = TargetPortfolioProcessor()
        self.signal_processor = SignalProcessor()
        self.external_processor = ExternalTradesProcessor()

        # Simple trade optimizer
        self.simple_optimizer = SimpleTradeOptimizer(config.max_adv_participation)

        # Full optimizer (for use case 3)
        self.portfolio_optimizer = PortfolioOptimizer(
            cost_model=self.cost_model,
            risk_model=self.risk_model,
            method=config.optimizer_method,
            max_iter=config.optimizer_max_iter,
            tolerance=config.optimizer_tolerance
        )

        # State
        self.state: Optional[BacktestState] = None

    def run(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        use_case: Literal[1, 2, 3],
        inputs: Dict,
        show_progress: bool = True
    ) -> BacktestResults:
        """
        Run backtest simulation.

        Parameters:
        -----------
        start_date : pd.Timestamp
            Start date
        end_date : pd.Timestamp
            End date
        use_case : int
            Use case (1, 2, or 3)
        inputs : Dict
            Use-case specific inputs
        show_progress : bool
            Show progress bar

        Returns:
        --------
        BacktestResults
            Backtest results
        """
        print(f"\n{'='*60}")
        print(f"Starting Backtest - Use Case {use_case}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"{'='*60}\n")

        # Load required data
        print("Loading data...")
        self._load_data()

        # Initialize portfolio
        self._initialize_portfolio(start_date)

        # Get trading dates
        all_dates = self.data_manager.load_prices().index
        trading_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

        print(f"Trading days: {len(trading_dates)}")
        print(f"Initial portfolio value: ${self.state.portfolio.get_market_value():,.2f}\n")

        # Run simulation
        if show_progress:
            date_iterator = tqdm(trading_dates, desc="Simulating")
        else:
            date_iterator = trading_dates

        for date in date_iterator:
            self._simulate_day(date, use_case, inputs)

        print(f"\n{'='*60}")
        print("Backtest Complete")
        print(f"Final portfolio value: ${self.state.portfolio.get_market_value():,.2f}")
        print(f"{'='*60}\n")

        # Create results object
        return self._create_results()

    def _load_data(self):
        """Load all required data."""
        self.data_manager.load_prices()
        self.data_manager.load_adv()

        if self.config.enable_beta_hedge:
            self.data_manager.load_betas()

        if self.config.enable_sector_hedge:
            self.data_manager.load_sector_mapping()

        # Load factor model data
        try:
            self.data_manager.load_factor_exposures()
            self.data_manager.load_factor_returns()
            self.data_manager.load_factor_covariance()
            self.data_manager.load_specific_variance()
        except Exception as e:
            print(f"Warning: Could not load factor model data: {e}")

    def _initialize_portfolio(self, start_date: pd.Timestamp):
        """Initialize portfolio state."""
        prices_data = self.data_manager.load_prices()
        print(prices_data.index, type(start_date))
        start_prices = prices_data.loc[start_date].to_dict()

        initial_positions = self.config.initial_positions or {}

        portfolio = Portfolio(
            date=start_date,
            positions=initial_positions,
            cash=self.config.initial_cash,
            prices=start_prices
        )

        self.state = BacktestState(
            current_date=start_date,
            portfolio=portfolio
        )

        # Record initial state
        self.state.dates.append(start_date)
        self.state.portfolio_values.append(portfolio.get_market_value())
        self.state.daily_pnl.append(0.0)
        self.state.daily_returns.append(0.0)
        self.state.transaction_costs.append(0.0)
        self.state.gross_exposures.append(portfolio.get_gross_exposure())
        self.state.net_exposures.append(portfolio.get_net_exposure())
        self.state.factor_exposures.append({})  # Empty dict for initial state

    def _simulate_day(
        self,
        date: pd.Timestamp,
        use_case: Literal[1, 2, 3],
        inputs: Dict
    ):
        """Simulate one trading day."""
        # Get data for the day
        day_data = self.data_manager.get_data_for_date(date)

        if 'prices' not in day_data:
            return  # Skip if no price data

        prices = day_data['prices']
        adv = day_data.get('adv', {})
        betas = day_data.get('betas', {})
        sector_mapping = day_data.get('sector_mapping', {})

        # Get factor data if available
        factor_loadings = day_data.get('factor_exposures')
        factor_returns = day_data.get('factor_returns', {})

        # Determine target positions based on use case
        if use_case == 1:
            target_positions = self._process_use_case_1(
                date, inputs, prices
            )
        elif use_case == 2:
            target_positions = self._process_use_case_2(
                date, inputs, prices
            )
        elif use_case == 3:
            target_positions = self._process_use_case_3(
                date, inputs, prices, adv, factor_loadings
            )
        else:
            raise ValueError(f"Invalid use case: {use_case}")

        # Apply hedging
        if self.beta_hedger and betas:
            target_positions, _ = self.beta_hedger.apply_hedge(
                target_positions, prices, betas
            )

        if self.sector_hedger and sector_mapping:
            target_positions, _ = self.sector_hedger.apply_hedge(
                target_positions, prices, sector_mapping
            )

        # Calculate trades
        from .utils import calculate_trades
        trades = calculate_trades(self.state.portfolio.positions, target_positions)

        # Apply ADV constraints
        constrained_trades, _ = self.adv_calculator.apply_constraints(trades, adv)

        # Execute trades
        trade_prices = day_data.get('trade_prices')
        new_portfolio, total_cost, trade_records = self.executor.execute_trades(
            self.state.portfolio,
            constrained_trades,
            prices,
            trade_prices,
            adv,
            date
        )

        # Calculate attribution and factor exposures if factor data available
        if factor_loadings is not None and factor_returns:
            try:
                factor_pnl, specific_pnl, _, _ = self.attributor.calculate_total_attribution(
                    self.state.portfolio.positions,
                    self.state.portfolio.prices,
                    prices,
                    factor_loadings,
                    factor_returns
                )
                self.attribution_tracker.add_period(date, factor_pnl, specific_pnl)
            except:
                pass  # Skip attribution if error

        # Calculate factor exposures for new portfolio
        factor_exp_dict = {}
        if factor_loadings is not None:
            try:
                factor_exp_array = self.risk_model.calculate_factor_exposures(
                    new_portfolio.positions,
                    prices,
                    factor_loadings
                )
                # Convert to dict with factor names
                factor_names = factor_loadings.columns.tolist()
                factor_exp_dict = {
                    factor_names[i]: float(factor_exp_array[i])
                    for i in range(len(factor_names))
                }
            except:
                pass  # Skip if error

        # Update state
        self.state.update(new_portfolio, total_cost)
        self.state.add_trades(trade_records)
        self.state.factor_exposures.append(factor_exp_dict)

    def _process_use_case_1(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Process use case 1: target positions."""
        input_type = inputs.get('type', 'shares')  # 'shares', 'notional', or 'weights'
        targets_by_date = inputs.get('targets', {})

        if date not in targets_by_date:
            return self.state.portfolio.positions.copy()

        target_data = targets_by_date[date]

        if input_type == 'shares':
            shares, _, _ = self.target_processor.process_target_shares(
                target_data, prices
            )
        elif input_type == 'notional':
            shares, _, _ = self.target_processor.process_target_notional(
                target_data, prices
            )
        elif input_type == 'weights':
            portfolio_value = self.state.portfolio.get_market_value()
            shares, _, _ = self.target_processor.process_target_weights(
                target_data, prices, portfolio_value
            )
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        return shares

    def _process_use_case_2(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Process use case 2: signals."""
        signals_by_date = inputs.get('signals', {})

        if date not in signals_by_date:
            return self.state.portfolio.positions.copy()

        signals = signals_by_date[date]
        portfolio_value = self.state.portfolio.get_market_value()

        shares, _, _ = self.signal_processor.process_signals(
            signals, prices, portfolio_value
        )

        return shares

    def _process_use_case_3(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float],
        adv: Dict[str, float],
        factor_loadings: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Process use case 3: external trades with optimization."""
        external_trades_by_date = inputs.get('external_trades', {})

        # Apply external trades
        if date in external_trades_by_date:
            external_trades = external_trades_by_date[date]
            new_positions = self.external_processor.apply_external_trades(
                self.state.portfolio.positions, external_trades
            )
        else:
            new_positions = self.state.portfolio.positions.copy()

        # Check if optimization is needed (if constraints provided)
        needs_optimization = (
            self.config.max_portfolio_variance is not None or
            self.config.max_factor_exposure is not None
        )

        if not needs_optimization or factor_loadings is None:
            return new_positions

        # Run optimization to satisfy risk constraints
        try:
            factor_cov = self.data_manager.load_factor_covariance().values
            specific_var_data = self.data_manager.load_specific_variance()
            specific_var = specific_var_data.loc[date].to_dict() if date in specific_var_data.index else {}

            optimal_trades, _ = self.portfolio_optimizer.optimize_trades(
                new_positions,
                prices,
                adv,
                factor_loadings,
                factor_cov,
                specific_var,
                max_variance=self.config.max_portfolio_variance,
                max_factor_exposure=self.config.max_factor_exposure,
                max_adv_participation=self.config.max_adv_participation
            )

            # Apply optimal trades
            for ticker, trade_qty in optimal_trades.items():
                new_positions[ticker] = new_positions.get(ticker, 0.0) + trade_qty

        except Exception as e:
            print(f"Warning: Optimization failed on {date}: {e}")

        return new_positions

    def _create_results(self) -> BacktestResults:
        """Create results object from state."""
        return BacktestResults(
            dates=self.state.dates,
            portfolio_values=self.state.portfolio_values,
            daily_pnl=self.state.daily_pnl,
            daily_returns=self.state.daily_returns,
            transaction_costs=self.state.transaction_costs,
            gross_exposures=self.state.gross_exposures,
            net_exposures=self.state.net_exposures,
            trades=self.state.trades,
            attribution_tracker=self.attribution_tracker,
            risk_free_rate=self.config.risk_free_rate,
            factor_exposures=self.state.factor_exposures
        )

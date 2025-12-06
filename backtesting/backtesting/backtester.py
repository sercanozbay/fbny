"""
Main backtester engine.

This module contains the Backtester class that orchestrates
the entire backtest simulation.
"""

import pandas as pd
from typing import Dict, Optional, Literal, Tuple
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
from .stop_loss import StopLossManager, StopLossLevel


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

        self.sector_hedger = SectorHedger(
            target_exposures=config.sector_target_exposures,
            hedge_method=config.sector_hedge_method,
            sector_etf_mapping=config.sector_etf_mapping
        ) if config.enable_sector_hedge else None

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

        # Stop loss manager (for use cases 1 and 2)
        self.stop_loss_manager: Optional[StopLossManager] = None
        if config.stop_loss_levels is not None:
            levels = []
            for level_tuple in config.stop_loss_levels:
                if len(level_tuple) == 2:
                    dd, gr = level_tuple
                    levels.append(StopLossLevel(
                        drawdown_threshold=dd,
                        gross_reduction=gr,
                        recovery_threshold=None,
                        threshold_type='percent'
                    ))
                elif len(level_tuple) == 3:
                    # Check if 3rd element is a string (old format) or recovery threshold (new format)
                    if isinstance(level_tuple[2], str):
                        # Old format: (dd, gr, 'dollar')
                        dd, gr, threshold_type = level_tuple
                        levels.append(StopLossLevel(
                            drawdown_threshold=dd,
                            gross_reduction=gr,
                            recovery_threshold=None,
                            threshold_type=threshold_type
                        ))
                    else:
                        # New format: (dd, gr, recovery) - assumes percent
                        dd, gr, recovery = level_tuple
                        levels.append(StopLossLevel(
                            drawdown_threshold=dd,
                            gross_reduction=gr,
                            recovery_threshold=recovery,
                            threshold_type='percent'
                        ))
                elif len(level_tuple) == 4:
                    dd, gr, recovery, threshold_type = level_tuple
                    levels.append(StopLossLevel(
                        drawdown_threshold=dd,
                        gross_reduction=gr,
                        recovery_threshold=recovery,
                        threshold_type=threshold_type
                    ))
            self.stop_loss_manager = StopLossManager(levels)

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

        # Load corporate actions if available
        try:
            self.data_manager.load_corporate_actions()
        except Exception as e:
            print(f"Note: No corporate actions loaded: {e}")

    def _initialize_portfolio(self, start_date: pd.Timestamp):
        """Initialize portfolio state."""
        prices_data = self.data_manager.load_prices()
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

        # PnL breakdown for initial state
        self.state.external_trade_pnl.append(0.0)
        self.state.executed_trade_pnl.append(0.0)
        self.state.overnight_pnl.append(0.0)

        # Track previous prices for overnight PnL calculation
        self.prev_prices = None

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

        # Apply corporate actions BEFORE trading (splits and dividends)
        # Create adjusted portfolio WITHOUT modifying state
        current_positions = self.state.portfolio.positions.copy()
        current_cash = self.state.portfolio.cash

        if 'corporate_actions' in day_data:
            current_positions, current_cash = self._apply_corporate_actions(
                date, day_data['corporate_actions'], current_positions, current_cash
            )

        prices = day_data['prices']

        # Create temporary portfolio with adjusted positions/cash for use case processing
        from .config import Portfolio
        current_portfolio = Portfolio(
            date=date,
            positions=current_positions,
            cash=current_cash,
            prices=prices
        )

        adv = day_data.get('adv', {})
        betas = day_data.get('betas', {})
        sector_mapping = day_data.get('sector_mapping', {})

        # Get factor data if available
        factor_loadings = day_data.get('factor_exposures')
        factor_returns = day_data.get('factor_returns', {})

        # Handle use case 3 with PnL breakdown
        if use_case == 3:
            self._simulate_day_use_case_3(
                date, inputs, prices, adv, betas, sector_mapping,
                factor_loadings, factor_returns, day_data,
                current_portfolio
            )
            return

        # Determine target positions for use cases 1 and 2
        if use_case == 1:
            target_positions = self._process_use_case_1(
                date, inputs, prices, current_portfolio
            )
        elif use_case == 2:
            target_positions = self._process_use_case_2(
                date, inputs, prices, current_portfolio
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

        # Apply stop loss (if enabled) - update manager and reduce gross if needed
        if self.stop_loss_manager is not None:
            # Get current portfolio value
            current_portfolio_value = self.state.portfolio_values[-1] if self.state.portfolio_values else self.config.initial_cash

            # Update stop loss manager with current value
            gross_multiplier, level_changed = self.stop_loss_manager.update(current_portfolio_value)

            # Apply gross reduction to target positions if stop loss is active
            if gross_multiplier < 1.0:
                target_positions = self.stop_loss_manager.apply_to_positions(
                    target_positions,
                    current_portfolio.positions
                )

        # Calculate trades based on current (post-corporate-action) positions
        from .utils import calculate_trades
        trades = calculate_trades(current_portfolio.positions, target_positions)

        # Apply ADV constraints
        constrained_trades, _ = self.adv_calculator.apply_constraints(trades, adv)

        # Calculate overnight PnL (holding returns from previous day's positions)
        overnight_pnl = 0.0
        if self.prev_prices is not None:
            for ticker, shares in self.state.portfolio.positions.items():
                prev_px = self.prev_prices.get(ticker, 0.0)
                curr_px = prices.get(ticker, 0.0)
                overnight_pnl += shares * (curr_px - prev_px)

        # Execute trades using current (post-corporate-action) portfolio
        trade_prices = day_data.get('trade_prices')
        new_portfolio, total_cost, trade_records = self.executor.execute_trades(
            current_portfolio,
            constrained_trades,
            prices,
            trade_prices,
            adv,
            date
        )

        # Calculate executed trade PnL (difference between execution and close)
        executed_pnl = 0.0
        if trade_prices and self.config.use_trade_prices:
            for ticker, qty in constrained_trades.items():
                exec_px = trade_prices.get(ticker, prices.get(ticker, 0.0))
                close_px = prices.get(ticker, 0.0)
                executed_pnl += qty * (close_px - exec_px)

        external_pnl = 0.0  # No external trades in use cases 1 and 2

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

        # Update state with PnL breakdown
        self.state.update(
            new_portfolio,
            tc=total_cost,
            external_pnl=external_pnl,
            executed_pnl=executed_pnl,
            overnight_pnl=overnight_pnl
        )
        self.state.add_trades(trade_records)
        self.state.factor_exposures.append(factor_exp_dict)

        # Update previous prices for next day
        self.prev_prices = prices

    def _process_use_case_1(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float],
        current_portfolio: 'Portfolio'
    ) -> Dict[str, float]:
        """Process use case 1: target positions."""
        input_type = inputs.get('type', 'shares')  # 'shares', 'notional', or 'weights'
        targets_by_date = inputs.get('targets', {})

        if date not in targets_by_date:
            return current_portfolio.positions.copy()

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
            shares, _, _ = self.target_processor.process_target_weights(
                target_data, prices, current_portfolio.portfolio_value
            )
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        return shares

    def _process_use_case_2(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float],
        current_portfolio: 'Portfolio'
    ) -> Dict[str, float]:
        """Process use case 2: signals."""
        signals_by_date = inputs.get('signals', {})

        if date not in signals_by_date:
            return current_portfolio.positions.copy()

        signals = signals_by_date[date]

        shares, _, _ = self.signal_processor.process_signals(
            signals, prices, current_portfolio.portfolio_value
        )

        return shares

    def _process_use_case_3(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float],
        adv: Dict[str, float],
        factor_loadings: Optional[pd.DataFrame],
        current_portfolio: 'Portfolio'
    ) -> Dict:
        """
        Process use case 3: external trades with optimization.

        Returns:
        --------
        Dict with keys:
            'new_positions': Positions after external trades
            'target_positions': Optimized target positions (if optimization enabled)
        """
        external_trades_by_date = inputs.get('external_trades', {})

        # Apply external trades to current positions
        if date in external_trades_by_date:
            external_trades = external_trades_by_date[date]
            new_positions = self.external_processor.apply_external_trades(
                current_portfolio.positions, external_trades
            )
        else:
            new_positions = current_portfolio.positions.copy()

        # Check if optimization is needed (if constraints provided)
        needs_optimization = (
            self.config.max_portfolio_variance is not None or
            self.config.max_factor_exposure is not None
        )

        if not needs_optimization or factor_loadings is None:
            return {
                'new_positions': new_positions,
                'target_positions': new_positions
            }

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

            # Calculate target positions
            target_positions = new_positions.copy()
            for ticker, trade_qty in optimal_trades.items():
                target_positions[ticker] = target_positions.get(ticker, 0.0) + trade_qty

        except Exception as e:
            print(f"Warning: Optimization failed on {date}: {e}")
            target_positions = new_positions

        return {
            'new_positions': new_positions,
            'target_positions': target_positions
        }

    def _simulate_day_use_case_3(
        self,
        date: pd.Timestamp,
        inputs: Dict,
        prices: Dict[str, float],
        adv: Dict[str, float],
        betas: Dict[str, float],
        sector_mapping: Dict[str, str],
        factor_loadings: Optional[pd.DataFrame],
        factor_returns: Dict[str, float],
        day_data: Dict,
        current_portfolio: 'Portfolio'
    ):
        """
        Simulate day for use case 3 with external trades and PnL breakdown.

        This method handles external trades with their own execution prices
        and separates PnL into external, executed, and overnight components.

        External trades can be provided as:
        1. Dict[date, trades] - pre-generated trades
        2. Callable - function to generate trades dynamically based on state
        """
        external_trades_by_date = inputs.get('external_trades', {})
        external_exec_prices_by_date = inputs.get('external_exec_prices', {})

        # Check if external_trades is a callable (signal generator)
        if callable(external_trades_by_date):
            # Create context for the callable with current (post-corporate-action) portfolio
            context = {
                'date': date,
                'portfolio': current_portfolio,
                'prices': prices,
                'adv': adv,
                'betas': betas,
                'sector_mapping': sector_mapping,
                'factor_loadings': factor_loadings,
                'factor_returns': factor_returns,
                'portfolio_value': self.state.portfolio_values[-1] if self.state.portfolio_values else self.config.initial_cash,
                'dates': self.state.dates,
                'daily_returns': self.state.daily_returns,
                'daily_pnl': self.state.daily_pnl
            }

            # Call the function to generate trades for this date
            external_trades = external_trades_by_date(context)
            if external_trades is None:
                external_trades = {}
        else:
            # Get external trades from pre-generated dict
            external_trades = external_trades_by_date.get(date, {})

        external_exec_prices = external_exec_prices_by_date.get(date, None)

        # Apply external trades to get intermediate positions
        if external_trades:
            positions_after_external, external_trade_records = self.external_processor.apply_external_trades(
                current_portfolio.positions, external_trades
            )
            # Record external trades with tags for attribution
            self.state.add_external_trades_with_tags(date, external_trade_records)
        else:
            positions_after_external = current_portfolio.positions.copy()
            external_trade_records = []

        # Check if optimization is needed
        needs_optimization = (
            self.config.max_portfolio_variance is not None or
            self.config.max_factor_exposure is not None
        )

        # Calculate optimal trades (internal rebalancing)
        internal_trades = {}
        if needs_optimization and factor_loadings is not None:
            try:
                factor_cov = self.data_manager.load_factor_covariance().values
                specific_var_data = self.data_manager.load_specific_variance()
                specific_var = specific_var_data.loc[date].to_dict() if date in specific_var_data.index else {}

                optimal_trades, _ = self.portfolio_optimizer.optimize_trades(
                    positions_after_external,
                    prices,
                    adv,
                    factor_loadings,
                    factor_cov,
                    specific_var,
                    max_variance=self.config.max_portfolio_variance,
                    max_factor_exposure=self.config.max_factor_exposure,
                    max_adv_participation=self.config.max_adv_participation
                )
                internal_trades = optimal_trades
            except Exception as e:
                print(f"Warning: Optimization failed on {date}: {e}")

        # Apply hedging to internal trades if needed
        target_positions = positions_after_external.copy()
        for ticker, trade_qty in internal_trades.items():
            target_positions[ticker] = target_positions.get(ticker, 0.0) + trade_qty

        if self.beta_hedger and betas:
            target_positions, _ = self.beta_hedger.apply_hedge(
                target_positions, prices, betas
            )

        if self.sector_hedger and sector_mapping:
            target_positions, _ = self.sector_hedger.apply_hedge(
                target_positions, prices, sector_mapping
            )

        # Recalculate internal trades after hedging
        from .utils import calculate_trades
        internal_trades = calculate_trades(positions_after_external, target_positions)

        # Apply ADV constraints to internal trades
        internal_trades, _ = self.adv_calculator.apply_constraints(internal_trades, adv)

        # Get internal execution prices
        trade_prices = day_data.get('trade_prices')
        internal_exec_prices = trade_prices if self.config.use_trade_prices else None

        # Execute all trades with breakdown
        new_portfolio, total_cost, trade_records, pnl_breakdown = \
            self.executor.execute_trades_with_breakdown(
                self.state.portfolio,
                external_trades,
                internal_trades,
                prices,
                external_exec_prices,
                internal_exec_prices,
                adv,
                date,
                prev_close_prices=self.prev_prices
            )

        # Calculate attribution if factor data available
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
                pass

        # Calculate factor exposures for new portfolio
        factor_exp_dict = {}
        if factor_loadings is not None:
            try:
                factor_exp_array = self.risk_model.calculate_factor_exposures(
                    new_portfolio.positions,
                    prices,
                    factor_loadings
                )
                factor_names = factor_loadings.columns.tolist()
                factor_exp_dict = {
                    factor_names[i]: float(factor_exp_array[i])
                    for i in range(len(factor_names))
                }
            except:
                pass

        # Update state with PnL breakdown
        self.state.update(
            new_portfolio,
            tc=total_cost,
            external_pnl=pnl_breakdown['external'],
            executed_pnl=pnl_breakdown['executed'],
            overnight_pnl=pnl_breakdown['overnight']
        )
        self.state.add_trades(trade_records)
        self.state.factor_exposures.append(factor_exp_dict)

        # Aggregate PnL by tag for external trades
        pnl_by_tag = {}
        for trade_record in trade_records:
            if trade_record.get('type') == 'external':
                tag = trade_record.get('tag', 'untagged')
                trade_pnl = trade_record.get('pnl', 0.0)
                pnl_by_tag[tag] = pnl_by_tag.get(tag, 0.0) + trade_pnl

        # Record PnL for each tag
        for tag, pnl in pnl_by_tag.items():
            self.state.record_external_pnl_by_tag(tag, pnl)

        # Update previous prices for next day
        self.prev_prices = prices

    def _apply_corporate_actions(
        self,
        date: pd.Timestamp,
        actions_df: pd.DataFrame,
        positions: Dict[str, float],
        cash: float
    ) -> Tuple[Dict[str, float], float]:
        """
        Apply corporate actions (splits and dividends) to positions and cash.

        Corporate actions are applied BEFORE any trading:
        - Splits: Adjust share positions by split ratio
        - Dividends: Add cash payments based on positions held

        Parameters:
        -----------
        date : pd.Timestamp
            Current trading date (ex-date for corporate actions)
        actions_df : pd.DataFrame
            DataFrame with columns [action_type, value] and ticker as index
        positions : Dict[str, float]
            Current positions (will not be modified)
        cash : float
            Current cash balance

        Returns:
        --------
        Tuple[Dict[str, float], float]
            (adjusted_positions, adjusted_cash)
        """
        if actions_df.empty:
            return positions.copy(), cash

        # Work on copies to avoid modifying state
        adjusted_positions = positions.copy()
        adjusted_cash = cash

        # Process each corporate action
        for ticker, row in actions_df.iterrows():
            action_type = row['action_type']
            value = row['value']

            # Skip if we don't hold this ticker
            if ticker not in adjusted_positions or adjusted_positions[ticker] == 0:
                continue

            shares_held = adjusted_positions[ticker]

            if action_type == 'split':
                # Apply split: multiply shares by split ratio
                new_shares = shares_held * value
                adjusted_positions[ticker] = new_shares
                print(f"  Corporate Action: {ticker} {value:.2f}-for-1 split "
                      f"({shares_held:.2f} → {new_shares:.2f} shares)")

            elif action_type == 'dividend':
                # Apply dividend: add cash based on shares held
                dividend_received = shares_held * value
                adjusted_cash += dividend_received
                print(f"  Corporate Action: {ticker} ${value:.4f} dividend "
                      f"(${dividend_received:.2f} received on {shares_held:.2f} shares)")

        return adjusted_positions, adjusted_cash

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
            factor_exposures=self.state.factor_exposures,
            external_trade_pnl=self.state.external_trade_pnl,
            executed_trade_pnl=self.state.executed_trade_pnl,
            overnight_pnl=self.state.overnight_pnl,
            external_trades_by_tag=self.state.external_trades_by_tag,
            external_pnl_by_tag=self.state.external_pnl_by_tag
        )

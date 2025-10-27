# Use Case 3: External Trades with Multiple Executions

This document explains how to use Use Case 3 with external trades, supporting multiple trades per ticker with different quantities and prices.

## Overview

Use Case 3 allows you to input external trades (e.g., from an external system, manual trading, or algorithmic execution) and have the backtester:
1. Apply the external trades at specified execution prices
2. Optionally run optimization to satisfy risk constraints
3. Track PnL breakdown (external trades, internal rebalancing, overnight holding returns)
4. Analyze execution quality with comprehensive reporting and visualizations

## Input Format

External trades must be provided as a **list of dictionaries** for each ticker, where each dictionary contains 'qty' and 'price' keys.

You can either **manually specify trades** or **generate them from signals** using the trade generator.

### Example: Multiple Trades per Ticker

```python
inputs = {
    'external_trades': {
        pd.Timestamp('2023-01-03'): {
            'AAPL': [
                {'qty': 100, 'price': 150.25},   # First trade: buy 100 @ 150.25
                {'qty': 50, 'price': 150.50},    # Second trade: buy 50 @ 150.50
                {'qty': -25, 'price': 151.00}    # Third trade: sell 25 @ 151.00
            ],
            'MSFT': [
                {'qty': -30, 'price': 250.50},
                {'qty': -20, 'price': 250.75}
            ],
            'GOOGL': [
                {'qty': 75, 'price': 95.50}      # Single trade (still as list)
            ]
        },
        pd.Timestamp('2023-01-04'): {
            'AAPL': [
                {'qty': 200, 'price': 151.10},
                {'qty': -100, 'price': 151.20}
            ]
        }
    }
}
```

**Important Notes:**
- Each ticker MUST have a **list** of trade dictionaries
- Each trade dictionary MUST have `'qty'` and `'price'` keys
- Positive qty = buy, negative qty = sell
- Net position for ticker = sum of all trades
- Each trade's PnL is calculated separately
- Perfect for VWAP, TWAP, or algorithmic executions with multiple fills
- `external_exec_prices` parameter is deprecated and ignored

## Generating Trades from Signals

The framework provides utilities to automatically generate external trades from various signal types while accounting for current portfolio state.

### Quick Start: Convenience Function

The simplest way to generate trades is using the `generate_external_trades_from_signals()` function:

```python
from backtesting import generate_external_trades_from_signals

# Example: Generate from target weights
signals = {
    'AAPL': 0.30,   # 30% of portfolio
    'MSFT': 0.25,   # 25% of portfolio
    'GOOGL': -0.10  # -10% (short)
}

trades = generate_external_trades_from_signals(
    signals=signals,
    current_positions={'AAPL': 100, 'MSFT': 200},  # Current holdings
    close_prices={'AAPL': 150.0, 'MSFT': 250.0, 'GOOGL': 95.0},
    portfolio_value=100000,
    signal_type='weights',  # 'weights', 'positions', 'deltas', or 'scores'
    price_impact_bps=5.0,   # Slippage: 5 basis points
    num_fills=1             # Number of fills per ticker
)

# Result: trades ready for Use Case 3
# trades = {
#     'AAPL': [{'qty': 100, 'price': 150.075}],   # Buy to reach 30% weight
#     'MSFT': [{'qty': -100, 'price': 249.875}],  # Sell to reach 25% weight
#     'GOOGL': [{'qty': -105, 'price': 94.9525}]  # Short to reach -10% weight
# }
```

### Signal Types

#### 1. Target Weights (signal_type='weights')

Most common: specify desired portfolio weights.

```python
# Want 30% in AAPL, 20% in MSFT
target_weights = {'AAPL': 0.30, 'MSFT': 0.20}

trades = generate_external_trades_from_signals(
    signals=target_weights,
    current_positions={'AAPL': 50, 'MSFT': 100},
    close_prices=prices,
    portfolio_value=100000,
    signal_type='weights'
)
```

#### 2. Target Positions (signal_type='positions')

Specify exact share counts you want to hold.

```python
# Want to hold exactly 500 AAPL, 200 MSFT
target_positions = {'AAPL': 500, 'MSFT': 200}

trades = generate_external_trades_from_signals(
    signals=target_positions,
    current_positions={'AAPL': 300, 'MSFT': 250},
    close_prices=prices,
    portfolio_value=100000,
    signal_type='positions'
)
# Result: Buy 200 AAPL, Sell 50 MSFT
```

#### 3. Trade Deltas (signal_type='deltas')

Specify exact trade quantities (ignores current positions).

```python
# Want to buy 100 AAPL, sell 50 MSFT regardless of current holdings
trade_deltas = {'AAPL': 100, 'MSFT': -50}

trades = generate_external_trades_from_signals(
    signals=trade_deltas,
    current_positions={},  # Not used for deltas
    close_prices=prices,
    portfolio_value=100000,
    signal_type='deltas'
)
```

#### 4. Signal Scores (signal_type='scores')

Convert alpha signals or z-scores to trades using rank-based allocation.

```python
# Alpha scores (higher = more bullish)
alpha_scores = {
    'AAPL': 2.5,   # Strong buy signal
    'MSFT': 1.2,   # Moderate buy
    'GOOGL': -1.8, # Sell signal
    'TSLA': 0.3    # Weak buy
}

trades = generate_external_trades_from_signals(
    signals=alpha_scores,
    current_positions=current_holdings,
    close_prices=prices,
    portfolio_value=100000,
    signal_type='scores',
    target_notional=100000  # Total $ to allocate (can be > portfolio for leverage)
)
# Weights allocated proportional to |score|, direction from sign
```

### Advanced: ExternalTradeGenerator Class

For more control, use the `ExternalTradeGenerator` class directly:

```python
from backtesting import ExternalTradeGenerator, TradeGeneratorConfig

# Configure trade generation
config = TradeGeneratorConfig(
    price_impact_bps=5.0,        # Slippage in basis points
    use_random_fills=False,       # Deterministic vs random prices
    num_fills_per_ticker=5,       # Split into multiple fills (VWAP simulation)
    min_trade_size=1.0,           # Minimum shares to trade
    round_lots=True,              # Round to whole shares
    max_adv_participation=0.10    # Max 10% of ADV per ticker
)

generator = ExternalTradeGenerator(config)

# Generate from target positions with ADV constraints
trades = generator.from_target_positions(
    target_positions={'AAPL': 1000, 'MSFT': 500},
    current_positions={'AAPL': 500},
    close_prices=prices,
    adv={'AAPL': 10000000, 'MSFT': 5000000}  # Avg daily volume
)

# Result with 5 fills per ticker:
# trades = {
#     'AAPL': [
#         {'qty': 100, 'price': 150.00},
#         {'qty': 100, 'price': 150.0125},
#         {'qty': 100, 'price': 150.025},
#         {'qty': 100, 'price': 150.0375},
#         {'qty': 100, 'price': 150.05}
#     ],
#     ...
# }
```

### Multi-Day Trade Generation

Generate trades for multiple days from a time series of signals:

```python
from backtesting import ExternalTradeGenerator

generator = ExternalTradeGenerator()

# Create target positions by date
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
target_positions_by_date = {}

for date in dates:
    # Your signal logic here
    signals = calculate_signals(date)  # Returns dict of target positions
    target_positions_by_date[date] = signals

# Generate all trades
all_trades = generator.generate_multi_day_trades(
    dates=dates,
    target_positions_by_date=target_positions_by_date,
    prices_df=prices_df,  # DataFrame with dates as index, tickers as columns
    initial_positions={'AAPL': 100, 'MSFT': 50},
    adv_df=adv_df  # Optional: ADV constraints
)

# Result: trades for every day
# all_trades = {
#     pd.Timestamp('2023-01-03'): {'AAPL': [{'qty': 50, 'price': 150.0}], ...},
#     pd.Timestamp('2023-01-04'): {'MSFT': [{'qty': -25, 'price': 250.0}], ...},
#     ...
# }

# Use directly in backtest
results = backtester.run(
    use_case=3,
    inputs={'external_trades': all_trades}
)
```

### Complete Workflow Example

Putting it all together - generate trades from signals and run backtest:

```python
import pandas as pd
from backtesting import (
    Backtester, BacktestConfig, DataManager,
    generate_external_trades_from_signals
)

# 1. Setup
data_manager = DataManager('data')
backtester = Backtester(data_manager, BacktestConfig())

# 2. Calculate signals (your logic)
def calculate_daily_signals(date, prices, returns):
    """Your signal generation logic."""
    # Example: momentum signals
    signals = {}
    for ticker in prices.columns:
        mom_10d = returns[ticker].rolling(10).mean().loc[date]
        if mom_10d > 0.01:
            signals[ticker] = 0.2  # 20% weight
        elif mom_10d < -0.01:
            signals[ticker] = -0.1  # -10% weight
    return signals

# 3. Generate trades for all dates
prices_df = data_manager.get_prices()
returns_df = prices_df.pct_change()
dates = prices_df.index

all_external_trades = {}
current_positions = {}

for date in dates:
    # Calculate signals for this date
    signals = calculate_daily_signals(date, prices_df, returns_df)

    if not signals:
        continue

    # Get close prices and portfolio value
    close_prices = prices_df.loc[date].to_dict()
    portfolio_value = 1000000  # Starting capital

    # Generate trades
    daily_trades = generate_external_trades_from_signals(
        signals=signals,
        current_positions=current_positions,
        close_prices=close_prices,
        portfolio_value=portfolio_value,
        signal_type='weights',
        price_impact_bps=5.0,
        num_fills=3  # Simulate VWAP with 3 fills
    )

    if daily_trades:
        all_external_trades[date] = daily_trades

        # Update tracking
        for ticker, trades in daily_trades.items():
            total_qty = sum(t['qty'] for t in trades)
            current_positions[ticker] = current_positions.get(ticker, 0) + total_qty

# 4. Run backtest with generated trades
results = backtester.run(
    start_date=dates[0],
    end_date=dates[-1],
    use_case=3,
    inputs={'external_trades': all_external_trades}
)

# 5. Analyze results
print(results.calculate_metrics())
exec_quality = results.get_execution_quality_analysis(prices_df)
print("\nExecution Quality:")
print(exec_quality)

# Generate reports
results.generate_full_report('output', formats=['html', 'pdf'])
```

### Configuration Options

**TradeGeneratorConfig parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `price_impact_bps` | float | 5.0 | Slippage in basis points |
| `use_random_fills` | bool | False | Random vs deterministic fill prices |
| `num_fills_per_ticker` | int | 1 | Number of fills to generate per ticker |
| `min_trade_size` | float | 1.0 | Minimum shares to trade |
| `round_lots` | bool | True | Round to whole shares |
| `max_adv_participation` | float | None | Max fraction of ADV (e.g., 0.1 = 10%) |

**Price Impact Modeling:**

- **Deterministic mode** (`use_random_fills=False`):
  - Buys: price = close + slippage
  - Sells: price = close - slippage
  - Represents market impact (unfavorable execution)

- **Random mode** (`use_random_fills=True`):
  - Prices vary randomly around close
  - Simulates realistic execution variance
  - Useful for Monte Carlo analysis

**Multiple Fills:**

When `num_fills_per_ticker > 1`:
- Trade is split into equal parts
- Each fill gets slightly different price
- Simulates VWAP/TWAP algorithms
- Better models realistic execution

## Dynamic Trade Generation (Inside Simulation Loop)

For more realistic strategies, you can generate trades **dynamically during backtesting** based on the current portfolio state. This allows your signals to react to portfolio performance, positions, and market conditions in real-time.

### Method 1: Simple Callable Function

Pass a function that generates trades based on current state:

```python
def generate_daily_trades(context):
    """
    Generate trades based on current portfolio state.

    Parameters:
    -----------
    context : dict
        Contains: date, portfolio, prices, adv, portfolio_value,
                 dates, daily_returns, daily_pnl, etc.

    Returns:
    --------
    Dict[str, List[Dict]]
        External trades for this date
    """
    # Access current state
    date = context['date']
    current_positions = context['portfolio'].positions
    prices = context['prices']
    portfolio_value = context['portfolio_value']

    # Your signal logic here
    signals = {}

    # Example: Simple momentum
    if len(context['daily_returns']) >= 10:
        recent_return = sum(context['daily_returns'][-10:])

        if recent_return > 0.05:  # Portfolio up 5%
            signals = {'AAPL': 0.30, 'MSFT': 0.20}  # Increase exposure
        elif recent_return < -0.05:  # Portfolio down 5%
            signals = {'AAPL': 0.10, 'MSFT': 0.10}  # Reduce exposure

    if not signals:
        return {}

    # Convert signals to trades
    from backtesting import generate_external_trades_from_signals

    trades = generate_external_trades_from_signals(
        signals=signals,
        current_positions=current_positions,
        close_prices=prices,
        portfolio_value=portfolio_value,
        signal_type='weights'
    )

    return trades

# Use the function in backtest
results = backtester.run(
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_case=3,
    inputs={'external_trades': generate_daily_trades}  # Pass function, not dict!
)
```

### Method 2: Signal Generator Classes

Use pre-built signal generator classes for common patterns:

#### Quick Start with create_simple_signal_generator

```python
from backtesting import create_simple_signal_generator

def my_target_weights(context):
    """Calculate target weights based on current state."""
    # Your logic here
    portfolio_return = sum(context['daily_returns'][-30:]) if len(context['daily_returns']) >= 30 else 0

    if portfolio_return > 0.1:  # Strong performance
        return {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.2}
    else:
        return {'AAPL': 0.2, 'MSFT': 0.2, 'GOOGL': 0.1}

# Create generator
signal_gen = create_simple_signal_generator(
    signal_function=my_target_weights,
    signal_type='weights'  # 'weights', 'positions', or 'scores'
)

# Use in backtest
results = backtester.run(
    use_case=3,
    inputs={'external_trades': signal_gen}
)

# Access signal history
history = signal_gen.get_history()
print(history.head())
```

#### Target Weight Signal Generator

Most common: generate target portfolio weights.

```python
from backtesting import TargetWeightSignalGenerator, TradeGeneratorConfig

def calculate_target_weights(context):
    """Your weight calculation logic."""
    date = context['date']
    portfolio_value = context['portfolio_value']
    current_positions = context['portfolio'].positions

    # Example: Rebalance based on portfolio drift
    weights = {}
    for ticker, shares in current_positions.items():
        if ticker in context['prices']:
            current_weight = (shares * context['prices'][ticker]) / portfolio_value

            # If weight drifted too much, target original allocation
            if abs(current_weight - 0.2) > 0.05:
                weights[ticker] = 0.2

    return weights

# Configure trade generation
trade_config = TradeGeneratorConfig(
    price_impact_bps=5.0,
    num_fills_per_ticker=3,
    max_adv_participation=0.1
)

# Create signal generator
signal_gen = TargetWeightSignalGenerator(
    signal_function=calculate_target_weights,
    trade_generator_config=trade_config
)

# Run backtest
results = backtester.run(use_case=3, inputs={'external_trades': signal_gen})
```

#### Alpha Signal Generator

Convert alpha scores to trades.

```python
from backtesting import AlphaSignalGenerator

def calculate_alpha_scores(context):
    """Calculate alpha/z-scores for each ticker."""
    scores = {}

    # Example: Mean reversion signals
    prices = context['prices']

    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']:
        if ticker in prices:
            # Simplified mean reversion
            # In practice, you'd calculate proper z-scores
            score = np.random.randn()  # Your alpha model here
            scores[ticker] = score

    return scores

signal_gen = AlphaSignalGenerator(
    signal_function=calculate_alpha_scores,
    target_notional=1000000  # Allocate $1M based on scores
)

results = backtester.run(use_case=3, inputs={'external_trades': signal_gen})
```

#### Conditional Signal Generator

Only trade when certain conditions are met.

```python
from backtesting import ConditionalSignalGenerator

def should_trade(context):
    """Decide if we should trade today."""
    # Example: Only rebalance on month-end
    date = context['date']
    is_month_end = date.is_month_end

    # Or: Only trade if portfolio has drifted
    if context['portfolio_value'] > 0:
        return True

    return is_month_end

def my_signals(context):
    """Generate signals when condition is met."""
    return {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.15}

signal_gen = ConditionalSignalGenerator(
    signal_function=my_signals,
    condition_function=should_trade,
    signal_type='weights'
)

results = backtester.run(use_case=3, inputs={'external_trades': signal_gen})
```

### Context Dictionary Reference

The `context` dictionary passed to your signal functions contains:

| Key | Type | Description |
|-----|------|-------------|
| `date` | pd.Timestamp | Current simulation date |
| `portfolio` | Portfolio | Current portfolio state (positions, cash) |
| `prices` | Dict[str, float] | Close prices for current date |
| `adv` | Dict[str, float] | Average daily volume |
| `betas` | Dict[str, float] | Market betas |
| `sector_mapping` | Dict[str, str] | Ticker to sector mapping |
| `factor_loadings` | pd.DataFrame | Factor exposures |
| `factor_returns` | Dict[str, float] | Factor returns |
| `portfolio_value` | float | Current portfolio value |
| `dates` | List[pd.Timestamp] | Historical dates so far |
| `daily_returns` | List[float] | Historical daily returns |
| `daily_pnl` | List[float] | Historical daily PnL |

### Complete Example: Dynamic Rebalancing Strategy

```python
from backtesting import (
    Backtester, BacktestConfig, DataManager,
    TargetWeightSignalGenerator, TradeGeneratorConfig
)
import numpy as np

# Define strategy logic
def dynamic_weights(context):
    """
    Calculate target weights based on portfolio performance and volatility.
    """
    # Don't trade first 30 days (need history)
    if len(context['daily_returns']) < 30:
        return {}

    # Calculate recent metrics
    recent_returns = context['daily_returns'][-30:]
    recent_vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
    cumulative_return = np.prod([1 + r for r in recent_returns]) - 1

    # Get current positions
    current_positions = context['portfolio'].positions
    portfolio_value = context['portfolio_value']

    # Calculate current allocations
    current_allocations = {}
    for ticker, shares in current_positions.items():
        if ticker in context['prices']:
            value = shares * context['prices'][ticker]
            current_allocations[ticker] = value / portfolio_value

    # Adjust based on volatility
    if recent_vol > 0.3:  # High vol - reduce exposure
        target_leverage = 0.5
    elif recent_vol < 0.15:  # Low vol - increase exposure
        target_leverage = 1.2
    else:
        target_leverage = 1.0

    # Adjust based on performance
    if cumulative_return < -0.10:  # Down 10% - reduce risk
        target_leverage *= 0.7
    elif cumulative_return > 0.15:  # Up 15% - take profits
        target_leverage *= 0.8

    # Define base weights
    base_weights = {
        'AAPL': 0.25,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'AMZN': 0.15,
        'TSLA': 0.15
    }

    # Scale by target leverage
    target_weights = {k: v * target_leverage for k, v in base_weights.items()}

    # Only rebalance if drift is significant
    needs_rebalance = False
    for ticker, target_weight in target_weights.items():
        current_weight = current_allocations.get(ticker, 0)
        if abs(current_weight - target_weight) > 0.05:  # 5% drift
            needs_rebalance = True
            break

    return target_weights if needs_rebalance else {}

# Setup
data_manager = DataManager('data')
config = BacktestConfig(
    initial_cash=1000000,
    tc_fixed=0.001,  # 10 bps fixed transaction cost (0.001 = 10 bps)
    max_portfolio_variance=0.015
)
backtester = Backtester(config, data_manager)  # config first, then data_manager

# Create signal generator
trade_config = TradeGeneratorConfig(
    price_impact_bps=5.0,
    num_fills_per_ticker=3,
    max_adv_participation=0.10
)

signal_gen = TargetWeightSignalGenerator(
    signal_function=dynamic_weights,
    trade_generator_config=trade_config
)

# Run backtest
results = backtester.run(
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_case=3,
    inputs={'external_trades': signal_gen}
)

# Analyze
metrics = results.calculate_metrics()
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# View signal history
signal_history = signal_gen.get_history()
print(f"\nGenerated {len(signal_history)} signals")
print(signal_history.head())

# Detailed analysis
pnl_breakdown = results.get_pnl_breakdown_dataframe()
print("\nPnL Breakdown:")
print(pnl_breakdown.tail())
```

### Advantages of Dynamic Generation

1. **State-Aware**: React to portfolio performance, positions, volatility
2. **Realistic**: Mimics how traders actually make decisions
3. **Memory Efficient**: Don't need to pre-generate all trades
4. **Flexible**: Easy to implement complex strategies (mean reversion, momentum, risk parity)
5. **Interactive**: Can use current portfolio metrics to adjust strategy
6. **Conditional**: Only trade when conditions are met (month-end, significant drift)

### Tips

- **Start simple**: Use `create_simple_signal_generator()` for quick prototyping
- **Use history**: Access `context['daily_returns']`, `context['daily_pnl']` for lookback
- **Check positions**: Use `context['portfolio'].positions` to avoid overtrading
- **Handle edge cases**: Return `{}` when you don't want to trade
- **Test conditions**: Use `ConditionalSignalGenerator` for rule-based trading
- **Track signals**: Use `signal_gen.get_history()` to analyze your signals

## Complete Example

```python
from backtesting import Backtester, BacktestConfig, DataManager
import pandas as pd

# Load data
data_manager = DataManager('data/')

# Configure with risk limits
config = BacktestConfig(
    enable_beta_hedge=False,
    enable_sector_hedge=False,
    max_adv_participation=0.05,
    tc_coefficient=0.001,
    tc_power=1.5,
    # Risk constraints for optimization
    max_factor_exposure={'Momentum': 0.1, 'Value': 0.1},
    max_portfolio_variance=0.01
)

# Initialize backtester
backtester = Backtester(config, data_manager)

# Define external trades (mixing formats)
inputs = {
    'external_trades': {
        pd.Timestamp('2023-01-03'): {
            # Multiple fills for AAPL (VWAP execution)
            'AAPL': [
                {'qty': 500, 'price': 150.10},   # 9:30 AM
                {'qty': 300, 'price': 150.25},   # 10:00 AM
                {'qty': 200, 'price': 150.45},   # 11:00 AM
            ],
            # Multiple fills for MSFT
            'MSFT': [
                {'qty': -400, 'price': 250.80},  # Sell in two parts
                {'qty': -100, 'price': 250.60}
            ],
            # Single trade for GOOGL (using close price)
            'GOOGL': 150
        },
        pd.Timestamp('2023-01-04'): {
            'TSLA': [
                {'qty': 100, 'price': 180.50},
                {'qty': 100, 'price': 181.00},
                {'qty': 100, 'price': 181.50}
            ]
        }
    }
}

# Run backtest
results = backtester.run(
    start_date=pd.Timestamp('2023-01-03'),
    end_date=pd.Timestamp('2023-12-31'),
    use_case=3,  # External trades with optimization
    inputs=inputs
)

# Analyze results
print("\nTotal PnL:", sum(results.daily_pnl))
print("External Trade PnL:", sum(results.external_trade_pnl))
print("Executed Trade PnL:", sum(results.executed_trade_pnl))
print("Overnight PnL:", sum(results.overnight_pnl))

# Examine trade records
trades_df = results.get_trades_dataframe()
print("\nExternal trades:")
print(trades_df[trades_df['type'] == 'external'])
```

## PnL Breakdown

The framework now splits PnL into three components:

### 1. External Trade PnL

Captures the execution quality of external trades (slippage vs close price).

**Calculation:**
```
For each external trade:
  PnL = qty * (close_price - execution_price)
```

**Interpretation:**
- **Positive**: Execution was favorable (bought below close, sold above close)
- **Negative**: Execution was unfavorable (paid premium)
- **Zero**: Executed at close price

**Example:**
```python
# Buy 100 AAPL @ 150.25, close = 150.50
external_pnl = 100 * (150.50 - 150.25) = 25.00  # Good execution!

# Sell 50 MSFT @ 250.50, close = 250.75
external_pnl = -50 * (250.75 - 250.50) = -12.50  # Good execution!
```

### 2. Executed Trade PnL

PnL from internal rebalancing trades (optimization, hedging, ADV constraints).

**These trades are made by the backtester to:**
- Satisfy risk constraints (factor limits, variance limits)
- Apply hedging (beta, sector)
- Enforce ADV constraints

**Example:**
If external trades push you over factor exposure limits, the optimizer will create offsetting trades.

### 3. Overnight PnL

Pure holding returns from price changes.

**Calculation:**
```
For each position held:
  PnL = shares * (today_close - yesterday_close)
```

**Interpretation:**
- Market return component
- Independent of trading activity
- "Buy and hold" performance

### Verification

The three components should sum to total PnL (minus transaction costs):

```python
import pandas as pd

pnl_check = pd.DataFrame({
    'date': results.dates,
    'total_pnl': results.daily_pnl,
    'external_pnl': results.external_trade_pnl,
    'executed_pnl': results.executed_trade_pnl,
    'overnight_pnl': results.overnight_pnl,
    'transaction_costs': results.transaction_costs
})

pnl_check['calculated'] = (
    pnl_check['external_pnl'] +
    pnl_check['executed_pnl'] +
    pnl_check['overnight_pnl']
)

pnl_check['diff'] = pnl_check['total_pnl'] - pnl_check['calculated']

print(pnl_check)
# diff should be near zero (allowing for floating point)
```

## Analysis and Reporting

The framework provides comprehensive analysis methods and visualizations for external trades.

### Analysis Methods

#### 1. PnL Breakdown DataFrame

Get detailed PnL breakdown by date:

```python
pnl_df = results.get_pnl_breakdown_dataframe()
print(pnl_df.head())
```

Output:
```
        date  external_pnl  executed_pnl  overnight_pnl  total_pnl
0 2023-01-03        125.50        -45.20         234.10     314.40
1 2023-01-04        -67.30         12.50         156.20     101.40
...
```

#### 2. External Trades Summary

Get statistics by ticker:

```python
summary = results.get_external_trades_summary()
print(summary)
```

Output:
```
  ticker  num_trades  total_qty    vwap  avg_price  total_cost
0   AAPL          15      1500  150.35     150.40       45.25
1   MSFT           8      -800  250.55     250.60       28.15
2  GOOGL          12       900   95.45      95.50       32.40
...
```

Columns:
- `num_trades`: Number of individual fills
- `total_qty`: Net shares traded (positive = bought, negative = sold)
- `vwap`: Volume-weighted average price across all fills
- `avg_price`: Simple average of fill prices
- `total_cost`: Total transaction costs

#### 3. Execution Quality Analysis

Analyze execution vs close prices (requires passing close_prices):

```python
# You need to pass close prices from your data
close_prices = data_manager.get_prices()  # Your price DataFrame

execution_quality = results.get_execution_quality_analysis(close_prices)
print(execution_quality)
```

Output:
```
  ticker  total_qty    vwap  avg_close  slippage  slippage_pct  execution_pnl
0   AAPL       1500  150.35     150.50      0.15          0.10         225.00
1   MSFT       -800  250.55     250.40     -0.15         -0.06        -120.00
2  GOOGL        900   95.45      95.50      0.05          0.05          45.00
...
```

Columns:
- `slippage`: Price difference (positive = favorable execution)
- `slippage_pct`: Slippage as percentage of close price
- `execution_pnl`: PnL from execution quality (qty * (close - exec_price))

#### 4. External Trades by Date

Get daily trading summary:

```python
daily_trades = results.get_external_trades_by_date()
print(daily_trades)
```

Output:
```
        date  num_trades  num_tickers  total_notional  total_cost
0 2023-01-03          25           15      1,250,000       425.50
1 2023-01-04          18           12        875,000       315.25
...
```

### Visualizations

The framework automatically generates comprehensive charts for external trades when you call `results.generate_charts()`.

#### 1. PnL Breakdown Chart

Visualizes the three PnL components over time:
- **Top panel**: Cumulative PnL by component (line chart)
- **Bottom panel**: Daily PnL breakdown (stacked bar chart)

```python
# Automatically generated when external trades exist
results.generate_charts('output/charts', close_prices=close_prices)
```

Saved as: `pnl_breakdown.png`

#### 2. External Trades Analysis

Four-panel chart showing:
- **Trade count by ticker** (top 15)
- **Total notional by ticker** (top 15)
- **Daily trade volume** over time
- **Transaction costs by ticker** (top 15)

Saved as: `external_trades_analysis.png`

#### 3. Execution Quality Chart

Four-panel chart showing:
- **Slippage by ticker** (green = favorable, red = unfavorable)
- **VWAP vs Average Close Price** comparison
- **Execution PnL by ticker** (green = profit, red = loss)
- **Slippage vs Trade Size** scatter plot

Saved as: `execution_quality.png`

### Full Report Generation

Generate complete reports with all charts and analysis:

```python
# Run backtest
results = backtester.run(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31'),
    use_case=3,
    inputs={'external_trades': external_trades}
)

# Get close prices for execution quality analysis
close_prices = data_manager.get_prices()

# Generate full report with all formats
results.generate_full_report(
    output_dir='output/reports',
    formats=['html', 'pdf', 'excel', 'csv']
)

# Or just generate charts with close prices
results.generate_charts('output/charts', close_prices=close_prices)
```

The report will include:
- All standard performance metrics and charts
- PnL breakdown analysis and visualization
- External trades summary tables
- Execution quality analysis
- Trade-by-trade records

### Example: Complete Analysis Workflow

```python
# Run backtest
results = backtester.run(use_case=3, inputs={'external_trades': trades})

# 1. View PnL breakdown
pnl_df = results.get_pnl_breakdown_dataframe()
print("\nPnL Breakdown:")
print(pnl_df.describe())

# 2. Analyze external trades
summary = results.get_external_trades_summary()
print("\nExternal Trades Summary:")
print(summary)

# 3. Execution quality (requires close prices)
close_prices = data_manager.get_prices()
exec_quality = results.get_execution_quality_analysis(close_prices)
print("\nExecution Quality:")
print(exec_quality[['ticker', 'vwap', 'avg_close', 'slippage_pct', 'execution_pnl']])

# 4. Daily trading activity
daily = results.get_external_trades_by_date()
print("\nDaily Trading Summary:")
print(daily)

# 5. Generate all visualizations
results.generate_charts('output/charts', close_prices=close_prices)

# 6. Generate full reports
results.generate_full_report('output/reports', formats=['html', 'pdf', 'excel'])
```

## Use Cases

### 1. Algorithmic Execution Analysis

Track VWAP/TWAP fill quality:

```python
inputs = {
    'external_trades': {
        date: {
            'AAPL': [
                {'qty': 100, 'price': 150.10},  # VWAP fills
                {'qty': 100, 'price': 150.25},
                {'qty': 100, 'price': 150.40},
                {'qty': 100, 'price': 150.55},
                {'qty': 100, 'price': 150.70}
            ]
        }
    }
}

results = backtester.run(...)

# Calculate VWAP vs close
trades = results.get_trades_dataframe()
aapl_trades = trades[(trades['ticker'] == 'AAPL') & (trades['date'] == date)]
vwap = (aapl_trades['quantity'] * aapl_trades['price']).sum() / aapl_trades['quantity'].sum()
close = prices.loc[date, 'AAPL']

print(f"VWAP: {vwap:.2f}, Close: {close:.2f}, Slippage: {(vwap - close):.2f}")
```

### 2. Manual Trading + Optimization

Manual trades followed by automated risk management:

```python
# You manually trade during the day
manual_trades = {
    date: {
        'AAPL': [
            {'qty': 500, 'price': 150.25},   # Manual buy
            {'qty': -200, 'price': 151.00}   # Partial sell
        ],
        'MSFT': [
            {'qty': 300, 'price': 250.50}
        ]
    }
}

# Backtester automatically rebalances to meet risk limits
config = BacktestConfig(
    max_factor_exposure={'Momentum': 0.1},  # Risk limit
    max_portfolio_variance=0.01
)

results = backtester.run(use_case=3, inputs={'external_trades': manual_trades})

# See what the optimizer did
optimizer_trades = results.get_trades_dataframe()
optimizer_trades = optimizer_trades[optimizer_trades['type'] == 'internal']
print("Optimizer adjustments:")
print(optimizer_trades)
```

### 3. External Signal + Execution Feed

Integrate with external execution system:

```python
# Load fills from execution broker
execution_fills = load_fills_from_broker()  # Your function

# Convert to backtester format
external_trades = {}
for fill in execution_fills:
    date = fill['date']
    ticker = fill['ticker']

    if date not in external_trades:
        external_trades[date] = {}
    if ticker not in external_trades[date]:
        external_trades[date][ticker] = []

    external_trades[date][ticker].append({
        'qty': fill['quantity'],
        'price': fill['fill_price']
    })

# Backtest with actual fills
inputs = {'external_trades': external_trades}
results = backtester.run(use_case=3, inputs=inputs)
```

## Trade Records

All trades are recorded with detailed information:

```python
trades_df = results.get_trades_dataframe()
print(trades_df.columns)
# ['date', 'ticker', 'quantity', 'price', 'cost', 'type']

# Filter by type
external = trades_df[trades_df['type'] == 'external']
internal = trades_df[trades_df['type'] == 'internal']

# Analyze costs
print(f"External trade costs: ${external['cost'].sum():.2f}")
print(f"Internal trade costs: ${internal['cost'].sum():.2f}")
```

## Best Practices

### 1. Validate Input Data

```python
def validate_external_trades(external_trades):
    """Validate external trades format."""
    for date, trades in external_trades.items():
        for ticker, trade_data in trades.items():
            if isinstance(trade_data, list):
                for trade in trade_data:
                    assert 'qty' in trade, f"Missing 'qty' for {ticker} on {date}"
                    assert 'price' in trade, f"Missing 'price' for {ticker} on {date}"
                    assert trade['price'] > 0, f"Invalid price for {ticker} on {date}"
    return True
```

### 2. Handle Missing Prices

```python
# Ensure all tickers have prices
for date, trades in inputs['external_trades'].items():
    for ticker in trades.keys():
        if ticker not in prices.columns:
            print(f"Warning: Missing price data for {ticker}")
```

### 3. Monitor PnL Components

```python
# Create time series analysis
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Cumulative PnL breakdown
axes[0].plot(results.dates, np.cumsum(results.external_trade_pnl), label='External')
axes[0].plot(results.dates, np.cumsum(results.executed_trade_pnl), label='Executed')
axes[0].plot(results.dates, np.cumsum(results.overnight_pnl), label='Overnight')
axes[0].legend()
axes[0].set_title('Cumulative PnL Components')

# Daily PnL components
width = 0.8
axes[1].bar(results.dates, results.external_trade_pnl, width, label='External', alpha=0.7)
axes[1].bar(results.dates, results.executed_trade_pnl, width, label='Executed', alpha=0.7, bottom=results.external_trade_pnl)
overnight_bottom = [e + x for e, x in zip(results.external_trade_pnl, results.executed_trade_pnl)]
axes[1].bar(results.dates, results.overnight_pnl, width, label='Overnight', alpha=0.7, bottom=overnight_bottom)
axes[1].legend()
axes[1].set_title('Daily PnL Breakdown')

plt.tight_layout()
plt.show()
```

### 4. Execution Quality Analysis

```python
def analyze_execution_quality(results):
    """Analyze execution vs close prices."""
    trades = results.get_trades_dataframe()
    external = trades[trades['type'] == 'external']

    # Group by ticker
    for ticker in external['ticker'].unique():
        ticker_trades = external[external['ticker'] == ticker]

        total_qty = ticker_trades['quantity'].sum()
        vwap = (ticker_trades['quantity'] * ticker_trades['price']).sum() / total_qty

        # Get average close price (weighted by trade date)
        # This is simplified - you'd want to weight by quantity per date

        print(f"{ticker}:")
        print(f"  Total quantity: {total_qty}")
        print(f"  VWAP: ${vwap:.2f}")
        print(f"  Number of fills: {len(ticker_trades)}")
        print(f"  Price range: ${ticker_trades['price'].min():.2f} - ${ticker_trades['price'].max():.2f}")
```

## Troubleshooting

### Issue: "External trades not being applied"

**Check:**
1. Date format matches data dates
2. Ticker symbols match price data
3. Trade format is correct (dict with 'qty' and 'price')

### Issue: "PnL components don't sum to total"

**This is expected!** Transaction costs are separate:
```python
total_pnl ≈ external_pnl + executed_pnl + overnight_pnl - transaction_costs
```

### Issue: "Optimizer creating large offsetting trades"

**This means:** External trades violate risk constraints.

**Solutions:**
1. Adjust risk limits in config
2. Review external trade sizes
3. Check factor exposures of external trades

## Advanced: Custom Position Sizing

Combine external trades with dynamic position sizing:

```python
# External trades provide signal, backtester sizes positions
for date in trade_dates:
    # Get external trade signal
    external_signal = external_trades[date]

    # Backtester applies:
    # 1. External trades
    # 2. Risk-based position sizing (optimization)
    # 3. Hedging
    # 4. ADV constraints

    # Result: Risk-managed execution of external signals
```

## Summary

### Input Format

All external trades must use **list format**:

```python
{
    'ticker': [
        {'qty': shares, 'price': execution_price},
        {'qty': shares, 'price': execution_price},
        ...
    ]
}
```

- Each ticker requires a list of trade dictionaries
- Each trade must have 'qty' and 'price' keys
- Perfect for algorithmic fills, VWAP/TWAP executions, multiple fills

### PnL Breakdown

```
Total PnL = External Trade PnL + Executed Trade PnL + Overnight PnL - Transaction Costs
```

- **External Trade PnL**: Execution quality vs close price
- **Executed Trade PnL**: Internal rebalancing to satisfy risk constraints
- **Overnight PnL**: Holding returns from price changes

### Analysis Methods

The framework provides:
- `get_pnl_breakdown_dataframe()` - Daily PnL components
- `get_external_trades_summary()` - Statistics by ticker (VWAP, costs, counts)
- `get_execution_quality_analysis()` - Slippage and execution PnL analysis
- `get_external_trades_by_date()` - Daily trading summary

### Visualizations

Automatically generated charts:
- **PnL Breakdown**: Cumulative and daily components
- **External Trades Analysis**: Trade counts, notional, costs by ticker
- **Execution Quality**: Slippage, VWAP vs close, PnL by ticker

### Trade Generation

The framework supports **two modes** of trade generation:

#### Mode 1: Static (Pre-Generated)
Generate all trades upfront before backtest:
```python
trades = generate_external_trades_from_signals(
    signals={'AAPL': 0.3, 'MSFT': 0.2},
    current_positions={'AAPL': 100},
    close_prices=prices,
    portfolio_value=100000,
    signal_type='weights'
)
results = backtester.run(use_case=3, inputs={'external_trades': trades})
```

#### Mode 2: Dynamic (Inside Simulation Loop)
Generate trades based on current portfolio state during backtest:
```python
def generate_trades(context):
    # Access current state
    portfolio_value = context['portfolio_value']
    current_positions = context['portfolio'].positions

    # Your strategy logic here
    if sum(context['daily_returns'][-10:]) > 0.05:
        signals = {'AAPL': 0.3, 'MSFT': 0.2}
    else:
        signals = {'AAPL': 0.1, 'MSFT': 0.1}

    return generate_external_trades_from_signals(
        signals, current_positions, context['prices'],
        portfolio_value, signal_type='weights'
    )

# Pass function directly
results = backtester.run(use_case=3, inputs={'external_trades': generate_trades})
```

Or use signal generator classes:
```python
from backtesting import create_simple_signal_generator

signal_gen = create_simple_signal_generator(
    signal_function=lambda ctx: {'AAPL': 0.3, 'MSFT': 0.2},
    signal_type='weights'
)
results = backtester.run(use_case=3, inputs={'external_trades': signal_gen})
```

**Four signal types supported:**
- `'weights'` - Target portfolio weights (most common)
- `'positions'` - Target share counts
- `'deltas'` - Direct trade quantities
- `'scores'` - Alpha signals converted to trades

**Dynamic generation advantages:**
- React to portfolio performance and positions
- Implement conditional trading logic
- Access full backtest history
- More realistic strategy modeling

### Key Features

- Multiple fills per ticker on same day
- Individual execution prices per fill
- **Automatic trade generation from signals**
- Comprehensive execution quality analysis
- Automatic risk management via optimization
- Detailed PnL attribution
- Professional reporting and visualization

## Related Documentation

- [README.md](README.md) - Main framework documentation
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [notebooks/04_use_case_3_risk_managed_portfolio.ipynb](notebooks/04_use_case_3_risk_managed_portfolio.ipynb) - Example notebook

# Corporate Actions Handling

## Overview

The backtester handles corporate actions (stock splits and cash dividends) by applying them **inside the backtesting loop** rather than adjusting historical prices. This approach:

- Maintains raw, unadjusted prices
- Updates share positions for splits/reverse splits
- Updates cash balance for dividend payments
- Applies actions on their ex-date before any trading

## Data Format

Corporate actions are stored in a CSV file with the following format:

```csv
date,ticker,action_type,value
2023-06-15,AAPL,split,2.0
2023-09-01,MSFT,dividend,0.75
2023-12-15,GOOGL,split,0.5
```

### Columns

- **date**: Ex-date for the corporate action (YYYY-MM-DD)
- **ticker**: Security ticker symbol
- **action_type**: Either `'split'` or `'dividend'`
- **value**:
  - For splits: Split ratio (e.g., 2.0 = 2-for-1 split, 0.5 = 1-for-2 reverse split)
  - For dividends: Cash amount per share in dollars

## How It Works

### 1. Data Loading

```python
from backtesting import DataManager

data_manager = DataManager('./data')
data_manager.load_prices()  # Load raw, unadjusted prices
data_manager.load_adv()
data_manager.load_corporate_actions()  # Load corporate actions
```

The corporate actions are stored as a DataFrame with MultiIndex `(date, ticker)`:

```python
                  action_type  value
date       ticker
2023-06-15 AAPL         split   2.00
2023-09-01 MSFT      dividend   0.75
```

### 2. Backtesting Loop

On each trading day, the backtester:

1. **Gets data for the date** (including any corporate actions)
2. **Applies corporate actions BEFORE trading**:
   - Splits: Multiply share positions by split ratio
   - Dividends: Add cash to account based on shares held
3. **Executes trades** at raw prices
4. **Updates portfolio value**

### 3. Corporate Action Application

The backtester applies corporate actions directly to the portfolio state:

```python
# Inside _simulate_day method:
day_data = self.data_manager.get_data_for_date(date)

if 'corporate_actions' in day_data:
    self._apply_corporate_actions(date, day_data['corporate_actions'])
```

The `_apply_corporate_actions` method:

```python
def _apply_corporate_actions(self, date, actions_df):
    positions = self.state.portfolio.positions
    cash = self.state.portfolio.cash

    for ticker, row in actions_df.iterrows():
        action_type = row['action_type']
        value = row['value']

        if ticker not in positions or positions[ticker] == 0:
            continue

        shares_held = positions[ticker]

        if action_type == 'split':
            # Multiply shares by split ratio
            positions[ticker] = shares_held * value

        elif action_type == 'dividend':
            # Add cash based on shares held
            cash += shares_held * value

    self.state.portfolio.cash = cash
```

## Example Usage

### Creating Corporate Actions File

```python
import pandas as pd

# Create corporate actions
actions = pd.DataFrame([
    {'date': '2023-06-15', 'ticker': 'AAPL', 'action_type': 'split', 'value': 2.0},
    {'date': '2023-09-01', 'ticker': 'MSFT', 'action_type': 'dividend', 'value': 0.75},
    {'date': '2023-12-15', 'ticker': 'GOOGL', 'action_type': 'split', 'value': 3.0},
])

actions.to_csv('data/corporate_actions.csv', index=False)
```

### Running Backtest with Corporate Actions

```python
from backtesting import Backtester, BacktestConfig, DataManager

# Load data
data_manager = DataManager('./data')
data_manager.load_prices()
data_manager.load_adv()
data_manager.load_corporate_actions()  # Will use raw prices

# Configure backtest
config = BacktestConfig(
    initial_cash=1_000_000,
    # ... other settings
)

# Run backtest
backtester = Backtester(config, data_manager)
results = backtester.run(...)
```

During the backtest, when the date reaches 2023-06-15, the backtester will:
1. Check for corporate actions on that date
2. Find AAPL 2-for-1 split
3. Double the AAPL share position
4. Continue with regular trading at raw prices

## Corporate Action Types

### Stock Splits

**2-for-1 Split** (value = 2.0):
- Before: 100 shares @ $200 = $20,000
- After: 200 shares @ $100 = $20,000 (same value, doubled shares)

**3-for-1 Split** (value = 3.0):
- Before: 100 shares @ $300 = $30,000
- After: 300 shares @ $100 = $30,000

**1-for-2 Reverse Split** (value = 0.5):
- Before: 200 shares @ $50 = $10,000
- After: 100 shares @ $100 = $10,000 (same value, halved shares)

### Cash Dividends

**$0.75 Dividend** (value = 0.75):
- Shares held: 200
- Cash received: 200 × $0.75 = $150
- Shares remain: 200 (unchanged)
- Cash balance increases by $150

## Important Notes

### 1. Use Raw Prices

Corporate actions work with **raw, unadjusted prices**. Do not use backward-adjusted prices.

```python
# CORRECT - Raw prices
prices = loader.load_prices_with_adjustments(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    apply_adjustments=False  # Use raw prices
)

# INCORRECT - Adjusted prices
prices = loader.load_prices_with_adjustments(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    apply_adjustments=True  # Don't do this with corporate actions
)
```

### 2. Ex-Date Timing

Corporate actions are applied on the **ex-date** before any trading:
- Day starts → Apply corporate actions → Execute trades → Day ends

### 3. Only Affects Held Positions

Corporate actions only apply to securities currently held in the portfolio. If you don't own a security when a split/dividend occurs, nothing happens.

### 4. Position Tracking

After a split, your position will show the new share count, but the total market value remains the same (shares change, price per share changes proportionally).

## Converting Adjustment Factors

If you have historical adjustment factors (for backward-adjusted prices), you can convert them:

```python
# Adjustment factor format (backward adjustment)
adjustments_df = pd.DataFrame([
    {'date': '2023-06-15', 'ticker': 'AAPL', 'adjustment_factor': 0.5}
])

# Convert to corporate action format (forward application)
# adjustment_factor = 0.5 → split_ratio = 2.0 (2-for-1 split)
actions_df = adjustments_df.copy()
actions_df['action_type'] = 'split'
actions_df['value'] = 1.0 / actions_df['adjustment_factor']
actions_df = actions_df.drop('adjustment_factor', axis=1)

actions_df.to_csv('data/corporate_actions.csv', index=False)
```

## Advantages of This Approach

1. **Realistic Simulation**: Mimics how corporate actions actually affect portfolios
2. **Raw Prices**: Works with unadjusted historical prices
3. **Clear Attribution**: Can see exactly when and how corporate actions affected P&L
4. **Cash Tracking**: Dividend payments directly visible in cash balance
5. **Position History**: Share count changes are part of the backtest history

## Limitations

- Corporate actions file must be manually created/maintained
- Only handles splits and cash dividends (not stock dividends, mergers, etc.)
- Assumes you have accurate corporate action data

## See Also

- `test_simple_corporate_actions.py` - Test examples
- `DataManager.load_corporate_actions()` - Loading method
- `Backtester._apply_corporate_actions()` - Application logic

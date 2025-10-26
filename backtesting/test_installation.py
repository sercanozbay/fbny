#!/usr/bin/env python
"""
Quick test script to verify the backtesting framework is working correctly.

Run this after installation to check everything is set up properly.
"""

import sys
import traceback

print("=" * 70)
print("Backtesting Framework - Installation Test")
print("=" * 70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    import pandas as pd
    import numpy as np
    from backtesting import Backtester, BacktestConfig, DataManager
    from backtesting.utils import get_date_range
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Generate sample data
print("\n2. Generating sample data...")
try:
    from generate_sample_data import generate_sample_data
    generate_sample_data(
        n_securities=50,  # Small for quick test
        n_days=20,        # Just 20 days
        n_factors=3,
        output_dir='./sample_data',
        seed=42
    )
    print("   ✓ Sample data generated")
except Exception as e:
    print(f"   ✗ Data generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load data
print("\n3. Loading data...")
try:
    data_manager = DataManager(data_dir='./sample_data', use_float32=True)
    prices = data_manager.load_prices()
    print(f"   ✓ Loaded {prices.shape[1]} securities, {prices.shape[0]} days")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Date alignment
print("\n4. Testing date alignment...")
try:
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    aligned_start, aligned_end = get_date_range(prices, start_date, end_date)
    print(f"   ✓ Dates aligned: {aligned_start.date()} to {aligned_end.date()}")
except Exception as e:
    print(f"   ✗ Date alignment failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Run backtest
print("\n5. Running quick backtest...")
try:
    # Configure
    config = BacktestConfig(
        initial_cash=1_000_000,
        max_adv_participation=0.05,
        enable_beta_hedge=False,
        risk_free_rate=0.02
    )

    # Prepare inputs
    target_weights = pd.read_csv('./sample_data/target_weights.csv',
                                  index_col=0, parse_dates=True)
    target_weights = target_weights.loc[aligned_start:aligned_end]

    targets_by_date = {
        date: target_weights.loc[date].to_dict()
        for date in target_weights.index
    }

    inputs = {
        'type': 'weights',
        'targets': targets_by_date
    }

    # Run backtest
    backtester = Backtester(config, data_manager)
    results = backtester.run(
        start_date=aligned_start,
        end_date=aligned_end,
        use_case=1,
        inputs=inputs,
        show_progress=False  # Quiet mode for test
    )

    print(f"   ✓ Backtest completed successfully")
except Exception as e:
    print(f"   ✗ Backtest failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Calculate metrics
print("\n6. Calculating metrics...")
try:
    metrics = results.calculate_metrics()
    print(f"   ✓ Metrics calculated")
    print(f"      - Total Return: {metrics['total_return']:.2%}")
    print(f"      - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"      - Max Drawdown: {metrics['max_drawdown']:.2%}")
except Exception as e:
    print(f"   ✗ Metrics calculation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Generate report
print("\n7. Testing report generation...")
try:
    results.generate_full_report(
        output_dir='./output/test_run',
        formats=['html', 'csv']  # Skip excel for speed
    )
    print(f"   ✓ Reports generated in ./output/test_run")
except Exception as e:
    print(f"   ✗ Report generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe backtesting framework is installed and working correctly.")
print("\nNext steps:")
print("  1. Check the reports in ./output/test_run")
print("  2. Run the example notebooks in ./notebooks")
print("  3. Try with your own data")
print("\nFor help, see:")
print("  - README.md for full documentation")
print("  - QUICKSTART.md for quick start guide")
print("  - TROUBLESHOOTING.md for common issues")
print("\nHappy backtesting! 🚀")

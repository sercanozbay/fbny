#!/usr/bin/env python
"""
Check what dates are actually in your data.

Run this to see what date range you should use for backtesting.
"""

import pandas as pd
import os

print("=" * 70)
print("DATA DATE CHECKER")
print("=" * 70)

# Check if sample data exists
if not os.path.exists('./sample_data/prices.csv'):
    print("\n❌ Sample data not found!")
    print("\nRun this first:")
    print("  python generate_sample_data.py")
    exit(1)

# Load prices
print("\n📅 Loading price data...")
prices = pd.read_csv('./sample_data/prices.csv', index_col=0, parse_dates=True)

print(f"\n✓ Found {len(prices)} trading days")
print(f"✓ Found {len(prices.columns)} securities")

print("\n" + "=" * 70)
print("DATE RANGE IN YOUR DATA:")
print("=" * 70)
print(f"\nFirst date: {prices.index[0].date()}")
print(f"Last date:  {prices.index[-1].date()}")

print("\n" + "=" * 70)
print("FIRST 10 DATES:")
print("=" * 70)
for i, date in enumerate(prices.index[:10], 1):
    day_name = date.strftime('%A')
    print(f"{i:2d}. {date.date()} ({day_name})")

print("\n" + "=" * 70)
print("LAST 10 DATES:")
print("=" * 70)
for i, date in enumerate(prices.index[-10:], len(prices)-9):
    day_name = date.strftime('%A')
    print(f"{i:3d}. {date.date()} ({day_name})")

print("\n" + "=" * 70)
print("HOW TO USE THESE DATES:")
print("=" * 70)
print("\n✓ OPTION 1: Use actual dates from data (RECOMMENDED)")
print("   start_date = prices.index[0]")
print("   end_date = prices.index[-1]")

print("\n✓ OPTION 2: Use date alignment utility")
print("   from backtesting.utils import get_date_range")
print("   aligned_start, aligned_end = get_date_range(prices, start_date, end_date)")

print("\n✓ OPTION 3: Pick specific dates that you know exist")
print(f"   start_date = pd.Timestamp('{prices.index[0].date()}')")
print(f"   end_date = pd.Timestamp('{prices.index[-1].date()}')")

print("\n" + "=" * 70)
print("READY TO RUN!")
print("=" * 70)
print("\nNow run one of these:")
print("  python simple_working_example.py")
print("  python quick_fix_example.py")
print("  python test_installation.py")

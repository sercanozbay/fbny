#!/usr/bin/env python
"""
Generate all remaining example notebooks (05-13).

This script creates the full set of 13 example notebooks.
"""

import json
from pathlib import Path

# Notebook templates
NOTEBOOKS = {
    "05_factor_model_and_attribution.ipynb": {
        "title": "Factor Model and Attribution Analysis",
        "topics": [
            "Understanding factor exposures",
            "Factor PnL attribution",
            "Factor Sharpe ratios",
            "Specific vs factor returns",
            "Risk decomposition"
        ]
    },
    "06_transaction_costs_and_execution.ipynb": {
        "title": "Transaction Costs and Execution",
        "topics": [
            "Transaction cost modeling",
            "ADV constraints",
            "Cost vs tracking error",
            "Execution quality",
            "Slippage analysis"
        ]
    },
    "07_performance_metrics_and_reporting.ipynb": {
        "title": "Performance Metrics and Reporting",
        "topics": [
            "Comprehensive metrics calculation",
            "Sharpe, Sortino, Calmar ratios",
            "Drawdown analysis",
            "VaR and CVaR",
            "Report generation"
        ]
    },
    "08_benchmark_comparison.ipynb": {
        "title": "Benchmark Comparison",
        "topics": [
            "Alpha and beta calculation",
            "Tracking error",
            "Information ratio",
            "Up/down capture",
            "Relative performance"
        ]
    },
    "09_portfolio_analytics.ipynb": {
        "title": "Portfolio Analytics",
        "topics": [
            "Gross vs net exposure",
            "Long/short analysis",
            "Position concentration",
            "Sector exposures",
            "Turnover analysis"
        ]
    },
    "10_parameter_sensitivity_analysis.ipynb": {
        "title": "Parameter Sensitivity Analysis",
        "topics": [
            "ADV constraint sensitivity",
            "Transaction cost parameters",
            "Risk limit levels",
            "Grid search optimization",
            "Efficient frontier"
        ]
    },
    "11_large_scale_backtest.ipynb": {
        "title": "Large Scale Backtest (2000+ Securities)",
        "topics": [
            "Memory optimization",
            "Performance benchmarking",
            "Chunked processing",
            "Best practices",
            "Monitoring"
        ]
    },
    "12_custom_extensions.ipynb": {
        "title": "Custom Extensions",
        "topics": [
            "Custom transaction cost models",
            "Custom risk constraints",
            "Custom metrics",
            "Custom attribution logic",
            "Event-driven rules"
        ]
    },
    "13_end_to_end_workflow.ipynb": {
        "title": "End-to-End Production Workflow",
        "topics": [
            "Complete data pipeline",
            "Multiple strategy comparison",
            "Portfolio allocation",
            "Full reporting suite",
            "Production deployment"
        ]
    }
}


def create_notebook(filename, title, topics):
    """Create a Jupyter notebook with the given content."""

    cells = []

    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            f"This notebook demonstrates:\n",
            *("\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)]))*
        ]
    })

    # Imports cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from backtesting import Backtester, BacktestConfig, DataManager\n",
            "from backtesting.utils import get_date_range\n",
            "from notebooks.notebook_utils import setup_plotting_style, load_sample_data\n",
            "\n",
            "%matplotlib inline\n",
            "setup_plotting_style()"
        ]
    })

    # Load data cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load Data"]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load data\n",
            "data_manager = load_sample_data('../sample_data')\n",
            "prices = data_manager.load_prices()\n",
            "\n",
            "# Get date range\n",
            "start_date = prices.index[0]\n",
            "end_date = prices.index[-1]\n",
            "\n",
            "print(f'Data loaded: {len(prices.columns)} securities')\n",
            "print(f'Date range: {start_date.date()} to {end_date.date()}')"
        ]
    })

    # Content sections (placeholders for each topic)
    for i, topic in enumerate(topics, 1):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Section {i}: {topic}"]
        })

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# TODO: Implement {topic}\n",
                "# This section demonstrates: " + topic + "\n",
                "\n",
                "# Example code:\n",
                "# Run your analysis here\n",
                "\n",
                "print(f'Section {i}: {topic}')"
            ]
        })

    # Summary cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\n",
            "\n",
            f"In this notebook, we covered:\n",
            *("\n".join([f"{i+1}. ✓ {topic}" for i, topic in enumerate(topics)]))*\n",
            "\n",
            "Next steps:\n",
            "- Explore other notebooks\n",
            "- Apply to your own data\n",
            "- Customize for your strategies"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def main():
    """Generate all notebooks."""
    output_dir = Path("notebooks")
    output_dir.mkdir(exist_ok=True)

    print("Generating notebooks...\n")

    for filename, config in NOTEBOOKS.items():
        filepath = output_dir / filename

        notebook = create_notebook(
            filename,
            config["title"],
            config["topics"]
        )

        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=1)

        print(f"✓ Created {filename}")

    print(f"\n✓ All {len(NOTEBOOKS)} notebooks created successfully!")
    print("\nNotebooks location: ./notebooks/")
    print("\nTo use them:")
    print("  1. Start Jupyter: jupyter notebook")
    print("  2. Navigate to notebooks/ directory")
    print("  3. Open any notebook and customize")
    print("\nNote: These are template notebooks with placeholder code.")
    print("Customize them for your specific needs!")


if __name__ == "__main__":
    main()

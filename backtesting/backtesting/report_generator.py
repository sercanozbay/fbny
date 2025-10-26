"""
Report generation module.

This module creates HTML, Excel, PDF, and text reports from backtest results.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np


class ReportGenerator:
    """
    Generate reports from backtest results.

    Creates HTML, Excel, and text-based reports.
    """

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate_html_report(
        self,
        metrics: Dict[str, float],
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        chart_dir: Optional[str],
        output_path: str
    ):
        """
        Generate HTML report.

        Parameters:
        -----------
        metrics : Dict[str, float]
            Performance metrics
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        chart_dir : str, optional
            Directory containing chart images
        output_path : str
            Path to save HTML report
        """
        html_content = self._create_html_template(metrics, dates, portfolio_values, chart_dir)

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved to {output_path}")

    def _create_html_template(
        self,
        metrics: Dict[str, float],
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        chart_dir: Optional[str]
    ) -> str:
        """Create HTML template."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-value {
            font-weight: bold;
            color: #4CAF50;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>

        <h2>Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
        # Add metrics
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                if 'return' in key.lower() or 'ratio' in key.lower():
                    formatted_value = f"{value:.4f}"
                elif 'rate' in key.lower():
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            html += f"            <tr><td>{formatted_key}</td><td class='metric-value'>{formatted_value}</td></tr>\n"

        html += """
        </table>

        <h2>Performance Period</h2>
        <p>Start Date: <strong>{}</strong></p>
        <p>End Date: <strong>{}</strong></p>
        <p>Initial Value: <strong>${:,.2f}</strong></p>
        <p>Final Value: <strong>${:,.2f}</strong></p>
""".format(
            dates[0].strftime('%Y-%m-%d') if dates else 'N/A',
            dates[-1].strftime('%Y-%m-%d') if dates else 'N/A',
            portfolio_values[0] if portfolio_values else 0,
            portfolio_values[-1] if portfolio_values else 0
        )

        # Add charts if available
        if chart_dir:
            chart_path = Path(chart_dir)
            if chart_path.exists():
                html += "\n        <h2>Charts</h2>\n"

                chart_files = [
                    ('cumulative_returns.png', 'Cumulative Returns'),
                    ('drawdown_underwater.png', 'Drawdown'),
                    ('rolling_sharpe.png', 'Rolling Sharpe Ratio'),
                    ('return_distribution.png', 'Return Distribution'),
                    ('exposures.png', 'Exposures'),
                    ('transaction_costs.png', 'Transaction Costs'),
                    ('factor_attribution.png', 'Factor Attribution')
                ]

                for filename, title in chart_files:
                    chart_file = chart_path / filename
                    if chart_file.exists():
                        html += f"""
        <div class="chart">
            <h3>{title}</h3>
            <img src="{filename}" alt="{title}">
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def generate_excel_report(
        self,
        metrics: Dict[str, float],
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        daily_pnl: List[float],
        gross_exposures: List[float],
        net_exposures: List[float],
        transaction_costs: List[float],
        trades: List[Dict],
        output_path: str
    ):
        """
        Generate Excel report with multiple sheets.

        Parameters:
        -----------
        metrics : Dict[str, float]
            Performance metrics
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        daily_pnl : List[float]
            Daily PnL
        gross_exposures : List[float]
            Gross exposures
        net_exposures : List[float]
            Net exposures
        transaction_costs : List[float]
            Transaction costs
        trades : List[Dict]
            Trade records
        output_path : str
            Path to save Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([metrics])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Performance sheet
            perf_df = pd.DataFrame({
                'Date': dates,
                'Portfolio_Value': portfolio_values,
                'Daily_PnL': daily_pnl,
                'Gross_Exposure': gross_exposures,
                'Net_Exposure': net_exposures,
                'Transaction_Costs': transaction_costs
            })
            perf_df.to_excel(writer, sheet_name='Performance', index=False)

            # Trades sheet
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_excel(writer, sheet_name='Trades', index=False)

        print(f"Excel report saved to {output_path}")

    def print_summary(self, metrics: Dict[str, float]):
        """
        Print summary to console.

        Parameters:
        -----------
        metrics : Dict[str, float]
            Performance metrics
        """
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                if 'return' in key.lower():
                    print(f"{formatted_key:.<40} {value:>15.2%}")
                elif 'ratio' in key.lower():
                    print(f"{formatted_key:.<40} {value:>15.2f}")
                elif 'rate' in key.lower():
                    print(f"{formatted_key:.<40} {value:>15.2%}")
                else:
                    print(f"{formatted_key:.<40} {value:>15.4f}")
            else:
                print(f"{formatted_key:.<40} {value:>15}")

        print("=" * 60 + "\n")

    def generate_pdf_report(
        self,
        metrics: Dict[str, float],
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        daily_pnl: List[float],
        daily_returns: List[float],
        gross_exposures: List[float],
        net_exposures: List[float],
        transaction_costs: List[float],
        chart_dir: Optional[str],
        output_path: str
    ):
        """
        Generate comprehensive PDF report.

        Parameters:
        -----------
        metrics : Dict[str, float]
            Performance metrics
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        daily_pnl : List[float]
            Daily PnL
        daily_returns : List[float]
            Daily returns
        gross_exposures : List[float]
            Gross exposures
        net_exposures : List[float]
            Net exposures
        transaction_costs : List[float]
            Transaction costs
        chart_dir : str, optional
            Directory containing chart images
        output_path : str
            Path to save PDF report
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph,
                Spacer, PageBreak, Image, KeepTogether
            )
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
        except ImportError:
            print("Warning: reportlab not installed. Install with: pip install reportlab")
            return

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Container for PDF elements
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=12
        )
        normal_style = styles['Normal']

        # Title
        elements.append(Paragraph("Backtest Performance Report", title_style))
        elements.append(Spacer(1, 0.2*inch))

        # Executive Summary Section
        elements.append(Paragraph("Executive Summary", heading_style))

        summary_data = [
            ['Period', f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"],
            ['Trading Days', str(len(dates))],
            ['Initial Value', f"${portfolio_values[0]:,.2f}"],
            ['Final Value', f"${portfolio_values[-1]:,.2f}"],
            ['Total Return', f"{((portfolio_values[-1] / portfolio_values[0]) - 1):.2%}"],
        ]

        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))

        # Performance Metrics Section
        elements.append(Paragraph("Performance Metrics", heading_style))

        # Split metrics into categories
        return_metrics = []
        risk_metrics = []
        other_metrics = []

        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()

            if isinstance(value, float):
                if 'return' in key.lower():
                    formatted_value = f"{value:.2%}"
                    return_metrics.append([formatted_key, formatted_value])
                elif 'ratio' in key.lower() or 'sharpe' in key.lower() or 'sortino' in key.lower() or 'calmar' in key.lower():
                    formatted_value = f"{value:.3f}"
                    risk_metrics.append([formatted_key, formatted_value])
                elif 'drawdown' in key.lower():
                    formatted_value = f"{value:.2%}"
                    risk_metrics.append([formatted_key, formatted_value])
                elif 'volatility' in key.lower():
                    formatted_value = f"{value:.2%}"
                    risk_metrics.append([formatted_key, formatted_value])
                elif 'rate' in key.lower():
                    formatted_value = f"{value:.2%}"
                    other_metrics.append([formatted_key, formatted_value])
                else:
                    formatted_value = f"{value:.4f}"
                    other_metrics.append([formatted_key, formatted_value])
            else:
                formatted_value = str(value)
                other_metrics.append([formatted_key, formatted_value])

        # Returns table
        if return_metrics:
            elements.append(Paragraph("<b>Returns</b>", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            returns_table = Table(return_metrics, colWidths=[3*inch, 2*inch])
            returns_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            elements.append(returns_table)
            elements.append(Spacer(1, 0.2*inch))

        # Risk metrics table
        if risk_metrics:
            elements.append(Paragraph("<b>Risk Metrics</b>", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            risk_table = Table(risk_metrics, colWidths=[3*inch, 2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            elements.append(risk_table)
            elements.append(Spacer(1, 0.2*inch))

        # Other metrics table
        if other_metrics:
            elements.append(Paragraph("<b>Additional Metrics</b>", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            other_table = Table(other_metrics, colWidths=[3*inch, 2*inch])
            other_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            elements.append(other_table)

        # Add page break before charts
        elements.append(PageBreak())

        # Charts section
        if chart_dir:
            chart_path = Path(chart_dir)
            if chart_path.exists():
                elements.append(Paragraph("Performance Charts", heading_style))
                elements.append(Spacer(1, 0.2*inch))

                chart_files = [
                    ('cumulative_returns.png', 'Cumulative Returns'),
                    ('drawdown_underwater.png', 'Drawdown Analysis'),
                    ('rolling_sharpe.png', 'Rolling Sharpe Ratio'),
                    ('return_distribution.png', 'Return Distribution'),
                    ('exposures.png', 'Portfolio Exposures'),
                    ('transaction_costs.png', 'Transaction Costs'),
                    ('factor_attribution.png', 'Factor Attribution'),
                    ('factor_exposures_timeseries.png', 'Factor Exposures Over Time'),
                    ('factor_exposures_heatmap.png', 'Factor Exposures Heatmap')
                ]

                for filename, title in chart_files:
                    chart_file = chart_path / filename
                    if chart_file.exists():
                        try:
                            # Add chart title
                            elements.append(Paragraph(f"<b>{title}</b>", normal_style))
                            elements.append(Spacer(1, 0.1*inch))

                            # Add image (scale to fit page width)
                            img = Image(str(chart_file), width=6.5*inch, height=4*inch)
                            elements.append(img)
                            elements.append(Spacer(1, 0.3*inch))
                        except Exception as e:
                            print(f"Warning: Could not add chart {filename} to PDF: {e}")

        # Performance Statistics Table
        elements.append(PageBreak())
        elements.append(Paragraph("Detailed Statistics", heading_style))
        elements.append(Spacer(1, 0.2*inch))

        # Calculate additional statistics
        returns_array = np.array(daily_returns)

        # Monthly statistics
        dates_series = pd.Series(dates)
        returns_series = pd.Series(returns_array, index=dates)
        monthly_returns = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        detailed_stats = [
            ['Statistic', 'Value'],
            ['Number of Trading Days', str(len(dates))],
            ['Number of Months', str(len(monthly_returns))],
            ['Positive Days', f"{(returns_array > 0).sum()} ({(returns_array > 0).sum() / len(returns_array):.1%})"],
            ['Negative Days', f"{(returns_array < 0).sum()} ({(returns_array < 0).sum() / len(returns_array):.1%})"],
            ['Average Daily Return', f"{np.mean(returns_array):.4%}"],
            ['Median Daily Return', f"{np.median(returns_array):.4%}"],
            ['Best Day', f"{np.max(returns_array):.2%}"],
            ['Worst Day', f"{np.min(returns_array):.2%}"],
            ['Average Positive Return', f"{np.mean(returns_array[returns_array > 0]):.4%}" if (returns_array > 0).any() else "N/A"],
            ['Average Negative Return', f"{np.mean(returns_array[returns_array < 0]):.4%}" if (returns_array < 0).any() else "N/A"],
            ['Total Transaction Costs', f"${sum(transaction_costs):,.2f}"],
            ['Average Daily Cost', f"${np.mean(transaction_costs):,.2f}"],
        ]

        stats_table = Table(detailed_stats, colWidths=[3.5*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        elements.append(stats_table)

        # Monthly returns table (if available)
        if len(monthly_returns) > 0:
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("Monthly Returns", heading_style))
            elements.append(Spacer(1, 0.1*inch))

            monthly_data = [['Month', 'Return']]
            for date, ret in monthly_returns.items():
                monthly_data.append([
                    date.strftime('%Y-%m'),
                    f"{ret:.2%}"
                ])

            # Limit to last 24 months if there are many
            if len(monthly_data) > 25:
                monthly_data = monthly_data[:1] + monthly_data[-24:]

            monthly_table = Table(monthly_data, colWidths=[2*inch, 2*inch])
            monthly_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            elements.append(monthly_table)

        # Footer
        elements.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Paragraph(
            f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            footer_style
        ))

        # Build PDF
        doc.build(elements)
        print(f"PDF report saved to {output_path}")

    def generate_pdf_report_simple(
        self,
        metrics: Dict[str, float],
        dates: List[pd.Timestamp],
        portfolio_values: List[float],
        output_path: str
    ):
        """
        Generate a simple PDF report without charts (minimal version).

        Parameters:
        -----------
        metrics : Dict[str, float]
            Performance metrics
        dates : List[pd.Timestamp]
            Dates
        portfolio_values : List[float]
            Portfolio values
        output_path : str
            Path to save PDF report
        """
        # Call the full version with minimal parameters
        self.generate_pdf_report(
            metrics=metrics,
            dates=dates,
            portfolio_values=portfolio_values,
            daily_pnl=[0] * len(dates),
            daily_returns=[0] * len(dates),
            gross_exposures=[0] * len(dates),
            net_exposures=[0] * len(dates),
            transaction_costs=[0] * len(dates),
            chart_dir=None,
            output_path=output_path
        )

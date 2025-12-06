"""
Stop Loss Manager

Manages stop loss functionality by reducing gross exposure when drawdown
hits specified levels.
"""

from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass


@dataclass
class StopLossLevel:
    """
    A single stop loss level.

    Supports both percentage and dollar drawdown thresholds.

    Attributes:
    -----------
    drawdown_threshold : float
        Drawdown level that triggers this stop loss.
        - If threshold_type='percent': value between 0-1 (e.g., 0.10 for 10% drawdown)
        - If threshold_type='dollar': absolute dollar amount (e.g., 5000 for $5,000 loss)
    gross_reduction : float
        Target gross exposure as a percentage (e.g., 0.5 means reduce to 50% of normal gross)
    threshold_type : Literal['percent', 'dollar']
        Type of drawdown threshold. Default: 'percent'
    """
    drawdown_threshold: float
    gross_reduction: float
    threshold_type: Literal['percent', 'dollar'] = 'percent'

    def __post_init__(self):
        if self.threshold_type == 'percent':
            if self.drawdown_threshold < 0 or self.drawdown_threshold > 1:
                raise ValueError(f"Percent drawdown threshold must be between 0 and 1, got {self.drawdown_threshold}")
        elif self.threshold_type == 'dollar':
            if self.drawdown_threshold < 0:
                raise ValueError(f"Dollar drawdown threshold must be non-negative, got {self.drawdown_threshold}")
        else:
            raise ValueError(f"threshold_type must be 'percent' or 'dollar', got {self.threshold_type}")

        if self.gross_reduction < 0 or self.gross_reduction > 1:
            raise ValueError(f"Gross reduction must be between 0 and 1, got {self.gross_reduction}")


class StopLossManager:
    """
    Manages stop loss functionality by tracking drawdown and reducing gross exposure.

    The manager tracks the peak portfolio value and current drawdown. When drawdown
    exceeds specified thresholds, it reduces the target gross exposure.

    Example:
    --------
    >>> levels = [
    ...     StopLossLevel(drawdown_threshold=0.05, gross_reduction=0.75),  # 5% DD -> 75% gross
    ...     StopLossLevel(drawdown_threshold=0.10, gross_reduction=0.50),  # 10% DD -> 50% gross
    ...     StopLossLevel(drawdown_threshold=0.15, gross_reduction=0.25),  # 15% DD -> 25% gross
    ... ]
    >>> manager = StopLossManager(levels)
    """

    def __init__(self, levels: List[StopLossLevel]):
        """
        Initialize the stop loss manager.

        Parameters:
        -----------
        levels : List[StopLossLevel]
            List of stop loss levels. Should be sorted by increasing drawdown threshold.
        """
        # Don't sort - user should provide levels in desired order
        # (especially important when mixing percent and dollar thresholds)
        self.levels = levels

        # Validate that gross reductions are decreasing (skip if mixed types)
        has_mixed_types = len(set(level.threshold_type for level in levels)) > 1
        if not has_mixed_types:
            for i in range(1, len(self.levels)):
                if self.levels[i].gross_reduction > self.levels[i-1].gross_reduction:
                    raise ValueError(
                        f"Gross reductions must be decreasing. Level {i} has higher "
                        f"reduction ({self.levels[i].gross_reduction}) than level {i-1} "
                        f"({self.levels[i-1].gross_reduction})"
                    )

        # Track peak portfolio value
        self.peak_value: Optional[float] = None
        self.current_drawdown_pct: float = 0.0
        self.current_drawdown_dollar: float = 0.0
        self.current_gross_multiplier: float = 1.0
        self.triggered_level: Optional[int] = None  # Index of triggered level

    def update(self, portfolio_value: float) -> Tuple[float, bool]:
        """
        Update the stop loss manager with current portfolio value.

        Parameters:
        -----------
        portfolio_value : float
            Current portfolio value

        Returns:
        --------
        Tuple[float, bool]
            (current_gross_multiplier, level_changed)
            - current_gross_multiplier: multiplier to apply to target positions (1.0 = no reduction)
            - level_changed: True if stop loss level changed on this update
        """
        # Initialize peak if first update
        if self.peak_value is None:
            self.peak_value = portfolio_value
            return 1.0, False

        # Update peak if new high
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calculate current drawdown (both percent and dollar)
        self.current_drawdown_dollar = self.peak_value - portfolio_value

        if self.peak_value > 0:
            self.current_drawdown_pct = self.current_drawdown_dollar / self.peak_value
        else:
            self.current_drawdown_pct = 0.0

        # Determine which level (if any) should be triggered
        new_triggered_level = None
        new_gross_multiplier = 1.0

        for i, level in enumerate(self.levels):
            # Check against appropriate threshold type
            if level.threshold_type == 'percent':
                if self.current_drawdown_pct >= level.drawdown_threshold:
                    new_triggered_level = i
                    new_gross_multiplier = level.gross_reduction
            elif level.threshold_type == 'dollar':
                if self.current_drawdown_dollar >= level.drawdown_threshold:
                    new_triggered_level = i
                    new_gross_multiplier = level.gross_reduction

        # Check if level changed
        level_changed = new_triggered_level != self.triggered_level

        if level_changed and new_triggered_level is not None:
            triggered_level_obj = self.levels[new_triggered_level]
            print(f"\n{'='*60}")
            print(f"STOP LOSS TRIGGERED")
            print(f"{'='*60}")
            print(f"Peak value: ${self.peak_value:,.2f}")
            print(f"Current value: ${portfolio_value:,.2f}")
            print(f"Drawdown: {self.current_drawdown_pct:.2%} (${self.current_drawdown_dollar:,.2f})")
            if triggered_level_obj.threshold_type == 'percent':
                print(f"Threshold: {triggered_level_obj.drawdown_threshold:.2%} (percent)")
            else:
                print(f"Threshold: ${triggered_level_obj.drawdown_threshold:,.2f} (dollar)")
            print(f"Reducing gross exposure to {new_gross_multiplier:.1%}")
            print(f"{'='*60}\n")
        elif level_changed and new_triggered_level is None and self.triggered_level is not None:
            print(f"\n{'='*60}")
            print(f"STOP LOSS CLEARED")
            print(f"{'='*60}")
            print(f"Drawdown recovered to {self.current_drawdown_pct:.2%} (${self.current_drawdown_dollar:,.2f})")
            print(f"Restoring full gross exposure")
            print(f"{'='*60}\n")

        self.triggered_level = new_triggered_level
        self.current_gross_multiplier = new_gross_multiplier

        return self.current_gross_multiplier, level_changed

    def apply_to_positions(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply stop loss gross reduction to target positions.

        This scales the target positions toward zero (or current positions)
        to achieve the desired gross reduction.

        Parameters:
        -----------
        target_positions : Dict[str, float]
            Target positions before stop loss
        current_positions : Dict[str, float]
            Current portfolio positions

        Returns:
        --------
        Dict[str, float]
            Adjusted target positions after stop loss
        """
        if self.current_gross_multiplier >= 1.0:
            # No stop loss active
            return target_positions.copy()

        # Scale positions by the gross multiplier
        adjusted_positions = {}
        for ticker, target_qty in target_positions.items():
            adjusted_positions[ticker] = target_qty * self.current_gross_multiplier

        return adjusted_positions

    def get_status(self) -> Dict:
        """
        Get current stop loss status.

        Returns:
        --------
        Dict
            Dictionary with status information
        """
        return {
            'peak_value': self.peak_value,
            'current_drawdown_pct': self.current_drawdown_pct,
            'current_drawdown_dollar': self.current_drawdown_dollar,
            'gross_multiplier': self.current_gross_multiplier,
            'triggered_level': self.triggered_level,
            'is_active': self.triggered_level is not None
        }

    def reset(self):
        """Reset the stop loss manager (e.g., for a new backtest)."""
        self.peak_value = None
        self.current_drawdown_pct = 0.0
        self.current_drawdown_dollar = 0.0
        self.current_gross_multiplier = 1.0
        self.triggered_level = None

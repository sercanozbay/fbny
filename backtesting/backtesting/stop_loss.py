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
    A single stop loss level with optional recovery threshold.

    Supports both percentage and dollar drawdown/recovery thresholds.

    Attributes:
    -----------
    drawdown_threshold : float
        Drawdown level that triggers reduction to this gross level.
        - If threshold_type='percent': value between 0-1 (e.g., 0.10 for 10% drawdown)
        - If threshold_type='dollar': absolute dollar amount (e.g., 5000 for $5,000 loss)

    gross_reduction : float
        Target gross exposure as a percentage (e.g., 0.5 means reduce to 50% of normal gross)

    recovery_threshold : Optional[float]
        Recovery level from drawdown bottom that triggers moving back to previous gross level.
        - If threshold_type='percent': recovery as percent from bottom (e.g., 0.50 = 50% recovery)
        - If threshold_type='dollar': dollar recovery from bottom (e.g., 2500 = $2,500 recovery)
        If None, no automatic recovery for this level.

    threshold_type : Literal['percent', 'dollar']
        Type of drawdown/recovery thresholds. Default: 'percent'
    """
    drawdown_threshold: float
    gross_reduction: float
    recovery_threshold: Optional[float] = None
    threshold_type: Literal['percent', 'dollar'] = 'percent'

    def __post_init__(self):
        if self.threshold_type == 'percent':
            if self.drawdown_threshold < 0 or self.drawdown_threshold > 1:
                raise ValueError(f"Percent drawdown threshold must be between 0 and 1, got {self.drawdown_threshold}")
            if self.recovery_threshold is not None:
                if self.recovery_threshold < 0 or self.recovery_threshold > 1:
                    raise ValueError(f"Percent recovery threshold must be between 0 and 1, got {self.recovery_threshold}")
        elif self.threshold_type == 'dollar':
            if self.drawdown_threshold < 0:
                raise ValueError(f"Dollar drawdown threshold must be non-negative, got {self.drawdown_threshold}")
            if self.recovery_threshold is not None and self.recovery_threshold < 0:
                raise ValueError(f"Dollar recovery threshold must be non-negative, got {self.recovery_threshold}")
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
            All levels must use the same threshold_type (either all 'percent' or all 'dollar').
        """
        if not levels:
            raise ValueError("Must provide at least one stop loss level")

        # Validate all levels use same threshold type
        threshold_types = set(level.threshold_type for level in levels)
        if len(threshold_types) > 1:
            raise ValueError(
                f"All stop loss levels must use the same threshold_type. "
                f"Found mixed types: {threshold_types}. "
                f"Use either all 'percent' or all 'dollar'."
            )

        self.threshold_type = levels[0].threshold_type
        self.levels = levels

        # Validate that gross reductions are decreasing
        for i in range(1, len(self.levels)):
            if self.levels[i].gross_reduction > self.levels[i-1].gross_reduction:
                raise ValueError(
                    f"Gross reductions must be decreasing. Level {i} has higher "
                    f"reduction ({self.levels[i].gross_reduction}) than level {i-1} "
                    f"({self.levels[i-1].gross_reduction})"
                )

        # Track peak portfolio value and drawdown
        self.peak_value: Optional[float] = None
        self.trough_value: Optional[float] = None  # Lowest value during current drawdown
        self.current_drawdown_pct: float = 0.0
        self.current_drawdown_dollar: float = 0.0
        self.current_recovery_pct: float = 0.0  # Recovery from trough
        self.current_recovery_dollar: float = 0.0  # Dollar recovery from trough
        self.current_gross_multiplier: float = 1.0
        self.triggered_level: Optional[int] = None  # Index of triggered level

    def update(self, portfolio_value: float) -> Tuple[float, bool]:
        """
        Update the stop loss manager with current portfolio value.

        Handles both drawdown triggers and recovery from drawdowns.

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
            self.trough_value = portfolio_value
            return 1.0, False

        # Update peak if new high (clears drawdown)
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.trough_value = portfolio_value  # Reset trough

        # Update trough if new low during drawdown
        if self.trough_value is None or portfolio_value < self.trough_value:
            self.trough_value = portfolio_value

        # Calculate current drawdown (from peak)
        self.current_drawdown_dollar = self.peak_value - portfolio_value
        if self.peak_value > 0:
            self.current_drawdown_pct = self.current_drawdown_dollar / self.peak_value
        else:
            self.current_drawdown_pct = 0.0

        # Calculate current recovery (from trough)
        self.current_recovery_dollar = portfolio_value - self.trough_value
        drawdown_amount = self.peak_value - self.trough_value
        if drawdown_amount > 0:
            self.current_recovery_pct = self.current_recovery_dollar / drawdown_amount
        else:
            self.current_recovery_pct = 0.0

        # Determine which level should be active
        new_triggered_level = None
        new_gross_multiplier = 1.0

        # First, check if we should trigger a deeper level (going down)
        for i, level in enumerate(self.levels):
            triggered = False
            if self.threshold_type == 'percent':
                triggered = self.current_drawdown_pct >= level.drawdown_threshold
            else:  # dollar
                triggered = self.current_drawdown_dollar >= level.drawdown_threshold

            if triggered:
                new_triggered_level = i
                new_gross_multiplier = level.gross_reduction

        # Second, check if we should recover to a shallower level (going up)
        # Only check recovery if we're currently at a level and recovering
        if self.triggered_level is not None and new_triggered_level == self.triggered_level:
            # Check if we can move to a less restrictive level
            current_level = self.levels[self.triggered_level]
            if current_level.recovery_threshold is not None:
                recovered = False
                if self.threshold_type == 'percent':
                    recovered = self.current_recovery_pct >= current_level.recovery_threshold
                else:  # dollar
                    recovered = self.current_recovery_dollar >= current_level.recovery_threshold

                if recovered:
                    # Move to previous (less restrictive) level, or clear if at first level
                    if self.triggered_level > 0:
                        new_triggered_level = self.triggered_level - 1
                        new_gross_multiplier = self.levels[new_triggered_level].gross_reduction
                    else:
                        new_triggered_level = None
                        new_gross_multiplier = 1.0

        # Check if level changed
        level_changed = new_triggered_level != self.triggered_level

        # Print notifications
        if level_changed:
            if new_triggered_level is not None and (self.triggered_level is None or new_triggered_level > self.triggered_level):
                # Moving to more restrictive level (drawdown)
                triggered_level_obj = self.levels[new_triggered_level]
                print(f"\n{'='*60}")
                print(f"STOP LOSS TRIGGERED - Level {new_triggered_level + 1}")
                print(f"{'='*60}")
                print(f"Peak value: ${self.peak_value:,.2f}")
                print(f"Trough value: ${self.trough_value:,.2f}")
                print(f"Current value: ${portfolio_value:,.2f}")
                print(f"Drawdown: {self.current_drawdown_pct:.2%} (${self.current_drawdown_dollar:,.2f})")
                if self.threshold_type == 'percent':
                    print(f"Threshold: {triggered_level_obj.drawdown_threshold:.2%}")
                else:
                    print(f"Threshold: ${triggered_level_obj.drawdown_threshold:,.2f}")
                print(f"Reducing gross exposure to {new_gross_multiplier:.1%}")
                print(f"{'='*60}\n")
            elif new_triggered_level is None or (self.triggered_level is not None and new_triggered_level < self.triggered_level):
                # Moving to less restrictive level or clearing (recovery)
                print(f"\n{'='*60}")
                if new_triggered_level is None:
                    print(f"STOP LOSS CLEARED - Full Recovery")
                else:
                    print(f"STOP LOSS RECOVERY - Moving to Level {new_triggered_level + 1}")
                print(f"{'='*60}")
                print(f"Peak value: ${self.peak_value:,.2f}")
                print(f"Trough value: ${self.trough_value:,.2f}")
                print(f"Current value: ${portfolio_value:,.2f}")
                print(f"Recovery: {self.current_recovery_pct:.2%} (${self.current_recovery_dollar:,.2f})")
                if new_triggered_level is None:
                    print(f"Restoring full gross exposure (100%)")
                else:
                    print(f"Increasing gross exposure to {new_gross_multiplier:.1%}")
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
            'trough_value': self.trough_value,
            'current_drawdown_pct': self.current_drawdown_pct,
            'current_drawdown_dollar': self.current_drawdown_dollar,
            'current_recovery_pct': self.current_recovery_pct,
            'current_recovery_dollar': self.current_recovery_dollar,
            'gross_multiplier': self.current_gross_multiplier,
            'triggered_level': self.triggered_level,
            'is_active': self.triggered_level is not None
        }

    def reset(self):
        """Reset the stop loss manager (e.g., for a new backtest)."""
        self.peak_value = None
        self.trough_value = None
        self.current_drawdown_pct = 0.0
        self.current_drawdown_dollar = 0.0
        self.current_recovery_pct = 0.0
        self.current_recovery_dollar = 0.0
        self.current_gross_multiplier = 1.0
        self.triggered_level = None

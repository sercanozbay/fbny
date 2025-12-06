"""
Stop Loss Manager

Manages stop loss functionality by reducing gross exposure when drawdown
hits specified dollar levels.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StopLossLevel:
    """
    A single stop loss level with optional recovery threshold.

    All thresholds are dollar-based for simplicity.

    Attributes:
    -----------
    drawdown_threshold : float
        Dollar loss that triggers reduction to this gross level.
        Example: 5000 means at $5,000 loss from peak

    gross_reduction : float
        Target gross exposure as a percentage (e.g., 0.5 means reduce to 50% of normal gross)

    recovery_threshold : Optional[float]
        Dollar recovery from drawdown bottom that triggers moving back to previous gross level.
        Example: 2500 means at $2,500 recovery from trough
        If None, no automatic recovery for this level.
    """
    drawdown_threshold: float
    gross_reduction: float
    recovery_threshold: Optional[float] = None

    def __post_init__(self):
        if self.drawdown_threshold < 0:
            raise ValueError(f"Dollar drawdown threshold must be non-negative, got {self.drawdown_threshold}")

        if self.recovery_threshold is not None and self.recovery_threshold < 0:
            raise ValueError(f"Dollar recovery threshold must be non-negative, got {self.recovery_threshold}")

        if self.gross_reduction < 0 or self.gross_reduction > 1:
            raise ValueError(f"Gross reduction must be between 0 and 1, got {self.gross_reduction}")


class StopLossManager:
    """
    Manages stop loss functionality by tracking dollar drawdown and reducing gross exposure.

    The manager tracks the peak portfolio value and current dollar drawdown. When drawdown
    exceeds specified dollar thresholds, it reduces the target gross exposure.

    Example:
    --------
    >>> levels = [
    ...     StopLossLevel(drawdown_threshold=5000, gross_reduction=0.75),   # $5k loss -> 75% gross
    ...     StopLossLevel(drawdown_threshold=10000, gross_reduction=0.50),  # $10k loss -> 50% gross
    ...     StopLossLevel(drawdown_threshold=15000, gross_reduction=0.25),  # $15k loss -> 25% gross
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
        if not levels:
            raise ValueError("Must provide at least one stop loss level")

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
        self.current_drawdown_dollar: float = 0.0
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

        # Calculate current recovery (from trough)
        self.current_recovery_dollar = portfolio_value - self.trough_value

        # Determine which level should be active
        new_triggered_level = None
        new_gross_multiplier = 1.0

        # First, check if we should trigger a deeper level (going down)
        for i, level in enumerate(self.levels):
            if self.current_drawdown_dollar >= level.drawdown_threshold:
                new_triggered_level = i
                new_gross_multiplier = level.gross_reduction

        # Second, check if we should recover to a shallower level (going up)
        # Only check recovery if we're currently at a level and recovering
        if self.triggered_level is not None and new_triggered_level == self.triggered_level:
            # Check if we can move to a less restrictive level
            current_level = self.levels[self.triggered_level]
            if current_level.recovery_threshold is not None:
                if self.current_recovery_dollar >= current_level.recovery_threshold:
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
                print(f"Dollar drawdown: ${self.current_drawdown_dollar:,.2f}")
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
                print(f"Dollar recovery: ${self.current_recovery_dollar:,.2f}")
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
            'current_drawdown_dollar': self.current_drawdown_dollar,
            'current_recovery_dollar': self.current_recovery_dollar,
            'gross_multiplier': self.current_gross_multiplier,
            'triggered_level': self.triggered_level,
            'is_active': self.triggered_level is not None
        }

    def reset(self):
        """Reset the stop loss manager (e.g., for a new backtest)."""
        self.peak_value = None
        self.trough_value = None
        self.current_drawdown_dollar = 0.0
        self.current_recovery_dollar = 0.0
        self.current_gross_multiplier = 1.0
        self.triggered_level = None

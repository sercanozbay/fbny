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
    A single stop loss level.

    All thresholds are dollar-based drawdown levels for consistency.

    Attributes:
    -----------
    drawdown_threshold : float
        Dollar drawdown from peak that triggers both entry and exit for this level.
        Example: 10000 means enter at $10,000 loss from peak, exit when DD < $10,000

    gross_reduction : float
        Target gross exposure as a percentage (e.g., 0.5 means reduce to 50% of normal gross)
    """
    drawdown_threshold: float
    gross_reduction: float

    def __post_init__(self):
        if self.drawdown_threshold < 0:
            raise ValueError(f"Drawdown threshold must be non-negative, got {self.drawdown_threshold}")

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
        self.current_drawdown_dollar: float = 0.0
        self.current_gross_multiplier: float = 1.0
        self.triggered_level: Optional[int] = None  # Index of currently active level

    def update(self, portfolio_value: float) -> Tuple[float, bool]:
        """
        Update the stop loss manager with current portfolio value.

        Simplified logic:
        - Enter a level when drawdown >= level's drawdown_threshold
        - Exit completely when drawdown < current level's drawdown_threshold (jump to no stop loss)
        - New peak clears all stop loss levels

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

        # Update peak if new high (clears all stop loss levels)
        if portfolio_value > self.peak_value:
            old_level = self.triggered_level
            self.peak_value = portfolio_value
            self.current_drawdown_dollar = 0.0
            self.triggered_level = None
            self.current_gross_multiplier = 1.0
            level_changed = old_level is not None

            if level_changed:
                print(f"\n{'='*60}")
                print(f"STOP LOSS CLEARED - New Peak")
                print(f"{'='*60}")
                print(f"New peak value: ${self.peak_value:,.2f}")
                print(f"Restoring full gross exposure (100%)")
                print(f"{'='*60}\n")

            return 1.0, level_changed

        # Calculate current drawdown from peak
        self.current_drawdown_dollar = self.peak_value - portfolio_value

        # Determine level using simplified logic
        new_triggered_level = self.triggered_level
        new_gross_multiplier = self.current_gross_multiplier

        if self.triggered_level is None:
            # Not at any level - check if we should enter one (deepest first)
            for i in range(len(self.levels) - 1, -1, -1):
                if self.current_drawdown_dollar >= self.levels[i].drawdown_threshold:
                    new_triggered_level = i
                    new_gross_multiplier = self.levels[i].gross_reduction
                    break
        else:
            # At a level - check if we should exit completely or go deeper
            current_threshold = self.levels[self.triggered_level].drawdown_threshold

            if self.current_drawdown_dollar < current_threshold:
                # Exit completely (jump to no stop loss)
                new_triggered_level = None
                new_gross_multiplier = 1.0
            else:
                # Check if we should enter a deeper level
                for i in range(self.triggered_level + 1, len(self.levels)):
                    if self.current_drawdown_dollar >= self.levels[i].drawdown_threshold:
                        new_triggered_level = i
                        new_gross_multiplier = self.levels[i].gross_reduction
                        break

        # Check if level changed
        level_changed = new_triggered_level != self.triggered_level

        # Print notifications
        if level_changed:
            if new_triggered_level is not None and (self.triggered_level is None or new_triggered_level > self.triggered_level):
                # Moving to more restrictive level (deeper drawdown)
                triggered_level_obj = self.levels[new_triggered_level]
                print(f"\n{'='*60}")
                print(f"STOP LOSS TRIGGERED - Level {new_triggered_level + 1}")
                print(f"{'='*60}")
                print(f"Peak value: ${self.peak_value:,.2f}")
                print(f"Current value: ${portfolio_value:,.2f}")
                print(f"Dollar drawdown: ${self.current_drawdown_dollar:,.2f}")
                print(f"Threshold: ${triggered_level_obj.drawdown_threshold:,.2f}")
                print(f"Reducing gross exposure to {new_gross_multiplier:.1%}")
                print(f"{'='*60}\n")
            elif new_triggered_level is None:
                # Clearing stop loss (recovery)
                print(f"\n{'='*60}")
                print(f"STOP LOSS CLEARED - Recovery")
                print(f"{'='*60}")
                print(f"Peak value: ${self.peak_value:,.2f}")
                print(f"Current value: ${portfolio_value:,.2f}")
                print(f"Dollar drawdown: ${self.current_drawdown_dollar:,.2f}")
                print(f"Restoring full gross exposure (100%)")
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
            'current_drawdown_dollar': self.current_drawdown_dollar,
            'gross_multiplier': self.current_gross_multiplier,
            'triggered_level': self.triggered_level,
            'is_active': self.triggered_level is not None
        }

    def reset(self):
        """Reset the stop loss manager (e.g., for a new backtest)."""
        self.peak_value = None
        self.current_drawdown_dollar = 0.0
        self.current_gross_multiplier = 1.0
        self.triggered_level = None

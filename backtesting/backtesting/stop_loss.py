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

    All thresholds are dollar-based drawdown levels for consistency.

    Attributes:
    -----------
    drawdown_threshold : float
        Dollar drawdown from peak that triggers entry to this level.
        Example: 10000 means at $10,000 loss from peak, enter this level

    gross_reduction : float
        Target gross exposure as a percentage (e.g., 0.5 means reduce to 50% of normal gross)

    recovery_drawdown : Optional[float]
        Dollar drawdown from peak that triggers exit (scale up) from this level.
        Must be GREATER than drawdown_threshold to scale up early during recovery.
        Example: If drawdown_threshold=10000 and recovery_drawdown=15000,
                 enter at $10k DD, scale up when DD reaches $15k (worse DD = recovering from even deeper loss)
        If None, defaults to drawdown_threshold (immediate scale up when DD improves)
    """
    drawdown_threshold: float
    gross_reduction: float
    recovery_drawdown: Optional[float] = None

    def __post_init__(self):
        if self.drawdown_threshold < 0:
            raise ValueError(f"Drawdown threshold must be non-negative, got {self.drawdown_threshold}")

        if self.gross_reduction < 0 or self.gross_reduction > 1:
            raise ValueError(f"Gross reduction must be between 0 and 1, got {self.gross_reduction}")

        if self.recovery_drawdown is not None:
            if self.recovery_drawdown < 0:
                raise ValueError(f"Recovery drawdown must be non-negative, got {self.recovery_drawdown}")
            if self.recovery_drawdown <= self.drawdown_threshold:
                raise ValueError(
                    f"Recovery drawdown ({self.recovery_drawdown}) must be greater than "
                    f"drawdown threshold ({self.drawdown_threshold}). "
                    f"Use None to default to drawdown_threshold for immediate scale up."
                )


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
        self.last_exited_level: Optional[int] = None  # Track most recently exited level for hysteresis

    def update(self, portfolio_value: float) -> Tuple[float, bool]:
        """
        Update the stop loss manager with current portfolio value.

        Uses "sticky" recovery logic:
        - Enter a level when drawdown >= level's drawdown_threshold
        - Exit a level when drawdown <= level's recovery_drawdown
        - Stay at current level if drawdown is between recovery and entry thresholds
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
            self.last_exited_level = None  # Clear last exited level on new peak
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

        # Clear last_exited_level if DD improves below its entry threshold
        if self.last_exited_level is not None:
            if self.current_drawdown_dollar < self.levels[self.last_exited_level].drawdown_threshold:
                self.last_exited_level = None

        # Determine level using proper hysteresis logic
        new_triggered_level = self.triggered_level
        new_gross_multiplier = self.current_gross_multiplier

        if self.triggered_level is None:
            # Not at any level - check if we should enter one
            # For initial entry, use drawdown_threshold
            for i in range(len(self.levels) - 1, -1, -1):
                if self.current_drawdown_dollar >= self.levels[i].drawdown_threshold:
                    new_triggered_level = i
                    new_gross_multiplier = self.levels[i].gross_reduction
                    break
        else:
            # Currently at a level - check for exit or deeper entry
            current_level_obj = self.levels[self.triggered_level]
            recovery_dd = current_level_obj.recovery_drawdown if current_level_obj.recovery_drawdown is not None else current_level_obj.drawdown_threshold

            # Exit when DD < recovery_drawdown (improved below recovery threshold)
            if self.current_drawdown_dollar < recovery_dd:
                # Exiting current level - track it for hysteresis
                self.last_exited_level = self.triggered_level

                # Check if we enter a shallower level
                new_triggered_level = None
                new_gross_multiplier = 1.0

                for i in range(self.triggered_level - 1, -1, -1):
                    level_recovery = self.levels[i].recovery_drawdown if self.levels[i].recovery_drawdown is not None else self.levels[i].drawdown_threshold
                    if self.current_drawdown_dollar >= level_recovery:
                        new_triggered_level = i
                        new_gross_multiplier = self.levels[i].gross_reduction
                        break
            else:
                # Still at current level (DD >= recovery_drawdown)
                # Check if we should enter a deeper level
                for i in range(self.triggered_level + 1, len(self.levels)):
                    # If this is the level we just exited, need DD >= recovery_drawdown (hysteresis)
                    # Otherwise, use drawdown_threshold for entry
                    if i == self.last_exited_level:
                        threshold = self.levels[i].recovery_drawdown if self.levels[i].recovery_drawdown is not None else self.levels[i].drawdown_threshold
                    else:
                        threshold = self.levels[i].drawdown_threshold

                    if self.current_drawdown_dollar >= threshold:
                        new_triggered_level = i
                        new_gross_multiplier = self.levels[i].gross_reduction
                        # Clear last_exited_level when we enter a deeper level
                        self.last_exited_level = None
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
                print(f"Entry threshold: ${triggered_level_obj.drawdown_threshold:,.2f}")
                if triggered_level_obj.recovery_drawdown is not None:
                    print(f"Recovery threshold: ${triggered_level_obj.recovery_drawdown:,.2f}")
                print(f"Reducing gross exposure to {new_gross_multiplier:.1%}")
                print(f"{'='*60}\n")
            elif new_triggered_level is None or (self.triggered_level is not None and new_triggered_level < self.triggered_level):
                # Moving to less restrictive level or clearing (recovery)
                print(f"\n{'='*60}")
                if new_triggered_level is None:
                    print(f"STOP LOSS RECOVERY - Cleared")
                else:
                    print(f"STOP LOSS RECOVERY - Moving to Level {new_triggered_level + 1}")
                print(f"{'='*60}")
                print(f"Peak value: ${self.peak_value:,.2f}")
                print(f"Current value: ${portfolio_value:,.2f}")
                print(f"Dollar drawdown: ${self.current_drawdown_dollar:,.2f}")
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
        self.last_exited_level = None

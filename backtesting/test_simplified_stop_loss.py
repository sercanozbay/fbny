"""
Test the simplified stop loss logic.

Tests that the stop loss uses immediate exit (no sticky recovery or hysteresis).
"""

from backtesting.stop_loss import StopLossManager, StopLossLevel


def test_immediate_exit():
    """Test that stop loss exits immediately when DD improves below threshold."""
    # Setup: 2 levels
    levels = [
        StopLossLevel(drawdown_threshold=5000, gross_reduction=0.75),
        StopLossLevel(drawdown_threshold=10000, gross_reduction=0.50)
    ]
    manager = StopLossManager(levels)

    # Day 1: Start at $100k
    gross, changed = manager.update(100000)
    assert gross == 1.0
    assert not changed
    assert manager.triggered_level is None

    # Day 2: Drop to $88k ($12k DD) -> should enter Level 2 (50% gross)
    gross, changed = manager.update(88000)
    assert gross == 0.50
    assert changed
    assert manager.triggered_level == 1

    # Day 3: Recover to $91k ($9k DD) -> should exit completely (DD < $10k threshold)
    gross, changed = manager.update(91000)
    assert gross == 1.0  # Jumped to 100% gross (not 75%)
    assert changed
    assert manager.triggered_level is None

    print("✓ Test passed: Immediate exit works correctly")


def test_no_gradual_scale_up():
    """Test that recovery jumps directly to no stop loss, not through levels."""
    levels = [
        StopLossLevel(drawdown_threshold=5000, gross_reduction=0.75),
        StopLossLevel(drawdown_threshold=10000, gross_reduction=0.50)
    ]
    manager = StopLossManager(levels)

    # Start at $100k
    manager.update(100000)

    # Drop to Level 2
    manager.update(88000)
    assert manager.triggered_level == 1
    assert manager.current_gross_multiplier == 0.50

    # Recover enough to exit Level 2 ($9k DD < $10k threshold)
    # Should jump directly to no stop loss, NOT to Level 1
    gross, changed = manager.update(91000)
    assert gross == 1.0
    assert manager.triggered_level is None  # Not at Level 1, completely cleared

    print("✓ Test passed: No gradual scale up")


def test_multiple_level_entries():
    """Test entering different levels as drawdown increases."""
    levels = [
        StopLossLevel(drawdown_threshold=5000, gross_reduction=0.75),
        StopLossLevel(drawdown_threshold=10000, gross_reduction=0.50)
    ]
    manager = StopLossManager(levels)

    # Start at $100k
    manager.update(100000)

    # Drop to $94k ($6k DD) -> Level 1
    gross, _ = manager.update(94000)
    assert gross == 0.75
    assert manager.triggered_level == 0

    # Drop further to $88k ($12k DD) -> Level 2
    gross, _ = manager.update(88000)
    assert gross == 0.50
    assert manager.triggered_level == 1

    # Recover to $96k ($4k DD) -> clear all
    gross, _ = manager.update(96000)
    assert gross == 1.0
    assert manager.triggered_level is None

    print("✓ Test passed: Multiple level entries work")


def test_new_peak_clears_all():
    """Test that reaching a new peak clears all stop loss levels."""
    levels = [
        StopLossLevel(drawdown_threshold=5000, gross_reduction=0.75)
    ]
    manager = StopLossManager(levels)

    # Start at $100k
    manager.update(100000)

    # Drop to $94k -> Level 1
    manager.update(94000)
    assert manager.triggered_level == 0

    # New peak at $102k -> should clear
    gross, changed = manager.update(102000)
    assert gross == 1.0
    assert changed
    assert manager.triggered_level is None
    assert manager.peak_value == 102000

    print("✓ Test passed: New peak clears stop loss")


def test_only_2_tuple_allowed():
    """Test that only 2-tuple format is allowed."""
    # Should work with 2-tuple
    level = StopLossLevel(drawdown_threshold=5000, gross_reduction=0.75)
    assert level.drawdown_threshold == 5000
    assert level.gross_reduction == 0.75

    # Verify no recovery_drawdown attribute exists
    assert not hasattr(level, 'recovery_drawdown')

    print("✓ Test passed: Only 2-tuple format supported")


def test_validation():
    """Test that validation works correctly."""
    # Negative drawdown threshold should fail
    try:
        StopLossLevel(drawdown_threshold=-1000, gross_reduction=0.75)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-negative" in str(e)

    # Gross reduction > 1 should fail
    try:
        StopLossLevel(drawdown_threshold=5000, gross_reduction=1.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between 0 and 1" in str(e)

    # Gross reduction < 0 should fail
    try:
        StopLossLevel(drawdown_threshold=5000, gross_reduction=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between 0 and 1" in str(e)

    print("✓ Test passed: Validation works")


if __name__ == '__main__':
    test_immediate_exit()
    test_no_gradual_scale_up()
    test_multiple_level_entries()
    test_new_peak_clears_all()
    test_only_2_tuple_allowed()
    test_validation()
    print("\n✅ All simplified stop loss tests passed!")

"""
Integration test for simplified stop loss with BacktestConfig.
"""

from backtesting.config import BacktestConfig
from backtesting.stop_loss import StopLossManager, StopLossLevel


def test_config_2_tuple_only():
    """Test that BacktestConfig only accepts 2-tuple stop loss levels."""
    # Valid 2-tuple config
    config = BacktestConfig(
        stop_loss_levels=[(5000, 0.75), (10000, 0.50)]
    )
    assert len(config.stop_loss_levels) == 2
    assert config.stop_loss_levels[0] == (5000, 0.75)
    assert config.stop_loss_levels[1] == (10000, 0.50)

    print("✓ Test passed: Config accepts 2-tuple")


def test_config_rejects_3_tuple():
    """Test that BacktestConfig rejects 3-tuple stop loss levels."""
    try:
        config = BacktestConfig(
            stop_loss_levels=[(5000, 0.75, 2000)]  # 3-tuple
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "2-tuple" in str(e)

    print("✓ Test passed: Config rejects 3-tuple")


def test_config_to_manager():
    """Test creating StopLossManager from config."""
    config = BacktestConfig(
        stop_loss_levels=[(5000, 0.75), (10000, 0.50)]
    )

    # Simulate what backtester does
    levels = []
    for level_tuple in config.stop_loss_levels:
        dd, gr = level_tuple
        levels.append(StopLossLevel(
            drawdown_threshold=dd,
            gross_reduction=gr
        ))
    manager = StopLossManager(levels)

    # Test it works
    gross, _ = manager.update(100000)
    assert gross == 1.0

    gross, _ = manager.update(88000)  # $12k DD
    assert gross == 0.50

    gross, _ = manager.update(91000)  # $9k DD
    assert gross == 1.0  # Jumped to no stop loss

    print("✓ Test passed: Config to Manager integration")


if __name__ == '__main__':
    test_config_2_tuple_only()
    test_config_rejects_3_tuple()
    test_config_to_manager()
    print("\n✅ All integration tests passed!")

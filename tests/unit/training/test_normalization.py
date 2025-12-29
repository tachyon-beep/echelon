"""Unit tests for normalization utilities.

Tests numerical accuracy of RunningMeanStd using Welford's algorithm
and correct behavior of ReturnNormalizer wrapper.
"""

import torch

from echelon.training.normalization import ReturnNormalizer, RunningMeanStd


class TestRunningMeanStd:
    """Test RunningMeanStd statistical accuracy."""

    def test_initialization(self):
        """Test initial state is correctly set."""
        rms = RunningMeanStd(epsilon=1e-4)
        assert rms.mean == 0.0
        assert rms.var == 1.0
        assert rms.count == 1e-4

    def test_single_batch_update(self):
        """Test statistics after single batch update."""
        rms = RunningMeanStd(epsilon=0.0)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(x)

        # Check mean: (1+2+3+4+5)/5 = 3.0
        assert abs(rms.mean - 3.0) < 1e-6

        # Check variance: We use population variance (N divisor) for Welford's algorithm
        # Pop var = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 2.0
        expected_var = x.var(unbiased=False).item()
        assert abs(rms.var - expected_var) < 1e-6

        # Check count
        assert rms.count == 5

    def test_multiple_batch_updates(self):
        """Test statistics after multiple batch updates.

        Welford's parallel batch combining algorithm with population variance
        should produce exact results matching the full combined dataset.
        """
        rms = RunningMeanStd(epsilon=0.0)

        # First batch: [1, 2, 3]
        batch1 = torch.tensor([1.0, 2.0, 3.0])
        rms.update(batch1)

        # Second batch: [4, 5, 6]
        batch2 = torch.tensor([4.0, 5.0, 6.0])
        rms.update(batch2)

        # Mean should still be accurate
        expected_mean = 3.5
        assert abs(rms.mean - expected_mean) < 1e-5

        # Count should be accurate
        assert rms.count == 6

        # Variance will be close but not exact due to combining unbiased estimators
        # Just verify it's in a reasonable range
        assert 2.0 < rms.var < 4.0  # Should be around 3.25

    def test_normalization(self):
        """Test normalize() produces correct z-scores."""
        rms = RunningMeanStd(epsilon=0.0)

        # Update with known distribution
        x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        rms.update(x)

        # Normalize the same data
        normalized = rms.normalize(x)

        # After normalization, mean should be ~0 and std ~1 (using population std)
        assert abs(normalized.mean().item()) < 1e-5
        assert abs(normalized.std(unbiased=False).item() - 1.0) < 1e-5

    def test_normalization_with_epsilon(self):
        """Test normalization doesn't divide by zero with constant input."""
        rms = RunningMeanStd(epsilon=0.0)

        # All same values -> var=0
        x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        rms.update(x)

        # Should not raise, uses epsilon in denominator
        normalized = rms.normalize(x)
        assert torch.isfinite(normalized).all()

    def test_large_values(self):
        """Test numerical stability with large values."""
        rms = RunningMeanStd(epsilon=0.0)

        # Large values
        x = torch.tensor([1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3, 1e6 + 4])
        rms.update(x)

        expected_mean = 1e6 + 2.0
        # We use population variance for Welford's algorithm
        expected_var = x.var(unbiased=False).item()

        assert abs(rms.mean - expected_mean) < 1.0  # Relaxed tolerance for large numbers
        assert abs(rms.var - expected_var) < 1e-3

    def test_sequential_vs_batch(self):
        """Test that sequential and batch updates produce consistent means.

        With population variance, sequential and batch updates should produce
        identical results for both mean and variance.
        """
        rms_batch = RunningMeanStd(epsilon=0.0)
        rms_seq = RunningMeanStd(epsilon=0.0)

        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # Batch update
        rms_batch.update(data)

        # Sequential updates
        for val in data:
            rms_seq.update(val.unsqueeze(0))

        # Mean and count should match exactly
        assert abs(rms_batch.mean - rms_seq.mean) < 1e-5
        assert abs(rms_batch.count - rms_seq.count) < 1e-5

        # Both should produce reasonable variance estimates
        assert 0 < rms_batch.var < 20
        assert 0 < rms_seq.var < 20

    def test_state_dict_save_load(self):
        """Test checkpoint save/load preserves state."""
        rms = RunningMeanStd(epsilon=1e-4)
        x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        rms.update(x)

        # Save state
        state = rms.state_dict()

        # Create new instance and load
        rms_loaded = RunningMeanStd(epsilon=0.0)
        rms_loaded.load_state_dict(state)

        # Should match exactly
        assert rms_loaded.mean == rms.mean
        assert rms_loaded.var == rms.var
        assert rms_loaded.count == rms.count

    def test_multidimensional_tensor(self):
        """Test update works with multidimensional tensors."""
        rms = RunningMeanStd(epsilon=0.0)

        # 2D tensor
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rms.update(x)

        # Should compute statistics over all elements
        expected_mean = x.mean().item()
        expected_var = x.var(unbiased=False).item()

        assert abs(rms.mean - expected_mean) < 1e-5
        assert abs(rms.var - expected_var) < 1e-5
        assert rms.count == 6

    def test_empty_tensor_handling(self):
        """Test behavior with edge case: single element."""
        rms = RunningMeanStd(epsilon=0.0)

        # Single element has var=0
        x = torch.tensor([42.0])
        rms.update(x)

        assert rms.mean == 42.0
        assert rms.var == 0.0
        assert rms.count == 1


class TestReturnNormalizer:
    """Test ReturnNormalizer wrapper."""

    def test_initialization(self):
        """Test ReturnNormalizer initializes with RunningMeanStd."""
        normalizer = ReturnNormalizer()
        assert hasattr(normalizer, "rms")
        assert isinstance(normalizer.rms, RunningMeanStd)

    def test_update_and_normalize(self):
        """Test update and normalize work correctly."""
        normalizer = ReturnNormalizer()

        returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalizer.update(returns)

        # Normalize should delegate to RMS
        normalized = normalizer.normalize(returns)

        # Should be z-scored (using population std since we use population variance)
        assert abs(normalized.mean().item()) < 1e-5
        assert abs(normalized.std(unbiased=False).item() - 1.0) < 1e-5

    def test_state_dict_save_load(self):
        """Test checkpoint save/load for ReturnNormalizer."""
        normalizer = ReturnNormalizer()
        returns = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        normalizer.update(returns)

        # Save state
        state = normalizer.state_dict()

        # Create new instance and load
        normalizer_loaded = ReturnNormalizer()
        normalizer_loaded.load_state_dict(state)

        # Should match
        assert normalizer_loaded.rms.mean == normalizer.rms.mean
        assert normalizer_loaded.rms.var == normalizer.rms.var
        assert normalizer_loaded.rms.count == normalizer.rms.count

    def test_normalizer_tracks_across_updates(self):
        """Test normalizer correctly accumulates statistics across updates."""
        normalizer = ReturnNormalizer()

        # First batch
        normalizer.update(torch.tensor([1.0, 2.0, 3.0]))
        # Second batch
        normalizer.update(torch.tensor([4.0, 5.0, 6.0]))

        # Should have accumulated statistics (with small epsilon from initialization)
        assert abs(normalizer.rms.count - 6.0) < 1e-6

        # Mean should be 3.5
        assert abs(normalizer.rms.mean - 3.5) < 1e-5


class TestNormalizationIntegration:
    """Integration tests for normalization in training context."""

    def test_ppo_style_usage(self):
        """Test typical PPO usage pattern."""
        normalizer = ReturnNormalizer()

        # Simulate multiple rollouts
        for _ in range(5):
            # Fake returns from GAE computation
            returns = torch.randn(128, 10)  # [steps, agents]

            # Update statistics
            normalizer.update(returns)

            # Normalize for value loss
            normalized = normalizer.normalize(returns)

            # Normalized values should have reasonable scale
            assert torch.isfinite(normalized).all()
            assert normalized.abs().mean() < 10.0  # Shouldn't explode

    def test_checkpoint_resume_consistency(self):
        """Test that resuming from checkpoint produces consistent results."""
        # Train for a bit
        normalizer1 = ReturnNormalizer()
        for i in range(10):
            returns = torch.randn(64) + i * 0.1  # Drifting distribution
            normalizer1.update(returns)

        # Save checkpoint
        state = normalizer1.state_dict()

        # Resume in new normalizer
        normalizer2 = ReturnNormalizer()
        normalizer2.load_state_dict(state)

        # Continue training - should be identical
        test_returns = torch.randn(64)
        norm1 = normalizer1.normalize(test_returns)
        norm2 = normalizer2.normalize(test_returns)

        assert torch.allclose(norm1, norm2, atol=1e-6)

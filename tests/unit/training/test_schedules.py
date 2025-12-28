"""Tests for training schedule functions."""

import sys
from pathlib import Path

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

from train_ppo import compute_linear_decay


def test_decay_at_zero_progress():
    """At progress=0, value equals initial."""
    result = compute_linear_decay(initial=3e-4, decay_factor=0.9, progress=0.0, floor=1e-5)
    assert result == 3e-4


def test_decay_at_full_progress():
    """At progress=1, value is initial * (1 - decay_factor)."""
    result = compute_linear_decay(initial=3e-4, decay_factor=0.9, progress=1.0, floor=1e-5)
    assert abs(result - 3e-5) < 1e-10  # 3e-4 * 0.1 = 3e-5


def test_decay_respects_floor():
    """Decay never goes below floor."""
    result = compute_linear_decay(initial=1e-4, decay_factor=0.99, progress=1.0, floor=5e-5)
    # Without floor: 1e-4 * 0.01 = 1e-6, but floor is 5e-5
    assert result == 5e-5


def test_decay_at_half_progress():
    """At progress=0.5, value is halfway decayed."""
    result = compute_linear_decay(initial=0.05, decay_factor=0.9, progress=0.5, floor=0.005)
    # 0.05 * (1 - 0.9 * 0.5) = 0.05 * 0.55 = 0.0275
    assert abs(result - 0.0275) < 1e-10


def test_zero_decay_factor_returns_initial():
    """With decay_factor=0, value stays at initial."""
    result = compute_linear_decay(initial=3e-4, decay_factor=0.0, progress=1.0, floor=1e-5)
    assert result == 3e-4

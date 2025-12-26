"""Tests for the extracted observations module.

TDD-first tests that verify the ObservationBuilder interface
and that extraction preserves behavior exactly.
"""

import numpy as np
import pytest

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


# RED: These imports will fail until we create observations.py
def test_import_observation_builder():
    """ObservationBuilder can be imported from observations module."""
    from echelon.env.observations import ObservationBuilder

    assert ObservationBuilder is not None


def test_import_observation_context():
    """ObservationContext can be imported from observations module."""
    from echelon.env.observations import ObservationContext

    assert ObservationContext is not None


def test_import_helper_functions():
    """Helper functions can be imported from observations module."""
    from echelon.env.observations import (
        compute_acoustic_intensities,
        compute_contact_features,
        compute_ewar_levels,
        compute_local_map,
    )

    assert compute_contact_features is not None
    assert compute_ewar_levels is not None
    assert compute_local_map is not None
    assert compute_acoustic_intensities is not None


class TestObservationBuilderInterface:
    """Verify ObservationBuilder has the correct interface."""

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            max_episode_seconds=10.0,
        )
        return EchelonEnv(cfg)

    def test_builder_creation(self, env):
        """ObservationBuilder can be created with env config."""
        from echelon.env.observations import ObservationBuilder

        builder = ObservationBuilder(
            config=env.config,
            max_contact_slots=env.MAX_CONTACT_SLOTS,
            contact_dim=env.CONTACT_DIM,
            local_map_r=env.LOCAL_MAP_R,
            comm_dim=env.comm_dim,
        )
        assert builder is not None

    def test_builder_obs_dim(self, env):
        """ObservationBuilder.obs_dim() matches env._obs_dim()."""
        from echelon.env.observations import ObservationBuilder

        builder = ObservationBuilder(
            config=env.config,
            max_contact_slots=env.MAX_CONTACT_SLOTS,
            contact_dim=env.CONTACT_DIM,
            local_map_r=env.LOCAL_MAP_R,
            comm_dim=env.comm_dim,
        )
        assert builder.obs_dim() == env._obs_dim()

    def test_builder_produces_valid_observations(self, env):
        """ObservationBuilder.build() produces valid observation dict."""
        from echelon.env.observations import ObservationBuilder, ObservationContext

        env.reset(seed=0)
        builder = ObservationBuilder(
            config=env.config,
            max_contact_slots=env.MAX_CONTACT_SLOTS,
            contact_dim=env.CONTACT_DIM,
            local_map_r=env.LOCAL_MAP_R,
            comm_dim=env.comm_dim,
        )

        # Build context from env state
        ctx = ObservationContext.from_env(env)
        obs = builder.build(ctx)

        # Basic validity checks
        assert isinstance(obs, dict)
        assert len(obs) == len(env.agents)
        for _aid, agent_obs in obs.items():
            assert agent_obs.shape == (builder.obs_dim(),)
            assert np.all(np.isfinite(agent_obs))


class TestObservationEquivalence:
    """Verify extracted ObservationBuilder produces identical results to env._obs()."""

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            max_episode_seconds=10.0,
        )
        return EchelonEnv(cfg)

    def test_observations_match_after_reset(self, env):
        """Observations from builder match env._obs() after reset."""
        from echelon.env.observations import ObservationBuilder, ObservationContext

        env.reset(seed=42)
        builder = ObservationBuilder(
            config=env.config,
            max_contact_slots=env.MAX_CONTACT_SLOTS,
            contact_dim=env.CONTACT_DIM,
            local_map_r=env.LOCAL_MAP_R,
            comm_dim=env.comm_dim,
        )

        # Get observations both ways
        env_obs = env._obs()
        ctx = ObservationContext.from_env(env)
        builder_obs = builder.build(ctx)

        # Compare
        for aid in env.agents:
            np.testing.assert_allclose(
                env_obs[aid],
                builder_obs[aid],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Observations differ for {aid}",
            )

    def test_observations_match_after_steps(self, env):
        """Observations from builder match env._obs() after steps."""
        from echelon.env.observations import ObservationBuilder, ObservationContext

        env.reset(seed=42)
        builder = ObservationBuilder(
            config=env.config,
            max_contact_slots=env.MAX_CONTACT_SLOTS,
            contact_dim=env.CONTACT_DIM,
            local_map_r=env.LOCAL_MAP_R,
            comm_dim=env.comm_dim,
        )

        # Run a few steps
        for _ in range(5):
            actions = {aid: np.random.uniform(-1, 1, env.ACTION_DIM).astype(np.float32) for aid in env.agents}
            obs, _, terms, truncs, _ = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break

            # Compare observations
            env_obs = env._obs()
            ctx = ObservationContext.from_env(env)
            builder_obs = builder.build(ctx)

            for aid in env.agents:
                np.testing.assert_allclose(
                    env_obs[aid],
                    builder_obs[aid],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Observations differ for {aid}",
                )


class TestContactFeatures:
    """Test isolated contact feature computation."""

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            max_episode_seconds=10.0,
        )
        return EchelonEnv(cfg)

    def test_contact_features_dimension(self, env):
        """Contact features have correct dimension."""
        from echelon.env.observations import compute_contact_features

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]
        other = env.sim.mechs["red_0"]

        feat = compute_contact_features(
            viewer=viewer,
            other=other,
            world=env.world,
            relation="hostile",
            painted_by_pack=False,
            contact_dim=env.CONTACT_DIM,
        )

        assert feat.shape == (env.CONTACT_DIM,)

    def test_contact_features_visible_flag(self, env):
        """Visible flag is set correctly."""
        from echelon.env.observations import compute_contact_features

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]
        other = env.sim.mechs["red_0"]

        feat = compute_contact_features(
            viewer=viewer,
            other=other,
            world=env.world,
            relation="hostile",
            painted_by_pack=False,
            contact_dim=env.CONTACT_DIM,
        )

        # visible flag is at index 21
        assert feat[21] == 1.0

    def test_contact_features_relation_encoding(self, env):
        """Relation one-hot encoding is correct."""
        from echelon.env.observations import compute_contact_features

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]
        other = env.sim.mechs["red_0"]

        # hostile relation
        feat_hostile = compute_contact_features(
            viewer=viewer,
            other=other,
            world=env.world,
            relation="hostile",
            painted_by_pack=False,
            contact_dim=env.CONTACT_DIM,
        )
        # relation one-hot at indices 13-15: [friendly, hostile, neutral]
        assert feat_hostile[13] == 0.0  # not friendly
        assert feat_hostile[14] == 1.0  # hostile
        assert feat_hostile[15] == 0.0  # not neutral

        # friendly relation
        friendly = env.sim.mechs["blue_1"]
        feat_friendly = compute_contact_features(
            viewer=viewer,
            other=friendly,
            world=env.world,
            relation="friendly",
            painted_by_pack=False,
            contact_dim=env.CONTACT_DIM,
        )
        assert feat_friendly[13] == 1.0  # friendly
        assert feat_friendly[14] == 0.0  # not hostile
        assert feat_friendly[15] == 0.0  # not neutral


class TestEwarLevels:
    """Test EWAR level computation."""

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            max_episode_seconds=10.0,
        )
        return EchelonEnv(cfg)

    def test_ewar_levels_returns_tuple(self, env):
        """compute_ewar_levels returns (sensor_quality, jam_level, eccm_level)."""
        from echelon.env.observations import compute_ewar_levels

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]

        result = compute_ewar_levels(viewer, env.sim)

        assert isinstance(result, tuple)
        assert len(result) == 3
        sensor_quality, jam_level, eccm_level = result
        assert sensor_quality >= 0.0
        assert 0.0 <= jam_level <= 1.0
        assert 0.0 <= eccm_level <= 1.0

    def test_ewar_levels_dead_mech(self, env):
        """Dead mech has minimum sensor quality."""
        from echelon.config import SENSOR_QUALITY_MIN
        from echelon.env.observations import compute_ewar_levels

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]
        viewer.alive = False

        sensor_quality, jam_level, eccm_level = compute_ewar_levels(viewer, env.sim)

        assert sensor_quality == float(SENSOR_QUALITY_MIN)
        assert jam_level == 0.0
        assert eccm_level == 0.0


class TestLocalMap:
    """Test local map computation."""

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            max_episode_seconds=10.0,
        )
        return EchelonEnv(cfg)

    def test_local_map_dimension(self, env):
        """Local map has correct dimension."""
        from echelon.env.observations import compute_local_map

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]

        # Need occupancy_2d - use cached from env
        local_map = compute_local_map(
            viewer=viewer,
            world=env.world,
            occupancy_2d=env._cached_occupancy_2d,
            local_map_r=env.LOCAL_MAP_R,
        )

        expected_dim = env.LOCAL_MAP_DIM
        assert local_map.shape == (expected_dim,)

    def test_local_map_bounded(self, env):
        """Local map values are bounded [0, 1]."""
        from echelon.env.observations import compute_local_map

        env.reset(seed=0)
        viewer = env.sim.mechs["blue_0"]

        local_map = compute_local_map(
            viewer=viewer,
            world=env.world,
            occupancy_2d=env._cached_occupancy_2d,
            local_map_r=env.LOCAL_MAP_R,
        )

        assert np.all(local_map >= 0.0)
        assert np.all(local_map <= 1.0)


class TestObservationBuilderIntegration:
    """Test that EchelonEnv uses ObservationBuilder internally.

    These tests verify the refactoring was completed correctly:
    - Env has _obs_builder attribute
    - Inline helper methods are removed
    - Observations still work correctly
    """

    @pytest.fixture
    def env(self):
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            max_episode_seconds=10.0,
        )
        return EchelonEnv(cfg)

    def test_env_has_obs_builder_attribute(self, env):
        """EchelonEnv has _obs_builder attribute after init."""
        from echelon.env.observations import ObservationBuilder

        assert hasattr(env, "_obs_builder")
        assert isinstance(env._obs_builder, ObservationBuilder)

    def test_env_obs_builder_initialized_correctly(self, env):
        """ObservationBuilder is initialized with correct parameters."""
        builder = env._obs_builder
        assert builder.max_contact_slots == env.MAX_CONTACT_SLOTS
        assert builder.contact_dim == env.CONTACT_DIM
        assert builder.local_map_r == env.LOCAL_MAP_R
        assert builder.comm_dim == env.comm_dim

    def test_inline_helpers_removed(self, env):
        """Inline observation helper methods are removed from env.

        After refactoring, these methods should no longer exist on env.
        The logic now lives in observations.py module functions.
        """
        # These methods should NOT exist after refactoring
        assert not hasattr(env, "_contact_features")
        assert not hasattr(env, "_ewar_levels")
        assert not hasattr(env, "_local_map")

    def test_obs_still_works_after_refactoring(self, env):
        """_obs() produces valid observations after refactoring."""
        env.reset(seed=123)

        obs = env._obs()

        assert isinstance(obs, dict)
        assert len(obs) == len(env.agents)
        for _aid, agent_obs in obs.items():
            assert agent_obs.shape == (env._obs_dim(),)
            assert np.all(np.isfinite(agent_obs))

    def test_obs_dim_delegated_to_builder(self, env):
        """_obs_dim() returns same value as builder.obs_dim()."""
        assert env._obs_dim() == env._obs_builder.obs_dim()

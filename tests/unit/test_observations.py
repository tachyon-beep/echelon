"""Observation system tests.

Verify observations are always valid:
- No NaN or Inf values
- Dimensions match declared space
- Contact slots have valid structure
- Self-state information is present
"""

import numpy as np
import pytest

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


@pytest.fixture
def obs_env():
    """Environment for observation testing."""
    cfg = EnvConfig(
        world=WorldConfig(size_x=30, size_y=30, size_z=15),
        num_packs=1,
        seed=0,
        max_episode_seconds=10.0,
    )
    return EchelonEnv(cfg)


class TestObservationSanitization:
    """Verify observations are always valid."""

    def test_no_nan_in_observations(self, obs_env):
        """Observations never contain NaN."""
        env = obs_env
        obs, _ = env.reset(seed=0)

        for aid, agent_obs in obs.items():
            assert np.all(np.isfinite(agent_obs)), f"NaN/Inf in {aid} observation"

        # Run a few steps
        for _ in range(10):
            actions = {aid: np.random.uniform(-1, 1, env.ACTION_DIM).astype(np.float32) for aid in obs}
            obs, _, terms, truncs, _ = env.step(actions)

            for aid, agent_obs in obs.items():
                assert np.all(np.isfinite(agent_obs)), f"NaN/Inf in {aid} observation after step"

            if all(terms.values()) or all(truncs.values()):
                break

    def test_observation_dimension_matches_space(self, obs_env):
        """Observation dimension matches declared observation space."""
        env = obs_env
        obs, _ = env.reset(seed=0)

        expected_dim = env._obs_dim()
        for aid, agent_obs in obs.items():
            assert agent_obs.shape == (
                expected_dim,
            ), f"{aid} obs shape {agent_obs.shape} != expected ({expected_dim},)"

    def test_observations_bounded(self, obs_env):
        """Observations don't have extreme values."""
        env = obs_env
        obs, _ = env.reset(seed=0)

        MAX_REASONABLE = 1e6  # Observations should be normalized/reasonable

        for aid, agent_obs in obs.items():
            assert np.all(
                np.abs(agent_obs) < MAX_REASONABLE
            ), f"{aid} has extreme values: max={np.abs(agent_obs).max()}"


class TestContactSlots:
    """Verify contact slot structure."""

    def test_contact_slots_valid_structure(self, obs_env):
        """Contact slots have expected dimensions."""
        env = obs_env
        obs, _ = env.reset(seed=0)

        for _aid, agent_obs in obs.items():
            contacts_total = env.CONTACT_SLOTS * env.CONTACT_DIM
            contact_data = agent_obs[:contacts_total]

            # Should be able to reshape without error
            contacts = contact_data.reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)
            assert contacts.shape == (env.CONTACT_SLOTS, env.CONTACT_DIM)

    def test_visible_contacts_have_nonzero_features(self, obs_env):
        """Visible contacts have meaningful data."""
        env = obs_env
        obs, _ = env.reset(seed=0)

        # At reset, some contacts should be visible (teammates)
        for _aid, agent_obs in obs.items():
            contacts_total = env.CONTACT_SLOTS * env.CONTACT_DIM
            contacts = agent_obs[:contacts_total].reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)

            # visible flag is at index 21
            visible_mask = contacts[:, 21] > 0.5

            if visible_mask.any():
                # Visible contacts should have non-zero relative position
                visible_contacts = contacts[visible_mask]
                # rel_x, rel_y, rel_z at indices 0, 1, 2
                rel_pos = visible_contacts[:, 0:3]
                assert np.any(rel_pos != 0), "Visible contacts should have non-zero relative position"


class TestSelfState:
    """Verify self-state information in observations."""

    def test_self_state_present(self, obs_env):
        """Agent's own state is encoded in observation."""
        env = obs_env
        obs, _ = env.reset(seed=0)

        # Observations should contain self-features at the end
        # The exact structure depends on implementation, but the obs should be non-trivial
        for _aid, agent_obs in obs.items():
            # Self features are after contacts, comm, local_map, telemetry
            # Just verify the observation is long enough to contain self-state
            min_expected = env.CONTACT_SLOTS * env.CONTACT_DIM + 10  # Some self features
            assert (
                len(agent_obs) > min_expected
            ), f"Observation too short to contain self-state: {len(agent_obs)}"

    def test_different_positions_different_observations(self, obs_env):
        """Mechs at different positions have different observations."""
        env = obs_env
        env.reset(seed=0)

        # Get observations for two mechs at different positions
        mech_a = env.sim.mechs["blue_0"]
        mech_b = env.sim.mechs["blue_1"]

        # Ensure they're at different positions
        mech_a.pos[:] = [10.0, 10.0, 1.0]
        mech_b.pos[:] = [25.0, 25.0, 1.0]

        obs = env._obs()
        obs_a = obs["blue_0"]
        obs_b = obs["blue_1"]

        # Observations should be different
        assert not np.allclose(
            obs_a, obs_b
        ), "Mechs at different positions should have different observations"


class TestPaintUsageObservation:
    """Verify paint usage feedback in observations.

    The my_paint_used observation combines three signals:
    - paint_applied: scout successfully painted a target
    - paint_assists: painted target took damage from teammates
    - paint_kills: painted target was killed
    """

    def test_my_paint_used_updates_on_paint_activity(self, obs_env):
        """my_paint_used observation reflects any paint-related activity."""
        env = obs_env
        env.reset(seed=0)

        scout_id = "blue_0"

        # Manually set up paint state - could be paint applied, assist, or kill
        env._paint_used_this_step = {scout_id: 1}

        obs = env._obs()
        scout_obs = obs[scout_id]

        # my_paint_used is in self features, before formation_mode_onehot (self_dim = 51)
        # Layout: ...my_paint_used, formation_close, formation_standard, formation_loose
        self_features_end = scout_obs[-51:]
        my_paint_used_value = self_features_end[-4]  # 4th from end (before 3 formation one-hots)
        assert my_paint_used_value == 1.0, f"my_paint_used should be 1.0, got {my_paint_used_value}"

    def test_my_paint_used_zero_without_activity(self, obs_env):
        """my_paint_used is 0 when no paint activity this step."""
        env = obs_env
        env.reset(seed=0)

        scout_id = "blue_0"

        # No paint activity
        env._paint_used_this_step = {}

        obs = env._obs()
        scout_obs = obs[scout_id]

        self_features_end = scout_obs[-51:]
        my_paint_used_value = self_features_end[-4]  # 4th from end
        assert my_paint_used_value == 0.0, f"my_paint_used should be 0.0, got {my_paint_used_value}"

    def test_my_paint_used_combines_multiple_events(self, obs_env):
        """my_paint_used reflects combined count of paint events."""
        env = obs_env
        env.reset(seed=0)

        scout_id = "blue_0"

        # Multiple paint events: 1 applied + 2 assists = 3 total
        env._paint_used_this_step = {scout_id: 3}

        obs = env._obs()
        scout_obs = obs[scout_id]

        self_features_end = scout_obs[-51:]
        my_paint_used_value = self_features_end[-4]  # 4th from end
        # Observation is binary (> 0), so still 1.0
        assert my_paint_used_value == 1.0, f"my_paint_used should be 1.0, got {my_paint_used_value}"


class TestFormationModeObservation:
    """Verify formation mode appears in observations."""

    def test_formation_mode_in_observation(self, obs_env):
        """Formation mode one-hot is in self-features."""

        env = obs_env
        # Default is STANDARD
        env.reset(seed=0)

        obs = env._obs()
        scout_obs = obs["blue_0"]

        # self_dim increased by 3 (one-hot for formation)
        # Formation mode is in self-features, after my_paint_used
        self_features = scout_obs[-51:]  # 48 + 3 = 51

        # STANDARD = [0, 1, 0]
        formation_one_hot = self_features[-3:]
        assert formation_one_hot[0] == 0.0  # not CLOSE
        assert formation_one_hot[1] == 1.0  # STANDARD
        assert formation_one_hot[2] == 0.0  # not LOOSE

    def test_formation_mode_close_encoding(self, obs_env):
        """CLOSE formation encodes as [1, 0, 0]."""
        from echelon import EchelonEnv
        from echelon.config import EnvConfig, FormationMode, WorldConfig

        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            formation_mode=FormationMode.CLOSE,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)

        obs = env._obs()
        scout_obs = obs["blue_0"]
        formation_one_hot = scout_obs[-3:]

        assert formation_one_hot[0] == 1.0  # CLOSE
        assert formation_one_hot[1] == 0.0
        assert formation_one_hot[2] == 0.0

    def test_formation_mode_loose_encoding(self, obs_env):
        """LOOSE formation encodes as [0, 0, 1]."""
        from echelon import EchelonEnv
        from echelon.config import EnvConfig, FormationMode, WorldConfig

        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            formation_mode=FormationMode.LOOSE,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)

        obs = env._obs()
        scout_obs = obs["blue_0"]
        formation_one_hot = scout_obs[-3:]

        assert formation_one_hot[0] == 0.0
        assert formation_one_hot[1] == 0.0
        assert formation_one_hot[2] == 1.0  # LOOSE

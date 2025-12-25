"""Tests for Combat Suite system."""

from __future__ import annotations

import pytest
import torch

from echelon.rl.suite import (
    COMBAT_SUITES,
    SUITE_DESCRIPTOR_DIM,
    DeepSetEncoder,
    SuiteType,
    build_suite_descriptor,
    get_suite_for_role,
)


class TestDeepSetEncoder:
    """Test DeepSetEncoder functionality."""

    def test_output_shape(self):
        """Encoder produces correct output shape."""
        encoder = DeepSetEncoder(entity_dim=25, hidden_dim=32, output_dim=64)
        batch_size = 4
        max_entities = 10

        entities = torch.randn(batch_size, max_entities, 25)
        mask = torch.zeros(batch_size, max_entities)
        # Mark last 3 entities as padding
        mask[:, 7:] = 1.0

        output = encoder(entities, mask)

        assert output.shape == (batch_size, 64)

    def test_permutation_invariance(self):
        """Encoder is invariant to entity order."""
        encoder = DeepSetEncoder(entity_dim=10, hidden_dim=16, output_dim=8)
        encoder.eval()

        # Create 5 entities
        entities = torch.randn(1, 5, 10)
        mask = torch.zeros(1, 5)

        # Get output for original order
        out1 = encoder(entities, mask)

        # Permute entities
        perm = torch.tensor([3, 1, 4, 0, 2])
        entities_perm = entities[:, perm, :]
        out2 = encoder(entities_perm, mask)

        # Outputs should be identical
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

    def test_masking_works(self):
        """Masked entities don't affect output."""
        encoder = DeepSetEncoder(entity_dim=10, hidden_dim=16, output_dim=8)
        encoder.eval()

        # 3 real entities, 2 padding
        entities = torch.randn(1, 5, 10)
        mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])

        out1 = encoder(entities, mask)

        # Change padding values - should not affect output
        entities_modified = entities.clone()
        entities_modified[:, 3:, :] = torch.randn(1, 2, 10) * 1000
        out2 = encoder(entities_modified, mask)

        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

    def test_empty_set_handling(self):
        """Encoder handles empty sets (all masked)."""
        encoder = DeepSetEncoder(entity_dim=10, hidden_dim=16, output_dim=8)

        entities = torch.randn(1, 5, 10)
        mask = torch.ones(1, 5)  # All padding

        # Should not crash, should return zeros (for mean/sum)
        output = encoder(entities, mask)
        assert output.shape == (1, 8)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("aggregation", ["sum", "max", "mean"])
    def test_aggregation_modes(self, aggregation):
        """All aggregation modes work."""
        encoder = DeepSetEncoder(entity_dim=10, hidden_dim=16, output_dim=8, aggregation=aggregation)

        entities = torch.randn(2, 5, 10)
        mask = torch.zeros(2, 5)
        mask[:, 3:] = 1.0  # 3 real, 2 padding

        output = encoder(entities, mask)
        assert output.shape == (2, 8)
        assert not torch.isnan(output).any()


class TestCombatSuiteSpec:
    """Test CombatSuiteSpec dataclass."""

    def test_all_suites_defined(self):
        """All suite types have definitions."""
        for suite_type in SuiteType:
            assert suite_type in COMBAT_SUITES

    def test_scout_has_most_contacts(self):
        """Scout has the most contact slots."""
        scout = COMBAT_SUITES[SuiteType.SCOUT_RECON]
        for suite_type, spec in COMBAT_SUITES.items():
            if suite_type != SuiteType.SCOUT_RECON:
                assert (
                    scout.visual_contact_slots >= spec.visual_contact_slots
                ), f"Scout should have >= contacts than {suite_type}"

    def test_heavy_has_fewest_contacts(self):
        """Heavy has the fewest contact slots among line mechs."""
        heavy = COMBAT_SUITES[SuiteType.HEAVY_FIRE_SUPPORT]
        line_suites = [
            SuiteType.SCOUT_RECON,
            SuiteType.LIGHT_SKIRMISH,
            SuiteType.MEDIUM_ASSAULT,
            SuiteType.HEAVY_FIRE_SUPPORT,
        ]
        for suite_type in line_suites:
            if suite_type != SuiteType.HEAVY_FIRE_SUPPORT:
                spec = COMBAT_SUITES[suite_type]
                assert (
                    heavy.visual_contact_slots <= spec.visual_contact_slots
                ), f"Heavy should have <= contacts than {suite_type}"

    def test_command_suites_can_issue_orders(self):
        """Command suites can issue orders."""
        pack_cmd = COMBAT_SUITES[SuiteType.PACK_COMMAND]
        squad_cmd = COMBAT_SUITES[SuiteType.SQUAD_COMMAND]

        assert pack_cmd.issues_orders
        assert pack_cmd.order_scope == "pack"
        assert squad_cmd.issues_orders
        assert squad_cmd.order_scope == "squad"

    def test_line_suites_cannot_issue_orders(self):
        """Line mechs cannot issue orders."""
        line_suites = [
            SuiteType.SCOUT_RECON,
            SuiteType.LIGHT_SKIRMISH,
            SuiteType.MEDIUM_ASSAULT,
            SuiteType.HEAVY_FIRE_SUPPORT,
        ]
        for suite_type in line_suites:
            spec = COMBAT_SUITES[suite_type]
            assert not spec.issues_orders
            assert spec.order_scope == "none"

    def test_command_suites_see_squad_engaged(self):
        """Command suites can see what squadmates are fighting."""
        pack_cmd = COMBAT_SUITES[SuiteType.PACK_COMMAND]
        squad_cmd = COMBAT_SUITES[SuiteType.SQUAD_COMMAND]

        assert pack_cmd.sees_squad_engaged
        assert squad_cmd.sees_squad_engaged

    def test_squad_command_has_full_squad_awareness(self):
        """Squad commander can see entire squad."""
        squad_cmd = COMBAT_SUITES[SuiteType.SQUAD_COMMAND]
        assert squad_cmd.squad_position_slots == 12  # Full squad minus self
        assert squad_cmd.squad_detail_level == 2  # Full detail


class TestGetSuiteForRole:
    """Test suite assignment logic."""

    def test_squad_leader_gets_squad_command(self):
        """Squad leader role overrides mech class."""
        suite = get_suite_for_role("medium", "squad_leader")
        assert suite.suite_type == SuiteType.SQUAD_COMMAND

    def test_pack_leader_gets_pack_command(self):
        """Pack leader role overrides mech class."""
        suite = get_suite_for_role("light", "pack_leader")
        assert suite.suite_type == SuiteType.PACK_COMMAND

    def test_line_mechs_get_class_appropriate_suites(self):
        """Line mechs get suites based on their class."""
        assert get_suite_for_role("scout", None).suite_type == SuiteType.SCOUT_RECON
        assert get_suite_for_role("light", None).suite_type == SuiteType.LIGHT_SKIRMISH
        assert get_suite_for_role("medium", None).suite_type == SuiteType.MEDIUM_ASSAULT
        assert get_suite_for_role("heavy", None).suite_type == SuiteType.HEAVY_FIRE_SUPPORT


class TestSuiteDescriptor:
    """Test suite descriptor vector generation."""

    def test_descriptor_dimension(self):
        """Descriptor has correct dimension."""
        for suite in COMBAT_SUITES.values():
            desc = build_suite_descriptor(suite)
            assert len(desc) == SUITE_DESCRIPTOR_DIM

    def test_descriptor_has_one_hot(self):
        """Descriptor starts with valid one-hot encoding."""
        for suite in COMBAT_SUITES.values():
            desc = build_suite_descriptor(suite)
            one_hot = desc[:6]

            # Exactly one element is 1.0
            assert sum(one_hot) == 1.0
            assert all(v in (0.0, 1.0) for v in one_hot)

    def test_descriptor_values_normalized(self):
        """Descriptor values are in reasonable ranges."""
        for suite in COMBAT_SUITES.values():
            desc = build_suite_descriptor(suite)

            # All values should be in [0, 2] range (sensor_range_mult can be > 1)
            assert all(-0.1 <= v <= 2.1 for v in desc)

    def test_different_suites_have_different_descriptors(self):
        """Each suite type produces a unique descriptor."""
        descriptors = {}
        for suite_type, suite in COMBAT_SUITES.items():
            desc_tuple = tuple(build_suite_descriptor(suite))
            assert desc_tuple not in descriptors.values(), f"{suite_type} has duplicate descriptor"
            descriptors[suite_type] = desc_tuple

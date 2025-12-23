import numpy as np
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from echelon.actions import ACTION_DIM, ActionIndex
from echelon.config import MechClassConfig
from echelon.sim.mech import MechState
from echelon.sim.sim import Sim
from echelon.sim.world import VoxelWorld

# Define strategies for inputs
mech_class_names = st.sampled_from(["scout", "light", "medium", "heavy"])


def random_mech_config(name):
    # Minimal config for testing
    return MechClassConfig(
        name=name,
        size_voxels=(1.0, 1.0, 1.0),
        max_speed=5.0,
        max_yaw_rate=1.0,
        max_jet_accel=10.0,
        hp=100.0,
        leg_hp=50.0,
        heat_cap=100.0,
        heat_dissipation=10.0,
    )


class SimStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.world = VoxelWorld(voxels=np.zeros((10, 10, 10), dtype=np.uint8), voxel_size_m=5.0)
        self.sim = Sim(self.world, dt_sim=0.1, rng=np.random.default_rng(42))
        self.mech_ids = []

    @initialize(count=st.integers(min_value=1, max_value=5))
    def init_mechs(self, count):
        mechs = {}
        for i in range(count):
            mid = f"mech_{i}"
            mechs[mid] = MechState(
                mech_id=mid,
                team="blue" if i % 2 == 0 else "red",
                spec=random_mech_config("medium"),
                pos=np.array([5.0, 5.0, 1.0], dtype=np.float32),
                vel=np.zeros(3, dtype=np.float32),
                yaw=0.0,
                hp=100.0,
                leg_hp=50.0,
                heat=0.0,
                stability=100.0,
            )
            self.mech_ids.append(mid)
        self.sim.reset(mechs)

    @rule(
        mech_idx=st.integers(min_value=0, max_value=4),  # Max 5 mechs
        action=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=ACTION_DIM, max_size=ACTION_DIM),
    )
    def step_mech(self, mech_idx, action):
        if not self.mech_ids or mech_idx >= len(self.mech_ids):
            return

        mid = self.mech_ids[mech_idx]
        actions = {mid: np.array(action, dtype=np.float32)}

        # Pre-state capture
        mech_pre = self.sim.mechs[mid]
        # heat_pre = mech_pre.heat
        # hp_pre = mech_pre.hp

        self.sim.step(actions, num_substeps=1)

        # Post-state checks (Invariants)
        mech_post = self.sim.mechs[mid]

        # 1. Heat non-negative
        assert mech_post.heat >= 0.0, f"Heat went negative: {mech_post.heat}"

        # 2. Stability bounded
        assert (
            0.0 <= mech_post.stability <= mech_post.max_stability + 1e-6
        ), f"Stability OOB: {mech_post.stability}"

        # 3. Dead mechs state
        if not mech_post.alive:
            pass

        # 4. Shutdown mechs don't move (except gravity)
        if mech_post.shutdown and mech_post.fallen_time <= 0.0 and abs(action[ActionIndex.YAW_RATE]) > 0.1:
            assert abs(mech_post.yaw - mech_pre.yaw) < 1e-4, "Shutdown mech rotated!"


TestSim = SimStateMachine.TestCase

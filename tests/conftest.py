import pytest
import numpy as np
from echelon.sim.mech import MechState
from echelon.config import MechClassConfig

@pytest.fixture
def make_mech():
    def _make(mid: str, team: str, pos: list, cls_name: str) -> MechState:
        spec = MechClassConfig(
            name=cls_name,
            size_voxels=(1,1,1) if cls_name != "heavy" else (2,2,2),
            max_speed=5.0, max_yaw_rate=2.0, max_jet_accel=5.0,
            hp=100.0, leg_hp=50.0, heat_cap=100.0, heat_dissipation=10.0
        )
        return MechState(
            mech_id=mid, team=team, spec=spec,
            pos=np.array(pos, dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32),
            yaw=0.0, hp=spec.hp, leg_hp=spec.leg_hp, heat=0.0, stability=100.0
        )
    return _make

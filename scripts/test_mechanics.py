from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon.sim.sim import Sim, MISSILE
from echelon.sim.world import VoxelWorld, WorldConfig
from echelon.sim.mech import MechState
from echelon.config import MechClassConfig

def make_mech(mid: str, team: str, pos: list, cls_name: str) -> MechState:
    # Minimal config for test
    spec = MechClassConfig(
        name=cls_name,
        size_voxels=(1,1,1) if cls_name != "heavy" else (2,2,2),
        max_speed=5.0, max_yaw_rate=2.0, max_jet_accel=5.0,
        hp=100.0, heat_cap=100.0, heat_dissipation=10.0
    )
    return MechState(
        mech_id=mid, team=team, spec=spec,
        pos=np.array(pos, dtype=np.float32),
        vel=np.zeros(3, dtype=np.float32),
        yaw=0.0, hp=100.0, heat=0.0
    )

def test_indirect_fire():
    # Setup world
    # 20x20x20
    # Wall at x=10
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    # Clear world
    world.solid[:] = False
    # Build wall
    world.set_box_solid(10, 0, 0, 11, 20, 10, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    # Heavy at (5, 10, 0) facing +X (towards wall)
    heavy = make_mech("heavy", "blue", [5.0, 10.0, 1.0], "heavy")
    # Enemy at (15, 10, 0) behind wall
    enemy = make_mech("enemy", "red", [15.0, 10.0, 1.0], "medium")
    
    sim.reset({"heavy": heavy, "enemy": enemy})

    # Action to fire missile: [..., missile=1.0, ...]
    fire_action = np.zeros(8, dtype=np.float32)
    fire_action[6] = 1.0 # Missile

    # 1. Try fire without LOS or Paint
    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Should not fire without LOS or Paint"

    # 2. Paint the enemy
    enemy.painted_remaining = 5.0
    
    # 3. Try fire again
    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 1, "Should fire at painted target despite no LOS"
    assert len(sim.projectiles) == 1
    
    # 4. Verify projectile homing
    # Projectile should have velocity towards enemy
    p = sim.projectiles[0]
    assert p.target_id == "enemy"
    
    print("Indirect fire test passed!")

if __name__ == "__main__":
    test_indirect_fire()

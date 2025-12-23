# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig
from echelon.actions import ACTION_DIM, ActionIndex
from echelon.config import MechClassConfig, WorldConfig
from echelon.constants import PACK_SIZE
from echelon.sim.los import has_los
from echelon.sim.mech import MechState
from echelon.sim.sim import Sim
from echelon.sim.world import VoxelWorld


def make_mech(mid: str, team: str, pos: list, cls_name: str) -> MechState:
    # Minimal config for test
    spec = MechClassConfig(
        name=cls_name,
        size_voxels=(1, 1, 1) if cls_name != "heavy" else (2, 2, 2),
        max_speed=5.0,
        max_yaw_rate=2.0,
        max_jet_accel=5.0,
        hp=100.0,
        leg_hp=50.0,
        heat_cap=100.0,
        heat_dissipation=10.0,
    )
    return MechState(
        mech_id=mid,
        team=team,
        spec=spec,
        pos=np.array(pos, dtype=np.float32),
        vel=np.zeros(3, dtype=np.float32),
        yaw=0.0,
        hp=spec.hp,
        leg_hp=spec.leg_hp,
        heat=0.0,
        stability=100.0,
    )


def test_indirect_fire():
    # Setup world
    # 20x20x20
    # Wall at x=10
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    # Clear world
    world.voxels.fill(VoxelWorld.AIR)
    # Build wall
    world.set_box_solid(10, 0, 0, 11, 20, 10, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    # Heavy at (5, 10, 0) facing +X (towards wall)
    heavy = make_mech("blue_0", "blue", [5.0, 10.0, 1.0], "heavy")
    painter = make_mech("blue_1", "blue", [6.0, 10.0, 1.0], "light")
    # Enemy at (15, 10, 0) behind wall
    enemy = make_mech("red_0", "red", [15.0, 10.0, 1.0], "medium")

    sim.reset({"blue_0": heavy, "blue_1": painter, "red_0": enemy})

    # Action to fire missile: SECONDARY slot for heavy
    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0  # Missile

    # 1. Try fire without LOS or Paint
    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Should not fire without LOS or Paint"

    # 2. Paint the enemy
    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter.mech_id

    # 3. Try fire again
    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 1, "Should fire at painted target despite no LOS"
    assert len(sim.projectiles) == 1

    # 4. Verify projectile homing
    # Projectile should have velocity towards enemy
    p = sim.projectiles[0]
    assert p.target_id == "red_0"

    print("Indirect fire test passed!")


def test_missile_arc_blocks_rear_shots():
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    heavy = make_mech("blue_0", "blue", [10.0, 10.0, 1.0], "heavy")
    heavy.yaw = 0.0  # facing +X
    painter = make_mech("blue_1", "blue", [9.0, 10.0, 1.0], "light")
    enemy = make_mech("red_0", "red", [5.0, 10.0, 1.0], "medium")  # behind heavy
    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter.mech_id

    sim.reset({"blue_0": heavy, "blue_1": painter, "red_0": enemy})

    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0  # Missile

    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Paint should not bypass missile firing arc"
    assert len(sim.projectiles) == 0
    print("Missile arc blocks rear shots test passed!")


def test_paint_lock_is_pack_scoped():
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    # Wall blocks LOS.
    world.set_box_solid(10, 0, 0, 11, 20, 10, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    heavy = make_mech("blue_0", "blue", [5.0, 10.0, 1.0], "heavy")
    painter_other_pack = make_mech("blue_10", "blue", [6.0, 10.0, 1.0], "light")  # different pack (idx 10)
    enemy = make_mech("red_0", "red", [15.0, 10.0, 1.0], "medium")
    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter_other_pack.mech_id

    sim.reset({"blue_0": heavy, "blue_10": painter_other_pack, "red_0": enemy})

    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0  # Missile

    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Paint from a different pack should not grant indirect lock"
    assert len(sim.projectiles) == 0
    print("Paint lock is pack scoped test passed!")


def test_shutdown_keeps_physics():
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    sim = Sim(world, 0.05, np.random.default_rng(0))
    mech = make_mech("m", "blue", [10.0, 10.0, 10.0], "heavy")
    mech.heat = mech.spec.heat_cap + 100.0
    mech.vel = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    sim.reset({"m": mech})

    pos0 = mech.pos.copy()
    vel0 = mech.vel.copy()
    sim.step({"m": np.zeros(ACTION_DIM, dtype=np.float32)}, num_substeps=20)

    assert mech.pos[2] < pos0[2], "Shutdown mech should fall under gravity"
    assert mech.pos[0] > pos0[0], "Shutdown mech should coast (not instantly stop)"
    assert mech.vel[0] != 0.0 and mech.vel[0] < vel0[0], "Shutdown mech should damp, not zero velocity"
    print("Shutdown keeps physics test passed!")


def test_autocannon_auto_pitch():
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    sim = Sim(world, 0.05, np.random.default_rng(0))
    medium = make_mech("medium", "blue", [5.0, 5.0, 1.0], "medium")
    medium.yaw = 0.0  # facing +X
    enemy = make_mech("enemy", "red", [15.0, 5.0, 5.0], "heavy")
    sim.reset({"medium": medium, "enemy": enemy})

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ActionIndex.TERTIARY] = 1.0  # Kinetic slot  # Kinetic slot (AC for medium)
    events = sim._try_fire_kinetic(medium, action)
    assert len(events) == 1
    assert len(sim.projectiles) == 1
    assert float(sim.projectiles[0].vel[2]) > 0.1, "Autocannon should auto-pitch toward elevated targets"
    print("Autocannon auto pitch test passed!")


def test_knockdown_immobilizes():
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    # Large dt so the 3s stun resolves quickly in test.
    sim = Sim(world, 1.0, np.random.default_rng(0))
    mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
    mech.stability = 0.0
    sim.reset({"m": mech})

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ActionIndex.FORWARD] = 1.0  # forward
    action[ActionIndex.YAW_RATE] = 1.0  # yaw_rate

    pos0 = mech.pos.copy()
    yaw0 = float(mech.yaw)
    sim.step({"m": action}, num_substeps=1)

    assert mech.fallen_time > 0.0, "Mech should enter fallen state when stability <= 0"
    assert float(mech.yaw) == yaw0, "Fallen mech must not rotate"
    assert np.allclose(mech.pos[:2], pos0[:2]), "Fallen mech must not translate in XY"

    # Still stunned: repeated attempts shouldn't move it.
    pos1 = mech.pos.copy()
    yaw1 = float(mech.yaw)
    sim.step({"m": action}, num_substeps=1)
    assert float(mech.yaw) == yaw1, "Fallen mech must remain unable to rotate"
    assert np.allclose(mech.pos[:2], pos1[:2]), "Fallen mech must remain unable to move in XY"

    # After enough time, it should stand up and be able to move.
    sim.step({"m": action}, num_substeps=3)
    assert mech.fallen_time == 0.0, "Mech should recover after fallen_time elapses"
    assert float(np.linalg.norm(mech.pos[:2] - pos0[:2])) > 0.5, "Recovered mech should be able to move again"
    print("Knockdown immobilizes test passed!")


def test_gauss_projectile_does_not_tunnel_through_wall():
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    # 1-voxel thick wall directly between shooter and target.
    world.set_box_solid(10, 0, 0, 11, 20, 20, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))
    # Choose non-integer positions so fast projectiles straddle the wall between frames.
    heavy = make_mech("heavy", "blue", [5.5, 10.0, 1.0], "heavy")
    heavy.yaw = 0.0  # facing +X
    enemy = make_mech("enemy", "red", [15.5, 10.0, 1.0], "heavy")
    sim.reset({"heavy": heavy, "enemy": enemy})

    assert not has_los(world, heavy.pos, enemy.pos), "Wall should block LOS in this setup"

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ActionIndex.TERTIARY] = 1.0  # Kinetic slot  # Gauss for heavy
    events = sim._try_fire_kinetic(heavy, action)
    assert len(events) == 1
    assert len(sim.projectiles) == 1

    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    events = sim.step({"heavy": zero, "enemy": zero}, num_substeps=25)

    assert float(enemy.hp) == float(enemy.spec.hp), "Projectile should not hit through a wall"
    assert not any(
        ev.get("type") == "projectile_hit" and ev.get("target") == "enemy" for ev in events
    ), "Should not record a hit through terrain"
    print("Gauss projectile tunneling test passed!")


def test_pack_comm_is_pack_scoped():
    # Two packs per team so we can verify comm does not leak across packs.
    cfg = EnvConfig(
        world=WorldConfig(size_x=30, size_y=30, size_z=20),
        num_packs=2,
        observation_mode="full",
        comm_dim=3,
        seed=0,
        max_episode_seconds=5.0,
    )
    env = EchelonEnv(cfg)
    _obs, _ = env.reset(seed=0)

    contacts_total = int(env.CONTACT_SLOTS * env.CONTACT_DIM)
    comm_total = PACK_SIZE * int(cfg.comm_dim)

    sender = "blue_0"  # pack 0
    same_pack = "blue_1"  # pack 0
    other_pack = "blue_10"  # pack 1

    msg = np.array([0.25, -0.5, 0.9], dtype=np.float32)
    a_sender = np.zeros(env.ACTION_DIM, dtype=np.float32)
    a_sender[env.COMM_START : env.COMM_START + cfg.comm_dim] = msg

    actions = {sender: a_sender}
    obs2, *_ = env.step(actions)

    comm_same_pack = obs2[same_pack][contacts_total : contacts_total + comm_total]
    comm_other_pack = obs2[other_pack][contacts_total : contacts_total + comm_total]

    # In pack 0, the sender (blue_0) is row 0.
    row0 = comm_same_pack[: int(cfg.comm_dim)]
    assert np.allclose(row0, msg), "Packmate should receive sender's message"
    assert np.allclose(comm_other_pack, 0.0), "Other pack should not receive sender's message"
    print("Pack comm is pack scoped test passed!")


def test_partial_visibility_is_pack_scoped():
    cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20, obstacle_fill=0.0, ensure_connectivity=False),
        num_packs=2,
        observation_mode="partial",
        comm_dim=0,
        seed=0,
        max_episode_seconds=5.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)
    assert env.world is not None and env.sim is not None

    # Construct a deterministic LOS-blocking wall between x<20 and x>=20.
    env.world.voxels.fill(VoxelWorld.AIR)
    env.world.set_box_solid(20, 0, 0, 21, env.world.size_y, env.world.size_z, True)

    viewer_id = "blue_0"
    packmate_id = "blue_1"  # same pack as blue_0
    other_pack_id = "blue_10"  # different pack

    viewer = env.sim.mechs[viewer_id]
    packmate = env.sim.mechs[packmate_id]
    other_pack = env.sim.mechs[other_pack_id]

    # Disable all other mechs so the contact table is deterministic.
    for mid, m in env.sim.mechs.items():
        if mid not in (viewer_id, packmate_id, other_pack_id):
            m.alive = False

    # Place packmate and other-pack teammate behind the wall and outside radar range.
    viewer.pos[0], viewer.pos[1] = 10.0, 20.0
    packmate.pos[0], packmate.pos[1] = 30.0, 20.0
    other_pack.pos[0], other_pack.pos[1] = 30.0, 22.0
    viewer.vel[:] = 0.0
    packmate.vel[:] = 0.0
    other_pack.vel[:] = 0.0

    obs, *_ = env.step({})

    obs_v = obs[viewer_id]
    contacts_total = int(env.CONTACT_SLOTS * env.CONTACT_DIM)
    contacts = obs_v[:contacts_total].reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)

    visible = contacts[:, 20]
    assert float(visible.sum()) == 0.0, "No contacts should be visible without LOS/radar/paint"

    # Bring the packmate within radar range (radar is range-based, not LOS-based).
    packmate.pos[0], packmate.pos[1] = 22.0, 20.0  # dist=12 from viewer
    obs2, *_ = env.step({})
    obs_v2 = obs2[viewer_id]
    contacts2 = obs_v2[:contacts_total].reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)
    visible2 = contacts2[:, 20]
    assert float(visible2.sum()) == 1.0, "Packmate should become visible when in radar range"

    rel = contacts2[:, 13:16]
    assert float(rel[:, 0].sum()) == 1.0, "Visible contact should be marked friendly"
    print("Partial visibility is pack scoped test passed!")


def test_topk_contact_quota_and_repurpose():
    cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20, obstacle_fill=0.0, ensure_connectivity=False),
        num_packs=1,
        observation_mode="full",
        comm_dim=0,
        seed=0,
        max_episode_seconds=5.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)
    assert env.sim is not None

    viewer_id = "blue_0"
    viewer = env.sim.mechs[viewer_id]

    # Keep only a controlled set of mechs alive:
    # - 6 friendlies (same team) and 2 hostiles (other team), all visible in full mode.
    keep = {viewer_id, "blue_1", "blue_2", "blue_3", "blue_4", "blue_5", "blue_6", "red_0", "red_1"}
    for mid, m in env.sim.mechs.items():
        m.alive = mid in keep

    # Place them around the viewer.
    viewer.pos[:] = np.array([10.0, 10.0, 1.0], dtype=np.float32)
    for i, mid in enumerate(["blue_1", "blue_2", "blue_3", "blue_4", "blue_5", "blue_6"], start=1):
        env.sim.mechs[mid].pos[:] = np.array([10.0 + i, 10.0, 1.0], dtype=np.float32)
    env.sim.mechs["red_0"].pos[:] = np.array([30.0, 10.0, 1.0], dtype=np.float32)
    env.sim.mechs["red_1"].pos[:] = np.array([31.0, 10.0, 1.0], dtype=np.float32)

    obs, *_ = env.step({})
    obs_v = obs[viewer_id]
    contacts_total = int(env.CONTACT_SLOTS * env.CONTACT_DIM)
    contacts = obs_v[:contacts_total].reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)

    # In this setup, neutral category is empty, so its slot should be repurposed.
    # Priority is hostiles first, so we should observe 2 hostiles and 3 friendlies (K=5).
    rel = contacts[:, 13:16]
    n_friendly = int((rel[:, 0] > 0.5).sum())
    n_hostile = int((rel[:, 1] > 0.5).sum())
    n_visible = int((contacts[:, 20] > 0.5).sum())
    assert n_visible == 5
    assert n_hostile == 2
    assert n_friendly == 3
    print("Top-K contact quota test passed!")


if __name__ == "__main__":
    test_indirect_fire()
    test_missile_arc_blocks_rear_shots()
    test_paint_lock_is_pack_scoped()
    test_shutdown_keeps_physics()
    test_autocannon_auto_pitch()
    test_knockdown_immobilizes()
    test_gauss_projectile_does_not_tunnel_through_wall()
    test_pack_comm_is_pack_scoped()
    test_partial_visibility_is_pack_scoped()
    test_topk_contact_quota_and_repurpose()

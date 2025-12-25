import numpy as np

from echelon import EchelonEnv, EnvConfig
from echelon.actions import ACTION_DIM, ActionIndex
from echelon.config import WorldConfig
from echelon.constants import PACK_SIZE
from echelon.sim.los import has_los
from echelon.sim.sim import Sim
from echelon.sim.world import VoxelWorld


def test_indirect_fire(make_mech):
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    world.set_box_solid(10, 0, 0, 11, 20, 10, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    heavy = make_mech("blue_0", "blue", [5.0, 10.0, 1.0], "heavy")
    painter = make_mech("blue_1", "blue", [6.0, 10.0, 1.0], "light")
    enemy = make_mech("red_0", "red", [15.0, 10.0, 1.0], "medium")

    sim.reset({"blue_0": heavy, "blue_1": painter, "red_0": enemy})

    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0

    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Should not fire without LOS or Paint"

    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter.mech_id

    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 1, "Should fire at painted target despite no LOS"
    assert len(sim.projectiles) == 1

    p = sim.projectiles[0]
    assert p.target_id == "red_0"


def test_missile_arc_blocks_rear_shots(make_mech):
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    heavy = make_mech("blue_0", "blue", [10.0, 10.0, 1.0], "heavy")
    heavy.yaw = 0.0
    painter = make_mech("blue_1", "blue", [9.0, 10.0, 1.0], "light")
    enemy = make_mech("red_0", "red", [5.0, 10.0, 1.0], "medium")
    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter.mech_id

    sim.reset({"blue_0": heavy, "blue_1": painter, "red_0": enemy})

    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0

    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Paint should not bypass missile firing arc"
    assert len(sim.projectiles) == 0


def test_paint_lock_is_pack_scoped(make_mech):
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    world.set_box_solid(10, 0, 0, 11, 20, 10, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))

    heavy = make_mech("blue_0", "blue", [5.0, 10.0, 1.0], "heavy")
    # Use blue_11 to be in a different pack (PACK_SIZE=11, so 11/11=1 vs 0/11=0)
    painter_other_pack = make_mech("blue_11", "blue", [6.0, 10.0, 1.0], "light")
    enemy = make_mech("red_0", "red", [15.0, 10.0, 1.0], "medium")
    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter_other_pack.mech_id

    sim.reset({"blue_0": heavy, "blue_11": painter_other_pack, "red_0": enemy})

    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0

    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 0, "Paint from a different pack should not grant indirect lock"
    assert len(sim.projectiles) == 0


def test_shutdown_keeps_physics(make_mech):
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
    assert mech.pos[0] > pos0[0], "Shutdown mech should coast"
    assert mech.vel[0] != 0.0 and mech.vel[0] < vel0[0], "Shutdown mech should damp"


def test_autocannon_auto_pitch(make_mech):
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    sim = Sim(world, 0.05, np.random.default_rng(0))
    medium = make_mech("medium", "blue", [5.0, 5.0, 1.0], "medium")
    medium.yaw = 0.0
    enemy = make_mech("enemy", "red", [15.0, 5.0, 5.0], "heavy")
    sim.reset({"medium": medium, "enemy": enemy})

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ActionIndex.TERTIARY] = 1.0
    events = sim._try_fire_kinetic(medium, action)
    assert len(events) == 1
    assert len(sim.projectiles) == 1
    assert float(sim.projectiles[0].vel[2]) > 0.1, "Autocannon should auto-pitch"


def test_knockdown_immobilizes(make_mech):
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)

    sim = Sim(world, 1.0, np.random.default_rng(0))
    mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
    mech.stability = 0.0
    sim.reset({"m": mech})

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ActionIndex.FORWARD] = 1.0
    action[ActionIndex.YAW_RATE] = 1.0

    pos0 = mech.pos.copy()
    yaw0 = float(mech.yaw)
    sim.step({"m": action}, num_substeps=1)

    assert mech.fallen_time > 0.0
    assert float(mech.yaw) == yaw0
    assert np.allclose(mech.pos[:2], pos0[:2])

    pos1 = mech.pos.copy()
    yaw1 = float(mech.yaw)
    sim.step({"m": action}, num_substeps=1)
    assert float(mech.yaw) == yaw1
    assert np.allclose(mech.pos[:2], pos1[:2])

    sim.step({"m": action}, num_substeps=3)
    assert mech.fallen_time == 0.0
    assert float(np.linalg.norm(mech.pos[:2] - pos0[:2])) > 0.5


def test_gauss_projectile_does_not_tunnel_through_wall(make_mech):
    world = VoxelWorld.generate(WorldConfig(), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    world.set_box_solid(10, 0, 0, 11, 20, 20, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))
    heavy = make_mech("heavy", "blue", [5.5, 10.0, 1.0], "heavy")
    heavy.yaw = 0.0
    enemy = make_mech("enemy", "red", [15.5, 10.0, 1.0], "heavy")
    sim.reset({"heavy": heavy, "enemy": enemy})

    assert not has_los(world, heavy.pos, enemy.pos)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ActionIndex.TERTIARY] = 1.0
    events = sim._try_fire_kinetic(heavy, action)
    assert len(events) == 1
    assert len(sim.projectiles) == 1

    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    events = sim.step({"heavy": zero, "enemy": zero}, num_substeps=25)

    assert float(enemy.hp) == float(enemy.spec.hp)
    assert not any(ev.get("type") == "projectile_hit" and ev.get("target") == "enemy" for ev in events)


def test_splash_damage_is_occluded_by_walls(make_mech):
    world = VoxelWorld.generate(WorldConfig(size_x=30, size_y=30, size_z=20), np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    world.set_box_solid(10, 0, 0, 11, 30, 20, True)

    sim = Sim(world, 0.05, np.random.default_rng(0))
    heavy = make_mech("blue_0", "blue", [5.5, 15.0, 2.0], "heavy")
    painter = make_mech("blue_1", "blue", [5.5, 14.0, 2.0], "scout")
    enemy = make_mech("red_0", "red", [11.5, 15.0, 2.0], "medium")
    enemy.painted_remaining = 5.0
    enemy.last_painter_id = painter.mech_id

    sim.reset({"blue_0": heavy, "blue_1": painter, "red_0": enemy})

    fire_action = np.zeros(ACTION_DIM, dtype=np.float32)
    fire_action[ActionIndex.SECONDARY] = 1.0
    events = sim._try_fire_missile(heavy, fire_action)
    assert len(events) == 1
    assert len(sim.projectiles) == 1

    initial_hp = float(enemy.hp)
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    events = sim.step({"blue_0": zero, "blue_1": zero, "red_0": zero}, num_substeps=50)

    assert any(ev.get("type") == "explosion" for ev in events)
    assert float(enemy.hp) == initial_hp
    assert not any(ev.get("type") == "projectile_hit" and ev.get("target") == "red_0" for ev in events)


def test_pack_comm_is_pack_scoped():
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

    sender = "blue_0"
    same_pack = "blue_1"
    other_pack = f"blue_{PACK_SIZE}"  # First agent in second pack

    msg = np.array([0.25, -0.5, 0.9], dtype=np.float32)
    a_sender = np.zeros(env.ACTION_DIM, dtype=np.float32)
    a_sender[env.COMM_START : env.COMM_START + cfg.comm_dim] = msg

    actions = {sender: a_sender}
    obs2, *_ = env.step(actions)

    comm_same_pack = obs2[same_pack][contacts_total : contacts_total + comm_total]
    comm_other_pack = obs2[other_pack][contacts_total : contacts_total + comm_total]

    row0 = comm_same_pack[: int(cfg.comm_dim)]
    assert np.allclose(row0, msg)
    assert np.allclose(comm_other_pack, 0.0)


def test_shutdown_zeroes_observed_velocity():
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10, obstacle_fill=0.0, ensure_connectivity=False),
        num_packs=1,
        observation_mode="full",
        comm_dim=0,
        seed=0,
        max_episode_seconds=5.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)

    viewer_id = "blue_0"
    viewer = env.sim.mechs[viewer_id]
    viewer.heat = float(viewer.spec.heat_cap + 1.0)
    viewer.vel[:] = np.array([3.0, -2.0, 1.0], dtype=np.float32)

    obs = env._obs()
    obs_v = obs[viewer_id]
    contacts_total = int(env.CONTACT_SLOTS * env.CONTACT_DIM)
    comm_total = PACK_SIZE * int(cfg.comm_dim)
    telemetry_dim = 16 * 16
    offset = contacts_total + comm_total + int(env.LOCAL_MAP_DIM) + telemetry_dim

    self_features = obs_v[offset:]
    assert self_features.size >= 30

    self_vel_offset = 27  # acoustic(4) + hull(4) + 19 slots before self_vel
    self_vel = self_features[self_vel_offset : self_vel_offset + 3]
    assert np.allclose(self_vel, 0.0)


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

    env.world.voxels.fill(VoxelWorld.AIR)
    env.world.set_box_solid(20, 0, 0, 21, env.world.size_y, env.world.size_z, True)

    viewer_id = "blue_0"
    packmate_id = "blue_1"
    other_pack_id = f"blue_{PACK_SIZE}"  # First agent in second pack

    viewer = env.sim.mechs[viewer_id]
    packmate = env.sim.mechs[packmate_id]
    other_pack = env.sim.mechs[other_pack_id]

    for mid, m in env.sim.mechs.items():
        if mid not in (viewer_id, packmate_id, other_pack_id):
            m.alive = False

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
    # visible flag is at index 21 in contact features (see _contact_features)
    visible = contacts[:, 21]
    assert float(visible.sum()) == 0.0

    packmate.pos[0], packmate.pos[1] = 22.0, 20.0
    obs2, *_ = env.step({})
    obs_v2 = obs2[viewer_id]
    contacts2 = obs_v2[:contacts_total].reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)
    visible2 = contacts2[:, 21]
    assert float(visible2.sum()) == 1.0

    rel = contacts2[:, 13:16]
    assert float(rel[:, 0].sum()) == 1.0


def test_topk_contact_quota_and_repurpose():
    cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20, obstacle_fill=0.0, ensure_connectivity=False),
        num_packs=2,  # Need enough agents for contact quota test
        observation_mode="full",
        comm_dim=0,
        seed=0,
        max_episode_seconds=5.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)

    viewer_id = "blue_0"
    viewer = env.sim.mechs[viewer_id]

    keep = {viewer_id, "blue_1", "blue_2", "blue_3", "blue_4", "blue_5", "blue_6", "red_0", "red_1"}
    for mid, m in env.sim.mechs.items():
        m.alive = mid in keep

    viewer.pos[:] = np.array([10.0, 10.0, 1.0], dtype=np.float32)
    for i, mid in enumerate(["blue_1", "blue_2", "blue_3", "blue_4", "blue_5", "blue_6"], start=1):
        env.sim.mechs[mid].pos[:] = np.array([10.0 + i, 10.0, 1.0], dtype=np.float32)
    env.sim.mechs["red_0"].pos[:] = np.array([30.0, 10.0, 1.0], dtype=np.float32)
    env.sim.mechs["red_1"].pos[:] = np.array([31.0, 10.0, 1.0], dtype=np.float32)

    obs, *_ = env.step({})
    obs_v = obs[viewer_id]
    contacts_total = int(env.CONTACT_SLOTS * env.CONTACT_DIM)
    contacts = obs_v[:contacts_total].reshape(env.CONTACT_SLOTS, env.CONTACT_DIM)

    rel = contacts[:, 13:16]
    n_friendly = int((rel[:, 0] > 0.5).sum())
    n_hostile = int((rel[:, 1] > 0.5).sum())
    # visible flag is at index 21 in contact features (see _contact_features)
    n_visible = int((contacts[:, 21] > 0.5).sum())
    assert n_visible == 5
    assert n_hostile == 2
    assert n_friendly == 3

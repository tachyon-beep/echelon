from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldConfig:
    size_x: int = 20
    size_y: int = 20
    size_z: int = 20
    voxel_size_m: float = 1.0
    obstacle_fill: float = 0.06
    ensure_connectivity: bool = True
    connectivity_clearance_z: int = 4
    connectivity_obstacle_inflate_radius: int = 1
    connectivity_wall_cost: float = 80.0
    connectivity_penalty_radius: int = 3
    connectivity_penalty_cost: float = 20.0
    connectivity_carve_width: int = 5


@dataclass(frozen=True)
class MechClassConfig:
    name: str
    size_voxels: tuple[float, float, float]
    max_speed: float
    max_yaw_rate: float
    max_jet_accel: float
    hp: float
    leg_hp: float
    heat_cap: float
    heat_dissipation: float
    tonnage: float = 20.0


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    range_vox: float
    damage: float
    stability_damage: float
    heat: float
    cooldown_s: float
    arc_deg: float
    speed_vox: float = 0.0  # 0 for hitscan
    guidance: str = "none"  # "homing", "ballistic", "linear"
    splash_rad_vox: float = 0.0
    splash_dmg_scale: float = 0.0


LASER = WeaponSpec(
    name="laser",
    range_vox=8.0,
    damage=14.0,
    stability_damage=0.0,
    heat=22.0,
    cooldown_s=0.6,
    arc_deg=120.0,
)

FLAMER = WeaponSpec(
    name="flamer",
    range_vox=4.5,
    damage=4.0,
    stability_damage=0.0,
    heat=10.0,  # Self heat
    cooldown_s=0.15,
    arc_deg=90.0,
)

MISSILE = WeaponSpec(
    name="missile",
    range_vox=50.0,  # Long range ONLY with paint lock from teammate
    damage=40.0,
    stability_damage=15.0,
    heat=45.0,
    cooldown_s=3.0,
    arc_deg=180.0,
    speed_vox=12.0,
    guidance="homing",
    splash_rad_vox=2.5,
    splash_dmg_scale=0.5,
)

# Onboard painter range for missiles without external paint lock
MISSILE_ONBOARD_RANGE_VOX = 12.0

SMOKE = WeaponSpec(
    name="smoke",
    range_vox=20.0,
    damage=0.0,
    stability_damage=0.0,
    heat=10.0,
    cooldown_s=5.0,
    arc_deg=180.0,
    speed_vox=15.0,
    guidance="linear",
    splash_rad_vox=4.0,
)

GAUSS = WeaponSpec(
    name="gauss",
    range_vox=20.0,  # Was 60 - heavy sniper but not cross-map
    damage=50.0,
    stability_damage=60.0,  # Huge impact
    heat=15.0,
    cooldown_s=4.0,
    arc_deg=60.0,
    speed_vox=40.0,  # Very fast
    guidance="ballistic",
    splash_rad_vox=1.5,
    splash_dmg_scale=0.5,
)

AUTOCANNON = WeaponSpec(
    name="autocannon",
    range_vox=12.0,  # Was 20 - medium range suppression
    damage=8.0,
    stability_damage=5.0,
    heat=6.0,
    cooldown_s=0.2,  # Rapid fire
    arc_deg=90.0,
    speed_vox=25.0,
    guidance="linear",
    splash_rad_vox=0.0,
)

PAINTER = WeaponSpec(
    name="painter",
    range_vox=10.0,  # Was 15 - scout needs to get close
    damage=0.0,
    stability_damage=0.0,
    heat=5.0,
    cooldown_s=1.0,
    arc_deg=60.0,
)


# Lightweight "role" extensions (kept simple and readable in replays).
LASER_HEAT_TRANSFER = 6.0  # Heat applied to the target on laser hit.
FLAME_HEAT_TRANSFER = 15.0  # High heat for flamer.

# Autocannon suppression: slows stability regen briefly after AC hits.
SUPPRESS_DURATION_S = 1.2
SUPPRESS_REGEN_SCALE = 0.25

# Electronic warfare: ECM reduces sensor quality; ECCM restores it.
ECM_RADIUS_VOX = 50.0
ECCM_RADIUS_VOX = 50.0
ECM_WEIGHT = 0.85
ECCM_WEIGHT = 0.65
SENSOR_QUALITY_MIN = 0.25
SENSOR_QUALITY_MAX = 1.5
PAINT_LOCK_MIN_QUALITY = 0.70

ECM_HEAT_PER_S = 3.0
ECCM_HEAT_PER_S = 2.0

# Simple self-defense point-defense against homing missiles.
AMS_RANGE_VOX = 5.5
AMS_COOLDOWN_S = 1.0
AMS_INTERCEPT_PROB = 0.60

# Hazards
LAVA_HEAT_PER_S = 40.0
LAVA_DMG_PER_S = 5.0
WATER_COOLING_PER_S = 30.0
WATER_SPEED_MULT = 0.6


@dataclass(frozen=True)
class EnvConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    num_packs: int = 1
    dt_sim: float = 0.05
    decision_repeat: int = 5
    max_episode_seconds: float = 60.0
    observation_mode: str = "full"  # "full" | "partial"
    # Optional, pack-scoped communication side-channel.
    # Each mech can emit a message vector as part of its action, and receives the
    # last messages from packmates in its observation (1 decision-tick delay).
    comm_dim: int = 8  # 0 disables
    # Feature toggles (useful for ablations/curriculum without changing shapes).
    enable_target_selection: bool = True
    enable_ewar: bool = True
    enable_obs_control: bool = True
    enable_comm: bool = True
    nav_mode: str = "off"  # "off" | "assist" | "planner"
    record_replay: bool = False
    seed: int | None = None
    # Team reward mixing: alpha=1.0 is fully individual, alpha=0.0 is fully team-based
    # Mix helps credit assignment in cooperative multi-agent setting
    # Reduced from 1.0 to 0.7: 30% team reward improves coordination (2025-12-27 rebalance)
    team_reward_alpha: float = 0.7
    # Discount factor for PBRS-compliant approach shaping: r = gamma*phi(s') - phi(s)
    # Should match training gamma for strict PBRS compliance (Ng et al., 1999)
    shaping_gamma: float = 0.99

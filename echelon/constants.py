from __future__ import annotations

# ==============================================================================
# Unit Composition
# ==============================================================================

# Pack: 6 mechs - the basic tactical element
#   - 1 Scout (recon, painting)
#   - 1 Light (flanking, mobility)
#   - 1 Medium (line infantry)
#   - 1 Heavy (fire support, protected asset)
#   - 1 Pack Leader (light-equivalent chassis, pack command suite)
# Pack Leader sees aggregated pack sensor data and can issue orders to pack members.
PACK_SIZE = 6

# Squad: 13 mechs - two packs + squad leader
#   - 2 Packs (12 mechs)
#   - 1 Squad Leader (medium-equivalent chassis, squad command suite)
# Squad Leader sees full squad telemetry and can issue orders to anyone.
SQUAD_SIZE = 13

# Indices within a pack (for roster assignment)
# Two fire teams for "fix" and "assault" flexibility:
#   Fire Team A: Scout + Medium + Pack Leader (overwatch)
#   Fire Team B: Scout + Light + Heavy (maneuver element)
PACK_SCOUT_A_IDX = 0  # Fire team A recon
PACK_SCOUT_B_IDX = 1  # Fire team B recon
PACK_LIGHT_IDX = 2  # Flanker/mobility
PACK_MEDIUM_IDX = 3  # Line infantry
PACK_HEAVY_IDX = 4  # Fire support (protected asset)
PACK_LEADER_IDX = 5  # Light-equivalent with pack command suite

# Squad leader is always the last mech in the squad
# Squad layout: [Pack0: 0-5] [Pack1: 6-11] [SL: 12]

# ==============================================================================
# Physics Constants
# ==============================================================================

# Gravity in meters per second squared (Earth standard)
GRAVITY_M_S2 = 9.81

# Rear arc hit detection threshold: cos(45°) ≈ 0.707
# Attack directions with dot product > this value hit the rear arc
REAR_ARC_COS_THRESHOLD = 0.707

# Damage multiplier for hits to the rear armor arc
REAR_ARMOR_DAMAGE_MULT = 1.5

# Body damage multiplier when hitting legs (reduced due to leg armor absorption)
LEG_HIT_BODY_DAMAGE_MULT = 0.5

# ==============================================================================
# Stability & Knockdown
# ==============================================================================

# Duration in seconds that a mech remains fallen after knockdown
FALLEN_DURATION_S = 3.0

# Stability regeneration rate per second (base, before modifiers)
STABILITY_REGEN_PER_S = 10.0

# Stability recovered when standing up (fraction of max)
STANDUP_STABILITY_FRACTION = 0.5

# Stability regen penalty when legged (multiplier)
LEGGED_STABILITY_MULT = 0.5

# Stability regen penalty when moving (multiplier)
MOVING_STABILITY_MULT = 0.5

# Movement acceleration penalty when airborne or on unstable footing
UNSTABLE_ACCEL_MULT = 0.5

# Vent heat dissipation multiplier (2x normal dissipation rate)
VENT_HEAT_MULT = 2.0

# ==============================================================================
# Acoustic Noise (for sensor detection)
# ==============================================================================

# Mass factors for acoustic noise calculation (noise = speed * mass_factor)
NOISE_MASS_FACTORS: dict[str, float] = {
    "scout": 1.0,
    "light": 1.5,
    "medium": 2.5,
    "heavy": 4.0,
}

# ==============================================================================
# Projectile Physics
# ==============================================================================

# Initial vertical velocity component for missiles (upward arc)
MISSILE_VERTICAL_VEL = 0.5

# Spawn offset above mech center for projectiles (voxels)
PROJECTILE_SPAWN_OFFSET_Z = 0.5

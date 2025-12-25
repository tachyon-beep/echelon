from __future__ import annotations

# ==============================================================================
# Unit Composition
# ==============================================================================

# A "pack" is the standard deployed unit: 5 mechs (1 Heavy, 2 Medium, 1 Light, 1 Scout).
# Two packs form a squad (10 mechs) with a squad leader who can mix/match composition.
PACK_SIZE = 5

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

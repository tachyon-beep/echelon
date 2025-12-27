from __future__ import annotations

from enum import IntEnum

ACTION_DIM = 9


class ActionIndex(IntEnum):
    FORWARD = 0
    STRAFE = 1
    VERTICAL = 2
    YAW_RATE = 3
    PRIMARY = 4  # Laser (Med/Heavy), Flamer (Light), Paint (Scout)
    VENT = 5  # Universal
    SECONDARY = 6  # Missile (Heavy), Laser (Light), unused (Scout)
    TERTIARY = 7  # Paint (Light/Med), Gauss/AC (Heavy/Med)
    SPECIAL = 8  # Smoke (Med/Heavy/Scout), ECM/ECCM toggle (Light)


class OrderType(IntEnum):
    """Order types that command mechs can issue."""

    NONE = 0  # No order (default)
    FOCUS_FIRE = 1  # Attack the designated target
    ADVANCE = 2  # Move toward objective
    HOLD = 3  # Hold current position
    RALLY = 4  # Rally to commander's position
    COVER = 5  # Provide covering fire / overwatch


class OrderOverride(IntEnum):
    """Valid reasons for not following an order.

    Orders are contracts with escape clauses. Units can override orders
    but must provide a valid reason. Invalid overrides are penalized.
    """

    NONE = 0  # Following order normally
    BLOCKED_PATH = 1  # Can't reach objective (nav failure)
    CRITICAL_THREAT = 2  # Must handle immediate danger first
    TARGET_DESTROYED = 3  # Mission complete (target killed by others)
    RESOURCE_DEPLETED = 4  # Out of ammo/heat capacity
    COMMS_LOST = 5  # Can't confirm order still valid


# Command action dimensions
ORDER_TYPE_DIM = 6  # One-hot for order type selection
ORDER_RECIPIENT_DIM = 6  # Which pack member (0-5) or broadcast
# ORDER_PARAM_DIM must match MAX_CONTACT_SLOTS (20) so commanders can target any visible contact
ORDER_PARAM_DIM = 20  # Order-specific parameter (e.g., contact slot for FOCUS_FIRE)
COMMAND_ACTION_DIM = ORDER_TYPE_DIM + ORDER_RECIPIENT_DIM + ORDER_PARAM_DIM  # 32 total

# Status report action dimensions (for subordinates reporting to command)
# acknowledge(1) + override_reason(6) + progress(1) + flags(3) = 11
STATUS_ACKNOWLEDGE_DIM = 1  # Acknowledge receipt of order
STATUS_OVERRIDE_DIM = 6  # One-hot for override reason (NONE means following)
STATUS_PROGRESS_DIM = 1  # Order progress [0, 1]
STATUS_FLAGS_DIM = 3  # is_engaged, is_blocked, needs_support
STATUS_REPORT_ACTION_DIM = (
    STATUS_ACKNOWLEDGE_DIM + STATUS_OVERRIDE_DIM + STATUS_PROGRESS_DIM + STATUS_FLAGS_DIM
)  # 11 total

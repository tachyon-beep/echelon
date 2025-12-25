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
    SECONDARY = 6  # Missile (Heavy), Laser (Light), ECM/ECCM toggle (Scout)
    TERTIARY = 7  # Paint (Light/Med), Gauss/AC (Heavy/Med)
    SPECIAL = 8  # Smoke (Universal)


class OrderType(IntEnum):
    """Order types that command mechs can issue."""

    NONE = 0  # No order (default)
    FOCUS_FIRE = 1  # Attack the designated target
    ADVANCE = 2  # Move toward objective
    HOLD = 3  # Hold current position
    RALLY = 4  # Rally to commander's position
    COVER = 5  # Provide covering fire / overwatch


# Command action dimensions
ORDER_TYPE_DIM = 6  # One-hot for order type selection
ORDER_RECIPIENT_DIM = 6  # Which pack member (0-5) or broadcast
ORDER_PARAM_DIM = 5  # Order-specific parameter (e.g., contact slot for FOCUS_FIRE)
COMMAND_ACTION_DIM = ORDER_TYPE_DIM + ORDER_RECIPIENT_DIM + ORDER_PARAM_DIM  # 17 total

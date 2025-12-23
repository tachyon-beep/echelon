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

from __future__ import annotations

from enum import IntEnum


ACTION_DIM = 9


class ActionIndex(IntEnum):
    FORWARD = 0
    STRAFE = 1
    VERTICAL = 2
    YAW_RATE = 3
    FIRE_LASER = 4
    VENT = 5
    FIRE_MISSILE = 6
    PAINT = 7
    FIRE_KINETIC = 8


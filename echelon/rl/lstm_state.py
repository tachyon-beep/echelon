"""Shared LSTM state type for actor-critic models.

Uses NamedTuple instead of dataclass for torch.compile compatibility.
Dataclasses with tensor fields can cause graph breaks.
"""

from typing import NamedTuple

from torch import Tensor


class LSTMState(NamedTuple):
    """Hidden state for LSTM.

    Attributes:
        h: Hidden state tensor [1, batch, hidden_dim]
        c: Cell state tensor [1, batch, hidden_dim]
    """

    h: Tensor
    c: Tensor

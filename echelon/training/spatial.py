"""Spatial data accumulation for heatmap visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


class SpatialAccumulator:
    """Accumulates 2D spatial events for heatmap generation.

    Tracks spatial distribution of events (deaths, damage, movement) on a
    discretized grid for visualization as W&B heatmaps.

    Attributes:
        grid_size: Resolution of the accumulation grid.
        death_locations: 2D array counting deaths per cell.
        damage_locations: 2D array summing damage per cell.
        movement_density: 2D array counting position samples per cell.
    """

    def __init__(self, grid_size: int = 32) -> None:
        """Initialize accumulator with given grid resolution.

        Args:
            grid_size: Number of cells in each dimension (default 32).
        """
        self.grid_size = grid_size
        self.death_locations = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.damage_locations = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.movement_density = np.zeros((grid_size, grid_size), dtype=np.float32)

    def _to_grid(self, x: float, y: float, world_size: tuple[float, float]) -> tuple[int, int]:
        """Convert world coordinates to grid indices.

        Args:
            x: World x-coordinate.
            y: World y-coordinate.
            world_size: (width, height) of the world.

        Returns:
            (gx, gy) grid indices, clamped to valid range.
        """
        gx = int(np.clip(x / world_size[0] * self.grid_size, 0, self.grid_size - 1))
        gy = int(np.clip(y / world_size[1] * self.grid_size, 0, self.grid_size - 1))
        return gx, gy

    def record_death(self, x: float, y: float, world_size: tuple[float, float]) -> None:
        """Record a death event at the given world position.

        Args:
            x: World x-coordinate.
            y: World y-coordinate.
            world_size: (width, height) of the world.
        """
        gx, gy = self._to_grid(x, y, world_size)
        self.death_locations[gy, gx] += 1.0

    def record_damage(self, x: float, y: float, damage: float, world_size: tuple[float, float]) -> None:
        """Record damage dealt at the given world position.

        Args:
            x: World x-coordinate.
            y: World y-coordinate.
            damage: Amount of damage dealt.
            world_size: (width, height) of the world.
        """
        gx, gy = self._to_grid(x, y, world_size)
        self.damage_locations[gy, gx] += damage

    def record_position(self, x: float, y: float, world_size: tuple[float, float]) -> None:
        """Record an agent position sample for movement density.

        Args:
            x: World x-coordinate.
            y: World y-coordinate.
            world_size: (width, height) of the world.
        """
        gx, gy = self._to_grid(x, y, world_size)
        self.movement_density[gy, gx] += 1.0

    def reset(self) -> None:
        """Clear all accumulated data."""
        self.death_locations.fill(0.0)
        self.damage_locations.fill(0.0)
        self.movement_density.fill(0.0)

    def to_images(self) -> dict[str, Any]:
        """Convert accumulators to W&B images with colormaps.

        Returns:
            Dict mapping heatmap names to wandb.Image objects.
            Empty dict if matplotlib is not available or no data recorded.
        """
        import wandb

        try:
            import matplotlib.pyplot as plt

            images: dict[str, Any] = {}

            for name, data in [
                ("deaths", self.death_locations),
                ("damage", self.damage_locations),
                ("movement", self.movement_density),
            ]:
                if data.max() > 0:
                    normalized = data / data.max()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(normalized, cmap="hot", origin="lower")
                    ax.set_title(f"{name.title()} Heatmap")
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()
                    images[name] = wandb.Image(fig)  # type: ignore[attr-defined]
                    plt.close(fig)

            return images

        except ImportError:
            return {}

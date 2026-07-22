"""
The occupancy grid the evacuation logic reasons over.

Cell values: 0 = free, 1 = wall, 2 = fire.

The old `draw_map()` rendered a 300x300 PNG minimap with cv2 drawing calls. That
now happens on the client (static/js/app.js) against the same grid, which keeps
the feature but saves the server from rasterising and shipping an image on every
single frame. This module is therefore free of cv2 entirely.
"""

from __future__ import annotations

import config


def create_grid():
    """Fresh grid with the static building walls stamped in."""
    grid = [[0 for _ in range(config.GRID_COLS)] for _ in range(config.GRID_ROWS)]

    for c in range(config.WALL_COL_START, config.WALL_COL_END):
        if 0 <= config.WALL_ROW < config.GRID_ROWS and 0 <= c < config.GRID_COLS:
            grid[config.WALL_ROW][c] = 1

    return grid


def mark_fire(grid, fire_pos):
    """Flag the fire cell and its 8 neighbours as unsafe (value 2)."""
    r, c = fire_pos

    for i in range(-1, 2):
        for j in range(-1, 2):
            nr, nc = r + i, c + j
            if 0 <= nr < config.GRID_ROWS and 0 <= nc < config.GRID_COLS:
                grid[nr][nc] = 2

    return grid

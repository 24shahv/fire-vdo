"""Small helpers shared by the pipeline."""

from __future__ import annotations

import config


def map_to_grid(x, y, width, height):
    """Convert a pixel position into a (row, col) grid cell, clamped in range."""
    grid_x = int(x / max(width, 1) * config.GRID_COLS)
    grid_y = int(y / max(height, 1) * config.GRID_ROWS)

    grid_x = min(max(grid_x, 0), config.GRID_COLS - 1)
    grid_y = min(max(grid_y, 0), config.GRID_ROWS - 1)

    return (grid_y, grid_x)


def remove_duplicate_people(global_people, distance_thresh=None):
    """
    Collapse people seen by more than one camera into a single count.

    `global_people` is a list of (grid_pos, (px, py)). Unchanged from the
    original: it compares pixel centres, so it works best when the cameras
    overlap and share a framing.
    """
    if distance_thresh is None:
        distance_thresh = config.DUPLICATE_DISTANCE

    unique_people = []

    for person in global_people:
        (grid_pos, (px, py)) = person

        is_duplicate = False

        for u in unique_people:
            (_, (ux, uy)) = u

            dist = ((px - ux) ** 2 + (py - uy) ** 2) ** 0.5

            if dist < distance_thresh:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_people.append(person)

    return unique_people

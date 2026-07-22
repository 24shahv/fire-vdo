"""
Breadth-first search for a route that avoids walls (1) and fire (2).

Algorithm unchanged. It was dead code in the original main.py; the pipeline now
calls it so the minimap can draw real escape routes.
"""

from __future__ import annotations

from collections import deque


def find_safe_path(grid, start, exits):
    """
    Args:
        grid : list[list[int]] — 0 free, 1 wall, 2 fire
        start: (row, col)
        exits: container of (row, col) goal cells

    Returns:
        list[(row, col)] from `start` to the first exit reached, or None.
    """
    rows, cols = len(grid), len(grid[0])
    exits = set(exits)

    visited = {tuple(start)}
    queue = deque([(tuple(start), [])])

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) in exits:
            return path + [(r, c)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                # An exit cell is always enterable, even if fire is adjacent.
                if grid[nr][nc] != 0 and (nr, nc) not in exits:
                    continue

                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(r, c)]))

    return None

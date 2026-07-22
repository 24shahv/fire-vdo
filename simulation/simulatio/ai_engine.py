import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal, fire=None):
    rows, cols = len(grid), len(grid[0])

    open_set = []
    heapq.heappush(open_set, (0, start))

    came = {}
    g = {start:0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            return path[::-1]

        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = cur[0]+dr, cur[1]+dc

            if not (0<=nr<rows and 0<=nc<cols):
                continue

            # 🧱 wall
            if grid[nr][nc] == 1:
                continue

            # 🔥 MULTIPLE FIRE SUPPORT
            penalty = 0
            if fire:
                for f in fire:   # loop through all fire points
                    d = abs(nr - f[0]) + abs(nc - f[1])
                    if d < 4:
                        penalty += (4 - d) * 3   # stronger avoidance

            nxt = (nr,nc)
            ng = g[cur] + 1 + penalty

            if nxt not in g or ng < g[nxt]:
                came[nxt] = cur
                g[nxt] = ng
                f_score = ng + heuristic(nxt, goal)
                heapq.heappush(open_set, (f_score, nxt))

    return []
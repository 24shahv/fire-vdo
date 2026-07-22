import pygame
import random
import heapq

pygame.init()

# 📐 GRID SIZE
ROWS, COLS = 18, 18
CELL = 35
WIDTH, HEIGHT = COLS*CELL + 250, ROWS*CELL

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SAFEX AI - Hotel Evacuation")

clock = pygame.time.Clock()

# 🎨 COLORS
WHITE = (240,240,240)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (255,0,0)
BLUE = (0,100,255)
GRAY = (40,40,40)

# 🧱 GRID
grid = [[0]*COLS for _ in range(ROWS)]

# 🧱 WALLS (hotel structure simplified)
walls = []

# outer border
for i in range(COLS):
    walls.append((0,i))
    walls.append((ROWS-1,i))
for i in range(ROWS):
    walls.append((i,0))
    walls.append((i,COLS-1))

# vertical corridor
for r in range(2,16):
    walls.append((r,8))

# horizontal corridor
for c in range(2,16):
    walls.append((10,c))

# rooms (left)
for r in range(2,10):
    walls.append((r,3))
for r in range(10,16):
    walls.append((r,5))

# rooms (right)
for r in range(2,10):
    walls.append((r,13))
for r in range(10,16):
    walls.append((r,15))

for r,c in walls:
    grid[r][c] = 1

# 🚪 EXITS (stairs)
exits = [(1,1), (16,16)]

# 👥 PEOPLE (16)
people = []
for _ in range(16):
    while True:
        r = random.randint(3,14)
        c = random.randint(2,15)
        if grid[r][c] == 0:
            people.append({"pos":[r,c], "path":[]})
            break

# 🔥 FIRE
fire = None

# 🧠 HEURISTIC
def h(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# 🧠 A*
def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came = {}
    g = {start:0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came:
                path.append(current)
                current = came[current]
            return path[::-1]

        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = current[0]+dr, current[1]+dc

            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if grid[nr][nc] == 1:
                continue

            if fire and abs(nr-fire[0])+abs(nc-fire[1]) < 2:
                continue

            neighbor = (nr,nc)
            new_g = g[current] + 1

            if neighbor not in g or new_g < g[neighbor]:
                came[neighbor] = current
                g[neighbor] = new_g
                f = new_g + h(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return []

# 📊 EXIT COUNT
def exit_count():
    counts = {ex:0 for ex in exits}
    for p in people:
        pos = tuple(p["pos"])
        if pos in counts:
            counts[pos] += 1
    return counts

# 🧠 EXIT CHOICE WITH CONGESTION
def choose_exit(p):
    counts = exit_count()

    best = None
    best_score = 999

    for ex in exits:
        # 🚫 congestion rule
        if counts[ex] >= 8:
            continue

        path = astar(tuple(p), ex)
        if path and len(path) < best_score:
            best_score = len(path)
            best = ex

    # fallback (if both congested)
    if best is None:
        best = exits[0]

    return best

# 🎮 LOOP
running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                fire = (9,9)

    # 🔲 DRAW GRID
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c*CELL, r*CELL, CELL, CELL)
            pygame.draw.rect(screen, (200,200,200), rect, 1)

            if grid[r][c] == 1:
                pygame.draw.rect(screen, BLACK, rect)

    # 🔥 FIRE
    if fire:
        pygame.draw.rect(screen, RED,
            (fire[1]*CELL, fire[0]*CELL, CELL, CELL))

    # 🚪 EXITS
    counts = exit_count()

    for ex in exits:
        color = GREEN if counts[ex] < 8 else RED
        pygame.draw.rect(screen, color,
            (ex[1]*CELL, ex[0]*CELL, CELL, CELL))

    # 👥 PEOPLE
    for person in people:
        p = person["pos"]

        if fire:
            if not person["path"]:
                target = choose_exit(p)
                person["path"] = astar(tuple(p), target)

            if person["path"]:
                step = person["path"].pop(0)
                p[0], p[1] = step

        pygame.draw.circle(screen, BLUE,
            (p[1]*CELL + CELL//2, p[0]*CELL + CELL//2), 8)

    # 📊 PANEL
    panel_x = COLS*CELL
    pygame.draw.rect(screen, GRAY, (panel_x, 0, 250, HEIGHT))

    font = pygame.font.SysFont(None, 28)

    text1 = font.render(f"Exit A: {counts[exits[0]]}", True, WHITE)
    text2 = font.render(f"Exit B: {counts[exits[1]]}", True, WHITE)
    text3 = font.render("Congestion ≥ 8", True, WHITE)

    screen.blit(text1, (panel_x+20, 50))
    screen.blit(text2, (panel_x+20, 100))
    screen.blit(text3, (panel_x+20, 150))

    pygame.display.flip()
    clock.tick(6)

pygame.quit()
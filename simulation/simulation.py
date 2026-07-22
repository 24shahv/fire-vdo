import pygame
import random

pygame.init()

# 📐 GRID SETTINGS
ROWS, COLS = 12, 12
CELL_SIZE = 50

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SAFEX AI - Grid Evacuation")

clock = pygame.time.Clock()

# 🎨 COLORS
WHITE = (240,240,240)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (255,0,0)
BLUE = (0,100,255)

# 🧱 GRID (0=empty, 1=wall)
grid = [[0]*COLS for _ in range(ROWS)]

# 🧱 CREATE WALLS (like your image)
walls = [
    (1,1),(2,1),(4,1),(5,1),(7,1),
    (5,2),(7,2),
    (1,3),(2,3),(3,3),(5,3),(7,3),(9,3),
    (5,4),(7,4),
    (2,5),(3,5),(4,5),(5,5),(7,5),(9,5),
    (9,6),(9,7),(9,8),
    (0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10),(7,10),(8,10),(9,10)
]

for r,c in walls:
    grid[r][c] = 1

# 🚪 EXITS
exits = [(0,0), (0,10)]

# 👥 PEOPLE
people = []
for _ in range(10):
    while True:
        r = random.randint(6,11)
        c = random.randint(0,11)
        if grid[r][c] == 0:
            people.append([r,c])
            break

# 🔥 FIRE
fire = None

# 🧠 DISTANCE
def dist(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# 🧠 CROWD
def crowd_at_exit(ex):
    return sum(1 for p in people if p == list(ex))

# 🧠 CHOOSE EXIT
def choose_exit(p):
    best = None
    best_score = 999

    for ex in exits:
        score = dist(p, ex)

        # 🔥 avoid fire
        if fire and dist(ex, fire) < 3:
            score += 100

        # 👥 crowd
        score += crowd_at_exit(ex) * 5

        if score < best_score:
            best_score = score
            best = ex

    return best

# 🧠 MOVE PERSON (GRID STEP)
def move(p):
    target = choose_exit(p)

    r, c = p
    tr, tc = target

    options = []

    if tr > r: options.append((r+1, c))
    if tr < r: options.append((r-1, c))
    if tc > c: options.append((r, c+1))
    if tc < c: options.append((r, c-1))

    random.shuffle(options)

    for nr, nc in options:
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            if grid[nr][nc] == 0:
                p[0], p[1] = nr, nc
                return

# 🎮 LOOP
running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                fire = (5,6)

    # 🔲 DRAW GRID
    for r in range(ROWS):
        for c in range(COLS):
            x = c * CELL_SIZE
            y = r * CELL_SIZE

            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            pygame.draw.rect(screen, (200,200,200), rect, 1)

            if grid[r][c] == 1:
                pygame.draw.rect(screen, BLACK, rect)

    # 🟩 EXITS
    for ex in exits:
        r,c = ex
        pygame.draw.rect(screen, GREEN,
            (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # 🔥 FIRE
    if fire:
        r,c = fire
        pygame.draw.rect(screen, RED,
            (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # 👥 PEOPLE
    for p in people:

        if fire:
            move(p)

        r,c = p
        pygame.draw.circle(screen, BLUE,
            (c*CELL_SIZE + CELL_SIZE//2,
             r*CELL_SIZE + CELL_SIZE//2),
            10)

    pygame.display.flip()
    clock.tick(5)

pygame.quit()
import pygame
from layout import GRID, EXITS
from ai_engine import astar

pygame.init()

CELL = 30
ROWS, COLS = len(GRID), len(GRID[0])

screen = pygame.display.set_mode((COLS*CELL+200, ROWS*CELL))
clock = pygame.time.Clock()

# 🎨 COLORS
WHITE=(240,240,240)
BLACK=(0,0,0)
GREEN=(0,255,0)
RED=(255,0,0)
BLUE=(0,100,255)
GRAY=(40,40,40)

# 👥 INIT
people=[]
fire_positions = []   # 🔥 store all fire cells from layout
fire_on = False

for r in range(ROWS):
    for c in range(COLS):
        if GRID[r][c]==2:
            people.append({
                "pos":[r,c],
                "path":[],
                "delay":0
            })
        elif GRID[r][c]==3:
            fire_positions.append((r,c))   # store fire from layout

# 🚪 EXIT SYSTEM
exit_count={ex:0 for ex in EXITS}
MAX=2

running=True
while running:
    screen.fill(WHITE)

    # 🎮 EVENTS
    for e in pygame.event.get():
        if e.type==pygame.QUIT:
            running=False

        if e.type==pygame.KEYDOWN:
            if e.key==pygame.K_f:
                fire_on=True
            
                for p in people:
                    p["path"]=[]

    # ⚠️ OCCUPANCY SET (STRICT ONE PERSON PER BLOCK)
    occupied = {tuple(p["pos"]) for p in people}

    # 🔲 DRAW GRID
    for r in range(ROWS):
        for c in range(COLS):
            rect=pygame.Rect(c*CELL,r*CELL,CELL,CELL)
            pygame.draw.rect(screen,(200,200,200),rect,1)

            if GRID[r][c]==1:
                pygame.draw.rect(screen,BLACK,rect)

    # 🔥 FIRE
    if fire_on:
        for  f in fire_positions:
            pygame.draw.rect(screen,RED,(f[1]*CELL,f[0]*CELL,CELL,CELL))

    # 🚪 EXITS
    for ex in EXITS:
        col=GREEN if exit_count[ex]<MAX else RED
        pygame.draw.rect(screen,col,(ex[1]*CELL,ex[0]*CELL,CELL,CELL))

    new_people=[]
    new_occupied=set()   # 🔥 IMPORTANT (updated positions)

    # 👥 PEOPLE LOOP
    for p in people:
        pos=p["pos"]

        # 🚫 BEFORE FIRE → NO MOVEMENT
        if not fire_on:
            pygame.draw.circle(screen,BLUE,(pos[1]*CELL+15,pos[0]*CELL+15),6)
            new_people.append(p)
            new_occupied.add(tuple(pos))
            continue

        # 🎯 dynamic exit selection (EVERY FRAME)
        exits_sorted = sorted(EXITS, key=lambda ex: abs(pos[0]-ex[0]) + abs(pos[1]-ex[1]))

        new_target = None
        for ex in exits_sorted:
            if exit_count[ex] < MAX:
                new_target = ex
                break

        if new_target is None:
            new_target = exits_sorted[0]

            # 🔥 IF TARGET CHANGED → RESET PATH
        if "target" not in p or p["target"] != new_target:
            p["target"] = new_target
            p["path"] = []   # force reroute

        target = p["target"]
        # 🧠 compute path
        if not p["path"] or exit_count[target] >= MAX:
            p["path"]=astar(GRID,tuple(pos),target,fire_positions)

        # 🐢 CONTROL SPEED
        p["delay"]+=1

        if p["delay"]>=6:
            p["delay"]=0

            if p["path"]:
                nxt=p["path"][0]

                # 🔥 STRICT OCCUPANCY CHECK
                if nxt not in occupied and nxt not in new_occupied:
                    p["path"].pop(0)
                    pos[0],pos[1]=nxt

        # 🚪 ENTER EXIT
        if abs(pos[0]-target[0])+abs(pos[1]-target[1])<=1:
            if exit_count[target]<MAX:
                exit_count[target]+=1
                continue

        new_people.append(p)
        new_occupied.add(tuple(pos))

        pygame.draw.circle(screen,BLUE,(pos[1]*CELL+15,pos[0]*CELL+15),6)

    people=new_people

    # 🐢 SLOW EXIT CLEARING (CONTROLLED)
    if "release_timer" not in globals():
        release_timer = 0

    release_timer += 0.5

    if release_timer >= 30:   # 🔥 increase for more congestion (try 40–80)
        release_timer = 0

        for ex in EXITS:
            if exit_count[ex] > 0:
                exit_count[ex] -= 1

    # 📊 PANEL
    pygame.draw.rect(screen,GRAY,(COLS*CELL,0,200,ROWS*CELL))
    font=pygame.font.SysFont(None,24)

    y=20
    for i,ex in enumerate(EXITS):
        screen.blit(font.render(f"Exit {i+1}: {int(exit_count[ex])}",True,WHITE),(COLS*CELL+10,y))
        y+=25

    screen.blit(font.render("Press F = Fire",True,WHITE),(COLS*CELL+10,y+20))

    pygame.display.flip()
    clock.tick(20)

pygame.quit()
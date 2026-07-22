# Desktop simulations (not deployed)

The original pygame evacuation simulations, kept for reference and coursework:

- `simulation.py` — 12x12 grid evacuation
- `hotel_simulation.py` — 18x18 hotel floor with A* routing
- `simulatio/` — layout + A* engine + runner

The web service never imports these, and `pygame` is deliberately absent from
`requirements.txt` — they open a desktop window, which a Render container has
no display for.

Run one locally:

```bash
pip install pygame
python simulation/hotel_simulation.py
```

The A* logic in `simulatio/ai_engine.py` is a richer version of the BFS in the
deployed `pathfinding.py` — it adds a proximity penalty around fire cells
rather than treating them as hard walls. Worth porting into the live pipeline
if you want routes that *prefer* distance from fire instead of merely avoiding
burning cells.

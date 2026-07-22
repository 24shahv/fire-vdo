# Desktop simulations (not deployed)

These are the original pygame evacuation simulations from the project:

- `simulation.py` — 12×12 grid evacuation
- `hotel_simulation.py` — 18×18 hotel floor with A* routing
- `simulatio/` — layout + A* engine + runner

They are kept for reference and coursework. **The web service never imports
them**, and `pygame` is deliberately absent from `requirements.txt` — these
open a desktop window, which a Render container has no display for.

To run one locally:

```bash
pip install pygame
python simulation/hotel_simulation.py
```

The A* logic in `simulatio/ai_engine.py` is a richer version of the BFS in the
deployed `pathfinding.py` — it adds a proximity penalty around fire cells
rather than treating them as hard walls. Worth porting into the live pipeline
if you want routes that *prefer* distance from fire instead of merely avoiding
burning cells.

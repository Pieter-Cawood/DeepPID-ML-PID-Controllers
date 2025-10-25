
# Usage Guide

- Install editable: `pip install -e .`
- Run examples: `python examples/run_pid_vs_gru.py`
- Train the GRU scheduler (toy): `python examples/train_gru_scheduler.py`

## API
- `deeppid.PID` — classic PID controller.
- `deeppid.AdaptiveGRUPID` — wrapper around `GRUGainNet` for adaptive gains.
- `deeppid.envs.plants` — FOPDT, SOPDT, NonlinearTank.
- `deeppid.utils.sim.simulate_closed_loop` — simple simulator.

## Testing
```bash
pip install pytest
pytest -q
```

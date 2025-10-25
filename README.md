
# DeepPID — PID + ML Experiments

A professional, documented playground for classic PID control and ML-enhanced PID.
This repo provides:
- Clean PID baseline
- GRU-based adaptive gain scheduling (neural PID)
- Simple process models (FOPDT, SOPDT, nonlinear tank)
- Reproducible examples and matplotlib visualizations
- PyPI-style packaging so imports **work**: `import deeppid`

## Install (editable)
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate  
# OR Windows PowerShell
.venv\Scripts\activate   
pip install -e .
```

## Quick start
```bash
python examples/run_pid_vs_gru.py
```

This launches a step-response comparison on a nonlinear tank.

## Project layout
```text
deeppid/
  controllers/
    pid.py                # Classic PID with anti-windup + bumpless transfer
    adaptive_gru_pid.py   # GRU gain scheduler for Kp, Ki, Kd
  envs/
    plants.py             # FOPDT, SOPDT, NonlinearTank
  scheduling/
    setpoint.py           # Step/ramp/sine profiles with seeds
  utils/
    sim.py                # Fixed-step simulator + plots
examples/
  run_pid_vs_gru.py       # Side-by-side simulation + plots
  train_gru_scheduler.py  # Lightweight training loop (PyTorch)
tests/
  test_pid.py
  test_plants.py
```

## Notes on ML + PID
- We keep the classical PID structure for interpretability and stability.
- The GRU predicts per-step gains that are **constrained** to practical ranges.
- Loss is integrated tracking error + control effort penalty.

## References
- Deep reinforcement learning enhanced PID (DDPG-based): Hao et al., 2025 (Scientific Reports).
- DRL + PID supervisor control: Wang et al., 2023 (Engineering Applications of AI).
- PID tuning with learning methods (survey & methods): Bujgoi et al., 2025 (MDPI Processes).

## License
MIT

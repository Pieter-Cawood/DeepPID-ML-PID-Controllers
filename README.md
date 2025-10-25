# DeepPID — A PID Playground and Deep Learning–Based Adaptive PID Controller

[![CI](https://github.com/Pieter-Cawood/DeepPID/actions/workflows/ci.yml/badge.svg)](https://github.com/Pieter-Cawood/DeepPID/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

<div style="display: flex; align-items: flex-start;">
  <div style="flex: 1;">

A **playground** for experimenting with **PID** and **machine-learning-based controllers**.  
DeepPID provides both **traditional** and **neural adaptive controllers** in a single, consistent framework, complete with a live **Tkinter + Matplotlib GUI** for interactive benchmarking.

Through extensive simulation and real-time tests on **nonlinear**, **coupled**, and **time-varying plants**, it is demonstrated that the **ML-based adaptive models** (*GRU*, *MLP*, and *Transformer* variants) consistently **outperform conventional PID and Cascade-PID controllers** in both transient and steady-state performance.

The adaptive models achieve:
- ⚡ **Faster convergence** with minimal overshoot  
- 🎯 **Near-zero steady-state error** across diverse process conditions  
- 🧩 **Robustness** to parameter drift and actuator limits without manual re-tuning  

These results confirm that **data-driven adaptation—when combined with physical constraints—generalizes PID control** beyond fixed-gain heuristics while maintaining interpretability and stability.

  </div>
  <div style="margin-left: 20px; flex-shrink: 0;">
    <img src="docs/deeppid.png" alt="DeepPID Architecture" width="340"><br>
    <em>DeepPID — Hybrid classical & ML-based control framework.</em>
  </div>
</div>

---

### GRUController — Adaptive Neural Controller (PID-inspired)

A gated recurrent unit (GRU) network that directly predicts actuator speeds based on recent state history.  
It embeds **PID-like control objectives**—composition matching, total flow regulation, smoothness, and bounded actuation—into its online loss function.  
While not using explicit PID equations, it behaves as a **hybrid adaptive controller**, combining physical constraints with data-driven prediction.  
This approach consistently **outperforms fixed-gain PID** under nonlinear, coupled, or drifting plant conditions, achieving **near-zero steady-state error** and **smoother transients**.

---

The GUI (`examples/test.py`) lets you:
- Choose different **plant problems** (tank, flow, quadcopter-like, etc.)
- Switch between **controllers** (PID, CascadePID, MLP, GRU, Transformer, etc.)
- Observe **real-time set-point tracking**, **mean absolute error (MAE)** curves, and **controller outputs**
- See which approach adapts fastest to nonlinear or coupled dynamics

<p align="center">
  <img src="docs/gui.png" alt="DeepPID GUI"><br>
  <em>Interactive GUI — live comparison of controller performance.</em>
</p>

---

## What’s inside

- **PID**: IMC‑style auto‑tuned PID with anti‑windup, bumpless transfer, and online refinement.  
- **CascadePID**: stabilized inner PID with outer composition/total loops.  
- **Neural controllers**: MLP, GRU, Transformer, PINN‑flavored, hybrid MPC stub, and safety‑wrapped RL stub.  
- **GUI**: real‑time MAE table + history plot for apples‑to‑apples comparisons.  
- **Packaging**: imports work (`import deeppid`) and examples run out of the box.

## Install (editable)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# OR Windows PowerShell
.venv\Scripts\activate
pip install -e .
```

## Quick start (GUI)

```bash
python examples/test.py
```

This launches the controller shoot‑out app. Choose any plant from the dropdown, pick a driver
(controller), and watch the **real‑time mix error** and **MAE history** update as the set‑point moves.

## Project layout

```text
deeppid/
  controllers/
    controllers.py        # PID, CascadePID, MLP, GRU, Transformer, etc.
  utils/
    utils.py              # Utility functions 
  envs/
    problems.py           # “Problems”/plants with labels, units, limits
examples/
  test.py                 # Tk + Matplotlib live comparison app
tests/                    # (optional) put your pytest tests here
```

## How the GRU controller works (and why it’s different from PID)

**Conventional PID** computes the next actuation using fixed (or slowly tuned) gains `Kp, Ki, Kd`
around an interpretable structure with anti‑windup and filters. It’s great when the plant can be
reasonably approximated by first/second‑order dynamics and the operating point doesn’t move too much.

**GRU controller (adaptive & live)** takes a different tack:

- **State** each tick: `[target ratio, total set‑point, recent measured flows, previous speeds]`
- **Sequence model**: a GRU processes the recent context to estimate the next speeds in one shot
- **Hard safety layer**: speeds are **slew‑limited** and **clamped** to `[min, max]`
- **Online objective** (optimized every few steps):
  - match **composition** (fractions) to target
  - match **total** output to the requested value
  - keep **smooth** changes (actuator wellness)
  - stay inside bounds with a **soft barrier**
  - optionally track a reference/baseline (e.g., PID suggestion)
- **Why it helps**: when the plant is nonlinear, coupled, or operating conditions drift, the GRU
  can “learn” mappings PID would need re‑tuning for. You still keep the same safety rails as PID.

You can inspect all loss terms and constraints in `controllers.py` (classes `GRUController`, `MLPController`).
Everything is implemented to be **stable‑by‑construction**: we never bypass slew/clamp and we bias to
baseline allocations when signals are missing or become non‑finite.

---

## How to add a new *Problem* (plant)

Problems live in `deeppid/envs/problems.py` and are registered in `AVAILABLE_PROBLEMS` so the GUI can discover them.

**1) Implement a class** with the following minimal API (feel free to copy an existing one and tweak):

```python
# deeppid/envs/problems.py

import torch

class MyCustomProblem:
    def __init__(self, Ts: float):
        # Dimensions / labels
        self.N = 3
        self.labels = [f"Source {i+1}" for i in range(self.N)]

        # Metadata (optional; affects GUI labels)
        self.output_name = "Flow"
        self.output_unit = "L/min"
        self.entity_title = "Material"

        # Constraints and nominal model parameters
        self.k_coeff = torch.tensor([0.9, 1.1, 0.8], dtype=torch.float64)
        self.speed_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        self.speed_max = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        self.slew_rate = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)  # % per step
        self.Ts = Ts
        self.alpha = 0.4  # single-pole plant step factor for filtering

        # Defaults
        self.default_total_flow = 60.0
        self.default_target_ratio = torch.ones(self.N, dtype=torch.float64) / self.N

        # Internal filtered measurement
        self._y = torch.zeros(self.N, dtype=torch.float64)

    def baseline_allocation(self, ratio: torch.Tensor, F_total: torch.Tensor) -> torch.Tensor:
        \"\"\"Feedforward speeds (simple inverse of k within bounds).\"\"\"
        s = (ratio * F_total) / (self.k_coeff + 1e-12)
        return torch.clamp(s, self.speed_min, self.speed_max)

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        \"\"\"One simulation step. Update and return filtered measured outputs.\"\"\"
        y_raw = self.k_coeff * speeds_cmd
        self._y = self._y + self.alpha * (y_raw - self._y)
        return self._y.clone()

    def comp_from_speeds(self, speeds: torch.Tensor) -> torch.Tensor:
        \"\"\"Return composition (fractions) implied by nominal model for display/metrics.\"\"\"
        flow = self.k_coeff * speeds
        tot = flow.sum() + 1e-12
        return flow / tot
```

**2) Register it** at the bottom of `problems.py`:

```python
AVAILABLE_PROBLEMS = {
    # existing ones...
    "MyCustomProblem": MyCustomProblem,
}
```

Your new problem will now appear in the GUI's *Problem* dropdown.

---

## How to add a new *Controller*

Controllers live in `deeppid/controllers/controllers.py`. The GUI expects them to be discoverable via the
package registry `deeppid.AVAILABLE` (set up in `deeppid/__init__.py`). The easiest path is to implement
a class with a **PID-like interface** and wrap it with `CtrlAdapter` automatically:

**Minimum contract (any of these works):**
- Provide `step(flows_meas_filt, target_ratio, F_total, speeds_direct)` → returns speeds (Tensor of length N); or
- Provide `forward(target_ratio, F_total, flows_meas_filt)` → returns speeds; and optionally
- Provide `train_step(...)` if your controller learns online; and
- Optionally `sync_to(speeds_now, flows_now)` to initialize internal state when the problem changes.

**Constructor signature** should accept (at least) the common parameters so the GUI can instantiate it:
`(N, k, speed_min, speed_max, Ts, slew_rate)` — extra arguments are fine.

**Skeleton:**

```python
# deeppid/controllers/controllers.py
import torch

class MyFancyController(torch.nn.Module):
    def __init__(self, N, k, speed_min, speed_max, Ts, slew_rate, **kwargs):
        super().__init__()
        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate
        self.register_buffer("prev_speeds", torch.ones(N, dtype=torch.float64) * 25.0)
        # your nets/params here...
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N + 1 + N + N, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, N)
        )

    def _state(self, tr, Ft, y):
        Ft_n = (Ft / (torch.sum(self.k * self.speed_max).clamp_min(1.0))).clamp(0, 2.0)
        prev = ((self.prev_speeds - self.speed_min) / (self.speed_max - self.speed_min).clamp_min(1e-6)).clamp(0, 1)
        flows_n = (y / (Ft + 1e-6)).clamp(0, 2.0)
        return torch.cat([tr, Ft_n.view(1), flows_n, prev], dim=0)

    def forward(self, target_ratio, F_total, flows_meas_filt):
        x = self._state(target_ratio, F_total, flows_meas_filt)
        raw = self.net(x)
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        s = self.speed_min + span * torch.sigmoid(raw)
        s = torch.clamp(
            self.prev_speeds + torch.clamp(s - self.prev_speeds, -self.slew, self.slew),
            self.speed_min, self.speed_max
        )
        self.prev_speeds = s.clone()
        return s

    # Optional, for adaptive controllers
    def train_step(self, target_ratio, F_total, flows_meas_filt, **_):
        # do an update; return the new speeds (or None)
        return self.forward(target_ratio, F_total, flows_meas_filt)
```

**Register the controller** name in `deeppid/__init__.py` (registry `AVAILABLE`):  

```python
from .controllers.controllers import MyFancyController

AVAILABLE["MyFancy"] = MyFancyController
```

It will then show up in the GUI *Driver* combo-box automatically.

> 🧩 Tip: If your controller already conforms to the `step(...)` signature, the GUI will call it directly.
> Otherwise it will fall back to `forward(...)`. `CtrlAdapter` normalizes those differences for you.

---

## Testing

- Quick import smoke test: `python -c "import deeppid; print('OK')"`  
- Run GUI: `python examples/test.py`  
- Add lightweight unit tests in `tests/` (e.g., `pytest -q`).

## License

MIT

# ─────────────────────────────────────────────────────────────────────────────
# File: app.py
# GUI app (Tkinter + Matplotlib) that compares controllers from models.py
# against interchangeable "Problem" dynamics from problems.py.
#
# Now supports variable N and per-problem units/titles:
#   • problem.labels            -> row labels (e.g., ["Source 1", ...] or ["Rotor 1", ...])
#   • problem.output_name       -> "Flow" / "Thrust" / etc. (optional, defaults to "Flow")
#   • problem.output_unit       -> "L/min" / "N" / etc. (optional, defaults to "L/min")
#   • problem.entity_title      -> "Material" / "Rotor" / etc. (optional, defaults to "Material")
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import numpy as np
import torch
import importlib

# --- Matplotlib for interactive chart ---
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import problems (plants)
from .problems import *

torch.set_default_dtype(torch.float64)

# -------------------------- Import controllers robustly --------------------------
AVAILABLE = {}

def _safe_import(class_name, menu_name=None):
    """Try absolute then relative import so this works both as a script and a package."""
    try:
        mod = importlib.import_module("models")
        cls = getattr(mod, class_name)
        AVAILABLE[menu_name or class_name] = cls
        return
    except Exception:
        pass
    try:
        mod = importlib.import_module(".models", package=__package__)
        cls = getattr(mod, class_name)
        AVAILABLE[menu_name or class_name] = cls
        return
    except Exception:
        pass

# Core controllers (names must match your models.py)
_safe_import("PIDController", "PID")
_safe_import("CascadePIDController", "CascadePID")
_safe_import("MLPController", "MLP")
_safe_import("GRUController", "GRU")

# Advanced (map to your class names)
_safe_import("HybridMPCController", "HybridMPC")
_safe_import("PIDResidualNN", "PID+Residual")
_safe_import("TransformerCtrl", "Transformer")
_safe_import("RLSafetyCtrl", "SafeRL")
_safe_import("PINNCtrl", "PINN")
_safe_import("AdaptiveHierCtrl", "AdaptiveHier")

# -------------------------- Configuration --------------------------
CONTROL_PERIOD_MS = 250   # the UI tick; each problem uses Ts passed in here
Ts = CONTROL_PERIOD_MS / 1000.0

# For MLP/GRU defaults (only passed through to their constructors if present)
W_COMP, W_TOTAL, W_SMOOTH, W_BOUND, W_REF, W_MEAS = 3.0, 6.0, 0.10, 0.08, 0.03, 0.25
SEQ_LEN, GRU_HID, GRU_TRAIN_STEPS, MLP_TRAIN_STEPS = 20, 128, 6, 12

# Chart history cap
MAX_POINTS = 1200  # ~5 minutes at 250 ms per step

# -------------------------- Utilities --------------------------

def clip(v, lo, hi):
    return torch.min(torch.max(v, lo), hi)

def slew_limit(prev, desired, max_delta, lo, hi):
    return clip(prev + torch.clamp(desired - prev, -max_delta, max_delta), lo, hi)

def sanitize(x, fallback):
    """Replace NaN/Inf in tensor x with values from fallback (same shape)."""
    bad = ~torch.isfinite(x)
    if bad.any():
        x = x.clone()
        x[bad] = fallback[bad]
    return x

def ensure_shape(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Ensure x has the same length as 'like'. If longer -> truncate; if shorter -> pad with 'like'.
    Always returns float64 1-D tensor of len N.
    """
    like = like.detach().clone().to(torch.float64).reshape(-1)
    N = like.numel()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    x = x.detach().clone().to(torch.float64).reshape(-1)
    if x.numel() == N:
        return x
    y = like.clone()
    m = min(N, x.numel())
    if m > 0:
        y[:m] = x[:m]
    return y

def to_f64_scalar_tensor(x):
    """Return a float64 0-D tensor from a float or a tensor (no re-wrapping warning)."""
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(torch.float64).reshape(())
    return torch.tensor(float(x), dtype=torch.float64)


# Normalize controller interfaces
class CtrlAdapter:
    def __init__(self, name, inst):
        self.name = name
        self.inst = inst

    def sync_to(self, speeds_now, flows_now):
        if hasattr(self.inst, "sync_to"):
            self.inst.sync_to(speeds_now, flows_now)
        if hasattr(self.inst, "reset_hidden"):
            self.inst.reset_hidden()

    def train_once(self, target_ratio, F_total, flows_meas_filt, ref_speeds=None, measured_flows=None):
        if hasattr(self.inst, "train_step"):
            try:
                return self.inst.train_step(target_ratio, F_total, flows_meas_filt,
                                            ref_speeds=ref_speeds, measured_flows=measured_flows)
            except TypeError:
                try:
                    return self.inst.train_step(target_ratio, F_total, flows_meas_filt)
                except Exception:
                    return None
        return None

    def suggest(self, flows_meas_filt, target_ratio, F_total, speeds_direct):
        if hasattr(self.inst, "step"):
            try:
                out = self.inst.step(flows_meas_filt, target_ratio, F_total, speeds_direct)
                return out[0] if isinstance(out, (tuple, list)) else out
            except TypeError:
                try:
                    out = self.inst.step(flows_meas_filt, target_ratio, F_total)
                    return out[0] if isinstance(out, (tuple, list)) else out
                except Exception:
                    pass
        if hasattr(self.inst, "forward"):
            try:
                return self.inst.forward(target_ratio, F_total, flows_meas_filt)
            except Exception:
                pass
        if hasattr(self.inst, "prev_speeds"):
            return self.inst.prev_speeds.clone()
        return speeds_direct.clone()


# -------------------------- App --------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Controllers Comparison — Interchangeable Plant Problems")

        # --- Choose problem ---
        self.problem_names = list(AVAILABLE_PROBLEMS.keys())
        if not self.problem_names:
            raise RuntimeError("No problems available. Ensure problems.py defines AVAILABLE_PROBLEMS.")
        self.problem_var = tk.StringVar(value=self.problem_names[0])
        self.problem = AVAILABLE_PROBLEMS[self.problem_var.get()](Ts)

        # Per-problem metadata (defaults for older problems)
        self.output_name = getattr(self.problem, "output_name", "Flow")         # e.g., "Flow" or "Thrust"
        self.output_unit = getattr(self.problem, "output_unit", "L/min")        # e.g., "L/min" or "N"
        self.entity_title = getattr(self.problem, "entity_title", "Material")   # e.g., "Material" or "Rotor")

        # --- Global target (shared across problems) ---
        self.target_ratio = self.problem.default_target_ratio.clone()
        self.F_total_target = to_f64_scalar_tensor(self.problem.default_total_flow)

        # Build controllers for the current problem (important for variable N!)
        self.controllers = {}
        self.controller_names = []
        self._build_controllers_for_current_problem()

        # Plant state
        self.speeds_cmd = torch.ones(self.problem.N, dtype=torch.float64) * 25.0
        self.flows_meas_filt = torch.zeros(self.problem.N, dtype=torch.float64)

        # Histories for chart
        self.step = 0
        self.steps_hist = []
        self.mae_hist = {name: [] for name in self.controllers.keys()}

        # UI state
        self.running = True
        self._updating_ratios = False  # guard flag for spinbox updates

        # Fixed-width font for tables
        self.fixed = tkfont.nametofont("TkFixedFont")

        # ─────────────────── Top Controls (packed) ───────────────────
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Problem:").grid(row=0, column=0, sticky="w")
        problem_cb = ttk.Combobox(ctrl, textvariable=self.problem_var, values=self.problem_names, width=28, state="readonly")
        problem_cb.grid(row=0, column=1, sticky="w", padx=(6, 12))
        problem_cb.bind("<<ComboboxSelected>>", self.on_problem_change)

        ttk.Label(ctrl, text="Driver:").grid(row=0, column=2, sticky="w")
        self.driver_var = tk.StringVar(value=self._default_driver_name())
        driver = ttk.Combobox(ctrl, textvariable=self.driver_var, values=self.controller_names, width=14, state="readonly")
        driver.grid(row=0, column=3, sticky="w", padx=(6, 12))
        driver.bind("<<ComboboxSelected>>", self.on_driver_change)

        # Dynamic label for total target with units
        self.total_label_text = tk.StringVar(value=f"Target Total {self.output_name} ({self.output_unit}):")
        ttk.Label(ctrl, textvariable=self.total_label_text).grid(row=0, column=4, sticky="w")
        self.total_flow_var = tk.DoubleVar(value=float(self.F_total_target.item()))
        ttk.Entry(ctrl, textvariable=self.total_flow_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 12))

        if "MLP" in self.controllers:
            ttk.Button(ctrl, text="Train MLP once", command=self.optimize_mlp_once).grid(row=0, column=6, padx=6)
        if "GRU" in self.controllers:
            ttk.Button(ctrl, text="Train GRU once", command=self.optimize_gru_once).grid(row=0, column=7, padx=6)

        self.run_btn = ttk.Button(ctrl, text="Stop", command=self.toggle_run)
        self.run_btn.grid(row=0, column=8, padx=6)
        ttk.Button(ctrl, text="Randomize Target", command=self.randomize_target).grid(row=0, column=9, padx=6)

        # Loaded controllers label + Step counter (packed above main area)
        self.loaded_label = ttk.Label(root, text=f"Loaded: {', '.join(self.controller_names)}")
        self.loaded_label.pack(fill="x", padx=10)
        self.step_label = ttk.Label(root, text="Step: 0")
        self.step_label.pack(fill="x", padx=10, pady=(0, 4))

        # ─────────────────── Main 2-column area (grid) ───────────────────
        main = ttk.Frame(root, padding=(10, 0))
        self.main = main  # keep a handle for width syncing
        main.pack(fill="both", expand=True)
        # Left column fixed like the MAE summary; right column expands
        main.grid_columnconfigure(0, weight=0)  # fixed/narrow, minsize will be set to summary width
        main.grid_columnconfigure(1, weight=1, uniform="cols")

        # Left column, Row 0: Target Ratio Editor
        self.ratio_frame = ttk.LabelFrame(main, text="Target Ratio (%) — type/edit; others adjust to keep 100%", padding=10)
        self.ratio_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self._build_ratio_controls()

        # Right column, Row 0: Applied Driver (Speeds & Outputs)
        self.applied_frame = ttk.LabelFrame(main, text="Applied Driver: Speeds & Measured Outputs", padding=10)
        self.applied_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))

        # Left column, Row 1: Measured Share
        self.bars_frame = ttk.LabelFrame(main, text="Measured Share (%) — Applied", padding=10)
        self.bars_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))

        # Right column, Row 1: Controllers — Real-time Mix Error
        self.cmpf = ttk.LabelFrame(main, text="Controllers — Real-time Mix Error (pp)", padding=10)
        self.cmpf.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))

        # Row 2 (side by side): MAE summary table and chart
        bottom = ttk.Frame(main, padding=(0, 0))
        self.bottom = bottom  # keep a handle for width syncing
        bottom.grid(row=2, column=0, columnspan=2, sticky="nsew")
        bottom.grid_columnconfigure(0, weight=0)  # summary (fixed)
        bottom.grid_columnconfigure(1, weight=1)  # chart (expands)

        # ── MAE Summary table on the left
        self.summary_table_frame = ttk.LabelFrame(bottom, text="MAE Summary (pp)", padding=8)
        self.summary_table_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 8), pady=(0, 10))
        self.summary_table_frame.grid_columnconfigure(0, weight=0)
        self.summary_table_frame.grid_columnconfigure(1, weight=1)

        header_style = {"font": self.fixed}
        ttk.Label(self.summary_table_frame, text="Controller", **header_style).grid(row=0, column=0, sticky="w", padx=(2, 8))
        ttk.Label(self.summary_table_frame, text="MAE (pp) / Max (pp)", **header_style).grid(row=0, column=1, sticky="w")

        self.summary_rows = {}  # name -> (name_label, value_label)
        for i, name in enumerate(self.controller_names, start=1):
            name_lbl = ttk.Label(self.summary_table_frame, text=name, font=self.fixed)
            val_lbl = ttk.Label(self.summary_table_frame, text="-- / --", font=self.fixed)
            name_lbl.grid(row=i, column=0, sticky="w", padx=(2, 8))
            val_lbl.grid(row=i, column=1, sticky="w")
            self.summary_rows[name] = (name_lbl, val_lbl)

        # Right-aligned tail (applied total & step)
        self.summary_tail = ttk.Label(
            self.summary_table_frame,
            text=f"Applied total {self.output_name.lower()} ~ -- {self.output_unit} (target --)   | Step 0",
            font=self.fixed
        )
        tail_row = len(self.controller_names) + 1
        self.summary_tail.grid(row=tail_row, column=0, columnspan=2, sticky="e", pady=(6, 0))

        # Match left column width to MAE summary width
        self._sync_left_column_width_with_summary()

        # ── MAE History chart on the right
        chart_frame = ttk.LabelFrame(bottom, text="MAE History (pp) vs Step", padding=6)
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=(0, 10))
        bottom.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(7.0, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("MAE (pp)")
        self.ax.grid(True, alpha=0.3)

        self.lines = {}  # name -> Line2D
        for name in self.controller_names:
            line, = self.ax.plot([], [], label=name)
            self.lines[name] = line
        self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, chart_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        btns = ttk.Frame(chart_frame)
        btns.pack(side="bottom", fill="x")
        ttk.Button(btns, text="Clear Chart", command=self._clear_chart).pack(side="right", padx=4)

        # Build all dynamic tables/rows once initially
        self._build_applied_table()
        self._build_share_bars()
        self._build_cmp_table()

        # Sync controllers to current plant state
        for ctrl in self.controllers.values():
            ctrl.sync_to(self.speeds_cmd, self.flows_meas_filt)

        # Loop
        self.root.after(CONTROL_PERIOD_MS, self.loop)

    # --- controller building / driver helper ---

    def _default_driver_name(self):
        names = list(self.controllers.keys())
        if not names:
            return ""
        return "GRU" if "GRU" in names else names[0]

    def _build_controllers_for_current_problem(self):
        """(Re)build controller instances to match the CURRENT problem's N, gains, limits."""
        self.controllers.clear()
        dd = dict(
            N=self.problem.N,
            k=self.problem.k_coeff,
            speed_min=self.problem.speed_min,
            speed_max=self.problem.speed_max,
            Ts=Ts,
            plant_alpha=getattr(self.problem, "alpha", 0.5),
            slew_rate=self.problem.slew_rate,
        )

        def try_add(name, cls):
            if cls is None:
                return
            try:
                inst = None
                try:
                    inst = cls(**dd)  # PID/Cascade keyword signature
                except TypeError:
                    if name == "MLP":
                        inst = cls(self.problem.N, self.problem.k_coeff, self.problem.speed_min, self.problem.speed_max,
                                   Ts, self.problem.slew_rate,
                                   hidden=128, train_steps=MLP_TRAIN_STEPS,
                                   W_COMP=W_COMP, W_TOTAL=W_TOTAL, W_SMOOTH=W_SMOOTH,
                                   W_BOUND=W_BOUND, W_REF=W_REF, W_MEAS=W_MEAS)
                    elif name == "GRU":
                        inst = cls(self.problem.N, self.problem.k_coeff, self.problem.speed_min, self.problem.speed_max,
                                   Ts, self.problem.slew_rate,
                                   seq_len=SEQ_LEN, hidden=GRU_HID, train_steps=GRU_TRAIN_STEPS,
                                   W_COMP=W_COMP, W_TOTAL=W_TOTAL, W_SMOOTH=W_SMOOTH,
                                   W_BOUND=W_BOUND, W_REF=W_REF, W_MEAS=W_MEAS)
                    else:
                        inst = cls(self.problem.N, self.problem.k_coeff, self.problem.speed_min, self.problem.speed_max,
                                   Ts, getattr(self.problem, "alpha", 0.5), self.problem.slew_rate)
                self.controllers[name] = CtrlAdapter(name, inst)
            except Exception:
                pass

        # prefer PID/Cascade first in display order
        order = []
        for preferred in ["PID", "CascadePID"]:
            if preferred in AVAILABLE:
                try_add(preferred, AVAILABLE[preferred])
                if preferred in self.controllers:
                    order.append(preferred)
        for name, cls in AVAILABLE.items():
            if name not in ("PID", "CascadePID"):
                try_add(name, cls)
                if name in self.controllers:
                    order.append(name)

        self.controller_names = [n for n in order if n in self.controllers]

    # --- UI helpers ---

    def _sync_left_column_width_with_summary(self):
        """Make the left column (Target Ratio + Measured Share) match the MAE summary column width."""
        try:
            self.root.update_idletasks()
            w = max(self.summary_table_frame.winfo_reqwidth(), 280)
            if hasattr(self, "main"):
                self.main.grid_columnconfigure(0, minsize=w, weight=0)
            if hasattr(self, "bottom"):
                self.bottom.grid_columnconfigure(0, minsize=w, weight=0)
        except Exception:
            pass

    def _stability_is_static(self) -> bool:
        """True when the stability slider is effectively 100%."""
        try:
            return float(self.stability_var.get()) >= 99.9
        except Exception:
            return True

    def _update_ratio_controls_state(self):
        """Enable editing only when stability is static; otherwise lock spinboxes."""
        if not hasattr(self, "ratio_spinboxes"):
            return
        state = "normal" if self._stability_is_static() else "disabled"
        for sb in self.ratio_spinboxes:
            try:
                sb.config(state=state)
            except Exception:
                pass

    def _build_ratio_controls(self):
        """Create spinboxes for per-channel target ratio editing (sum locked to 100%) + stability slider."""
        # Clear old widgets if rebuilding
        for child in self.ratio_frame.winfo_children():
            child.destroy()

        self.ratio_vars = []
        self.ratio_spinboxes = []

        header = ttk.Frame(self.ratio_frame)
        header.pack(fill="x")
        ttk.Label(header, text=self.entity_title, width=18).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Target (%)", width=12).grid(row=0, column=1, sticky="w")

        for i, name in enumerate(self.problem.labels):
            row = ttk.Frame(self.ratio_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=18).grid(row=0, column=0, sticky="w")

            var = tk.DoubleVar(value=float(self.target_ratio[i].item() * 100.0))
            self.ratio_vars.append(var)

            sb = ttk.Spinbox(row, from_=0.0, to=100.0, increment=0.1, textvariable=var, width=8,
                             command=lambda idx=i: self._on_ratio_change(idx))
            sb.grid(row=0, column=1, sticky="w", padx=(0, 8))
            sb.bind("<Return>", lambda e, idx=i: self._on_ratio_change(idx))
            sb.bind("<FocusOut>", lambda e, idx=i: self._on_ratio_change(idx))
            self.ratio_spinboxes.append(sb)

        # Footer total readout
        self.total_label = ttk.Label(self.ratio_frame, text=self._ratio_total_text(), foreground="#555")
        self.total_label.pack(anchor="w", pady=(4, 6))

        # Stability slider
        current_stability = 100.0
        if hasattr(self, "stability_var"):
            try:
                current_stability = float(self.stability_var.get())
            except Exception:
                current_stability = 100.0

        stability_row = ttk.Frame(self.ratio_frame)
        stability_row.pack(fill="x", pady=(4, 0))
        ttk.Label(stability_row, text="Stability (Static targets for 100%):").grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.stability_var = tk.DoubleVar(value=current_stability)
        self.stability_scale = ttk.Scale(
            stability_row, from_=0, to=100, orient="horizontal",
            variable=self.stability_var, command=lambda _=None: self._on_stability_change()
        )
        self.stability_scale.grid(row=0, column=1, sticky="we")
        stability_row.grid_columnconfigure(1, weight=1)

        self.stability_label = ttk.Label(stability_row, text=f"{current_stability:.0f}%")
        self.stability_label.grid(row=0, column=2, sticky="w", padx=(8, 0))

        self._update_ratio_controls_state()

    def _ratio_total_text(self):
        vals = [v.get() for v in self.ratio_vars] if hasattr(self, "ratio_vars") else []
        return f"Sum: {sum(vals):.1f}% (locked to 100%)"

    def _on_ratio_change(self, idx):
        if self._updating_ratios or not self._stability_is_static():
            return
        try:
            v = max(0.0, min(100.0, float(self.ratio_vars[idx].get())))
        except Exception:
            v = 0.0
        self._updating_ratios = True
        try:
            vals = [float(var.get()) for var in self.ratio_vars]
            N = len(vals)
            vals[idx] = v
            remaining = 100.0 - v
            if N > 1:
                others_idx = [j for j in range(N) if j != idx]
                current_others = [max(0.0, vals[j]) for j in others_idx]
                total_others = sum(current_others)
                if total_others <= 1e-9:
                    even = remaining / (N - 1)
                    for j in others_idx:
                        vals[j] = max(0.0, even)
                else:
                    scale = remaining / total_others
                    for k, j in enumerate(others_idx):
                        vals[j] = max(0.0, current_others[k] * scale)

            for j, var in enumerate(self.ratio_vars):
                var.set(round(vals[j], 1))

            arr = np.array(vals, dtype=np.float64) / 100.0
            s = arr.sum()
            if s > 0:
                arr = arr / s
            self.target_ratio = torch.tensor(arr, dtype=torch.float64)

            self.total_label.config(text=self._ratio_total_text())
        finally:
            self._updating_ratios = False

    def toggle_run(self):
        self.running = not self.running
        self.run_btn.config(text="Stop" if self.running else "Start")

    def randomize_target(self):
        rnd = np.random.dirichlet(np.ones(self.problem.N), size=1)[0]
        self.target_ratio = torch.tensor(rnd, dtype=torch.float64)
        for ctrl in self.controllers.values():
            if hasattr(ctrl.inst, "reset_hidden"):
                ctrl.inst.reset_hidden()
        # Update spinboxes to match new random target
        self._updating_ratios = True
        try:
            for i in range(self.problem.N):
                self.ratio_vars[i].set(round(float(self.target_ratio[i].item() * 100.0), 1))
            self.total_label.config(text=self._ratio_total_text())
        finally:
            self._updating_ratios = False

    def optimize_mlp_once(self):
        if "MLP" not in self.controllers:
            return
        F_target = to_f64_scalar_tensor(self.total_flow_var.get())
        ref_speeds = self.problem.baseline_allocation(self.target_ratio, F_target)
        self.controllers["MLP"].train_once(self.target_ratio, F_target, self.flows_meas_filt,
                                           ref_speeds=ref_speeds, measured_flows=self.flows_meas_filt)

    def optimize_gru_once(self):
        if "GRU" not in self.controllers:
            return
        F_target = to_f64_scalar_tensor(self.total_flow_var.get())
        ref_speeds = self.problem.baseline_allocation(self.target_ratio, F_target)
        self.controllers["GRU"].train_once(self.target_ratio, F_target, self.flows_meas_filt,
                                           ref_speeds=ref_speeds, measured_flows=self.flows_meas_filt)

    def on_driver_change(self, _evt=None):
        for ctrl in self.controllers.values():
            ctrl.sync_to(self.speeds_cmd, self.flows_meas_filt)

    # --- Rebuild dynamic tables for current problem ---

    def _clear_children(self, frame):
        for c in frame.winfo_children():
            c.destroy()

    def _build_applied_table(self):
        self._clear_children(self.applied_frame)
        hdr = ttk.Frame(self.applied_frame)
        hdr.pack(fill="x")
        for j, h in enumerate([self.entity_title, "Speed (%)", f"{self.output_name} ({self.output_unit})"]):
            ttk.Label(hdr, text=h, width=16).grid(row=0, column=j, sticky="w")

        self.rows_applied = []
        for i, name in enumerate(self.problem.labels):
            r = ttk.Frame(self.applied_frame)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=name, width=16).grid(row=0, column=0, sticky="w")
            spd = ttk.Label(r, text="--", width=16, font=self.fixed, anchor="e"); spd.grid(row=0, column=1, sticky="e")
            flw = ttk.Label(r, text="--", width=16, font=self.fixed, anchor="e"); flw.grid(row=0, column=2, sticky="e")
            self.rows_applied.append((spd, flw))

    def _build_share_bars(self):
        self._clear_children(self.bars_frame)
        self.flow_bars = []
        self.flow_bar_vals = []
        for i, name in enumerate(self.problem.labels):
            row = ttk.Frame(self.bars_frame)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=name, width=12).pack(side="left")
            bar = ttk.Progressbar(row, orient="horizontal", length=280, mode="determinate", maximum=100)
            bar.pack(side="left", padx=8)
            val = ttk.Label(row, text="0.0%")
            val.pack(side="left")
            self.flow_bars.append(bar)
            self.flow_bar_vals.append(val)

    def _build_cmp_table(self):
        self._clear_children(self.cmpf)
        top = ttk.Frame(self.cmpf); top.pack(fill="x")

        # Fixed widths for alignment
        MAT_W, TAR_W, ERR_W = 12, 10, 12
        col = 0
        ttk.Label(top, text=self.entity_title, width=MAT_W, font=self.fixed, anchor="w").grid(row=0, column=col, sticky="w"); col += 1
        ttk.Label(top, text="Target (%)", width=TAR_W, font=self.fixed, anchor="e").grid(row=0, column=col, sticky="e"); col += 1

        self.err_header_labels = {}
        self.err_cols = {}
        for name in self.controller_names:
            text = f"{name}"
            lbl = ttk.Label(top, text=text, width=ERR_W, font=self.fixed, anchor="e")
            lbl.grid(row=0, column=col, sticky="e")
            self.err_header_labels[text] = lbl
            self.err_cols[name] = col
            col += 1

        self.rows_cmp = []
        for i, name in enumerate(self.problem.labels):
            r = ttk.Frame(self.cmpf)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=name, width=MAT_W, font=self.fixed, anchor="w").grid(row=0, column=0, sticky="w")
            t_lbl = ttk.Label(r, text="--", width=TAR_W, font=self.fixed, anchor="e"); t_lbl.grid(row=0, column=1, sticky="e")

            err_labels = []
            for mname in self.controller_names:
                col_idx = self.err_cols[mname]
                el = ttk.Label(r, text="--", width=ERR_W, font=self.fixed, anchor="e")
                el.grid(row=0, column=col_idx, sticky="e")
                err_labels.append(el)

            self.rows_cmp.append((t_lbl, err_labels))

    # --- UI helper: write target to spinboxes and keep sum == 100.0% ---
    def _set_spinboxes_from_target_exact_sum(self, target_ratio):
        if not hasattr(self, "ratio_vars") or not self.ratio_vars:
            return
        N = self.problem.N
        vals = [round(float(target_ratio[i].item() * 100.0), 1) for i in range(N)]
        total_others = round(sum(vals[:-1]), 1)
        last_needed = round(100.0 - total_others, 1)
        vals[-1] = last_needed

        def adjust(deficit):
            step = 0.1 if deficit > 0 else -0.1
            remaining = round(abs(deficit), 1)
            idx = 0
            while remaining > 1e-9 and idx < N - 1:
                can = (100.0 - vals[idx]) if step > 0 else vals[idx]
                can_steps = int(round(can / 0.1))
                take_steps = min(int(round(remaining / 0.1)), can_steps)
                if take_steps > 0:
                    delta = step * take_steps
                    vals[idx] = round(vals[idx] + delta, 1)
                    remaining = round(remaining - abs(delta), 1)
                idx += 1

        if vals[-1] < 0.0:
            adjust(-vals[-1])
            vals[-1] = 0.0
        elif vals[-1] > 100.0:
            adjust(100.0 - vals[-1])
            vals[-1] = 100.0

        self._updating_ratios = True
        try:
            for i, var in enumerate(self.ratio_vars):
                var.set(vals[i])
            if hasattr(self, "total_label"):
                self.total_label.config(text=self._ratio_total_text())
        finally:
            self._updating_ratios = False

    # --- Target jitter driven by the Stability slider ---
    def _maybe_jitter_target_from_slider(self):
        s = float(self.stability_var.get()) / 100.0
        if s >= 0.999:
            return
        strength = 1.0 - s
        mix = strength ** 2

        N = self.problem.N
        r = self.target_ratio.detach().clone()
        u = torch.tensor(np.random.dirichlet(np.ones(N)), dtype=torch.float64)
        new_r = (1.0 - mix) * r + mix * u
        new_r = torch.clamp(new_r, min=1e-9)
        new_r = new_r / torch.sum(new_r)

        self.target_ratio = new_r
        self._set_spinboxes_from_target_exact_sum(new_r)

    def _on_stability_change(self):
        val = float(self.stability_var.get())
        self.stability_label.config(text=f"{val:.0f}%")
        self._update_ratio_controls_state()

    def on_problem_change(self, _evt=None):
        # swap problem & reset state; rebuild controllers to match new N
        name = self.problem_var.get()
        self.problem = AVAILABLE_PROBLEMS[name](Ts)

        # Update per-problem metadata
        self.output_name = getattr(self.problem, "output_name", "Flow")
        self.output_unit = getattr(self.problem, "output_unit", "L/min")
        self.entity_title = getattr(self.problem, "entity_title", "Material")

        self.target_ratio = self.problem.default_target_ratio.clone()
        self.F_total_target = to_f64_scalar_tensor(self.problem.default_total_flow)
        self.total_label_text.set(f"Target Total {self.output_name} ({self.output_unit}):")
        self.total_flow_var.set(float(self.F_total_target.item()))

        # Plant state for new N
        self.speeds_cmd = torch.ones(self.problem.N, dtype=torch.float64) * 25.0
        self.flows_meas_filt = torch.zeros(self.problem.N, dtype=torch.float64)

        # Rebuild controllers for the new problem (fixes the 5->4 size mismatch)
        current_driver = self.driver_var.get()
        self._build_controllers_for_current_problem()
        # If previous driver missing (unlikely), pick default
        if current_driver not in self.controllers:
            current_driver = self._default_driver_name()
        self.driver_var.set(current_driver)

        # Update labels for what's actually loaded
        if hasattr(self, "loaded_label"):
            self.loaded_label.config(text=f"Loaded: {', '.join(self.controller_names)}")

        # Rebuild dynamic views for new N and labels/units
        self._build_ratio_controls()
        self._build_applied_table()
        self._build_share_bars()
        self._build_cmp_table()
        self._sync_left_column_width_with_summary()

        # Reset chart + counters
        self._clear_chart()
        self.step = 0
        self.step_label.config(text="Step: 0")
        self.summary_tail.config(
            text=f"Applied total {self.output_name.lower()} ~ -- {self.output_unit} (target --)   | Step 0"
        )

        # Sync controllers to fresh plant state
        for ctrl in self.controllers.values():
            ctrl.sync_to(self.speeds_cmd, self.flows_meas_filt)

    def _clear_chart(self):
        self.steps_hist.clear()
        for name in self.mae_hist:
            self.mae_hist[name].clear()
            if name in self.lines:
                self.lines[name].set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    # --- Control + Plant Loop ---
    def loop(self):
        self.step += 1
        self.step_label.config(text=f"Step: {self.step}")

        # Apply stability-driven target jitter *before* computing the baseline
        self._maybe_jitter_target_from_slider()

        F_target = to_f64_scalar_tensor(self.total_flow_var.get())
        speeds_direct = self.problem.baseline_allocation(self.target_ratio, F_target)

        # train/update learnable controllers (off-policy)
        for name in self.controller_names:
            if name in ("MLP", "GRU", "Transformer", "HybridMPC", "SafeRL", "PINN", "AdaptiveHier", "PID+Residual"):
                try:
                    self.controllers[name].train_once(self.target_ratio, F_target, self.flows_meas_filt,
                                                      ref_speeds=speeds_direct, measured_flows=self.flows_meas_filt)
                except Exception:
                    pass

        # get suggestions (sanitize to avoid NaNs) and FORCE correct shape
        suggested = {}
        for name in self.controller_names:
            try:
                s = self.controllers[name].suggest(self.flows_meas_filt, self.target_ratio, F_target, speeds_direct)
                s = s.detach().clone()
            except Exception:
                s = speeds_direct.clone()
            s = sanitize(s, speeds_direct)                    # replace non-finite with baseline
            s = ensure_shape(s, speeds_direct)                # enforce correct N (guards custom controllers)
            suggested[name] = s

        # apply selected driver
        drv = self.driver_var.get()
        if self.running:
            speeds_des = suggested.get(drv, speeds_direct)
            # Ensure shape again for safety
            speeds_des = ensure_shape(speeds_des, speeds_direct)
            self.speeds_cmd = slew_limit(self.speeds_cmd, speeds_des, self.problem.slew_rate,
                                         self.problem.speed_min, self.problem.speed_max)

        # plant sim step -> filtered measured outputs
        self.flows_meas_filt = self.problem.step(self.speeds_cmd)

        # comps for table (based on nominal model to compare suggestions fairly)
        comps = {name: self.problem.comp_from_speeds(spd) for name, spd in suggested.items()}
        self.render(F_target, comps)

        # schedule next tick
        self.root.after(int(self.problem.Ts * 1000), self.loop)

    def render(self, F_target, comps_by_name):
        flows_meas = self.flows_meas_filt.detach()
        speeds_applied = self.speeds_cmd.detach()

        # applied table
        for i in range(self.problem.N):
            spd_lbl, flw_lbl = self.rows_applied[i]
            spd_lbl.config(text=f"{float(speeds_applied[i].item()):5.1f}")
            flw_lbl.config(text=f"{float(flows_meas[i].item()):7.2f}")

        # share bars
        total_meas = float(torch.sum(flows_meas).item() + 1e-12)
        comp_applied = (flows_meas / total_meas).numpy() if total_meas > 0 else np.zeros(self.problem.N)
        shares_pct = (comp_applied * 100.0).tolist()
        for i in range(self.problem.N):
            self.flow_bars[i]["value"] = shares_pct[i]
            self.flow_bar_vals[i].config(text=f"{shares_pct[i]:.1f}%")

        # errors for table and chart (NaN-safe)
        targ_pct = (self.target_ratio.numpy() * 100.0)
        err_arrays = {}
        for name in self.controller_names:
            arr = (comps_by_name[name].detach().cpu().numpy() * 100.0)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            err_arrays[name] = arr - targ_pct

        # per-row: show errors & highlight smallest |error|
        for i in range(self.problem.N):
            t_lbl, err_labels = self.rows_cmp[i]
            t_lbl.config(text=f"{targ_pct[i]:6.2f}")

            best_abs, best_idx = None, None
            for j, name in enumerate(self.controller_names):
                val = err_arrays[name][i]
                txt = "--" if not np.isfinite(val) else f"{val:+6.2f}"
                err_labels[j].config(text=txt, foreground="black")
                if np.isfinite(val):
                    av = abs(val)
                    if best_abs is None or av < best_abs:
                        best_abs, best_idx = av, j
            if best_idx is not None:
                err_labels[best_idx].config(foreground="red")

        def mae(arr):
            arr = np.asarray(arr, dtype=float)
            return float(np.nanmean(np.abs(arr)))

        # Compute MAE and max|error| per controller
        maes = {name: mae(err_arrays[name]) for name in self.controller_names}
        maxes = {name: float(np.nanmax(np.abs(err_arrays[name]))) for name in self.controller_names}

        # Color header for best MAE
        for lbl in self.err_header_labels.values():
            lbl.config(foreground="black")
        best_name = min(maes.keys(), key=lambda n: maes[n] if np.isfinite(maes[n]) else np.inf)
        best_header = f"{best_name}"
        if best_header in self.err_header_labels:
            self.err_header_labels[best_header].config(foreground="red")

        # ── Update two-column MAE table (best row in red)
        for name, (_name_lbl, val_lbl) in self.summary_rows.items():
            val_lbl.config(
                text=f"{maes[name]:.2f} / {maxes[name]:.2f}",
                foreground=("red" if name == best_name else "black")
            )

        # Update trailing tail (total + step)
        self.summary_tail.config(
            text=f"Applied total {self.output_name.lower()} ~ {float(torch.sum(flows_meas).item()):.2f} {self.output_unit} "
                 f"(target {float(self.F_total_target.item()):.2f})   | Step {self.step}"
        )

        # ------------------ Update MAE chart ------------------
        self.steps_hist.append(self.step)
        if len(self.steps_hist) > MAX_POINTS:
            self.steps_hist = self.steps_hist[-MAX_POINTS:]

        for name in self.controller_names:
            self.mae_hist[name].append(maes[name])
            if len(self.mae_hist[name]) > MAX_POINTS:
                self.mae_hist[name] = self.mae_hist[name][-MAX_POINTS:]
            if len(self.mae_hist[name]) < len(self.steps_hist):
                pad = [np.nan] * (len(self.steps_hist) - len(self.mae_hist[name]))
                self.mae_hist[name] = pad + self.mae_hist[name]
            self.lines[name].set_data(self.steps_hist, self.mae_hist[name])

        if self.step % 2 == 0:
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()


# -------------------------- Main --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    App(root)
    root.mainloop()

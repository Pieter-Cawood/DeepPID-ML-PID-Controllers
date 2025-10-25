# ─────────────────────────────────────────────────────────────────────────────
# File: viz.py
# Process visualization panel for the app:
#   (1) Source input speeds (%)
#   (2) Target vs Actual composition per source (%)
#   (3) Tank total flow (target vs actual, L/min)
# ─────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk

class ProcessViz:
    def __init__(self, parent, problem):
        self.parent = parent
        self.problem = problem
        self._build_widgets()

    def _build_widgets(self):
        # Frame
        self.viz_frame = ttk.LabelFrame(self.parent, text="Process Visualization — Inputs, Targets, Actuals", padding=6)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Figure & axes
        self.fig_proc = Figure(figsize=(7.8, 6.0), dpi=100, constrained_layout=True)
        gs = self.fig_proc.add_gridspec(3, 1)
        self.ax_speed = self.fig_proc.add_subplot(gs[0, 0])
        self.ax_comp  = self.fig_proc.add_subplot(gs[1, 0])
        self.ax_tank  = self.fig_proc.add_subplot(gs[2, 0])

        # Axes setup
        x = np.arange(self.problem.N)
        self._x_source = x

        # --- (1) Speed axis
        self.ax_speed.set_title("Source Inputs (Speed %)", fontsize=10, pad=6)
        vmax = self._max_of(self.problem.speed_max, default=100.0)
        # give a little headroom and avoid 0-range
        vmax = max(1.0, vmax)
        vmax = math.ceil(vmax * 1.05)
        self.ax_speed.set_ylim(0, vmax)
        self.ax_speed.grid(True, alpha=0.25)
        self.ax_speed.set_xticks(x)
        self.ax_speed.set_xticklabels(self.problem.labels, fontsize=9, rotation=0)
        self.bars_speed = self.ax_speed.bar(x, np.zeros_like(x, dtype=float))

        # --- (2) Composition axis
        self.ax_comp.set_title("Composition per Source (% of total)", fontsize=10, pad=6)
        self.ax_comp.set_ylim(0, 100)
        self.ax_comp.grid(True, alpha=0.25)
        width = 0.38
        self._bar_width = width
        self.ax_comp.set_xticks(x)
        self.ax_comp.set_xticklabels(self.problem.labels, fontsize=9)

        self.bars_comp_target = self.ax_comp.bar(x - width/2, np.zeros(self.problem.N), width, label="Target %")
        self.bars_comp_actual = self.ax_comp.bar(x + width/2, np.zeros(self.problem.N), width, label="Actual %")
        self.ax_comp.legend(loc="upper right", fontsize=9)

        # --- (3) Tank axis
        self.ax_tank.set_title("Tank Total Flow (L/min): Target vs Actual", fontsize=10, pad=6)
        self.ax_tank.grid(True, alpha=0.25)
        self.ax_tank.set_xticks([0, 1])
        self.ax_tank.set_xticklabels(["Target", "Actual"], fontsize=9)

        self.bar_tank_target = self.ax_tank.bar([0], [0.0])
        self.bar_tank_actual = self.ax_tank.bar([1], [0.0])

        self.ax_tank.margins(y=0.15) 

        # Canvas + toolbar
        self.canvas_proc = FigureCanvasTkAgg(self.fig_proc, master=self.viz_frame)
        self.canvas_proc.draw()
        self.canvas_proc.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.toolbar_proc = NavigationToolbar2Tk(self.canvas_proc, self.viz_frame, pack_toolbar=False)
        self.toolbar_proc.update()
        self.toolbar_proc.pack(side="bottom", fill="x")

    def _max_of(self, v, default=0.0):
        """
        Return a float max from a python number, numpy array, or torch tensor.
        If tensor/array has multiple elements, take the max. Falls back to default.
        """
        try:
            if torch.is_tensor(v):
                if v.numel() == 0:
                    return float(default)
                return float(torch.max(v).item())
            else:
                # numpy arrays or python scalars
                try:
                    return float(np.max(v))
                except Exception:
                    return float(v)
        except Exception:
            return float(default)

    def update(self, speeds_applied, flows_meas, target_ratio, F_total_target):
        """Update bars every tick."""
        # Safety to avoid over-annotating the tank axis
        for artist in list(self.ax_tank.texts):
            artist.remove()

        # --- (1) Speed bars
        speeds_np = speeds_applied.detach().cpu().numpy().astype(float)
        for rect, val in zip(self.bars_speed, speeds_np):
            rect.set_height(val)

        # --- (2) Composition bars (%)
        total_meas = float(torch.sum(flows_meas).item() + 1e-12)
        actual_pct = (flows_meas.detach().cpu().numpy().astype(float) / total_meas) * 100.0 if total_meas > 0 else np.zeros(self.problem.N)
        target_pct = (target_ratio.detach().cpu().numpy().astype(float) * 100.0)

        for rect, val in zip(self.bars_comp_target, target_pct):
            rect.set_height(val)
        for rect, val in zip(self.bars_comp_actual, actual_pct):
            rect.set_height(val)

        # --- (3) Tank total flow (L/min)
        actual_total = float(torch.sum(flows_meas).item())
        # target can be tensor or scalar; handle both
        target_total = float(F_total_target.detach().cpu().item() if torch.is_tensor(F_total_target) else float(F_total_target))
        self.bar_tank_target[0].set_height(target_total)
        self.bar_tank_actual[0].set_height(actual_total)

        # annotate numbers on tank bars
        for r in (self.bar_tank_target[0], self.bar_tank_actual[0]):
            h = r.get_height()
            self.ax_tank.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2.0, h),
                                  xytext=(0, 3), textcoords="offset points",
                                  ha="center", va="bottom", fontsize=8)

        # autoscale tank axis
        max_h = max(self.bar_tank_target[0].get_height(), self.bar_tank_actual[0].get_height(), 1.0)
        self.ax_tank.set_ylim(0, max_h * 1.15)

        # redraw
        self.canvas_proc.draw_idle()

    def rebuild(self, problem):
        """Rebuild the entire panel if N/limits changed (e.g., on problem change)."""
        # remove old frame entirely and rebuild fresh
        try:
            self.viz_frame.destroy()
        except Exception:
            pass
        self.problem = problem
        self._build_widgets()

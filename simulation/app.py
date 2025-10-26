# File: app.py
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import numpy as np
import torch

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from deeppid import AVAILABLE, CtrlAdapter, TensorUtils
from deeppid.envs.problems import AVAILABLE_PROBLEMS

from visual_window import VisualWindow
from ui_parts import build_summary_table, build_cmp_table

# -------------------------- Configuration --------------------------
CONTROL_PERIOD_MS = 250
Ts = CONTROL_PERIOD_MS / 1000.0

W_COMP, W_TOTAL, W_SMOOTH, W_BOUND, W_REF, W_MEAS = 3.0, 6.0, 0.10, 0.08, 0.03, 0.25
SEQ_LEN, GRU_HID, GRU_TRAIN_STEPS, MLP_TRAIN_STEPS = 20, 128, 6, 12
MAX_POINTS = 1200


class App:
    def __init__(self, root):
        self.root = root
        root.title("DeepPID — Controllers vs Interchangeable Plant Problems")

        # Problem setup
        self.problem_names = list(AVAILABLE_PROBLEMS.keys())
        if not self.problem_names:
            raise RuntimeError("No problems available. Ensure problems.py defines AVAILABLE_PROBLEMS.")
        self.problem_var = tk.StringVar(value=self.problem_names[0])
        self.problem = AVAILABLE_PROBLEMS[self.problem_var.get()](Ts)

        # Metadata
        self.output_name = getattr(self.problem, "output_name", "Flow")
        self.output_unit = getattr(self.problem, "output_unit", "L/min")
        self.entity_title = getattr(self.problem, "entity_title", "Material")

        # Targets
        self.target_ratio = self.problem.default_target_ratio.clone()
        self.F_total_target = TensorUtils.to_f64_scalar_tensor(self.problem.default_total_flow)

        # Controllers
        self.controllers = {}
        self.controller_names = []
        self._build_controllers_for_current_problem()

        # Plant state
        self.speeds_cmd = torch.ones(self.problem.N, dtype=torch.float64) * 25.0
        self.flows_meas_filt = torch.zeros(self.problem.N, dtype=torch.float64)

        # UI state
        self.step = 0
        self.running = True
        self._updating_ratios = False
        self.fixed = tkfont.nametofont("TkFixedFont")

        # ─────────────────── Top Controls ───────────────────
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Problem:").grid(row=0, column=0, sticky="w")
        problem_cb = ttk.Combobox(
            ctrl, textvariable=self.problem_var, values=self.problem_names,
            width=28, state="readonly"
        )
        problem_cb.grid(row=0, column=1, sticky="w", padx=(6, 12))
        problem_cb.bind("<<ComboboxSelected>>", self.on_problem_change)

        ttk.Label(ctrl, text="Driver:").grid(row=0, column=2, sticky="w")
        self.driver_var = tk.StringVar(value=self._default_driver_name())
        driver = ttk.Combobox(
            ctrl, textvariable=self.driver_var, values=self.controller_names,
            width=14, state="readonly"
        )
        driver.grid(row=0, column=3, sticky="w", padx=(6, 12))
        driver.bind("<<ComboboxSelected>>", self.on_driver_change)

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
        ttk.Button(ctrl, text="Open Visual Window", command=self._show_visual_popup).grid(row=0, column=10, padx=6)

        self.loaded_label = ttk.Label(root, text=f"Loaded: {', '.join(self.controller_names)}")
        self.loaded_label.pack(fill="x", padx=10)
        self.step_label = ttk.Label(root, text="Step: 0")
        self.step_label.pack(fill="x", padx=10, pady=(0, 4))

        # ─────────────────── Main Grid (no embedded visual anymore) ───────────────────
        main = ttk.Frame(root, padding=(10, 0))
        self.main = main
        main.pack(fill="both", expand=True)

        # Left column holds Ratio + Controllers Error; Right column holds MAE Summary
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0)
        main.grid_rowconfigure(0, weight=1)  # Ratio
        main.grid_rowconfigure(1, weight=1)  # Controllers Error

        # Left, row 0: Ratio Editor
        self.ratio_frame = ttk.LabelFrame(
            main, text="Target Ratio (%) — type/edit; others adjust to keep 100%", padding=10
        )
        self.ratio_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self._build_ratio_controls()

        # Left, row 1: Controllers table
        self.cmpf, self.err_header_labels, self.err_cols, self.rows_cmp = build_cmp_table(
            parent=main, entity_title=self.entity_title,
            controller_names=self.controller_names, labels=self.problem.labels, fixed_font=self.fixed
        )
        self.cmpf.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        # RIGHT SIDE: MAE Summary (spans the height of Ratio + Controllers Error)
        self.summary_table_frame, self.summary_rows, self.summary_tail = build_summary_table(
            parent=main, controller_names=self.controller_names
        )
        self.summary_table_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 10), pady=(0, 8))

        # Bottom row: Chart only (full width)
        bottom = ttk.Frame(main, padding=(0, 0))
        self.bottom = bottom
        bottom.grid(row=2, column=0, columnspan=2, sticky="nsew")
        bottom.grid_columnconfigure(0, weight=0)
        bottom.grid_columnconfigure(1, weight=1)

        chart_frame = ttk.LabelFrame(bottom, text="MAE History (pp) vs Step", padding=6)
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=(0, 10))
        bottom.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(7.0, 3.6), dpi=100)
        self.fig.patch.set_facecolor("none")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("none")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("MAE (pp)")
        self.ax.grid(True, alpha=0.3)
        self.lines = {}
        for name in self.controller_names:
            line, = self.ax.plot([], [], label=name)
            self.lines[name] = line
        if self.controller_names:
            self.ax.legend(loc="upper right", framealpha=0.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        w = self.canvas.get_tk_widget()
        style = ttk.Style(chart_frame)
        bg = (
            style.lookup("TLabelframe", "background")
            or style.lookup("TFrame", "background")
            or chart_frame.master.cget("bg")
            or "#FFFFFF"
        )
        w.configure(bg=bg, highlightthickness=0)
        w.pack(side="top", fill="both", expand=True)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, chart_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        # Sync controllers
        for ctrl in self.controllers.values():
            ctrl.sync_to(self.speeds_cmd, self.flows_meas_filt)

        # Create the modeless visual popup
        self.visual = VisualWindow(self.root)
        self._ensure_visual_widget(recreate=True)
        self.visual.show()  # auto-show on startup

        # Start loop — bullet-proof against exceptions
        self.root.after(CONTROL_PERIOD_MS, self.loop)

    # --- Visual selection ---
    def _current_visual_kind(self) -> str:
        ent = str(getattr(self, "entity_title", "")).lower()
        out = str(getattr(self, "output_name", "")).lower()
        if ("rotor" in ent) or ("thrust" in out) or ("quad" in ent) or ("quadcopter" in ent):
            return "quad"
        return "tanks"

    def _ensure_visual_widget(self, recreate: bool = False):
        want = self._current_visual_kind()
        if recreate or getattr(self.visual, "kind", None) != want:
            n = self.problem.N
            title = "Quadcopter" if want == "quad" else "5-Tank Water System"
            self.visual.set_kind(want, n, title)

    def _show_visual_popup(self):
        self._ensure_visual_widget(recreate=False)
        self.visual.show()

    # --- Controller helpers ---
    def _default_driver_name(self):
        names = list(self.controllers.keys())
        if not names:
            return ""
        return "GRU" if "GRU" in names else names[0]

    def _build_controllers_for_current_problem(self):
        self.controllers.clear()
        dd = dict(
            N=self.problem.N, k=self.problem.k_coeff,
            speed_min=self.problem.speed_min, speed_max=self.problem.speed_max,
            Ts=Ts, plant_alpha=getattr(self.problem, "alpha", 0.5),
            slew_rate=self.problem.slew_rate
        )

        def try_add(name, cls):
            if cls is None:
                return
            try:
                try:
                    inst = cls(**dd)
                except TypeError:
                    if name == "MLP":
                        inst = cls(
                            self.problem.N, self.problem.k_coeff, self.problem.speed_min, self.problem.speed_max,
                            Ts, self.problem.slew_rate, hidden=128, train_steps=MLP_TRAIN_STEPS,
                            W_COMP=W_COMP, W_TOTAL=W_TOTAL, W_SMOOTH=W_SMOOTH,
                            W_BOUND=W_BOUND, W_REF=W_REF, W_MEAS=W_MEAS
                        )
                    elif name == "GRU":
                        inst = cls(
                            self.problem.N, self.problem.k_coeff, self.problem.speed_min, self.problem.speed_max,
                            Ts, self.problem.slew_rate, seq_len=SEQ_LEN, hidden=GRU_HID, train_steps=GRU_TRAIN_STEPS,
                            W_COMP=W_COMP, W_TOTAL=W_TOTAL, W_SMOOTH=W_SMOOTH,
                            W_BOUND=W_BOUND, W_REF=W_REF, W_MEAS=W_MEAS
                        )
                    else:
                        inst = cls(
                            self.problem.N, self.problem.k_coeff, self.problem.speed_min, self.problem.speed_max,
                            Ts, getattr(self.problem, "alpha", 0.5), self.problem.slew_rate
                        )
                self.controllers[name] = CtrlAdapter(name, inst)
            except Exception:
                pass

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

    # --- Ratio controls ---
    def _stability_is_static(self) -> bool:
        try:
            return float(self.stability_var.get()) >= 99.9
        except Exception:
            return True

    def _update_ratio_controls_state(self):
        if not hasattr(self, "ratio_spinboxes"):
            return
        state = "normal" if self._stability_is_static() else "disabled"
        for sb in self.ratio_spinboxes:
            try:
                sb.config(state=state)
            except Exception:
                pass

    def _build_ratio_controls(self):
        for child in self.ratio_frame.winfo_children():
            child.destroy()
        self.ratio_vars = []
        self.ratio_spinboxes = []
        header = ttk.Frame(self.ratio_frame); header.pack(fill="x")
        ttk.Label(header, text=self.entity_title, width=18).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Target (%)", width=12).grid(row=0, column=1, sticky="w")
        for i, name in enumerate(self.problem.labels):
            row = ttk.Frame(self.ratio_frame); row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=18).grid(row=0, column=0, sticky="w")
            var = tk.DoubleVar(value=float(self.target_ratio[i].item() * 100.0))
            self.ratio_vars.append(var)
            sb = ttk.Spinbox(
                row, from_=0.0, to=100.0, increment=0.1, textvariable=var, width=8,
                command=lambda idx=i: self._on_ratio_change(idx)
            )
            sb.grid(row=0, column=1, sticky="w", padx=(0, 8))
            sb.bind("<Return>", lambda e, idx=i: self._on_ratio_change(idx))
            sb.bind("<FocusOut>", lambda e, idx=i: self._on_ratio_change(idx))
            self.ratio_spinboxes.append(sb)
        self.total_label = ttk.Label(self.ratio_frame, text=self._ratio_total_text(), foreground="#555")
        self.total_label.pack(anchor="w", pady=(4, 6))
        current_stability = 100.0
        if hasattr(self, "stability_var"):
            try:
                current_stability = float(self.stability_var.get())
            except Exception:
                current_stability = 100.0
        stability_row = ttk.Frame(self.ratio_frame); stability_row.pack(fill="x", pady=(4, 0))
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
            N = len(vals); vals[idx] = v; remaining = 100.0 - v
            if N > 1:
                others_idx = [j for j in range(N) if j != idx]
                current_others = [max(0.0, vals[j]) for j in others_idx]
                total_others = sum(current_others)
                if total_others <= 1e-9:
                    even = remaining / (N - 1)
                    for j in others_idx: vals[j] = max(0.0, even)
                else:
                    scale = remaining / total_others
                    for k, j in enumerate(others_idx): vals[j] = max(0.0, current_others[k] * scale)
            for j, var in enumerate(self.ratio_vars): var.set(round(vals[j], 1))
            arr = np.array(vals, dtype=np.float64) / 100.0
            s = arr.sum();  arr = arr / s if s > 0 else arr
            self.target_ratio = torch.tensor(arr, dtype=torch.float64)
            self.total_label.config(text=self._ratio_total_text())
        finally:
            self._updating_ratios = False

    def _on_stability_change(self):
        val = float(self.stability_var.get())
        self.stability_label.config(text=f"{val:.0f}%")
        self._update_ratio_controls_state()

    def _set_spinboxes_from_target_exact_sum(self, target_ratio):
        if not hasattr(self, "ratio_vars") or not self.ratio_vars:
            return
        N = self.problem.N
        vals = [round(float(target_ratio[i].item() * 100.0), 1) for i in range(N)]
        total_others = round(sum(vals[:-1]), 1); last_needed = round(100.0 - total_others, 1); vals[-1] = last_needed
        def adjust(deficit):
            step = 0.1 if deficit > 0 else -0.1
            remaining = round(abs(deficit), 1); idx = 0
            while remaining > 1e-9 and idx < N - 1:
                can = (100.0 - vals[idx]) if step > 0 else vals[idx]
                can_steps = int(round(can / 0.1)); take_steps = min(int(round(remaining / 0.1)), can_steps)
                if take_steps > 0:
                    delta = step * take_steps; vals[idx] = round(vals[idx] + delta, 1); remaining = round(remaining - abs(delta), 1)
                idx += 1
        if vals[-1] < 0.0: adjust(-vals[-1]); vals[-1] = 0.0
        elif vals[-1] > 100.0: adjust(100.0 - vals[-1]); vals[-1] = 100.0
        self._updating_ratios = True
        try:
            for i, var in enumerate(self.ratio_vars): var.set(vals[i])
            if hasattr(self, "total_label"): self.total_label.config(text=self._ratio_total_text())
        finally:
            self._updating_ratios = False

    # --- Control + Plant Loop (bullet-proof) ---
    def toggle_run(self):
        self.running = not self.running
        self.run_btn.config(text="Stop" if self.running else "Start")

    def randomize_target(self):
        rnd = np.random.dirichlet(np.ones(self.problem.N), size=1)[0]
        self.target_ratio = torch.tensor(rnd, dtype=torch.float64)
        for ctrl in self.controllers.values():
            if hasattr(ctrl.inst, "reset_hidden"): ctrl.inst.reset_hidden()
        self._updating_ratios = True
        try:
            for i in range(self.problem.N): self.ratio_vars[i].set(round(float(self.target_ratio[i].item() * 100.0), 1))
            self.total_label.config(text=self._ratio_total_text())
        finally:
            self._updating_ratios = False

    def optimize_mlp_once(self):
        if "MLP" not in self.controllers: return
        F_target = TensorUtils.to_f64_scalar_tensor(self.total_flow_var.get())
        ref_speeds = self.problem.baseline_allocation(self.target_ratio, F_target)
        self.controllers["MLP"].train_once(self.target_ratio, F_target, self.flows_meas_filt,
                                           ref_speeds=ref_speeds, measured_flows=self.flows_meas_filt)

    def optimize_gru_once(self):
        if "GRU" not in self.controllers: return
        F_target = TensorUtils.to_f64_scalar_tensor(self.total_flow_var.get())
        ref_speeds = self.problem.baseline_allocation(self.target_ratio, F_target)
        self.controllers["GRU"].train_once(self.target_ratio, F_target, self.flows_meas_filt,
                                           ref_speeds=ref_speeds, measured_flows=self.flows_meas_filt)

    def on_driver_change(self, _evt=None):
        for ctrl in self.controllers.values():
            ctrl.sync_to(self.speeds_cmd, self.flows_meas_filt)

    def on_problem_change(self, _evt=None):
        name = self.problem_var.get()
        self.problem = AVAILABLE_PROBLEMS[name](Ts)

        # Update per-problem metadata
        self.output_name = getattr(self.problem, "output_name", "Flow")
        self.output_unit = getattr(self.problem, "output_unit", "L/min")
        self.entity_title = getattr(self.problem, "entity_title", "Material")

        self.target_ratio = self.problem.default_target_ratio.clone()
        self.F_total_target = TensorUtils.to_f64_scalar_tensor(self.problem.default_total_flow)
        self.total_label_text.set(f"Target Total {self.output_name} ({self.output_unit}):")
        self.total_flow_var.set(float(self.F_total_target.item()))

        # Plant state for new N
        self.speeds_cmd = torch.ones(self.problem.N, dtype=torch.float64) * 25.0
        self.flows_meas_filt = torch.zeros(self.problem.N, dtype=torch.float64)

        # Rebuild controllers for the new problem
        current_driver = self.driver_var.get()
        self._build_controllers_for_current_problem()
        if current_driver not in self.controllers:
            current_driver = self._default_driver_name()
        self.driver_var.set(current_driver)

        # Update labels for what's loaded
        if hasattr(self, "loaded_label"):
            self.loaded_label.config(text=f"Loaded: {', '.join(self.controller_names)}")

        # Rebuild dynamic views for new N and labels/units
        self._build_ratio_controls()

        # Rebuild comparison table
        self.cmpf.destroy()
        self.cmpf, self.err_header_labels, self.err_cols, self.rows_cmp = build_cmp_table(
            parent=self.main, entity_title=self.entity_title,
            controller_names=self.controller_names, labels=self.problem.labels, fixed_font=self.fixed
        )
        self.cmpf.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        # Rebuild / relocate MAE Summary to right side
        try:
            self.summary_table_frame.destroy()
        except Exception:
            pass
        self.summary_table_frame, self.summary_rows, self.summary_tail = build_summary_table(
            parent=self.main, controller_names=self.controller_names
        )
        self.summary_table_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 10), pady=(0, 8))

        # Rebuild chart lines + legend
        self._rebuild_chart_lines()

        # Reset step/labels
        self.step = 0
        self.step_label.config(text="Step: 0")

        # Sync controllers to fresh plant state
        for ctrl in self.controllers.values():
            ctrl.sync_to(self.speeds_cmd, self.flows_meas_filt)

        # Switch the modeless visual
        self._ensure_visual_widget(recreate=True)
        self.visual.show()

    def _rebuild_chart_lines(self):
        # Clear histories
        self.steps_hist = []
        self.mae_hist = {name: [] for name in self.controller_names}
        # Delete all existing lines and rebuild
        for line in getattr(self, "lines", {}).values():
            try:
                line.remove()
            except Exception:
                pass
        self.lines = {}
        for name in self.controller_names:
            line, = self.ax.plot([], [], label=name)
            self.lines[name] = line
        self.ax.legend(loc="upper right", framealpha=0.0)
        self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw_idle()

    def _update_cmp_and_chart(self, F_target, comps_by_name):
        flows_meas = self.flows_meas_filt.detach()
        targ_pct = (self.target_ratio.numpy() * 100.0)
        err_arrays = {}
        for name in self.controller_names:
            arr = (comps_by_name[name].detach().cpu().numpy() * 100.0)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            err_arrays[name] = arr - targ_pct

        for i in range(len(getattr(self, 'rows_cmp', []))):
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
            arr = np.asarray(arr, dtype=float); return float(np.nanmean(np.abs(arr)))

        maes = {name: mae(err_arrays[name]) for name in self.controller_names}
        maxes = {name: float(np.nanmax(np.abs(err_arrays[name]))) for name in self.controller_names}

        for lbl in getattr(self, 'err_header_labels', {}).values():
            lbl.config(foreground="black")
        best_name = None
        if len(maes):
            best_name = min(maes.keys(), key=lambda n: maes[n] if np.isfinite(maes[n]) else np.inf)
            if best_name in getattr(self, 'err_header_labels', {}):
                self.err_header_labels[best_name].config(foreground="red")

        for name, (_name_lbl, val_lbl) in self.summary_rows.items():
            val_lbl.config(
                text=f"{maes.get(name, float('nan')):.2f} / {maxes.get(name, float('nan')):.2f}",
                foreground=("red" if name == best_name else "black")
            )

        self.summary_tail.config(
            text=f"Applied total {self.output_name.lower()} ~ {float(torch.sum(flows_meas).item()):.2f} {self.output_unit} "
                 f"(target {float(self.F_total_target.item()):.2f})   | Step {self.step}"
        )

        # Chart series
        if not hasattr(self, 'steps_hist'): self.steps_hist = []
        if not hasattr(self, 'mae_hist'): self.mae_hist = {name: [] for name in self.controller_names}
        self.steps_hist.append(self.step)
        if len(self.steps_hist) > MAX_POINTS: self.steps_hist = self.steps_hist[-MAX_POINTS:]
        for name in self.controller_names:
            self.mae_hist.setdefault(name, []).append(maes[name])
            if len(self.mae_hist[name]) > MAX_POINTS: self.mae_hist[name] = self.mae_hist[name][-MAX_POINTS:]
            if len(self.mae_hist[name]) < len(self.steps_hist):
                pad = [np.nan] * (len(self.steps_hist) - len(self.mae_hist[name]))
                self.mae_hist[name] = pad + self.mae_hist[name]
            self.lines[name].set_data(self.steps_hist, self.mae_hist[name])
        if self.step % 2 == 0:
            self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw_idle()

    # --- Target jitter ---
    def _maybe_jitter_target_from_slider(self):
        s = float(self.stability_var.get()) / 100.0
        if s >= 0.999: return
        strength = 1.0 - s; mix = strength ** 2
        N = self.problem.N
        r = self.target_ratio.detach().clone()
        u = torch.tensor(np.random.dirichlet(np.ones(N)), dtype=torch.float64)
        new_r = (1.0 - mix) * r + mix * u
        new_r = torch.clamp(new_r, min=1e-9)
        new_r = new_r / torch.sum(new_r)
        self.target_ratio = new_r
        self._set_spinboxes_from_target_exact_sum(new_r)

    # --- Main loop (never stalls now) ---
    def loop(self):
        try:
            self.step += 1
            self.step_label.config(text=f"Step: {self.step}")

            # Jitter targets if needed (driven by stability slider)
            self._maybe_jitter_target_from_slider()

            F_target = TensorUtils.to_f64_scalar_tensor(self.total_flow_var.get())
            speeds_direct = self.problem.baseline_allocation(self.target_ratio, F_target)

            # Train off-policy (best-effort)
            for name in self.controller_names:
                if name in ("MLP", "GRU", "Transformer", "HybridMPC", "SafeRL", "PINN", "AdaptiveHier", "PID+Residual"):
                    try:
                        self.controllers[name].train_once(
                            self.target_ratio, F_target, self.flows_meas_filt,
                            ref_speeds=speeds_direct, measured_flows=self.flows_meas_filt
                        )
                    except Exception:
                        pass

            # Suggestions
            suggested = {}
            for name in self.controller_names:
                try:
                    s = self.controllers[name].suggest(
                        self.flows_meas_filt, self.target_ratio,
                        F_total=F_target, speeds_direct=speeds_direct
                    ).detach().clone()
                except Exception:
                    s = speeds_direct.clone()
                s = TensorUtils.sanitize(s, speeds_direct)
                s = TensorUtils.ensure_shape(s, speeds_direct)
                suggested[name] = s

            # Apply selected driver
            drv = self.driver_var.get()
            if self.running:
                speeds_des = TensorUtils.ensure_shape(suggested.get(drv, speeds_direct), speeds_direct)
                self.speeds_cmd = TensorUtils.slew_limit(
                    self.speeds_cmd, speeds_des, self.problem.slew_rate,
                    self.problem.speed_min, self.problem.speed_max
                )

            # Plant step
            self.flows_meas_filt = self.problem.step(self.speeds_cmd)

            # Feed modeless visual
            try:
                F_tot = float(F_target.item())
                denom = max(1e-6, F_tot)

                # For tanks we still pass a compatible API, for quad levels are ignored
                levels = torch.clamp(self.flows_meas_filt / denom, 0.0, 1.0)
                inflow = torch.clamp(self.problem.comp_from_speeds(self.speeds_cmd) * F_tot, min=0.0)
                outflow = torch.clamp(self.flows_meas_filt, min=0.0)
                span = float(self.problem.speed_max - self.problem.speed_min) if hasattr(self.problem, "speed_max") else 1.0
                span = max(span, 1e-9)
                valves = torch.clamp((self.speeds_cmd - self.problem.speed_min) / span, 0.0, 1.0)

                self.visual.push_state(
                    levels=levels.detach().cpu().numpy(),
                    inflow=inflow.detach().cpu().numpy(),
                    outflow=outflow.detach().cpu().numpy(),
                    valves=valves.detach().cpu().numpy(),
                    units=self.output_unit,
                )
            except Exception:
                # Never let the visual update break the main loop
                pass

            # Update MAE comparisons + chart
            comps = {name: self.problem.comp_from_speeds(spd) for name, spd in suggested.items()}
            self._update_cmp_and_chart(F_target, comps)

        finally:
            # ALWAYS reschedule — protects against any exception (e.g., on quad switch)
            self.root.after(int(getattr(self.problem, 'Ts', Ts) * 1000), self.loop)

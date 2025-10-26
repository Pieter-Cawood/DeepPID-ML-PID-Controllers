# File: quadcopter_widget.py
# Quadcopter Canvas (Tkinter)
# ---------------------------
# Drop-in animated widget for quad/multicopter visualization.
#
# Public API (kept for compatibility with TanksWidget):
#     push_state(levels, inflow, outflow, valves, units)
#
# Notes for quads:
#   • `levels` are IGNORED (only kept to match the tanks widget signature).
#   • `inflow`  -> nominal thrust reference per rotor
#   • `outflow` -> measured thrust per rotor
#   • `valves`  -> 0..1 motor command, shown as "Throttle"
#   • `units`   -> thrust units string, e.g., "N"
#
# The widget animates at ~60 FPS, falls back to a smooth demo if no external
# push_state() is received for >1s, and never raises errors into the app.
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import tkinter as tk

# ----------------------------- Visual constants -----------------------------
CANVAS_W, CANVAS_H = 980, 420
PADDING = 12
FPS = 60
GRID_STEP = 22

# Drone layout
ARM_LEN = 110
HUB_R = 26
ROTOR_R = 30
BLADE_R = 36
THRUST_SCALE = 0.9   # scales the length of the thrust vector vs outflow value

# Colors
BG = "#0f172a"          # slate-900
CARD = "#111827"        # gray-900
STROKE = "#374151"      # gray-700
INK = "#e5e7eb"         # gray-200
SUB = "#9ca3af"         # gray-400
ACCENT = "#22c55e"      # green-500
ACCENT2 = "#22d3ee"     # cyan-400
ACCENT3 = "#a78bfa"     # violet-400

# Rotor positions (X pattern; index order: FL, FR, BR, BL)
ROTOR_POS = (
    (-ARM_LEN, -ARM_LEN),
    (+ARM_LEN, -ARM_LEN),
    (+ARM_LEN, +ARM_LEN),
    (-ARM_LEN, +ARM_LEN),
)

# ------------------------------ Data container ------------------------------
@dataclass
class RotorState:
    inflow: float = 0.0     # nominal thrust
    outflow: float = 0.0    # measured thrust
    throttle: float = 0.0   # 0..1 motor command
    phase: float = 0.0      # blade phase (animation)

# ------------------------------- Main widget --------------------------------
class QuadcopterWidget(tk.Frame):
    def __init__(self, parent: tk.Misc, n_rotors: int = 4, title: str = "Quadcopter"):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.n = max(1, int(n_rotors))
        self.title = title
        self.units = "N"

        # States (support >4 rotors by placing extras on a ring around the body)
        self.states = [RotorState(2.0, 1.6, 0.5, random.random() * 1000.0)
                       for _ in range(self.n)]

        self._running = True
        self._last_push = 0.0
        self._last_t = time.time()

        # Header
        self.header = tk.Label(self, text=self.title, fg=INK, bg=BG, font=("Segoe UI Semibold", 14))
        self.header.pack(side="top", anchor="w", padx=PADDING, pady=(PADDING, 6))

        # Canvas card
        self.card = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=STROKE)
        self.card.pack(fill="both", expand=True, padx=PADDING, pady=(0, PADDING))
        self.canvas = tk.Canvas(self.card, width=CANVAS_W, height=CANVAS_H, bg=CARD, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # Footer HUD for totals
        self.hud = tk.Label(self, fg=SUB, bg=BG, font=("Consolas", 10), anchor="e")
        self.hud.pack(fill="x", padx=PADDING, pady=(0, PADDING // 2))

        # Start animation
        self.after(int(1000 / FPS), self._tick)

    # ------------------------------ Public API --------------------------------
    def push_state(
        self,
        levels: Iterable[float],     # IGNORED (compat only)
        inflow: Iterable[float],
        outflow: Iterable[float],
        valves: Iterable[float],
        units: Optional[str] = None,
    ):
        try:
            fi = list(inflow)
            fo = list(outflow)
            va = list(valves)
            m = min(self.n, len(fi), len(fo), len(va))
            for i in range(m):
                st = self.states[i]
                st.inflow = float(max(0.0, fi[i]))
                st.outflow = float(max(0.0, fo[i]))
                st.throttle = float(_clip01(va[i]))
            if units:
                self.units = str(units)
            self._last_push = time.time()
        except Exception:
            # Never let visualization input kill the UI
            pass

    # --------------------------- Internal loop --------------------------------
    def _tick(self):
        if not self._running:
            return
        now = time.time()
        dt = max(1.0 / 240.0, now - self._last_t)
        self._last_t = now

        # Demo if idle (no external push_state recently)
        if now - self._last_push > 1.0:
            self._demo_step(dt)

        # Integrate blade phases from commanded throttle (RPM proxy)
        for st in self.states:
            # RPM ~ 5..25 rps mapped from throttle 0..1
            rps = 5.0 + 20.0 * st.throttle
            st.phase = (st.phase + rps * dt * 2.0 * math.pi) % (2.0 * math.pi)

            # Very light thrust smoothing toward inflow/throttle balance for visuals
            target_out = max(0.0, st.inflow * (0.7 + 0.6 * st.throttle))
            st.outflow += (target_out - st.outflow) * (1.0 - math.exp(-dt * 1.5))

        self._draw_scene()
        self.after(int(1000 / FPS), self._tick)

    def _demo_step(self, dt: float):
        # Gentle sinusoid on throttle and nominal thrust for a pleasant idle animation
        t = time.time()
        for i, st in enumerate(self.states):
            st.throttle = _clip01(0.45 + 0.35 * math.sin(0.8 * t + i * 0.9))
            nominal = 8.0 + 5.0 * math.sin(0.6 * t + i * 1.2)
            st.inflow = max(0.0, nominal)

    # ---------------------------- Drawing -------------------------------------
    def _draw_scene(self):
        c = self.canvas
        c.delete("all")
        w = int(c.winfo_width() or CANVAS_W)
        h = int(c.winfo_height() or CANVAS_H)

        self._draw_grid(c, w, h)
        self._draw_drone(c, w // 2, h // 2 - 6)
        self._draw_sidebar(c, w, h)
        self._draw_hud()

    def _draw_grid(self, c: tk.Canvas, w: int, h: int):
        # faint grid
        g = "#1f2937"
        for x in range(0, w, GRID_STEP):
            c.create_line(x, 0, x, h, fill=g)
        for y in range(0, h, GRID_STEP):
            c.create_line(0, y, w, y, fill=g)

    def _draw_drone(self, c: tk.Canvas, cx: int, cy: int):
        # Body hub
        self._rounded(c, cx - HUB_R, cy - HUB_R, cx + HUB_R, cy + HUB_R, 12,
                      fill="#0b1220", outline=STROKE, width=2)
        c.create_text(cx, cy, text="FC", fill=SUB, font=("Consolas", 10, "bold"))

        # Arms (X pattern)
        arm_color = "#243244"
        for dx, dy in ROTOR_POS[:min(4, self.n)]:
            c.create_line(cx, cy, cx + dx, cy + dy, fill=arm_color, width=8, capstyle=tk.ROUND)

        # For >4 rotors, arrange them in a ring
        extra = []
        if self.n > 4:
            k = self.n - 4
            rad = ARM_LEN + 36
            for j in range(k):
                ang = (2 * math.pi) * (j / k)
                extra.append((int(rad * math.cos(ang)), int(rad * math.sin(ang))))

        # Draw rotors
        positions = list(ROTOR_POS[:min(4, self.n)]) + extra
        for i in range(self.n):
            x = cx + positions[i][0]
            y = cy + positions[i][1]
            self._draw_rotor(c, i, x, y, self.states[i])

    def _draw_rotor(self, c: tk.Canvas, idx: int, x: int, y: int, st: RotorState):
        # Rotor base
        self._rounded(c, x - ROTOR_R, y - ROTOR_R, x + ROTOR_R, y + ROTOR_R, 12,
                      fill="#0b1220", outline=STROKE, width=2)

        # Spinning blades (two bars at phase and +90°)
        for k in (0.0, math.pi / 2.0):
            ang = st.phase + k
            bx = x + BLADE_R * math.cos(ang)
            by = y + BLADE_R * math.sin(ang)
            cx2 = x - BLADE_R * math.cos(ang)
            cy2 = y - BLADE_R * math.sin(ang)
            c.create_line(cx2, cy2, bx, by, fill=ACCENT2, width=3)

        # Throttle ring (progress)
        p = st.throttle
        c.create_oval(x - (ROTOR_R + 8), y - (ROTOR_R + 8), x + (ROTOR_R + 8), y + (ROTOR_R + 8),
                      outline="#1f2937", width=4)
        c.create_arc(x - (ROTOR_R + 8), y - (ROTOR_R + 8), x + (ROTOR_R + 8), y + (ROTOR_R + 8),
                     start=90, extent=-max(1, int(p * 360)), style=tk.ARC, outline=ACCENT, width=5)

        # Thrust vector (outflow)
        vec_len = 18 + 70 * THRUST_SCALE * _smooth(st.outflow)
        c.create_line(x, y, x, y - vec_len, fill=ACCENT3, width=3, arrow=tk.LAST)

        # Labels around rotor
        tag = f"R{idx + 1}"
        c.create_text(x, y + ROTOR_R + 16, text=tag, fill=INK, font=("Segoe UI Semibold", 11))
        c.create_text(x, y + ROTOR_R + 32, text=f"throttle {st.throttle * 100:4.0f}%",
                      fill=SUB, font=("Consolas", 10))

    def _draw_sidebar(self, c: tk.Canvas, w: int, h: int):
        # Right side panel with per-rotor readouts
        panel_w = 290
        x1 = w - panel_w
        y1 = 10
        x2 = w - 10
        y2 = h - 10
        self._rounded(c, x1, y1, x2, y2, 14, fill="#0b1220", outline=STROKE, width=2)
        c.create_text((x1 + x2) // 2, y1 + 18, text=f"Rotor Telemetry ({self.units})",
                      fill=INK, font=("Segoe UI Semibold", 12))

        # Table headers
        colx = [x1 + 10, x1 + 70, x1 + 140, x1 + 210]
        c.create_text(colx[0], y1 + 44, text="Rotor",    fill=SUB, font=("Consolas", 10), anchor="w")
        c.create_text(colx[1], y1 + 44, text="Throttle", fill=SUB, font=("Consolas", 10), anchor="w")
        c.create_text(colx[2], y1 + 44, text="Thrust",   fill=SUB, font=("Consolas", 10), anchor="w")
        c.create_text(colx[3], y1 + 44, text="Nominal",  fill=SUB, font=("Consolas", 10), anchor="w")

        y = y1 + 64
        row_h = 26
        tot_out = 0.0
        tot_nom = 0.0

        for i, st in enumerate(self.states):
            # text
            c.create_text(colx[0], y, text=f"R{i + 1}", fill=INK, font=("Consolas", 11), anchor="w")
            # throttle bar
            vb_w = 56
            self._rounded(c, colx[1], y - 8, colx[1] + vb_w, y + 8, 6, fill="#0b1524", outline="#1f2937", width=1)
            fill_w = int(vb_w * _clip01(st.throttle))
            if fill_w > 0:
                self._rounded(c, colx[1], y - 8, colx[1] + fill_w, y + 8, 6, fill=ACCENT, outline="", width=0)

            # thrust/nominal numbers
            c.create_text(colx[2], y, text=f"{st.outflow:7.2f}", fill=INK, font=("Consolas", 11), anchor="w")
            c.create_text(colx[3], y, text=f"{st.inflow:7.2f}",  fill=SUB, font=("Consolas", 11), anchor="w")

            tot_out += st.outflow
            tot_nom += st.inflow
            y += row_h

        # Totals
        y += 6
        c.create_line(x1 + 10, y, x2 - 10, y, fill="#1f2937")
        y += 10
        c.create_text(colx[1], y, text="Total:",       fill=SUB, font=("Consolas", 11), anchor="w")
        c.create_text(colx[2], y, text=f"{tot_out:7.2f}", fill=INK, font=("Consolas", 11), anchor="w")
        c.create_text(colx[3], y, text=f"{tot_nom:7.2f}", fill=SUB, font=("Consolas", 11), anchor="w")

    def _draw_hud(self):
        # Quick HUD: min/max/avg throttle, min/max/avg thrust
        throttles = [s.throttle for s in self.states]
        outs = [s.outflow for s in self.states]
        if not throttles or not outs:
            self.hud.config(text="")
            return
        tmin, tmax, tavg = min(throttles), max(throttles), sum(throttles) / len(throttles)
        omin, omax, oavg = min(outs), max(outs), sum(outs) / len(outs)
        self.hud.config(
            text=f"throttle min/avg/max: {tmin:0.2f} / {tavg:0.2f} / {tmax:0.2f}   |   "
                 f"thrust min/avg/max ({self.units}): {omin:0.2f} / {oavg:0.2f} / {omax:0.2f}"
        )

    # --------------------------- Primitives -----------------------------------
    def _rounded(self, c: tk.Canvas, x1, y1, x2, y2, r=10, **kw):
        r = max(2, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        # smoothed polygon performs better than four arcs
        pts = [
            (x1 + r, y1), (x2 - r, y1), (x2, y1), (x2, y1 + r),
            (x2, y2 - r), (x2, y2), (x2 - r, y2), (x1 + r, y2),
            (x1, y2), (x1, y2 - r), (x1, y1 + r), (x1, y1)
        ]
        return c.create_polygon(pts, smooth=True, **kw)

# ------------------------------ Helpers --------------------------------------
def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

def _smooth(x: float) -> float:
    # light nonlinearity to make small thrusts still visible in the vector
    return x / (1.0 + 0.6 * abs(x))

# ----------------------------- Standalone demo -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("QuadcopterWidget Demo")
    w = QuadcopterWidget(root, n_rotors=4, title="Quadcopter — Canvas Demo")
    w.pack(fill="both", expand=True)
    root.mainloop()

"""
Embedded 5‑Tank Canvas (Tkinter)
--------------------------------
Drop‑in widget version of the popup viewer. Renders a polished 5‑tank
simulation directly inside any Tkinter Frame. API is identical to the popup's
`push_state(...)` so you can reuse your loop logic.

Usage in your main GUI:

    from tanks_widget import TanksWidget

    self.tanks = TanksWidget(parent_frame, n_tanks=5, title="5‑Tank Water System")
    self.tanks.pack(fill="both", expand=True)
    # In your control loop:
    self.tanks.push_state(levels=..., inflow=..., outflow=..., valves=..., units="L/min")

This file borrows the drawing logic from tanks_popup, but manages its own
canvas and internal clock without creating a Toplevel window.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import tkinter as tk

# Visual constants
CANVAS_W, CANVAS_H = 1120, 420
PADDING = 12
TANK_W, TANK_H = 160, 220
TANK_SPACING = 24
PIPES_Y_OFFSET = 34
FPS = 30
PARTICLE_SPACING = 24

# palette
BG = "#0f172a"          # slate-900
CARD = "#111827"        # gray-900
STROKE = "#374151"      # gray-700
INK = "#e5e7eb"         # gray-200
ACCENT = "#22d3ee"      # cyan-400
ACCENT2 = "#34d399"     # green-400
WARNING = "#f59e0b"     # amber-500


@dataclass
class TankState:
    level: float   # 0..1
    inflow: float  # L/min
    outflow: float # L/min
    valve: float   # 0..1


class TanksWidget(tk.Frame):
    def __init__(self, parent: tk.Misc, n_tanks: int = 5, title: str = "5‑Tank Water System"):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.n = n_tanks
        self.title = title
        self.states = [TankState(0.25 + 0.1*i, 20.0, 18.0, 0.5) for i in range(self.n)]
        self.units = "L/min"
        self._phases = [random.random()*1000.0 for _ in range(self.n)]
        self._last_external_push = 0.0
        self._running = True
        self._last_time = time.time()

        # Header label
        self.header = tk.Label(self, text=self.title, fg=INK, bg=BG, font=("Segoe UI Semibold", 14))
        self.header.pack(side="top", anchor="w", padx=PADDING, pady=(PADDING, 4))

        # Canvas in a card
        self.card = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=STROKE)
        self.card.pack(fill="both", expand=True, padx=PADDING, pady=(0, PADDING))
        self.canvas = tk.Canvas(self.card, width=CANVAS_W, height=CANVAS_H, bg=CARD, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=12)

        self.after(int(1000/FPS), self._tick)

    # --------------------------- Public API ---------------------------
    def push_state(
        self,
        levels: Iterable[float],
        inflow: Iterable[float],
        outflow: Iterable[float],
        valves: Iterable[float],
        units: Optional[str] = None,
    ):
        lv = list(levels)
        fi = list(inflow)
        fo = list(outflow)
        va = list(valves)
        m = min(self.n, len(lv), len(fi), len(fo), len(va))
        for i in range(m):
            self.states[i].level = float(max(0.0, min(1.0, lv[i])))
            self.states[i].inflow = float(max(0.0, fi[i]))
            self.states[i].outflow = float(max(0.0, fo[i]))
            self.states[i].valve = float(max(0.0, min(1.0, va[i])))
        if units:
            self.units = units
        self._last_external_push = time.time()

    # ------------------------ Internal loop & draw --------------------
    def _tick(self):
        if not self._running:
            return
        now = time.time()
        dt = max(1.0/120.0, now - self._last_time)
        self._last_time = now

        if now - self._last_external_push > 1.0:
            self._demo_step(dt)

        self._draw_scene(now)
        self.after(int(1000/FPS), self._tick)

    def _demo_step(self, dt: float):
        for i, st in enumerate(self.states):
            v = 0.5 + 0.35 * math.sin(0.6 * i + time.time()*0.7)
            st.valve = max(0.05, min(0.95, v))
            st.inflow = 30.0 * st.valve
            st.outflow = st.outflow + (st.inflow - st.outflow) * (1.0 - math.exp(-dt*1.6))
            delta = (st.inflow - st.outflow) * 0.001 * dt
            st.level = max(0.0, min(1.0, st.level + delta))

    def _draw_scene(self, t: float):
        c = self.canvas
        c.delete("all")
        self._draw_grid(c)
        x0 = PADDING
        y0 = 16
        for i in range(self.n):
            bx = x0 + i * (TANK_W + TANK_SPACING)
            self._draw_tank_block(c, i, bx, y0, self.states[i], t)

    def _draw_grid(self, c: tk.Canvas):
        step = 24
        for x in range(0, int(c.winfo_width() or CANVAS_W), step):
            c.create_line(x, 0, x, int(c.winfo_height() or CANVAS_H), fill="#1f2937")
        for y in range(0, int(c.winfo_height() or CANVAS_H), step):
            c.create_line(0, y, int(c.winfo_width() or CANVAS_W), y, fill="#1f2937")

    def _draw_tank_block(self, c: tk.Canvas, idx: int, bx: int, by: int, st: TankState, t: float):
        label = f"Tank {idx+1}"
        c.create_text(bx + TANK_W/2, by + 10, text=label, fill=INK, font=("Segoe UI Semibold", 12))

        pipe_y = by + PIPES_Y_OFFSET
        c.create_line(bx - 26, pipe_y, bx, pipe_y, fill=ACCENT, width=4)
        c.create_line(bx + TANK_W, pipe_y, bx + TANK_W + 26, pipe_y, fill=ACCENT2, width=4)

        self._draw_valve(c, bx - 8, pipe_y, st.valve)
        self._draw_particles(c, bx - 26, pipe_y, 24, +1, speed=80 + 80*st.valve, phase=self._phases[idx]+t)
        self._draw_particles(c, bx + TANK_W, pipe_y, 24, +1, speed=60 + 60*st.valve, phase=self._phases[idx]+t*0.9, color=ACCENT2)

        x1, y1 = bx, by + 40
        x2, y2 = bx + TANK_W, by + 40 + TANK_H
        self._rounded_rect(c, x1, y1, x2, y2, r=14, outline=STROKE, width=2, fill="#0b1220")

        lvl_h = (y2 - y1 - 6) * st.level
        lvl_y = y2 - 3 - lvl_h
        self._rounded_rect(c, x1 + 3, lvl_y, x2 - 3, y2 - 3, r=12, outline="", width=0, fill="#0ea5e9")
        c.create_line(x1+2, y1+8, x1+2, y2-8, fill="#334155")

        panel_y = y2 + 10
        self._draw_readout(c, bx, panel_y, "Level", f"{st.level*100:5.1f}%")
        self._draw_readout(c, bx, panel_y+22, "Inflow", f"{st.inflow:6.1f} {self.units}")
        self._draw_readout(c, bx, panel_y+44, "Outflow", f"{st.outflow:6.1f} {self.units}")
        self._draw_readout(c, bx, panel_y+66, "Valve", f"{st.valve*100:5.1f}%")

        if st.level < 0.08 or st.level > 0.92:
            c.create_text((x1+x2)/2, y1-10, text="LEVEL ALERT", fill=WARNING, font=("Segoe UI Semibold", 11))

    # ---------------------- primitives ----------------------
    def _rounded_rect(self, c: tk.Canvas, x1, y1, x2, y2, r=10, **kw):
        r = max(2, min(r, (x2-x1)/2, (y2-y1)/2))
        pts = [
            (x1+r, y1), (x2-r, y1), (x2, y1), (x2, y1+r),
            (x2, y2-r), (x2, y2), (x2-r, y2), (x1+r, y2),
            (x1, y2), (x1, y2-r), (x1, y1+r), (x1, y1)
        ]
        return c.create_polygon(
            [pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7], pts[8], pts[9], pts[10], pts[11]],
            smooth=True, **kw
        )

    def _draw_readout(self, c: tk.Canvas, x: int, y: int, label: str, value: str):
        c.create_text(x + 6, y + 10, text=label, anchor="w", fill="#9ca3af", font=("Consolas", 10))
        c.create_text(x + TANK_W - 6, y + 10, text=value, anchor="e", fill=INK, font=("Consolas", 11))

    def _draw_valve(self, c: tk.Canvas, cx: int, cy: int, pos: float):
        pos = max(0.0, min(1.0, pos))
        r = 10
        c.create_oval(cx-r, cy-r, cx+r, cy+r, outline=STROKE, width=2, fill=CARD)
        angle = (pos * 90.0)
        rad = math.radians(angle)
        x1, y1 = cx - r*math.cos(rad), cy - r*math.sin(rad)
        x2, y2 = cx + r*math.cos(rad), cy + r*math.sin(rad)
        c.create_line(x1, y1, x2, y2, fill=ACCENT, width=3)

    def _draw_particles(self, c: tk.Canvas, x: int, y: int, length: int, dir_sign: int, speed: float, phase: float, color: str = ACCENT):
        n = max(2, length // PARTICLE_SPACING)
        for i in range(n):
            p = (phase*speed*0.01 + i*PARTICLE_SPACING) % length
            px = x + dir_sign * p
            c.create_oval(px-2, y-2, px+2, y+2, fill=color, outline="")


# Standalone demo
if __name__ == "__main__":
    root = tk.Tk()
    w = TanksWidget(root)
    w.pack(fill="both", expand=True)
    root.mainloop()

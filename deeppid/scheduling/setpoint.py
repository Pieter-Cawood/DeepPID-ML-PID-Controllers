
from __future__ import annotations
import numpy as np

def step(total_t=20.0, dt=0.02, t_step=1.0, level=1.0):
    t = np.arange(0.0, total_t, dt)
    sp = np.zeros_like(t)
    sp[t >= t_step] = level
    return t, sp

def multi_step(total_t=30.0, dt=0.02, levels=(0.5, 1.0, 0.2), every=10.0):
    t = np.arange(0.0, total_t, dt)
    sp = np.zeros_like(t)
    for i, lvl in enumerate(levels):
        sp[t >= i*every] = lvl
    return t, sp

def sine(total_t=20.0, dt=0.02, amp=0.5, bias=0.5, freq=0.1):
    t = np.arange(0.0, total_t, dt)
    sp = bias + amp * np.sin(2*np.pi*freq*t)
    return t, sp

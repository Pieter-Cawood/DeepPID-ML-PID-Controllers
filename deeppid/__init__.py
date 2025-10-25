"""
DeepPID: PID + ML playground
--------------------------------
A clean reference implementation for experimenting with classic PID control,
gain scheduling, and neural adaptive PID (GRU-based) on simple nonlinear plants.
"""

# --- Core scientific imports ---
import math
import numpy as np
import torch
import importlib

# --- Tkinter & Matplotlib setup (for GUI examples) ---
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

import matplotlib
matplotlib.use("TkAgg")
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Make transparency the default for all figures
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"

# --- Default torch precision ---
torch.set_default_dtype(torch.float64)

# --- Package imports ---
from .controllers.controllers import *
from .envs.problems import *
from .utils.utils import CtrlAdapter, TensorUtils

# =====================================================================
# Controller Registry (AVAILABLE)
# Dynamically imports controllers from new or legacy paths.
# =====================================================================
AVAILABLE = {}

def _safe_import(class_name, menu_name=None):
    """
    Try importing controller classes from canonical path:
      deeppid.controllers.controllers
    Falls back to legacy or local imports if not found.
    """
    # 1) Canonical path
    try:
        mod = importlib.import_module("deeppid.controllers.controllers")
        cls = getattr(mod, class_name)
        AVAILABLE[menu_name or class_name] = cls
        return
    except Exception:
        pass

    # 2) Local/legacy models.py (for backward compatibility)
    for path in ("models", ".models"):
        try:
            mod = importlib.import_module(path)
            cls = getattr(mod, class_name)
            AVAILABLE[menu_name or class_name] = cls
            return
        except Exception:
            continue

# Core controllers
_safe_import("PIDController", "PID")
_safe_import("CascadePIDController", "CascadePID")
_safe_import("MLPController", "MLP")
_safe_import("GRUController", "GRU")

# Advanced controllers
_safe_import("HybridMPCController", "HybridMPC")
_safe_import("PIDResidualNN", "PID+Residual")
_safe_import("TransformerCtrl", "Transformer")
_safe_import("RLSafetyCtrl", "SafeRL")
_safe_import("PINNCtrl", "PINN")
_safe_import("AdaptiveHierCtrl", "AdaptiveHier")

__all__ = [
    # exposed submodules
    "CtrlAdapter",
    "TensorUtils",
    "AVAILABLE",
    # controllers (imported above)
    "PIDController",
    "CascadePIDController",
    "MLPController",
    "GRUController",
    "HybridMPCController",
    "PIDResidualNN",
    "TransformerCtrl",
    "RLSafetyCtrl",
    "PINNCtrl",
    "AdaptiveHierCtrl",
]

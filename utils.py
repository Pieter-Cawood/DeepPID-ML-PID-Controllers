# ─────────────────────────────────────────────────────────────────────────────
# File: utils.py
# Small shared helpers for the app.
# ─────────────────────────────────────────────────────────────────────────────

import torch

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

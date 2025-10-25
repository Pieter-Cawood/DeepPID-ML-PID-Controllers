
import math
import numpy as np
import torch
from typing import Union

class TensorUtils:
    """
    Small, static tensor helpers used across controllers and GUI code.
    All methods are pure (no internal state) and operate on PyTorch tensors.
    """

    @staticmethod
    def ensure_shape(x: Union[torch.Tensor, float, int, list, tuple],
                     like: torch.Tensor) -> torch.Tensor:
        """
        Ensure `x` has the same length as `like`. If longer -> truncate; if shorter -> pad with `like`.
        Always returns float64 1-D tensor of length N.
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

    @staticmethod
    def to_f64_scalar_tensor(x: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """
        Return a float64 0-D tensor from a float/int or a tensor
        (no re-wrapping warning, keeps it scalar-shaped).
        """
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(torch.float64).reshape(())
        return torch.tensor(float(x), dtype=torch.float64)

    @staticmethod
    def clip(v: torch.Tensor, lo: Union[torch.Tensor, float], hi: Union[torch.Tensor, float]) -> torch.Tensor:
        """Clamp tensor `v` element-wise to [lo, hi]."""
        return torch.min(torch.max(v, lo), hi)

    @staticmethod
    def slew_limit(prev: torch.Tensor, desired: torch.Tensor, max_delta: Union[float, torch.Tensor],
                   lo: Union[torch.Tensor, float], hi: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        First-order rate limiter: move from `prev` towards `desired` by at most ±`max_delta`,
        then clamp to [lo, hi].
        """
        return TensorUtils.clip(prev + torch.clamp(desired - prev, -max_delta, max_delta), lo, hi)

    @staticmethod
    def sanitize(x: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        """
        Replace NaN/Inf entries in `x` with the corresponding entries from `fallback`.
        Shapes must match.
        """
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
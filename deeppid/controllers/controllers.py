# controllers.py
# ─────────────────────────────────────────────────────────────────────────────
# Controller zoo used by the demo GUI:
#   • PIDController            – IMC-style auto-tuned PID with online refinement
#   • CascadePIDController     – stabilized inner PID + adaptive outer composition/total PI
#   • MLPController            – feed-forward NN policy with physics-aware loss & safety layer
#   • GRUController            – sequence model with the same safety + objectives
#   • HybridMPCController      – short-horizon optimizer with a learned residual model
#   • PIDResidualNN            – classic PID with a small residual NN on top
#   • TransformerCtrl          – causal Transformer policy (demo scale)
#   • RLSafetyCtrl             – actor NN + safety (stubbed w/ supervised objective)
#   • PINNCtrl                 – MLP variant with stronger physics penalties
#   • AdaptiveHierCtrl         – CascadePID with a tiny NN tuner (safe, log-scaled gains)
#
# Design notes
#   - All controllers accept the SAME constructor signature used by the GUI.
#   - All produce *speeds* (actuator %), never bypassing slew-rate or bounds.
#   - Numerical edges are handled (NaNs, tiny denominators, log barriers).
#   - Default dtype is float64 for stable aggregation and training online.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

__all__ = [
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
    # helpers
    "clip",
    "slew_limit",
    "ema",
]

# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers (kept minimal and torch-native)
# ─────────────────────────────────────────────────────────────────────────────

def clip(v: torch.Tensor, lo: torch.Tensor | float, hi: torch.Tensor | float) -> torch.Tensor:
    """Clamp tensor v element-wise to [lo, hi]."""
    return torch.min(torch.max(v, lo), hi)

def slew_limit(
    prev: torch.Tensor,
    desired: torch.Tensor,
    max_delta: float | torch.Tensor,
    lo: torch.Tensor | float,
    hi: torch.Tensor | float,
) -> torch.Tensor:
    """
    First-order rate limiter: move from prev towards desired by at most ±max_delta,
    then clamp to [lo, hi].
    """
    return clip(prev + torch.clamp(desired - prev, -max_delta, max_delta), lo, hi)

def ema(x_prev: torch.Tensor, x_new: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving average with weight alpha on the new value."""
    return (1.0 - alpha) * x_prev + alpha * x_new

def _total_scale(k: torch.Tensor, speed_max: torch.Tensor) -> torch.Tensor:
    """Rough normalizer: maximum nominal total output ≈ sum(k * speed_max)."""
    return torch.sum(k * speed_max).clamp_min(1.0)

def _tau_from_alpha(Ts: float, alpha: float) -> float:
    """
    Map discrete plant step:
         y_{t+1} = y_t + alpha * (k * s - y_t)
    to an equivalent continuous time constant τ via z=1−α:
         τ ≈ Ts / -ln(1 - α),   with safe fallbacks for extreme α.
    """
    a = float(alpha)
    if a <= 1e-6:
        return 10.0 * Ts
    if a >= 0.999999:
        return 0.1 * Ts
    try:
        return Ts / max(-np.log(1.0 - a), 1e-6)
    except Exception:
        return Ts / max(a, 1e-6)

def _classify_fast(N: int, tau_p: float) -> bool:
    """
    Heuristic classifier: small N or small τ -> "fast" plant.
    Used to bias IMC tuning towards more aggressive/defensive gains.
    """
    return (N <= 4) or (tau_p <= 0.8)

# ─────────────────────────────────────────────────────────────────────────────
# PID (single loop) — IMC auto-tuned + online refinement
# ─────────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    IMC-style PID in the *flow domain* (gains expressed against flow set-point), mapped
    to the *speed domain* via division by k, with:
      • derivative on measurement,
      • set-point weight on proportional (β),
      • conditional integrator + back-calculation anti-windup,
      • online refinement of the plant alpha -> τ estimate from observed steps.

    Public step:
        step(flows_meas_filt, target_ratio, F_total, s_ff) -> speeds (N,)

    Where:
      - `target_ratio` sums to ~1 and represents the desired composition,
      - `F_total` is the requested total output (same unit as measured flows),
      - `s_ff` is a feed-forward baseline (e.g., from a “baseline allocation”).
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, plant_alpha, slew_rate):
        self.N, self.k = int(N), k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.alpha_p, self.slew = float(Ts), float(plant_alpha), float(slew_rate)

        d, dt = k.device, k.dtype

        # Internal states
        self.int       = torch.zeros(N, dtype=dt, device=d)
        self.d_y       = torch.zeros(N, dtype=dt, device=d)
        self.prev_meas = torch.zeros(N, dtype=dt, device=d)
        self.prev_out  = torch.ones (N, dtype=dt, device=d) * 25.0

        # IMC flow-domain gains (mapped to speed domain by /k on use)
        self.Kp_f = torch.full((N,), 1.0, dtype=dt, device=d)
        self.Ki_f = torch.full((N,), 0.0, dtype=dt, device=d)
        self.Kd_f = torch.full((N,), 0.0, dtype=dt, device=d)

        # Filters / anti-windup
        self.tau_d = 0.0  # derivative filter τ
        self.beta  = 0.8  # setpoint weight on proportional
        self.Tt    = 0.2  # back-calculation anti-windup time-constant

        # Online plant estimate (alpha -> τ) and “fast” bias
        self.alpha_hat = np.clip(self.alpha_p, 1e-3, 0.99)
        self.tau_p     = _tau_from_alpha(self.Ts, self.alpha_hat)
        self.fast_mode = _classify_fast(self.N, self.tau_p)

        self._imc_retune(hard=True)

    # ----- IMC tuning (flow domain) -----
    def _imc_retune(self, hard: bool = False) -> None:
        tau_p = max(self.tau_p, 1e-6)
        fast = self.fast_mode

        lam = (0.35 * tau_p) if fast else (1.20 * tau_p)
        Ti  = tau_p
        Td  = (0.25 * tau_p) if fast else (0.10 * tau_p)
        tau_d_filt = 0.10 * tau_p if fast else 0.05 * tau_p

        Kp_f_new = (tau_p / max(lam, 1e-6))
        Ki_f_new = Kp_f_new / max(Ti, 1e-6)
        Kd_f_new = Kp_f_new * Td
        Tt_new   = max(0.5 * Ti, 0.05)

        if hard:
            self.Kp_f.fill_(Kp_f_new); self.Ki_f.fill_(Ki_f_new); self.Kd_f.fill_(Kd_f_new)
            self.tau_d = tau_d_filt;   self.Tt    = Tt_new
        else:
            a = 0.10
            self.Kp_f = (1 - a) * self.Kp_f + a * Kp_f_new
            self.Ki_f = (1 - a) * self.Ki_f + a * Ki_f_new
            self.Kd_f = (1 - a) * self.Kd_f + a * Kd_f_new
            self.tau_d = (1 - a) * self.tau_d + a * tau_d_filt
            self.Tt    = (1 - a) * self.Tt    + a * Tt_new

    def _update_alpha_estimate(self, y_now: torch.Tensor) -> None:
        """
        Robust per-step alpha update from one-step response:
            Δy ≈ α (k * u_prev − y_prev)  ->  α_inst = Δy / (k*u_prev − y_prev)
        Aggregate via median across channels and smooth via EMA.
        """
        y_prev = self.prev_meas
        u_prev = self.prev_out
        num = (y_now - y_prev)
        den = (self.k * u_prev - y_prev)
        den = torch.where(torch.abs(den) < 1e-6, torch.full_like(den, 1e-6), den)
        alpha_inst = torch.median(torch.clamp(num / den, 0.01, 0.99)).item()
        self.alpha_hat = 0.9 * self.alpha_hat + 0.1 * alpha_inst

        new_tau = _tau_from_alpha(self.Ts, self.alpha_hat)
        fast_new = _classify_fast(self.N, new_tau)
        if fast_new != self.fast_mode:
            self.fast_mode = fast_new
        if abs(new_tau - self.tau_p) / max(self.tau_p, 1e-6) > 0.10:
            self.tau_p = 0.9 * self.tau_p + 0.1 * new_tau
            self._imc_retune(hard=False)

    def sync_to(self, speeds_now: torch.Tensor, flows_now: torch.Tensor) -> None:
        """
        Re-seed states during startup or plant switch.
        Keeps τ estimate and gently re-tunes IMC gains for the new operating point.
        """
        self.prev_out = speeds_now.detach().clone()
        self.prev_meas = flows_now.detach().clone()
        self.d_y.zero_()
        self.int = torch.clamp(self.int, -2000.0, 2000.0)
        self._update_alpha_estimate(self.prev_meas)
        self._imc_retune(hard=False)

    # ----- internal: PID pieces -----
    def _core(self, flow_sp: torch.Tensor, flow_meas: torch.Tensor, s_ff: torch.Tensor):
        """Build unsaturated control action (P+I+D + feed-forward), return also e and Ki_s."""
        e  = flow_sp - flow_meas
        eP = self.beta * flow_sp - flow_meas

        alpha = 0.0 if self.tau_d < 1e-9 else np.exp(-self.Ts / max(self.tau_d, 1e-9))
        dy_raw = (flow_meas - self.prev_meas) / self.Ts
        self.d_y = alpha * self.d_y + (1 - alpha) * dy_raw

        # Map flow-domain gains to speed-domain by dividing by k
        Kp_s = self.Kp_f / (self.k + 1e-12)
        Ki_s = self.Ki_f / (self.k + 1e-12)
        Kd_s = self.Kd_f / (self.k + 1e-12)

        u_p = Kp_s * eP
        u_d = -Kd_s * self.d_y
        u_unsat = s_ff + u_p + self.int + u_d
        return u_unsat, e, Ki_s

    def _int_aw(self, u_unsat: torch.Tensor, e: torch.Tensor, Ki_s: torch.Tensor) -> torch.Tensor:
        """Conditional integrator + back-calculation anti-windup."""
        u_clamped = clip(u_unsat, self.speed_min, self.speed_max)
        sat_err = u_clamped - u_unsat

        push = Ki_s * e
        tol = 1e-9
        sat_hi = (u_unsat > self.speed_max - tol)
        sat_lo = (u_unsat < self.speed_min + tol)
        advance = torch.ones_like(e)
        # Stop integrating if error would push further into saturation
        advance[(sat_hi & (push > 0)) | (sat_lo & (push < 0))] = 0.0

        self.int += advance * (Ki_s * e * self.Ts)
        self.int += (sat_err / max(self.Tt, 1e-6)) * self.Ts
        return u_clamped

    def _total_trim(self, speeds_cmd: torch.Tensor, F_total_target: torch.Tensor, flows_meas: torch.Tensor):
        """Small uniform bias to enforce total flow exactly (helps at limits)."""
        total_meas = torch.sum(flows_meas) + 1e-12
        dF = (F_total_target - total_meas).item()
        if abs(dF) < 1e-3:
            return speeds_cmd
        b = torch.tensor(dF / self.N, dtype=speeds_cmd.dtype, device=speeds_cmd.device) / (self.k + 1e-12)
        return clip(speeds_cmd + b, self.speed_min, self.speed_max)

    # ----- public: one control tick -----
    def step(self, flows_meas_filt: torch.Tensor, target_ratio: torch.Tensor, F_total: torch.Tensor, s_ff: torch.Tensor):
        self._update_alpha_estimate(flows_meas_filt)

        flow_sp = target_ratio * F_total
        u_unsat, e, Ki_s = self._core(flow_sp, flows_meas_filt, s_ff)
        u_cmd = self._int_aw(u_unsat, e, Ki_s)

        u_cmd = slew_limit(self.prev_out, u_cmd, self.slew, self.speed_min, self.speed_max)
        u_cmd = clip(u_cmd, self.speed_min, self.speed_max)
        u_cmd = self._total_trim(u_cmd, F_total, flows_meas_filt)

        self.prev_out = u_cmd.clone()
        self.prev_meas = flows_meas_filt.clone()
        return u_cmd

# ─────────────────────────────────────────────────────────────────────────────
# Cascade PID — IMC inner loop + adaptive outer loops (composition & total)
# ─────────────────────────────────────────────────────────────────────────────

class CascadePIDController:
    """
    Stabilized inner PID (IMC tuned) that tracks *per-channel* flow set-points,
    plus two outer PI loops:
      • total loop   – enforces the requested total output via a global gain γ,
      • composition  – small per-channel trim that steers measured composition to target.
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, plant_alpha, slew_rate):
        self.N, self.k = int(N), k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = float(Ts), float(slew_rate)
        self.alpha_p = float(plant_alpha)

        d, dt = k.device, k.dtype

        # Inner PID gains (flow-domain)
        self.Kp_f = torch.full((N,), 1.0, dtype=dt, device=d)
        self.Ki_f = torch.full((N,), 0.0, dtype=dt, device=d)
        self.Kd_f = torch.full((N,), 0.0, dtype=dt, device=d)
        # Baseline copies (for safe tuning in AdaptiveHierCtrl)
        self.Kp_base = self.Kp_f.clone()
        self.Ki_base = self.Ki_f.clone()
        self.Kd_base = self.Kd_f.clone()

        # Inner loop states
        self.int        = torch.zeros(N, dtype=dt, device=d)
        self.int_limit  = torch.full((N,), 2_000.0, dtype=dt, device=d)
        self.d_y        = torch.zeros(N, dtype=dt, device=d)
        self.prev_meas  = torch.zeros(N, dtype=dt, device=d)
        self.prev_out   = torch.ones(N, dtype=dt, device=d) * 25.0

        # Filters / anti-windup
        self.beta  = 0.8
        self.tau_d = 0.0
        self.Tt    = 0.2

        # Outer loops (γ for total; trim for composition)
        self.gamma = 1.0
        self.gamma_int = 0.0
        self.Kp_tot, self.Ki_tot = 0.25, 0.12
        self.gamma_min, self.gamma_max = 0.7, 1.3
        self.gamma_int_min, self.gamma_int_max = -0.5, 0.5
        self.gamma_leak = 1e-3

        self.comp_int = torch.zeros(N, dtype=dt, device=d)
        self.Kp_comp, self.Ki_comp = 0.12, 0.06
        self.comp_leak = 1e-3
        self.comp_int_limit = 0.25  # max additive fraction per channel

        # Online plant estimate
        self.alpha_hat = np.clip(self.alpha_p, 1e-3, 0.99)
        self.tau_p     = _tau_from_alpha(self.Ts, self.alpha_hat)
        self.fast_mode = _classify_fast(self.N, self.tau_p)

        self._imc_inner_retune(hard=True)
        self._retune_outer(hard=True)

    # ----- tuning helpers -----
    def _imc_inner_retune(self, hard: bool = False) -> None:
        tau_p = max(self.tau_p, 1e-6)
        fast = self.fast_mode
        lam = (0.45 * tau_p) if fast else (1.10 * tau_p)
        Ti  = tau_p
        Td  = (0.22 * tau_p) if fast else (0.08 * tau_p)
        tau_d_filt = 0.10 * tau_p if fast else 0.05 * tau_p

        Kp_f_new = (tau_p / max(lam, 1e-6))
        Ki_f_new = Kp_f_new / max(Ti, 1e-6)
        Kd_f_new = Kp_f_new * Td
        Tt_new   = max(0.5 * Ti, 0.05)

        if hard:
            self.Kp_f.fill_(Kp_f_new); self.Ki_f.fill_(Ki_f_new); self.Kd_f.fill_(Kd_f_new)
            self.Kp_base = self.Kp_f.clone(); self.Ki_base = self.Ki_f.clone(); self.Kd_base = self.Kd_f.clone()
            self.tau_d = tau_d_filt; self.Tt = Tt_new
        else:
            a = 0.10
            self.Kp_f = (1 - a) * self.Kp_f + a * Kp_f_new
            self.Ki_f = (1 - a) * self.Ki_f + a * Ki_f_new
            self.Kd_f = (1 - a) * self.Kd_f + a * Kd_f_new
            self.tau_d = (1 - a) * self.tau_d + a * tau_d_filt
            self.Tt    = (1 - a) * self.Tt    + a * Tt_new

    def _retune_outer(self, hard: bool = False) -> None:
        """Bias outer PI strengths based on inner plant speed."""
        fast = self.fast_mode
        Kp_tot_new = 0.35 if fast else 0.25
        Ki_tot_new = 0.18 if fast else 0.12
        Kp_comp_new = 0.14 if fast else 0.10
        Ki_comp_new = 0.07 if fast else 0.05

        if hard:
            self.Kp_tot, self.Ki_tot = Kp_tot_new, Ki_tot_new
            self.Kp_comp, self.Ki_comp = Kp_comp_new, Ki_comp_new
        else:
            a = 0.10
            self.Kp_tot = (1 - a) * self.Kp_tot + a * Kp_tot_new
            self.Ki_tot = (1 - a) * self.Ki_tot + a * Ki_tot_new
            self.Kp_comp = (1 - a) * self.Kp_comp + a * Kp_comp_new
            self.Ki_comp = (1 - a) * self.Ki_comp + a * Ki_comp_new

    def _update_alpha_estimate(self, y_now: torch.Tensor) -> None:
        y_prev = self.prev_meas
        u_prev = self.prev_out
        num = (y_now - y_prev)
        den = (self.k * u_prev - y_prev)
        den = torch.where(torch.abs(den) < 1e-6, torch.full_like(den, 1e-6), den)
        alpha_inst = torch.median(torch.clamp(num / den, 0.01, 0.99)).item()
        self.alpha_hat = 0.9 * self.alpha_hat + 0.1 * alpha_inst
        new_tau = _tau_from_alpha(self.Ts, self.alpha_hat)
        fast_new = _classify_fast(self.N, new_tau)
        if abs(new_tau - self.tau_p) / max(self.tau_p, 1e-6) > 0.10 or fast_new != self.fast_mode:
            self.fast_mode = fast_new
            self.tau_p = 0.9 * self.tau_p + 0.1 * new_tau
            self._imc_inner_retune(hard=False)
            self._retune_outer(hard=False)

    def sync_to(self, speeds_now: torch.Tensor, flows_now: torch.Tensor) -> None:
        self.prev_out = speeds_now.detach().clone()
        self.prev_meas = flows_now.detach().clone()
        self.d_y.zero_()
        self.int.zero_()
        self.gamma, self.gamma_int = 1.0, 0.0
        self.comp_int.zero_()
        self._update_alpha_estimate(self.prev_meas)
        self._imc_inner_retune(hard=False)
        self._retune_outer(hard=False)

    # ----- outer loops -----
    def _outer_total(self, F_total_sp: torch.Tensor, flows_meas: torch.Tensor) -> None:
        """PI on total flow; produces global multiplier γ with leakage and clamps."""
        err = float(F_total_sp - torch.sum(flows_meas))
        denom = max(float(F_total_sp), 1e-6)
        self.gamma_int += (self.Ki_tot * err / denom - self.gamma_leak * self.gamma_int) * self.Ts
        self.gamma_int = float(np.clip(self.gamma_int, self.gamma_int_min, self.gamma_int_max))
        p = self.Kp_tot * err / denom
        self.gamma = float(np.clip(1.0 + p + self.gamma_int, self.gamma_min, self.gamma_max))

    def _outer_comp(self, ratio_sp: torch.Tensor, flows_meas: torch.Tensor) -> torch.Tensor:
        """Small zero-sum trim vector that nudges composition to target."""
        total = float(torch.sum(flows_meas) + 1e-12)
        comp_meas = (flows_meas / total) if total > 0 else torch.full_like(ratio_sp, 1.0 / self.N)
        e = ratio_sp - comp_meas
        e = e - torch.mean(e)

        self.comp_int += (self.Ki_comp * e - self.comp_leak * self.comp_int) * self.Ts
        self.comp_int -= torch.mean(self.comp_int)
        self.comp_int = torch.clamp(self.comp_int, -self.comp_int_limit, self.comp_int_limit)

        trim = self.Kp_comp * e + self.comp_int
        trim -= torch.mean(trim)
        trim = torch.clamp(trim, -self.comp_int_limit, self.comp_int_limit)
        return trim

    # ----- inner loop -----
    def _inner(self, Fsp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha = 0.0 if self.tau_d < 1e-6 else np.exp(-self.Ts / self.tau_d)
        dy_raw = (y - self.prev_meas) / self.Ts
        self.d_y = alpha * self.d_y + (1 - alpha) * dy_raw

        Kp_s = self.Kp_f / (self.k + 1e-12)
        Ki_s = self.Ki_f / (self.k + 1e-12)
        Kd_s = self.Kd_f / (self.k + 1e-12)

        e  = Fsp - y
        eP = self.beta * Fsp - y

        u_p = Kp_s * eP
        u_d = -Kd_s * self.d_y
        u_unsat = (Fsp / (self.k + 1e-12)) + u_p + self.int + u_d

        u_clamped = clip(u_unsat, self.speed_min, self.speed_max)
        sat_err = u_clamped - u_unsat

        push = Ki_s * e
        tol = 1e-9
        sat_hi = (u_unsat > self.speed_max - tol)
        sat_lo = (u_unsat < self.speed_min + tol)
        advance = torch.ones_like(e)
        advance[(sat_hi & (push > 0)) | (sat_lo & (push < 0))] = 0.0

        self.int += advance * (Ki_s * e * self.Ts)
        self.int += (sat_err / max(self.Tt, 1e-6)) * self.Ts
        self.int = torch.clamp(self.int, -self.int_limit, self.int_limit)
        return u_clamped

    # ----- public -----
    def step(self, flows_meas_filt: torch.Tensor, ratio_sp: torch.Tensor, F_total_sp: torch.Tensor, _unused=None):
        self._update_alpha_estimate(flows_meas_filt)

        # Outer loops
        self._outer_total(F_total_sp, flows_meas_filt)
        trim = self._outer_comp(ratio_sp, flows_meas_filt)

        Ft = F_total_sp
        Fsp = torch.clamp(self.gamma * Ft * ratio_sp + Ft * trim, min=0.0)

        u = self._inner(Fsp, flows_meas_filt)
        u = slew_limit(self.prev_out, u, self.slew, self.speed_min, self.speed_max)
        u = clip(u, self.speed_min, self.speed_max)

        self.prev_out = u.clone()
        self.prev_meas = flows_meas_filt.clone()
        return u, Fsp

# ─────────────────────────────────────────────────────────────────────────────
# Neural controllers (safety-wrapped; physics-aware objectives)
# ─────────────────────────────────────────────────────────────────────────────

class MLPController(torch.nn.Module):
    """
    Feed-forward policy: one-shot speeds = f([target_ratio, F_total, flows, prev_speeds]).
    Loss terms:
      • composition error, total error,
      • smoothness on speeds,
      • bound soft-barrier (keeps away from hard min/max),
      • optional reference proximity, and measured-total consistency.
    Always passes through a rate limiter and clamp.
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, slew_rate,
                 hidden=128, train_steps=12,
                 W_COMP=3.0, W_TOTAL=6.0, W_SMOOTH=0.10, W_BOUND=0.08, W_REF=0.03, W_MEAS=0.25):
        super().__init__()
        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate

        in_dim = N + 1 + N + N
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, N)
        )

        d, dt = k.device, k.dtype
        self.register_buffer("prev_speeds", torch.ones(N, dtype=dt, device=d) * 25.0)

        self.to(d).double()
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-3)

        self.train_steps = train_steps
        self.W_COMP, self.W_TOTAL, self.W_SMOOTH = W_COMP, W_TOTAL, W_SMOOTH
        self.W_BOUND, self.W_REF, self.W_MEAS = W_BOUND, W_REF, W_MEAS

    # ----- feature + head -----
    def _state(self, target_ratio, F_total, y):
        F_norm    = (F_total / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        flows_norm= (y / (F_total + 1e-6)).clamp(0.0, 2.0)
        prev_norm = ((self.prev_speeds - self.speed_min) /
                     (self.speed_max - self.speed_min).clamp_min(1e-6)).clamp(0.0, 1.0)
        return torch.cat([target_ratio, F_norm.view(1), flows_norm, prev_norm], dim=0)

    def _map_logits_to_speed(self, raw):
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        return self.speed_min + span * torch.sigmoid(raw)

    def forward(self, target_ratio, F_total, y):
        x = self._state(target_ratio, F_total, y)
        raw = self.net(x)
        s = self._map_logits_to_speed(raw)
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        return clip(s, self.speed_min, self.speed_max)

    # ----- online training -----
    def train_step(self, target_ratio, F_total, y, ref_speeds=None, measured_flows=None, **_):
        for _ in range(self.train_steps):
            self.opt.zero_grad()
            s = self.forward(target_ratio, F_total, y)

            flow_pred  = self.k * s
            total_pred = flow_pred.sum() + 1e-12
            comp_pred  = flow_pred / total_pred

            comp_err = ((comp_pred - target_ratio) ** 2).sum()
            total_err= (total_pred - F_total).pow(2)
            smooth   = ((s - self.prev_speeds) ** 2).sum()

            span = (self.speed_max - self.speed_min).clamp_min(1e-6)
            t = (s - self.speed_min) / span
            bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()

            ref = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if ref_speeds is not None:
                ref = ((s - ref_speeds) ** 2).sum()

            meas_term = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if measured_flows is not None:
                meas_total = measured_flows.sum() + 1e-12
                meas_term = (total_pred - meas_total).pow(2)

            loss = (self.W_COMP * comp_err + self.W_TOTAL * total_err + self.W_SMOOTH * smooth +
                    self.W_BOUND * bound_pen + self.W_REF * ref + self.W_MEAS * meas_term)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            self.prev_speeds = self.forward(target_ratio, F_total, y).detach()
        return self.prev_speeds.clone()

class GRUController(torch.nn.Module):
    """
    GRU-based policy that consumes a short history (internally buffered) and outputs
    speeds each tick. Same objective terms and safety layer as the MLP version.

    Public methods:
      - forward(tr, Ft, y) -> speeds
      - train_step(tr, Ft, y, ref_speeds=None, measured_flows=None)
      - reset_hidden()
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, slew_rate,
                 seq_len=20, hidden=128, train_steps=6,
                 W_COMP=3.0, W_TOTAL=6.0, W_SMOOTH=0.10, W_BOUND=0.08, W_REF=0.03, W_MEAS=0.25):
        super().__init__()
        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate
        self.seq_len, self.train_steps = seq_len, train_steps
        self.W_COMP, self.W_TOTAL, self.W_SMOOTH = W_COMP, W_TOTAL, W_SMOOTH
        self.W_BOUND, self.W_REF, self.W_MEAS = W_BOUND, W_REF, W_MEAS

        in_dim = N + 1 + N + N
        self.gru  = torch.nn.GRU(in_dim, hidden, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 128), torch.nn.ReLU(), torch.nn.Linear(128, N)
        )

        d, dt = k.device, k.dtype
        self.register_buffer("prev_speeds", torch.ones(N, dtype=dt, device=d) * 25.0)
        self.register_buffer("h", torch.zeros(1, 1, hidden, dtype=dt, device=d))

        self.buf_inputs, self.buf_ref, self.buf_meas = [], [], []

        self.to(d).double()
        self.opt = torch.optim.Adam(self.parameters(), lr=2.5e-3)

    def reset_hidden(self):
        """Zero the recurrent hidden state (useful after target randomization)."""
        self.h.zero_()

    def _x(self, tr, Ft, y):
        F_norm     = (Ft / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        flows_norm = (y / (Ft + 1e-6)).clamp(0.0, 2.0)
        prev_norm  = ((self.prev_speeds - self.speed_min) /
                      (self.speed_max - self.speed_min).clamp_min(1e-6)).clamp(0.0, 1.0)
        return torch.cat([tr, F_norm.view(1), flows_norm, prev_norm], dim=0)

    def _map_logits_to_speed(self, raw):
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        return self.speed_min + span * torch.sigmoid(raw)

    def forward_step(self, x, h):
        out, h1 = self.gru(x, h)
        raw = self.head(out[:, -1, :])
        s = self._map_logits_to_speed(raw.squeeze(0))
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        return clip(s, self.speed_min, self.speed_max), h1

    def forward(self, tr, Ft, y):
        x = self._x(tr, Ft, y).view(1, 1, -1)
        s, self.h = self.forward_step(x, self.h)
        return s

    def _push(self, x, ref, meas):
        """FIFO buffers for a short context window used in training."""
        self.buf_inputs.append(x.detach())
        self.buf_ref.append((ref if ref is not None else torch.zeros_like(self.prev_speeds)).detach())
        self.buf_meas.append((meas if meas is not None else torch.zeros_like(self.prev_speeds)).detach())
        if len(self.buf_inputs) > self.seq_len:
            self.buf_inputs.pop(0); self.buf_ref.pop(0); self.buf_meas.pop(0)

    def train_step(self, tr, Ft, y, ref_speeds=None, measured_flows=None, **_):
        x = self._x(tr, Ft, y)
        self._push(x, ref_speeds, measured_flows)

        # Short warm-up to stabilize gradients from tiny contexts
        if len(self.buf_inputs) < 4:
            return self._warm(tr, Ft, y, ref_speeds, measured_flows)

        X    = torch.stack(self.buf_inputs).unsqueeze(0)     # (1, T, D)
        REFS = torch.stack(self.buf_ref).unsqueeze(0)        # (1, T, N)
        MEAS = torch.stack(self.buf_meas).unsqueeze(0)       # (1, T, N)

        for _ in range(self.train_steps):
            self.opt.zero_grad()
            h0 = torch.zeros_like(self.h)
            out, _ = self.gru(X, h0)
            logits = self.head(out)                           # (1, T, N)
            s = self._map_logits_to_speed(logits)             # (1, T, N)

            flow = self.k.view(1, 1, self.N) * s
            tot  = flow.sum(2, keepdim=True) + 1e-12
            comp = flow / tot

            trT = tr.view(1, 1, self.N).expand_as(comp)
            FtT = Ft.view(1, 1, 1).expand_as(tot)

            comp_err = ((comp - trT) ** 2).sum()
            total_err= ((tot - FtT) ** 2).sum()

            s_prev = torch.roll(s, 1, dims=1)
            s_prev[:, 0, :] = self.prev_speeds.view(1, 1, -1)
            smooth = ((s - s_prev) ** 2).sum()

            span = (self.speed_max - self.speed_min).clamp_min(1e-6).view(1, 1, self.N)
            t = (s - self.speed_min.view(1, 1, self.N)) / span
            bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()

            ref = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if ref_speeds is not None:
                ref = ((s - REFS) ** 2).sum()

            meas_term = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if measured_flows is not None:
                meas_total = MEAS.sum(2, keepdim=True) + 1e-12
                meas_term = ((tot - meas_total) ** 2).sum()

            loss = (self.W_COMP * comp_err + self.W_TOTAL * total_err + self.W_SMOOTH * smooth +
                    self.W_BOUND * bound_pen + self.W_REF * ref + self.W_MEAS * meas_term)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            s = self.forward(tr, Ft, y).detach()
            self.prev_speeds = s.clone()
        return s.clone()

    def _warm(self, tr, Ft, y, ref, meas):
        """Single-step variant of the objective for the very first few ticks."""
        self.opt.zero_grad()
        h0 = torch.zeros_like(self.h)
        x = self._x(tr, Ft, y).view(1, 1, -1)
        out, _ = self.gru(x, h0)
        raw = self.head(out)
        s = self._map_logits_to_speed(raw).squeeze(0).squeeze(0)

        flow = self.k * s
        tot  = flow.sum() + 1e-12
        comp = flow / tot

        comp_err = ((comp - tr) ** 2).sum()
        total_err= (tot - Ft).pow(2)
        smooth   = ((s - self.prev_speeds) ** 2).sum()

        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        t = (s - self.speed_min) / span
        bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()

        refL = torch.tensor(0.0, dtype=s.dtype, device=s.device)
        if ref is not None:
            refL = ((s - ref) ** 2).sum()

        meas_term = torch.tensor(0.0, dtype=s.dtype, device=s.device)
        if meas is not None:
            meas_total = meas.sum() + 1e-12
            meas_term = (tot - meas_total).pow(2)

        loss = (3.0 * comp_err + 6.0 * total_err + 0.10 * smooth + 0.08 * bound_pen +
                0.03 * refL + 0.25 * meas_term)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            s = self._map_logits_to_speed(self.head(out)[:, -1, :].squeeze(0))
            s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
            s = clip(s, self.speed_min, self.speed_max)
            self.prev_speeds = s.clone()
        return s.clone()

# ─────────────────────────────────────────────────────────────────────────────
# Hybrid MPC with a learned residual dynamics model (demo-scale)
# ─────────────────────────────────────────────────────────────────────────────

class HybridMPCController(torch.nn.Module):
    """
    Short-horizon (H steps) gradient-based optimizer:
      - physics model: y_{t+1} = y_t + α (k * s_t − y_t),
      - residual model NN refines actions,
      - objective: composition & total errors + smoothness + bound barriers,
      - enforces slew & clamps on the applied step (first action of the plan).
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, *args, horizon=5, opt_iters=8, lr=0.05):
        super().__init__()
        # Accept either original (Ts, slew_rate) or (Ts, plant_alpha, slew_rate)
        if len(args) == 1:
            plant_alpha = 0.5
            slew_rate = args[0]
        elif len(args) >= 2:
            plant_alpha = float(args[0])
            slew_rate = args[1]
        else:
            raise TypeError("HybridMPCController expects slew_rate (and optional plant_alpha).")

        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate
        self.alpha_p = float(plant_alpha)
        self.H, self.opt_iters, self.lr = horizon, opt_iters, lr

        in_dim = 2 * N  # [flows, speeds]
        hid = 128
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.Tanh(),
            torch.nn.Linear(hid, hid),   torch.nn.Tanh(),
            torch.nn.Linear(hid, N)
        )

        d, dt = k.device, k.dtype
        self.register_buffer("prev_speeds", torch.ones(N, dtype=dt, device=d) * 25.0)

        self.to(d).double()
        self.opt_res = torch.optim.Adam(self.residual.parameters(), lr=2.0e-3)

    def _phys(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return y + self.alpha_p * (self.k * s - y)

    def _map_logits_to_speed(self, raw):
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        return self.speed_min + span * torch.sigmoid(raw)

    def step(self, y, tr, Ft, s_ff=None, **_):
        """
        Plan a short sequence S[0:H] with gradient descent on the rollout objective,
        then apply the first action S[0] with slew/clamp safety.
        """
        # Warm start from residual NN around previous speeds
        with torch.no_grad():
            inp = torch.cat([y, self.prev_speeds], dim=0)
            bias = self._map_logits_to_speed(self.residual(inp))
            s0 = clip(self.prev_speeds + 0.2 * (bias - self.prev_speeds), self.speed_min, self.speed_max)

        S = s0.repeat(self.H, 1)   # (H, N)
        S.requires_grad_(True)
        opt = torch.optim.SGD([S], lr=self.lr, momentum=0.0)

        for _ in range(self.opt_iters):
            opt.zero_grad()
            y_sim = y.detach().clone()
            J = torch.tensor(0.0, dtype=y.dtype, device=y.device)

            for t in range(self.H):
                s_t = S[t]
                # residual correction
                inp = torch.cat([y_sim, s_t], dim=0)
                s_t_corr = clip(s_t + 0.1 * (self._map_logits_to_speed(self.residual(inp)) - s_t),
                                self.speed_min, self.speed_max)

                y_sim = self._phys(y_sim, s_t_corr)
                tot = y_sim.sum() + 1e-12
                comp = y_sim / tot

                J = J + 3.0 * ((comp - tr) ** 2).sum() + 6.0 * (tot - Ft).pow(2)
                if t > 0:
                    J = J + 0.05 * ((S[t] - S[t - 1]) ** 2).sum()

            # soft bound barriers over the entire plan
            span = (self.speed_max - self.speed_min).clamp_min(1e-6)
            tau = (S - self.speed_min) / span
            J = J + 0.08 * (-(torch.log(tau.clamp_min(0.02)) + torch.log((1 - tau).clamp_min(0.02)))).sum()

            J.backward()
            opt.step()
            with torch.no_grad():
                S[:] = torch.clamp(S, self.speed_min, self.speed_max)

        with torch.no_grad():
            s = S[0]
            s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
            s = clip(s, self.speed_min, self.speed_max)
            self.prev_speeds = s.clone()
            return s

# ─────────────────────────────────────────────────────────────────────────────
# PID + small residual NN (safe hybrid)
# ─────────────────────────────────────────────────────────────────────────────

class PIDResidualNN:
    """
    Wrap a base PID with a small NN that proposes *delta speeds*.
    The residual is rate-limited and clamped to keep the combined action safe.
    """
    def __init__(self, pid: PIDController, speed_min, speed_max, slew_rate):
        self.pid = pid
        self.speed_min, self.speed_max = speed_min, speed_max
        self.slew = slew_rate
        N = pid.N
        hid = 128

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(2 * N + 1 + N, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, N)
        )
        self.nn.to(pid.k.device).double()
        self.opt = torch.optim.Adam(self.nn.parameters(), lr=2.0e-3)
        self.prev_out = torch.ones(N, dtype=pid.k.dtype, device=pid.k.device) * 0.0

    def _x(self, tr, Ft, y, s_pid):
        # Normalize Ft using the *PID's* (k, speed_max) scale to be consistent
        Ft_n = (Ft / _total_scale(self.pid.k, self.speed_max)).clamp(0, 2.0)
        return torch.cat([tr, Ft_n.view(1), y, s_pid / 100.0], dim=0)

    def step(self, y, tr, Ft, s_ff=None, **_):
        s_pid = self.pid.step(y, tr, Ft, (s_ff if s_ff is not None else self.pid.prev_out))
        x = self._x(tr, Ft, y, s_pid)
        delta = self.nn(x)

        # Rate-limit residual and keep within a small ±band (safety)
        s_res = slew_limit(
            self.prev_out, delta, float(self.slew),
            -10.0 * torch.ones_like(delta), 10.0 * torch.ones_like(delta)
        )
        s = clip(s_pid + s_res, self.speed_min, self.speed_max)
        self.prev_out = s_res.clone()
        return s, s_pid, s_res

    def train(self, y, tr, Ft, s_pid, target_comp=None):
        self.opt.zero_grad()
        x = self._x(tr, Ft, y, s_pid).detach()
        delta = self.nn(x)

        flow_pred = self.pid.k * clip(s_pid + delta, self.speed_min, self.speed_max)
        tot = flow_pred.sum() + 1e-12
        comp = flow_pred / tot

        comp_err = ((comp - tr) ** 2).sum()
        total_err= (tot - Ft).pow(2)
        small    = (delta ** 2).sum()

        loss = 2.5 * comp_err + 4.0 * total_err + 0.01 * small
        loss.backward()
        self.opt.step()

# ─────────────────────────────────────────────────────────────────────────────
# Transformer controller (demo scale)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerCtrl(torch.nn.Module):
    """
    Simple causal Transformer encoder that consumes a recent feature window and
    outputs speeds. Trains online with the same objective as GRU/MLP.
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, slew_rate,
                 ctx=24, d_model=128, nhead=8, nlayers=2, train_steps=5):
        super().__init__()
        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate
        self.ctx, self.train_steps = ctx, train_steps

        self.in_dim = N + 1 + N + N
        self.embed = torch.nn.Linear(self.in_dim, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation="relu"
        )
        self.enc  = torch.nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = torch.nn.Sequential(torch.nn.Linear(d_model, 128), torch.nn.ReLU(), torch.nn.Linear(128, N))

        d, dt = k.device, k.dtype
        self.register_buffer("prev_speeds", torch.ones(N, dtype=dt, device=d) * 25.0)
        self.buf = []

        self.to(d).double()
        self.opt = torch.optim.Adam(self.parameters(), lr=2.5e-3)

    def _vec(self, tr, Ft, y):
        Ft_n  = (Ft / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        prev  = ((self.prev_speeds - self.speed_min) /
                 (self.speed_max - self.speed_min).clamp_min(1e-6)).clamp(0, 1)
        flows = (y / (Ft + 1e-6)).clamp(0, 2)
        return torch.cat([tr, Ft_n.view(1), flows, prev], dim=0)

    def _map_logits_to_speed(self, raw):
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        return self.speed_min + span * torch.sigmoid(raw)

    def forward(self, tr, Ft, y):
        v = self._vec(tr, Ft, y)
        self.buf.append(v.detach())
        if len(self.buf) > self.ctx:
            self.buf.pop(0)

        X = torch.stack(self.buf, dim=0).unsqueeze(0)  # (1, T, D)
        Z = self.enc(self.embed(X))
        raw = self.head(Z[:, -1, :])
        s = self._map_logits_to_speed(raw.squeeze(0))
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        s = clip(s, self.speed_min, self.speed_max)
        self.prev_speeds = s.clone()
        return s

    def train_step(self, tr, Ft, y, *_, **__):
        if len(self.buf) < max(4, self.ctx // 3):
            return self.prev_speeds.clone()

        X = torch.stack(self.buf, dim=0).unsqueeze(0)
        self.opt.zero_grad()
        Z = self.enc(self.embed(X))
        s = self._map_logits_to_speed(self.head(Z[:, -1, :]).squeeze(0))

        flow = self.k * s
        tot  = flow.sum() + 1e-12
        comp = flow / tot

        comp_err = ((comp - tr) ** 2).sum()
        total_err= (tot - Ft).pow(2)
        smooth   = ((s - self.prev_speeds) ** 2).sum()

        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        t = (s - self.speed_min) / span
        bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()

        loss = 3.0 * comp_err + 6.0 * total_err + 0.10 * smooth + 0.08 * bound_pen
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            self.prev_speeds = s.detach()
        return self.prev_speeds.clone()

# ─────────────────────────────────────────────────────────────────────────────
# RL + Safety (actor NN with supervised loss for the live demo)
# ─────────────────────────────────────────────────────────────────────────────

class RLSafetyCtrl(torch.nn.Module):
    """
    Actor network with the same supervised objective as MLP, wrapped by
    a slew limiter and clamp. In a full RL setup, you’d swap the loss for PPO/SAC.
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, slew_rate):
        super().__init__()
        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(N + 1 + N + N, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, N)
        )

        d, dt = k.device, k.dtype
        self.register_buffer("prev_speeds", torch.ones(N, dtype=dt, device=d) * 25.0)

        self.to(d).double()
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=2.0e-3)

    def _state(self, tr, Ft, y):
        Ft_n  = (Ft / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        prev  = ((self.prev_speeds - self.speed_min) /
                 (self.speed_max - self.speed_min).clamp_min(1e-6)).clamp(0, 1)
        flows = (y / (Ft + 1e-6)).clamp(0, 2)
        return torch.cat([tr, Ft_n.view(1), flows, prev], dim=0)

    def _map_logits_to_speed(self, raw):
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        return self.speed_min + span * torch.sigmoid(raw)

    def forward(self, tr, Ft, y):
        x = self._state(tr, Ft, y)
        raw = self.actor(x)
        s = self._map_logits_to_speed(raw)
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        s = clip(s, self.speed_min, self.speed_max)
        self.prev_speeds = s.clone()
        return s

    def train_step(self, tr, Ft, y, *_, **__):
        prev = self.prev_speeds.detach().clone()
        self.opt.zero_grad()

        s = self.forward(tr, Ft, y)
        flow = self.k * s
        tot  = flow.sum() + 1e-12
        comp = flow / tot

        smooth = ((s - prev) ** 2).sum()
        loss = 3.0 * ((comp - tr) ** 2).sum() + 6.0 * (tot - Ft).pow(2) + 0.05 * smooth
        loss.backward()
        self.opt.step()
        return self.prev_speeds.clone()

# ─────────────────────────────────────────────────────────────────────────────
# PINN-flavored MLP (adds physics barriers)
# ─────────────────────────────────────────────────────────────────────────────

class PINNCtrl(MLPController):
    """Same architecture as MLP but emphasizes physics consistency with extra penalties."""
    def train_step(self, tr, Ft, y, ref_speeds=None, measured_flows=None, **_):
        for _ in range(self.train_steps):
            self.opt.zero_grad()
            s = self.forward(tr, Ft, y)

            flow = self.k * s
            tot  = flow.sum() + 1e-12
            comp = flow / tot

            comp_err = ((comp - tr) ** 2).sum()
            total_err= (tot - Ft).pow(2)
            smooth_s = ((s - self.prev_speeds) ** 2).sum()

            # physics-motivated penalties
            pos_pen = -torch.log((flow / (Ft + 1e-9)).clamp_min(1e-3)).sum()
            span = (self.speed_max - self.speed_min).clamp_min(1e-6)
            t = (s - self.speed_min) / span
            bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()

            loss = 3.2 * comp_err + 6.2 * total_err + 0.12 * smooth_s + 0.06 * pos_pen + 0.08 * bound_pen
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            self.prev_speeds = self.forward(tr, Ft, y).detach()
        return self.prev_speeds.clone()

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive hierarchical PID — safe NN tuner over inner PID gains
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveHierCtrl(CascadePIDController):
    """
    Extends CascadePID with a small tuner NN that adjusts inner-loop gains.
    Updates are done in LOG-space relative to *baseline* gains, smoothed and
    tightly clamped to avoid runaway over long runs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N = self.N
        dt = self.Kp_f.dtype
        dev = self.Kp_f.device

        # Tuner network: features -> (Δlog Kp, Δlog Ki, Δlog Kd)
        self.tuner = torch.nn.Sequential(
            torch.nn.Linear(3 * N, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 3 * N)
        )
        self.opt_tuner = torch.optim.Adam(self.tuner.parameters(), lr=1.5e-3)

        # Error-like features: [|e|_ema, |int|, |d_meas|_ema]
        self.err_hist = torch.zeros(3, N, dtype=dt, device=dev)
        self.alpha_hist = 0.2

        # Log-scale offsets (relative to baseline)
        self.log_kp = torch.zeros(N, dtype=dt, device=dev)
        self.log_ki = torch.zeros(N, dtype=dt, device=dev)
        self.log_kd = torch.zeros(N, dtype=dt, device=dev)

        # Update smoothing and clamps
        self.tune_alpha = 0.05
        self.tune_scale = 0.05
        self.log_min = torch.full((N,), np.log(0.3), dtype=dt, device=dev)
        self.log_max = torch.full((N,), np.log(3.0), dtype=dt, device=dev)

    def _update_hist(self, e_abs: torch.Tensor, e_int: torch.Tensor, d_meas: torch.Tensor) -> None:
        self.err_hist[0] = ema(self.err_hist[0], e_abs, self.alpha_hist)
        self.err_hist[1] = ema(self.err_hist[1], e_int, self.alpha_hist)
        self.err_hist[2] = ema(self.err_hist[2], d_meas.abs(), self.alpha_hist)

    def _apply_log_gains(self) -> None:
        kp = self.Kp_base * torch.exp(self.log_kp.clamp(self.log_min, self.log_max))
        ki = self.Ki_base * torch.exp(self.log_ki.clamp(self.log_min, self.log_max))
        kd = self.Kd_base * torch.exp(self.log_kd.clamp(self.log_min, self.log_max))
        # Final safety clamps (very wide)
        self.Kp_f = torch.clamp(kp, 1e-6, 1e6)
        self.Ki_f = torch.clamp(ki, 1e-6, 1e6)
        self.Kd_f = torch.clamp(kd, 1e-6, 1e6)

    def _tune(self) -> None:
        x = torch.cat([self.err_hist[0], self.err_hist[1], self.err_hist[2]], dim=0).detach()
        delta = self.tuner(x)  # (3N,)
        dKp, dKi, dKd = delta[:self.N], delta[self.N:2 * self.N], delta[2 * self.N:]

        # Small, bounded log-steps (tanh in [-1, 1])
        step_kp = self.tune_scale * torch.tanh(dKp)
        step_ki = self.tune_scale * torch.tanh(dKi)
        step_kd = self.tune_scale * torch.tanh(dKd)

        # EMA in log-space (prevents oscillatory tuning)
        self.log_kp = (1 - self.tune_alpha) * self.log_kp + self.tune_alpha * step_kp
        self.log_ki = (1 - self.tune_alpha) * self.log_ki + self.tune_alpha * step_ki
        self.log_kd = (1 - self.tune_alpha) * self.log_kd + self.tune_alpha * step_kd

        self._apply_log_gains()

    def step(self, flows_meas_filt, ratio_sp, F_total_sp, _unused=None):
        # Features for tuner
        e_abs = (ratio_sp * F_total_sp - flows_meas_filt).abs()
        d_meas = (flows_meas_filt - self.prev_meas) / max(self.Ts, 1e-9)

        # Update statistics & tune gains, then run cascade step
        self._update_hist(e_abs, self.int.abs(), d_meas)
        self._tune()
        return super().step(flows_meas_filt, ratio_sp, F_total_sp, None)

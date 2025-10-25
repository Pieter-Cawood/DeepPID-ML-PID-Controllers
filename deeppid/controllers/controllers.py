# models.py
# Controller zoo: PID, CascadePID, MLP, GRU, HybridMPC, PIDResidualNN, TransformerCtrl,
# RLSafetyCtrl, PINNCtrl, AdaptiveHierCtrl.
#
# PID & CascadePID now have IMC-style auto-tuning that adapts to both
# fast (e.g., quadcopter) and slow (e.g., tank) plants and re-tunes online.

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# ---------- generic helpers ----------
def clip(v, lo, hi): return torch.min(torch.max(v, lo), hi)

def slew_limit(prev, desired, max_delta, lo, hi):
    return clip(prev + torch.clamp(desired - prev, -max_delta, max_delta), lo, hi)

def ema(x_prev, x_new, alpha):
    return (1 - alpha) * x_prev + alpha * x_new

def _total_scale(k, speed_max):
    # Max possible “total flow/thrust” under nominal gains
    return torch.sum(k * speed_max).clamp_min(1.0)

def _tau_from_alpha(Ts, alpha):
    """
    Map discrete 1st-order step y_{t+1} = y_t + alpha (k s - y_t)
    to an equivalent continuous time constant:
      pole z = 1 - alpha  ->  tau ≈ Ts / -ln(1 - alpha)
    Fallback to Ts/alpha for tiny alpha.
    """
    a = float(alpha)
    if a <= 1e-6: return 10.0 * Ts
    if a >= 0.999999: return 0.1 * Ts
    try:
        return Ts / max(-np.log(1.0 - a), 1e-6)
    except Exception:
        return Ts / max(a, 1e-6)

def _classify_fast(N, tau_p):
    """
    Heuristic classifier: quadcopter (N<=4) or generally fast dynamics (tau_p < ~0.8 s)
    gets 'fast' tuning; otherwise 'slow'.
    """
    return (N <= 4) or (tau_p <= 0.8)

# ======================================================================
#  PID (best-in-class single loop)  — IMC auto-tuned + online refinement
# ======================================================================
class PIDController:
    def __init__(self, N, k, speed_min, speed_max, Ts, plant_alpha, slew_rate):
        self.N, self.k = int(N), k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.alpha_p, self.slew = float(Ts), float(plant_alpha), float(slew_rate)

        d = k.device; dt = k.dtype

        # Internal states
        self.int = torch.zeros(N, dtype=dt, device=d)
        self.d_y = torch.zeros(N, dtype=dt, device=d)
        self.prev_meas = torch.zeros(N, dtype=dt, device=d)
        self.prev_out  = torch.ones (N, dtype=dt, device=d) * 25.0

        # Initial IMC tuning (and slots for gains)
        self.Kp_f = torch.full((N,), 1.0, dtype=dt, device=d)  # flow-domain gain (we divide by k later)
        self.Ki_f = torch.full((N,), 0.0, dtype=dt, device=d)
        self.Kd_f = torch.full((N,), 0.0, dtype=dt, device=d)

        # Filters/antiwindup
        self.tau_d = 0.0
        self.beta  = 0.8   # setpoint weight on P (on Fsp), derivative on measurement
        self.Tt    = 0.2   # anti-windup back-calc time constant; set in retune

        # Running estimates for online retuning
        self.alpha_hat = np.clip(self.alpha_p, 1e-3, 0.99)
        self.tau_p     = _tau_from_alpha(self.Ts, self.alpha_hat)
        self.fast_mode = _classify_fast(self.N, self.tau_p)

        self._imc_retune(hard=True)

    # ----- IMC tuning (flow-domain) -----
    def _imc_retune(self, hard=False):
        """
        Compute IMC-PID params from tau_p. We produce flow-domain gains:
          Kp_f = τp/λ
          Ki_f = (τp/λ)*(1/Ti),  with Ti = τp
          Kd_f = (τp/λ)*Td,      with Td = c_D * τp
        Controller applies /k to map into speed-domain.
        """
        tau_p = max(self.tau_p, 1e-6)
        fast = self.fast_mode

        # Choose IMC filter λ, integral/derivative times and derivative filter
        lam = (0.35 * tau_p) if fast else (1.20 * tau_p)
        Ti  = tau_p
        Td  = (0.25 * tau_p) if fast else (0.10 * tau_p)
        tau_d_filt = 0.10 * tau_p if fast else 0.05 * tau_p

        # Flow-domain "Kc" == τp/λ (since speed-domain Kc = (τp/(Kλ)); we divide by k later)
        Kp_f_new = (tau_p / max(lam, 1e-6))
        Ki_f_new = Kp_f_new / max(Ti, 1e-6)
        Kd_f_new = Kp_f_new * Td

        # Anti-windup time
        Tt_new = max(0.5 * Ti, 0.05)

        if hard:
            self.Kp_f.fill_(Kp_f_new)
            self.Ki_f.fill_(Ki_f_new)
            self.Kd_f.fill_(Kd_f_new)
            self.tau_d = tau_d_filt
            self.Tt    = Tt_new
        else:
            # Smooth update to avoid jerk
            a = 0.10  # smoothing
            self.Kp_f = (1 - a) * self.Kp_f + a * Kp_f_new
            self.Ki_f = (1 - a) * self.Ki_f + a * Ki_f_new
            self.Kd_f = (1 - a) * self.Kd_f + a * Kd_f_new
            self.tau_d = (1 - a) * self.tau_d + a * tau_d_filt
            self.Tt    = (1 - a) * self.Tt    + a * Tt_new

    def _update_alpha_estimate(self, y_now):
        """
        Update alpha_hat from one-step relation:
          Δy ≈ α (k*prev_out - prev_meas)
          => α_inst = Δy / (k*u_prev - y_prev)
        Use median across channels (robust) and EMA.
        """
        y_prev = self.prev_meas
        u_prev = self.prev_out
        num = (y_now - y_prev)
        den = (self.k * u_prev - y_prev)
        den = torch.where(torch.abs(den) < 1e-6, torch.full_like(den, 1e-6), den)
        alpha_inst = torch.median(torch.clamp(num / den, 0.01, 0.99)).item()
        self.alpha_hat = 0.9 * self.alpha_hat + 0.1 * alpha_inst
        new_tau = _tau_from_alpha(self.Ts, self.alpha_hat)

        # Re-classify 'fast' if needed and smoothly retune
        fast_new = _classify_fast(self.N, new_tau)
        if fast_new != self.fast_mode:
            self.fast_mode = fast_new
        # only retune if significant change
        if abs(new_tau - self.tau_p) / max(self.tau_p, 1e-6) > 0.10:
            self.tau_p = 0.9 * self.tau_p + 0.1 * new_tau
            self._imc_retune(hard=False)

    def sync_to(self, speeds_now, flows_now):
        # Called by app at start and problem switch
        self.prev_out = speeds_now.detach().clone()
        self.prev_meas = flows_now.detach().clone()
        self.d_y.zero_()
        self.int = torch.clamp(self.int, -2000.0, 2000.0)

        # Re-seed online estimator from current step
        # (keeps previous tau_p but nudges gains immediately)
        self._update_alpha_estimate(self.prev_meas)
        self._imc_retune(hard=False)

    # ----- core pieces -----
    def _core(self, flow_sp, flow_meas, s_ff):
        e  = flow_sp - flow_meas
        eP = self.beta * flow_sp - flow_meas

        alpha = 0.0 if self.tau_d < 1e-9 else np.exp(-self.Ts / max(self.tau_d, 1e-9))
        dy_raw = (flow_meas - self.prev_meas) / self.Ts
        self.d_y = alpha * self.d_y + (1 - alpha) * dy_raw

        Kp_s = self.Kp_f / (self.k + 1e-12)
        Ki_s = self.Ki_f / (self.k + 1e-12)
        Kd_s = self.Kd_f / (self.k + 1e-12)

        u_p = Kp_s * eP
        u_d = -Kd_s * self.d_y
        u_unsat = s_ff + u_p + self.int + u_d
        return u_unsat, e, Ki_s

    def _int_aw(self, u_unsat, e, Ki_s):
        u_clamped = clip(u_unsat, self.speed_min, self.speed_max)
        sat_err = u_clamped - u_unsat
        tol = 1e-9; push = Ki_s * e
        sat_hi = (u_unsat > self.speed_max - tol); sat_lo = (u_unsat < self.speed_min + tol)
        advance = torch.ones_like(e)
        advance[(sat_hi & (push > 0)) | (sat_lo & (push < 0))] = 0.0
        self.int += advance * (Ki_s * e * self.Ts)
        self.int += (sat_err / max(self.Tt, 1e-6)) * self.Ts
        return u_clamped

    def _total_trim(self, speeds_cmd, F_total_target, flows_meas):
        total_meas = torch.sum(flows_meas) + 1e-12
        dF = (F_total_target - total_meas).item()
        if abs(dF) < 1e-3: return speeds_cmd
        b = torch.tensor(dF / self.N, dtype=speeds_cmd.dtype, device=speeds_cmd.device) / (self.k + 1e-12)
        return clip(speeds_cmd + b, self.speed_min, self.speed_max)

    # ----- public step -----
    def step(self, flows_meas_filt, target_ratio, F_total, s_ff):
        # Online re-tune from observed step
        self._update_alpha_estimate(flows_meas_filt)

        flow_sp = target_ratio * F_total
        u_unsat, e, Ki_s = self._core(flow_sp, flows_meas_filt, s_ff)
        u_cmd = self._int_aw(u_unsat, e, Ki_s)
        u_cmd = slew_limit(self.prev_out, u_cmd, self.slew, self.speed_min, self.speed_max)
        u_cmd = clip(u_cmd, self.speed_min, self.speed_max)
        u_cmd = self._total_trim(u_cmd, F_total, flows_meas_filt)
        self.prev_out = u_cmd.clone(); self.prev_meas = flows_meas_filt.clone()
        return u_cmd

# ======================================================================
#  Cascade PID (stabilized) — IMC inner loop + adaptive outer loops
# ======================================================================
class CascadePIDController:
    def __init__(self, N, k, speed_min, speed_max, Ts, plant_alpha, slew_rate):
        self.N, self.k = int(N), k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = float(Ts), float(slew_rate)
        self.alpha_p = float(plant_alpha)

        d = k.device; dt = k.dtype

        # Inner PID gains (flow-domain) — filled by IMC retune
        self.Kp_f = torch.full((N,), 1.0, dtype=dt, device=d)
        self.Ki_f = torch.full((N,), 0.0, dtype=dt, device=d)
        self.Kd_f = torch.full((N,), 0.0, dtype=dt, device=d)
        self.Kp_base = self.Kp_f.clone()
        self.Ki_base = self.Ki_f.clone()
        self.Kd_base = self.Kd_f.clone()

        # Inner integrator & derivative state
        self.int = torch.zeros(N, dtype=dt, device=d)
        self.int_limit = torch.full((N,), 2_000.0, dtype=dt, device=d)
        self.d_y = torch.zeros(N, dtype=dt, device=d)
        self.prev_meas = torch.zeros(N, dtype=dt, device=d)
        self.prev_out = torch.ones(N, dtype=dt, device=d) * 25.0

        # Filters / antiwindup
        self.beta  = 0.8
        self.tau_d = 0.0
        self.Tt    = 0.2

        # Outer loops (total & composition)
        self.gamma = 1.0
        self.gamma_int = 0.0
        # these will be set by _retune_outer
        self.Kp_tot, self.Ki_tot = 0.25, 0.12
        self.gamma_min, self.gamma_max = 0.7, 1.3
        self.gamma_int_min, self.gamma_int_max = -0.5, 0.5
        self.gamma_leak = 1e-3

        self.comp_int = torch.zeros(N, dtype=dt, device=d)
        self.Kp_comp, self.Ki_comp = 0.12, 0.06
        self.comp_leak = 1e-3
        self.comp_int_limit = 0.25  # max additive fraction per channel

        # Running plant estimates
        self.alpha_hat = np.clip(self.alpha_p, 1e-3, 0.99)
        self.tau_p     = _tau_from_alpha(self.Ts, self.alpha_hat)
        self.fast_mode = _classify_fast(self.N, self.tau_p)

        self._imc_inner_retune(hard=True)
        self._retune_outer(hard=True)

    # ----- inner IMC tuning -----
    def _imc_inner_retune(self, hard=False):
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
            self.Kp_f.fill_(Kp_f_new)
            self.Ki_f.fill_(Ki_f_new)
            self.Kd_f.fill_(Kd_f_new)
            self.Kp_base = self.Kp_f.clone()
            self.Ki_base = self.Ki_f.clone()
            self.Kd_base = self.Kd_f.clone()
            self.tau_d = tau_d_filt
            self.Tt    = Tt_new
        else:
            a = 0.10
            self.Kp_f = (1 - a) * self.Kp_f + a * Kp_f_new
            self.Ki_f = (1 - a) * self.Ki_f + a * Ki_f_new
            self.Kd_f = (1 - a) * self.Kd_f + a * Kd_f_new
            self.tau_d = (1 - a) * self.tau_d + a * tau_d_filt
            self.Tt    = (1 - a) * self.Tt    + a * Tt_new

    def _retune_outer(self, hard=False):
        # Scale outer loops by speed of inner plant: faster plant -> stronger outer PI
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

    def _update_alpha_estimate(self, y_now):
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

    def sync_to(self, speeds_now, flows_now):
        self.prev_out = speeds_now.detach().clone()
        self.prev_meas = flows_now.detach().clone()
        self.d_y.zero_()
        self.int.zero_()
        self.gamma, self.gamma_int = 1.0, 0.0
        self.comp_int.zero_()
        self._update_alpha_estimate(self.prev_meas)
        self._imc_inner_retune(hard=False)
        self._retune_outer(hard=False)

    # ----- Outer loops -----
    def _outer_total(self, F_total_sp, flows_meas):
        err = float(F_total_sp - torch.sum(flows_meas))
        denom = max(float(F_total_sp), 1e-6)
        self.gamma_int += (self.Ki_tot * err / denom - self.gamma_leak * self.gamma_int) * self.Ts
        self.gamma_int = float(np.clip(self.gamma_int, self.gamma_int_min, self.gamma_int_max))
        p = self.Kp_tot * err / denom
        self.gamma = float(np.clip(1.0 + p + self.gamma_int, self.gamma_min, self.gamma_max))

    def _outer_comp(self, ratio_sp, flows_meas):
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

    # ----- Inner loop -----
    def _inner(self, Fsp, y):
        alpha = 0.0 if self.tau_d < 1e-6 else np.exp(-self.Ts / self.tau_d)
        dy_raw = (y - self.prev_meas) / self.Ts
        self.d_y = alpha * self.d_y + (1 - alpha) * dy_raw

        Kp_s = self.Kp_f / (self.k + 1e-12)
        Ki_s = self.Ki_f / (self.k + 1e-12)
        Kd_s = self.Kd_f / (self.k + 1e-12)

        e = Fsp - y
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

    # ----- Public step -----
    def step(self, flows_meas_filt, ratio_sp, F_total_sp, _unused=None):
        # Online retune from response
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

# ---------- MLP ----------
class MLPController(torch.nn.Module):
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
        self.prev_speeds = torch.ones(N, dtype=dt, device=d) * 25.0
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-3)
        self.train_steps = train_steps
        self.W_COMP, self.W_TOTAL, self.W_SMOOTH = W_COMP, W_TOTAL, W_SMOOTH
        self.W_BOUND, self.W_REF, self.W_MEAS = W_BOUND, W_REF, W_MEAS

    def _state(self, target_ratio, F_total, y):
        F_norm = (F_total / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        flows_norm = (y / (F_total + 1e-6)).clamp(0.0, 2.0)
        prev_norm = (self.prev_speeds / 100.0).clamp(0.0, 1.0)
        return torch.cat([target_ratio, F_norm.view(1), flows_norm, prev_norm], dim=0)

    def forward(self, target_ratio, F_total, y):
        x = self._state(target_ratio, F_total, y)
        raw = self.net(x)
        speeds = 100.0 * torch.sigmoid(raw)
        speeds = slew_limit(self.prev_speeds, speeds, self.slew, self.speed_min, self.speed_max)
        return clip(speeds, self.speed_min, self.speed_max)

    def train_step(self, target_ratio, F_total, y, ref_speeds=None, measured_flows=None):
        for _ in range(self.train_steps):
            self.opt.zero_grad()
            s = self.forward(target_ratio, F_total, y)
            flow_pred = self.k * s
            total_pred = flow_pred.sum() + 1e-12
            comp_pred = flow_pred / total_pred
            comp_err = ((comp_pred - target_ratio)**2).sum()
            total_err = (total_pred - F_total).pow(2)
            smooth = ((s - self.prev_speeds)**2).sum()
            span = (self.speed_max - self.speed_min).clamp_min(1e-6)
            t = (s - self.speed_min) / span
            bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()
            ref = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if ref_speeds is not None: ref = ((s - ref_speeds)**2).sum()
            meas_term = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if measured_flows is not None:
                meas_total = measured_flows.sum() + 1e-12
                meas_term = (total_pred - meas_total).pow(2)
            loss = (self.W_COMP*comp_err + self.W_TOTAL*total_err + self.W_SMOOTH*smooth +
                    self.W_BOUND*bound_pen + self.W_REF*ref + self.W_MEAS*meas_term)
            loss.backward(); self.opt.step()
        with torch.no_grad():
            self.prev_speeds = self.forward(target_ratio, F_total, y).detach()
        return self.prev_speeds.clone()

# ---------- GRU ----------
class GRUController(torch.nn.Module):
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
        self.gru = torch.nn.GRU(in_dim, hidden, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 128), torch.nn.ReLU(), torch.nn.Linear(128, N))
        d, dt = k.device, k.dtype
        self.prev_speeds = torch.ones(N, dtype=dt, device=d) * 25.0
        self.opt = torch.optim.Adam(self.parameters(), lr=2.5e-3)
        self.buf_inputs, self.buf_ref, self.buf_meas = [], [], []
        self.h = torch.zeros(1, 1, hidden, dtype=dt, device=d)

    def reset_hidden(self): self.h.zero_()

    def _x(self, tr, Ft, y):
        F_norm = (Ft / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        flows_norm = (y / (Ft + 1e-6)).clamp(0.0, 2.0)
        prev_norm = (self.prev_speeds / 100.0).clamp(0.0, 1.0)
        return torch.cat([tr, F_norm.view(1), flows_norm, prev_norm], dim=0)

    def forward_step(self, x, h):
        out, h1 = self.gru(x, h)
        raw = self.head(out[:, -1, :]); s = 100.0 * torch.sigmoid(raw.squeeze(0))
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        return clip(s, self.speed_min, self.speed_max), h1

    def forward(self, tr, Ft, y):
        x = self._x(tr, Ft, y).view(1,1,-1)
        s, self.h = self.forward_step(x, self.h)
        return s

    def _push(self, x, ref, meas):
        self.buf_inputs.append(x.detach())
        self.buf_ref.append((ref if ref is not None else torch.zeros_like(self.prev_speeds)).detach())
        self.buf_meas.append((meas if meas is not None else torch.zeros_like(self.prev_speeds)).detach())
        if len(self.buf_inputs) > self.seq_len:
            self.buf_inputs.pop(0); self.buf_ref.pop(0); self.buf_meas.pop(0)

    def train_step(self, tr, Ft, y, ref_speeds=None, measured_flows=None):
        x = self._x(tr, Ft, y); self._push(x, ref_speeds, measured_flows)
        if len(self.buf_inputs) < 4:
            return self._warm(tr, Ft, y, ref_speeds, measured_flows)
        X = torch.stack(self.buf_inputs).unsqueeze(0)
        REFS = torch.stack(self.buf_ref).unsqueeze(0); MEAS = torch.stack(self.buf_meas).unsqueeze(0)
        for _ in range(self.train_steps):
            self.opt.zero_grad()
            h0 = torch.zeros_like(self.h); out, _ = self.gru(X, h0)
            logits = self.head(out); s = 100.0 * torch.sigmoid(logits)
            flow = self.k.view(1,1,self.N) * s; tot = flow.sum(2, keepdim=True) + 1e-12; comp = flow / tot
            trT = tr.view(1,1,self.N).expand_as(comp); FtT = Ft.view(1,1,1).expand_as(tot)
            comp_err = ((comp - trT)**2).sum(); total_err = ((tot - FtT)**2).sum()
            s_prev = torch.roll(s, 1, dims=1); s_prev[:,0,:] = self.prev_speeds.view(1,1,-1)
            smooth = ((s - s_prev)**2).sum()
            span = (self.speed_max - self.speed_min).clamp_min(1e-6).view(1,1,self.N)
            t = (s - self.speed_min.view(1,1,self.N)) / span
            bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()
            ref = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if ref_speeds is not None: ref = ((s - REFS)**2).sum()
            meas_term = torch.tensor(0.0, dtype=s.dtype, device=s.device)
            if measured_flows is not None:
                meas_total = MEAS.sum(2, keepdim=True) + 1e-12
                meas_term = ((tot - meas_total)**2).sum()
            loss = (self.W_COMP*comp_err + self.W_TOTAL*total_err + self.W_SMOOTH*smooth +
                    self.W_BOUND*bound_pen + self.W_REF*ref + self.W_MEAS*meas_term)
            loss.backward(); self.opt.step()
        with torch.no_grad():
            s = self.forward(tr, Ft, y).detach(); self.prev_speeds = s.clone()
        return s.clone()

    def _warm(self, tr, Ft, y, ref, meas):
        self.opt.zero_grad()
        h0 = torch.zeros_like(self.h)
        x = self._x(tr, Ft, y).view(1,1,-1)
        out, _ = self.gru(x, h0); raw = self.head(out); s = 100.0 * torch.sigmoid(raw).squeeze(0).squeeze(0)
        flow = self.k * s; tot = flow.sum() + 1e-12; comp = flow / tot
        comp_err = ((comp - tr)**2).sum(); total_err = (tot - Ft).pow(2)
        smooth = ((s - self.prev_speeds)**2).sum()
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        t = (s - self.speed_min) / span
        bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()
        refL = torch.tensor(0.0, dtype=s.dtype, device=s.device)
        if ref is not None: refL = ((s - ref)**2).sum()
        meas_term = torch.tensor(0.0, dtype=s.dtype, device=s.device)
        if meas is not None:
            meas_total = meas.sum() + 1e-12
            meas_term = (tot - meas_total).pow(2)
        loss = (3.0*comp_err + 6.0*total_err + 0.10*smooth + 0.08*bound_pen + 0.03*refL + 0.25*meas_term)
        loss.backward(); self.opt.step()
        with torch.no_grad():
            s = 100.0 * torch.sigmoid(self.head(out)[:, -1, :].squeeze(0))
            s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
            s = clip(s, self.speed_min, self.speed_max); self.prev_speeds = s.clone()
        return s.clone()

# ---------- Hybrid MPC with learned residual model ----------
class HybridMPCController(torch.nn.Module):
    def __init__(self, N, k, speed_min, speed_max, Ts, *args, horizon=5, opt_iters=8, lr=0.05):
        super().__init__()
        # Accept either:
        #   (Ts, slew_rate)                          <- original
        #   (Ts, plant_alpha, slew_rate)             <- app.py fallback path
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
        self.alpha_p = float(plant_alpha)  # used by _phys()
        self.H, self.opt_iters, self.lr = horizon, opt_iters, lr

        in_dim = 2*N  # [flows, speeds]
        hid = 128
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.Tanh(),
            torch.nn.Linear(hid, hid),   torch.nn.Tanh(),
            torch.nn.Linear(hid, N)
        )
        self.prev_speeds = torch.ones(N, dtype=k.dtype, device=k.device) * 25.0
        self.opt_res = torch.optim.Adam(self.residual.parameters(), lr=2.0e-3)

    def _phys(self, y, s):
        # 1-step physics: y -> y + alpha*(k*s - y)
        return y + self.alpha_p * (self.k * s - y)

# ---------- PID + NN Residual ----------
class PIDResidualNN:
    """
    Base PID (single-loop) plus a small residual NN that outputs delta speeds.
    Residual is rate-limited and small-magnitude (via penalty) to ensure stability.
    """
    def __init__(self, pid: PIDController, speed_min, speed_max, slew_rate):
        self.pid = pid
        self.speed_min, self.speed_max = speed_min, speed_max
        self.slew = slew_rate
        N = pid.N; hid = 128
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(2*N + 1 + N, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, N)
        )
        self.opt = torch.optim.Adam(self.nn.parameters(), lr=2.0e-3)
        self.prev_out = torch.ones(N, dtype=pid.k.dtype, device=pid.k.device) * 0.0

    def _x(self, tr, Ft, y, s_pid):
        # FIX: normalize Ft using the PID's k/speed_max (self.pid.k), not self.k
        Ft_n = (Ft / _total_scale(self.pid.k, self.speed_max)).clamp(0, 2.0)
        return torch.cat([tr, Ft_n.view(1), y, s_pid/100.0], dim=0)

    def step(self, y, tr, Ft, s_ff):
        s_pid = self.pid.step(y, tr, Ft, s_ff)
        # residual action
        x = self._x(tr, Ft, y, s_pid)
        delta = self.nn(x)
        s_res = slew_limit(self.prev_out, delta, float(self.slew),
                           -10.0*torch.ones_like(delta), 10.0*torch.ones_like(delta))
        s = clip(s_pid + s_res, self.speed_min, self.speed_max)
        self.prev_out = s_res.clone()
        return s, s_pid, s_res

    def train(self, y, tr, Ft, s_pid, target_comp=None):
        self.opt.zero_grad()
        x = self._x(tr, Ft, y, s_pid).detach()
        delta = self.nn(x)
        flow_pred = self.pid.k * clip(s_pid + delta, self.speed_min, self.speed_max)
        tot = flow_pred.sum() + 1e-12; comp = flow_pred / tot
        comp_err = ((comp - tr)**2).sum(); total_err = (tot - Ft).pow(2)
        small = (delta**2).sum()
        loss = 2.5*comp_err + 4.0*total_err + 0.01*small
        loss.backward(); self.opt.step()

# ---------- Transformer-based controller ----------
class TransformerCtrl(torch.nn.Module):
    """
    Causal Transformer policy: context of last T states -> next speeds.
    Trains online with same supervised loss as GRU.
    """
    def __init__(self, N, k, speed_min, speed_max, Ts, slew_rate, ctx=24, d_model=128, nhead=8, nlayers=2, train_steps=5):
        super().__init__()
        self.N, self.k = N, k
        self.speed_min, self.speed_max = speed_min, speed_max
        self.Ts, self.slew = Ts, slew_rate
        self.ctx, self.train_steps = ctx, train_steps

        self.in_dim = N + 1 + N + N
        self.embed = torch.nn.Linear(self.in_dim, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='relu')
        self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = torch.nn.Sequential(torch.nn.Linear(d_model, 128), torch.nn.ReLU(), torch.nn.Linear(128, N))
        d, dt = k.device, k.dtype
        self.prev_speeds = torch.ones(N, dtype=dt, device=d) * 25.0
        self.opt = torch.optim.Adam(self.parameters(), lr=2.5e-3)
        self.buf = []

    def _vec(self, tr, Ft, y):
        Ft_n = (Ft / _total_scale(self.k, self.speed_max)).clamp(0, 2.0)
        prev = (self.prev_speeds / 100.0).clamp(0,1)
        flows_n = (y / (Ft + 1e-6)).clamp(0,2)
        return torch.cat([tr, Ft_n.view(1), flows_n, prev], dim=0)

    def forward(self, tr, Ft, y):
        v = self._vec(tr, Ft, y); self.buf.append(v.detach())
        if len(self.buf) > self.ctx: self.buf.pop(0)
        X = torch.stack(self.buf, dim=0).unsqueeze(0)  # (1,T,D)
        Z = self.enc(self.embed(X))
        raw = self.head(Z[:, -1, :]); s = 100.0 * torch.sigmoid(raw.squeeze(0))
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        s = clip(s, self.speed_min, self.speed_max); self.prev_speeds = s.clone()
        return s

    def train_step(self, tr, Ft, y):
        if len(self.buf) < max(4, self.ctx//3): return self.prev_speeds.clone()
        X = torch.stack(self.buf, dim=0).unsqueeze(0)
        self.opt.zero_grad()
        Z = self.enc(self.embed(X))
        s = 100.0 * torch.sigmoid(self.head(Z[:, -1, :]).squeeze(0))
        flow = self.k * s; tot = flow.sum() + 1e-12; comp = flow / tot
        comp_err = ((comp - tr)**2).sum(); total_err = (tot - Ft).pow(2)
        smooth = ((s - self.prev_speeds)**2).sum()
        span = (self.speed_max - self.speed_min).clamp_min(1e-6)
        t = (s - self.speed_min) / span
        bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()
        loss = 3.0*comp_err + 6.0*total_err + 0.10*smooth + 0.08*bound_pen
        loss.backward(); self.opt.step()
        with torch.no_grad():
            self.prev_speeds = s.detach()
        return self.prev_speeds.clone()

# ---------- RL + Safety (stubbed to supervised for real-time demo) ----------
class RLSafetyCtrl(torch.nn.Module):
    """
    Actor NN wrapped by a safety layer (slew + clamp). For demo we train with supervised loss;
    you can replace the loss with PPO/SAC update externally.
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
        self.prev_speeds = torch.ones(N, dtype=dt, device=d) * 25.0
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=2.0e-3)

    def _state(self, tr, Ft, y):
        Ft_n = (Ft / _total_scale(self.k, self.speed_max)).clamp(0, 2.0); prev = (self.prev_speeds/100.0).clamp(0,1)
        flows_n = (y / (Ft + 1e-6)).clamp(0,2)
        return torch.cat([tr, Ft_n.view(1), flows_n, prev], dim=0)

    def forward(self, tr, Ft, y):
        x = self._state(tr, Ft, y)
        raw = self.actor(x); s = 100.0 * torch.sigmoid(raw)
        s = slew_limit(self.prev_speeds, s, self.slew, self.speed_min, self.speed_max)
        s = clip(s, self.speed_min, self.speed_max); self.prev_speeds = s.clone()
        return s

    def train_step(self, tr, Ft, y):
        self.opt.zero_grad()
        s = self.forward(tr, Ft, y)
        flow = self.k * s; tot = flow.sum() + 1e-12; comp = flow / tot
        loss = 3.0*((comp - tr)**2).sum() + 6.0*(tot - Ft).pow(2) + 0.05*((s - self.prev_speeds)**2).sum()
        loss.backward(); self.opt.step()
        return self.prev_speeds.clone()

# ---------- Physics-Informed NN ----------
class PINNCtrl(MLPController):
    """
    Same architecture as MLP but adds stronger physics terms:
    - mass conservation already inherent; we add per-channel non-negativity soft barrier
    - optional equality on sum of ratios (implicitly holds) and smoothness in speeds/flows
    """
    def train_step(self, tr, Ft, y, ref_speeds=None, measured_flows=None):
        for _ in range(self.train_steps):
            self.opt.zero_grad()
            s = self.forward(tr, Ft, y)
            flow = self.k * s; tot = flow.sum() + 1e-12; comp = flow / tot
            comp_err = ((comp - tr)**2).sum(); total_err = (tot - Ft).pow(2)
            smooth_s = ((s - self.prev_speeds)**2).sum()
            # physics barriers
            pos_pen = -torch.log((flow/ (Ft + 1e-9)).clamp_min(1e-3)).sum()
            span = (self.speed_max - self.speed_min).clamp_min(1e-6); t = (s - self.speed_min) / span
            bound_pen = -(torch.log(t.clamp_min(0.02)) + torch.log((1 - t).clamp_min(0.02))).sum()
            loss = 3.2*comp_err + 6.2*total_err + 0.12*smooth_s + 0.06*pos_pen + 0.08*bound_pen
            loss.backward(); self.opt.step()
        with torch.no_grad():
            self.prev_speeds = self.forward(tr, Ft, y).detach()
        return self.prev_speeds.clone()

# ---------- Adaptive Hierarchical (safe gain tuner) ----------
class AdaptiveHierCtrl(CascadePIDController):
    """
    Cascade PID with a tiny tuner NN that adjusts inner-loop gains based on recent signals.
    Uses log-scale, smoothed, and tightly clamped updates relative to the *baseline* gains
    to avoid runaway growth/decay over long runs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N = self.N
        dt = self.Kp_f.dtype
        dev = self.Kp_f.device

        # Tuner network
        self.tuner = torch.nn.Sequential(
            torch.nn.Linear(3 * N, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 3 * N)  # outputs per-channel dKp, dKi, dKd (unscaled)
        )
        self.opt_tuner = torch.optim.Adam(self.tuner.parameters(), lr=1.5e-3)

        # Error history features: [|e|, e_int, |d_meas|]
        self.err_hist = torch.zeros(3, N, dtype=dt, device=dev)
        self.alpha_hist = 0.2

        # Log-scale gain offsets relative to baseline (start at 0)
        self.log_kp = torch.zeros(N, dtype=dt, device=dev)
        self.log_ki = torch.zeros(N, dtype=dt, device=dev)
        self.log_kd = torch.zeros(N, dtype=dt, device=dev)

        # Update smoothing and clamps
        self.tune_alpha = 0.05       # smoothing for log updates
        self.tune_scale = 0.05       # overall mapping from tuner output -> log step
        self.log_min = torch.full((N,), np.log(0.3), dtype=dt, device=dev)  # -1.204...
        self.log_max = torch.full((N,), np.log(3.0), dtype=dt, device=dev)  # +1.098...

    def _update_hist(self, e_abs, e_int, d_meas):
        # Exponential moving averages of magnitudes (keep them bounded & smooth)
        self.err_hist[0] = ema(self.err_hist[0], e_abs, self.alpha_hist)
        self.err_hist[1] = ema(self.err_hist[1], e_int, self.alpha_hist)
        self.err_hist[2] = ema(self.err_hist[2], d_meas.abs(), self.alpha_hist)

    def _apply_log_gains(self):
        # Gains = baseline * exp(log_offset), with clamping
        kp = self.Kp_base * torch.exp(self.log_kp.clamp(self.log_min, self.log_max))
        ki = self.Ki_base * torch.exp(self.log_ki.clamp(self.log_min, self.log_max))
        kd = self.Kd_base * torch.exp(self.log_kd.clamp(self.log_min, self.log_max))
        # Keep reasonable absolute bounds as a last resort
        self.Kp_f = torch.clamp(kp, 1e-6, 1e6)
        self.Ki_f = torch.clamp(ki, 1e-6, 1e6)
        self.Kd_f = torch.clamp(kd, 1e-6, 1e6)

    def _tune(self):
        # Features: [|e|_ema, |int|, |d_meas|_ema] -> deltas
        x = torch.cat([self.err_hist[0], self.err_hist[1], self.err_hist[2]], dim=0).detach()
        delta = self.tuner(x)  # (3N,)
        dKp, dKi, dKd = delta[:self.N], delta[self.N:2*self.N], delta[2*self.N:]

        # Map to small, smooth log-steps; tanh bounds to [-1,1]
        step_kp = self.tune_scale * torch.tanh(dKp)
        step_ki = self.tune_scale * torch.tanh(dKi)
        step_kd = self.tune_scale * torch.tanh(dKd)

        # EMA in log space
        self.log_kp = (1 - self.tune_alpha) * self.log_kp + self.tune_alpha * step_kp
        self.log_ki = (1 - self.tune_alpha) * self.log_ki + self.tune_alpha * step_ki
        self.log_kd = (1 - self.tune_alpha) * self.log_kd + self.tune_alpha * step_kd

        # Clamp and apply to actual gains
        self._apply_log_gains()

    def step(self, flows_meas_filt, ratio_sp, F_total_sp, _unused=None):
        # Prepare signals for tuner
        e_abs = (ratio_sp * F_total_sp - flows_meas_filt).abs()
        d_meas = (flows_meas_filt - self.prev_meas) / max(self.Ts, 1e-9)

        # Update history & tune gains
        self._update_hist(e_abs, self.int.abs(), d_meas)
        self._tune()

        # Run the stabilized cascade step
        return super().step(flows_meas_filt, ratio_sp, F_total_sp, None)

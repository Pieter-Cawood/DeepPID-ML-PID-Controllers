# ─────────────────────────────────────────────────────────────────────────────
# File: problems.py
# Defines interchangeable plant "problems" to validate controllers.
#
# Includes five 5-flow mixer scenarios (N=5) plus a 4-rotor quadcopter:
#  1) SingleTankMixerProblem              — baseline first-order lag + noise.
#  2) DeadtimeVaryingGainsProblem         — dead-time, actuator smoothing, drift.
#  3) NonlinearBackpressureProblem        — backpressure coupling + soft sat.
#  4) TwoTankCascadeProblem               — two-stage transport/mixing.
#  5) FaultySensorsActuatorsProblem       — stiction, outages, spikes.
#  6) QuadcopterAltYawProblem (N=4)       — altitude (total) + yaw (composition).
#
# Design notes:
# • All problems expose the same API: baseline_allocation(), comp_from_speeds(), step().
# • step() must return the measured per-source flows, which the app uses to compute
#   composition consistently across problems.
# • Mixers use "flow (L/min)" vocabulary; the quad uses "thrust (N)" but keeps
#   the same interface so controllers are unchanged.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch


def _clip(v, lo, hi):
    return torch.min(torch.max(v, lo), hi)


@dataclass
class BaseProblem:
    Ts: float
    N: int = 5

    def __post_init__(self):
        # Shared defaults (can be overridden in subclasses if needed)
        self.labels = [f"Source {i+1}" for i in range(self.N)]
        # Nominal linear gains from speed(%) to flow (L/min).
        self.k_coeff = torch.tensor([0.95, 0.75, 0.85, 0.60, 0.50], dtype=torch.float64)
        self.speed_min = torch.zeros(self.N, dtype=torch.float64)
        self.speed_max = torch.ones(self.N, dtype=torch.float64) * 100.0
        # Nominal target composition and total flow used by the app as defaults.
        self.default_target_ratio = torch.tensor([0.35, 0.25, 0.20, 0.12, 0.08], dtype=torch.float64)
        self.default_total_flow = 120.0
        # Max per-step change applied by the UI before entering the plant "actuator".
        self.slew_rate = 3.0
        # Optional plant lag constant some controllers read
        self.alpha = 0.5
        self._init_states()

    # --- Required API ---
    def _init_states(self):
        raise NotImplementedError

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        """Advance plant by one Ts. Returns filtered measured flows (per source)."""
        raise NotImplementedError

    def baseline_allocation(self, target_ratio: torch.Tensor, F_total: torch.Tensor) -> torch.Tensor:
        """Naive open-loop allocation used as a baseline and fallback."""
        speeds = (target_ratio * F_total) / (self.k_coeff + 1e-12)
        speeds = _clip(speeds, self.speed_min, self.speed_max)
        for _ in range(3):  # few passes to re-normalize total flow
            flows = self.k_coeff * speeds
            total = flows.sum()
            if total <= 1e-9:
                break
            scale = F_total / total
            speeds = _clip(speeds * scale, self.speed_min, self.speed_max)
        return speeds.detach()

    def comp_from_speeds(self, speeds: torch.Tensor) -> torch.Tensor:
        """Composition implied by speeds under *nominal* (k_coeff) assumptions."""
        speeds = speeds.clone()
        bad = ~torch.isfinite(speeds)
        if bad.any():
            speeds[bad] = 0.0
        flows = self.k_coeff * speeds
        total = flows.sum()
        if (not torch.isfinite(total)) or (float(total.item()) <= 1e-12):
            return torch.full_like(flows, 1.0 / self.N)
        comp = flows / (total + 1e-12)
        comp = torch.where(torch.isfinite(comp), comp, torch.full_like(comp, 1.0 / self.N))
        return comp


# ─────────────────────────────────────────────────────────────────────────────
# 1) Baseline: Single-tank with first-order lag, random fluctuation and noise
# ─────────────────────────────────────────────────────────────────────────────
class SingleTankMixerProblem(BaseProblem):
    def _init_states(self):
        self.alpha = 0.5       # plant smoothing on flows (higher -> faster response)
        self.NOISE_STD = 1.5   # additive sensor noise (L/min)
        self.RAPID_FLUCT = 0.12  # multiplicative short-term fluctuation
        self.MEAS_EWA = 0.3    # exponential moving average for measurement
        self.flows_meas = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        flows_pred = self.k_coeff * speeds_cmd
        noise = torch.randn(self.N, dtype=torch.float64) * self.NOISE_STD
        fluct = (torch.rand(self.N, dtype=torch.float64) - 0.5) * 2.0 * self.RAPID_FLUCT * flows_pred
        self.flows_meas = self.flows_meas + self.alpha * (flows_pred - self.flows_meas) + noise + fluct
        self.flows_meas = torch.clamp(self.flows_meas, min=0.0)
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * self.flows_meas
        return self.flows_meas_filt.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 2) Dead-time + drifting gains + actuator smoothing
# ─────────────────────────────────────────────────────────────────────────────
class DeadtimeVaryingGainsProblem(BaseProblem):
    def _init_states(self):
        self.actuator_alpha = 0.25  # actuator first-order smoothing
        self.NOISE_STD = 1.2
        self.MEAS_EWA = 0.25
        self.time = 0.0
        self.omega = 2.0 * np.pi / 120.0  # slow drift ~ period 120 steps
        self.drift_amp = 0.15

        # Per-source dead-time (steps).
        self.dead_steps = [2, 4, 6, 3, 5]
        self._fifo = [list([0.0] * d) for d in self.dead_steps]
        self._act = torch.zeros(self.N, dtype=torch.float64)  # actuator internal state (post-smoothing)

        self.flows_meas = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

        # Random per-source drift phase
        self._phi = torch.rand(self.N, dtype=torch.float64) * 2.0 * np.pi

    def _drifting_k(self):
        t = torch.tensor(self.time, dtype=torch.float64)
        k = self.k_coeff * (1.0 + self.drift_amp * torch.sin(self.omega * t + self._phi))
        k = torch.clamp(k, self.k_coeff * 0.6, self.k_coeff * 1.4)
        return k

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        # Actuator smoothing (pre-deadtime)
        self._act = self._act + self.actuator_alpha * (speeds_cmd - self._act)

        # Dead-time FIFO per source
        delayed = []
        for i in range(self.N):
            fifo = self._fifo[i]
            fifo.append(float(self._act[i].item()))
            out = fifo.pop(0) if len(fifo) > 0 else float(self._act[i].item())
            delayed.append(out)
        delayed = torch.tensor(delayed, dtype=torch.float64)

        # Time-varying gains
        k_now = self._drifting_k()

        # True flows
        flows_true = torch.clamp(k_now * delayed, min=0.0)

        # Sensor noise + measurement filter
        noise = torch.randn(self.N, dtype=torch.float64) * self.NOISE_STD
        self.flows_meas = torch.clamp(flows_true + noise, min=0.0)
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * self.flows_meas

        self.time += 1.0
        return self.flows_meas_filt.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 3) Nonlinearity & coupling: Backpressure + soft saturation
# ─────────────────────────────────────────────────────────────────────────────
class NonlinearBackpressureProblem(BaseProblem):
    def _init_states(self):
        self.beta = 0.01          # backpressure strength (coupling via total flow)
        self.sat_scale = 0.04     # governs softness of saturation (bigger -> harder saturation)
        self.alpha = 0.45         # plant intrinsic lag on flows
        self.NOISE_STD = 1.0
        self.MEAS_EWA = 0.28
        self.flows_meas = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

    def _soft_sat(self, u: torch.Tensor) -> torch.Tensor:
        # Soft valve curve: ~linear near zero, saturates near max speed.
        x = (u - 50.0) * self.sat_scale  # center around 50% for symmetry
        return 50.0 * (torch.tanh(x) + 1.0)  # maps to roughly [0,100]

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        # Apply valve nonlinearity
        sat_speed = self._soft_sat(_clip(speeds_cmd, self.speed_min, self.speed_max))

        # Backpressure coupling
        lin_flows = self.k_coeff * sat_speed
        F_total = torch.sum(lin_flows) + 1e-12
        k_eff_scale = 1.0 / (1.0 + self.beta * F_total)
        flows_pred = k_eff_scale * lin_flows

        # Plant lag + noise
        noise = torch.randn(self.N, dtype=torch.float64) * self.NOISE_STD
        self.flows_meas = self.flows_meas + self.alpha * (flows_pred - self.flows_meas) + noise
        self.flows_meas = torch.clamp(self.flows_meas, min=0.0)

        # Measurement filter
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * self.flows_meas
        return self.flows_meas_filt.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 4) Two-tank cascade (2nd-order dynamics)
# ─────────────────────────────────────────────────────────────────────────────
class TwoTankCascadeProblem(BaseProblem):
    def _init_states(self):
        self.alpha1 = 0.35  # Tank 1 time constant (larger -> faster)
        self.alpha2 = 0.25  # Tank 2 time constant
        self.recycle = 0.06 # fraction of previous outflow recycled to Tank 1
        self.NOISE_STD = 0.9
        self.MEAS_EWA = 0.30
        self.tank1 = torch.zeros(self.N, dtype=torch.float64)
        self.tank2 = torch.zeros(self.N, dtype=torch.float64)
        self.prev_out = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        inflow = torch.clamp(self.k_coeff * speeds_cmd, min=0.0)
        # Recycle a bit of the last outflow back to Tank 1 (per-source)
        inflow_re = inflow + self.recycle * self.prev_out

        # Tank 1 and Tank 2 lags
        self.tank1 = self.tank1 + self.alpha1 * (inflow_re - self.tank1)
        self.tank2 = self.tank2 + self.alpha2 * (self.tank1 - self.tank2)

        out = torch.clamp(self.tank2, min=0.0)
        self.prev_out = out.detach().clone()

        # Additive sensor noise + filter
        noise = torch.randn(self.N, dtype=torch.float64) * self.NOISE_STD
        meas = torch.clamp(out + noise, min=0.0)
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * meas
        return self.flows_meas_filt.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 5) Fault robustness: stiction, outages, spikes, dropouts
# ─────────────────────────────────────────────────────────────────────────────
class FaultySensorsActuatorsProblem(BaseProblem):
    def _init_states(self):
        # Actuator stiction parameters
        self.p_stick = 0.08    # probability to hold the previous actuator value
        self.actuator_alpha = 0.4  # when not stuck, smooth toward command
        self._act = torch.zeros(self.N, dtype=torch.float64)

        # Outage parameters (per source)
        self.p_outage_start = 0.01
        self.outage_len_steps = 15
        self._outage_remaining = torch.zeros(self.N, dtype=torch.int64)

        # Sensor issues
        self.p_dropout = 0.03  # replace with last filtered value
        self.p_spike = 0.01    # add a large positive spike
        self.SPIKE_STD = 8.0

        # Base plant
        self.alpha = 0.5
        self.NOISE_STD = 1.1
        self.MEAS_EWA = 0.32
        self.flows_meas = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

    def _apply_outages(self, flows: torch.Tensor) -> torch.Tensor:
        # Start new outages randomly where none are active
        start_mask = (self._outage_remaining == 0) & (torch.rand(self.N) < self.p_outage_start)
        self._outage_remaining[start_mask] = self.outage_len_steps
        # Decrement active outages
        active = self._outage_remaining > 0
        self._outage_remaining[active] -= 1
        # Force affected flows to a small leakage value
        flows = flows.clone()
        flows[active] = torch.minimum(flows[active], torch.tensor(0.8, dtype=torch.float64))
        return flows

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        # Actuator stiction & smoothing
        hold = torch.rand(self.N) < self.p_stick
        self._act[hold] = self._act[hold]  # hold
        self._act[~hold] = self._act[~hold] + self.actuator_alpha * (speeds_cmd[~hold] - self._act[~hold])

        # Nominal linear plant with lag
        flows_pred = self.k_coeff * self._act
        flows_pred = self._apply_outages(flows_pred)
        self.flows_meas = self.flows_meas + self.alpha * (flows_pred - self.flows_meas)

        # Sensor noise
        noise = torch.randn(self.N, dtype=torch.float64) * self.NOISE_STD
        meas = torch.clamp(self.flows_meas + noise, min=0.0)

        # Sensor dropouts & spikes
        dropout_mask = (torch.rand(self.N) < self.p_dropout)
        spike_mask = (torch.rand(self.N) < self.p_spike)
        meas[dropout_mask] = self.flows_meas_filt[dropout_mask]
        meas[spike_mask] = meas[spike_mask] + torch.abs(torch.randn(torch.sum(spike_mask), dtype=torch.float64)) * self.SPIKE_STD

        # Final measurement filter
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * meas
        return self.flows_meas_filt.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 6) Quadcopter (Altitude + Yaw) — N = 4
#     • Interprets "total flow" as total thrust [N] and "composition ratio"
#       as per-rotor thrust distribution (used by yaw).
#     • Keeps the same controller interface; optional autopilot maps (z, ψ) SPs
#       into (ratio, total) targets each tick.
# ─────────────────────────────────────────────────────────────────────────────
class QuadcopterAltYawProblem:
    """
    4-rotor quad. State: z (altitude), psi (yaw). Roll/pitch are hidden in disturbance.
    Per-rotor "flows" are thrusts (N). Speeds are 0..100 (%). Simple 1st-order motor model.

    Yaw convention (X-config):
      idx:   0      1      2      3
             FL(+)  FR(-)  RR(+)  RL(-)
      "(+)" contributes +yaw torque when increased (CW vs CCW grouping).
    """

    def __init__(self, Ts: float):
        self.Ts = float(Ts)

        # IO sizes & presentation
        self.N = 4
        self.labels = ["FL(+)", "FR(-)", "RR(+)", "RL(-)"]
        self.output_name, self.output_unit, self.entity_title = "Thrust", "N", "Rotor"

        # Speed→thrust map: thrust_i [N] ≈ k_i * speed_i[%]
        # Chosen so hover ~25% per rotor at m=1kg (9.81N total).
        self.k_coeff = torch.ones(self.N, dtype=torch.float64) * 0.10  # N per %speed

        # Actuator limits and slew (percent per control tick)
        self.speed_min = torch.zeros(self.N, dtype=torch.float64)
        self.speed_max = torch.ones(self.N, dtype=torch.float64) * 100.0
        self.slew_rate = 8.0  # % per step
        self.alpha = 0.5      # motor 1st-order response factor used by controllers

        # Physical params (toy scale, stable & responsive)
        self.m = 1.0
        self.g = 9.81
        self.Iz = 0.02
        self.c_tau = 0.015    # yaw torque coeff (N·m per N thrust differential)
        self.k_drag_z = 0.15  # vertical damping
        self.k_drag_yaw = 0.10
        self.noise_std = 0.02 # thrust noise (N)
        self.MEAS_EWA = 0.35  # measurement EWA for displayed per-rotor thrust

        # States
        self._thrust = torch.zeros(self.N, dtype=torch.float64)  # filtered motor thrust (pre-meas filter)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)  # returned "measured flows"
        self.z = 0.0; self.zd = 0.0
        self.psi = 0.0; self.psid = 0.0

        # Autopilot integrators (optional)
        self.z_int = 0.0
        self.psi_int = 0.0

        # App defaults (hover & equal split)
        self.default_target_ratio = torch.ones(self.N, dtype=torch.float64) / self.N
        self.default_total_flow  = torch.tensor(self.m * self.g, dtype=torch.float64)  # ≈ 9.81 N

    # --- Baseline allocation & composition (same API as mixers) -----------------
    def baseline_allocation(self, target_ratio: torch.Tensor, total_thrust: torch.Tensor) -> torch.Tensor:
        """Ideal speeds (percent) for given thrust split & total."""
        k = self.k_coeff
        Ft = total_thrust.double()
        r = torch.clamp(target_ratio.double(), min=1e-12)
        r = r / torch.sum(r)
        speeds = Ft * r / (k + 1e-12)
        return torch.clamp(speeds, self.speed_min, self.speed_max)

    def comp_from_speeds(self, speeds: torch.Tensor) -> torch.Tensor:
        """Predicted composition from instantaneous speeds (used for MAE table)."""
        y = (self.k_coeff * speeds.double()).clamp(min=0.0)
        s = torch.sum(y) + 1e-12
        return y / s

    # --- Plant step --------------------------------------------------------------
    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        """
        Apply speeds → update rotor thrust, then altitude & yaw.
        Returns filtered per-rotor thrust (N) as the "measured flows".
        """
        speeds = speeds_cmd.double()

        # Motor/thrust dynamics (1st order + noise)
        y_target = (self.k_coeff * speeds).clamp(min=0.0)
        self._thrust = self._thrust + self.alpha * (y_target - self._thrust)
        if self.noise_std > 0:
            self._thrust = torch.clamp(self._thrust + torch.randn_like(self._thrust) * self.noise_std, min=0.0)

        # Rigid-body vertical + yaw (very simplified)
        T = float(torch.sum(self._thrust).item())
        yaw_plus  = float((self._thrust[0] + self._thrust[2]).item())
        yaw_minus = float((self._thrust[1] + self._thrust[3]).item())
        tau_z = self.c_tau * (yaw_plus - yaw_minus)

        # z: m * zdd = T - m g - damping
        zdd = (T - self.m * self.g) / self.m - self.k_drag_z * self.zd
        self.zd += self.Ts * zdd
        self.z  += self.Ts * self.zd

        # psi: Iz * psidd = tau_z - damping
        psidd = (tau_z / self.Iz) - self.k_drag_yaw * self.psid
        self.psid += self.Ts * psidd
        self.psi  = self._wrap_pi(self.psi + self.Ts * self.psid)

        # Measurement filter (what the app "reads")
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * self._thrust
        return self.flows_meas_filt.clone()

    # --- Optional autopilot: set (ratio, total) from (z_sp, yaw_sp_deg) ----------
    def autopilot(self, z_sp_m: float, yaw_sp_deg: float):
        """
        Tiny PID-ish layer that outputs (ratio, total_thrust) for the controllers to track.
        Keeps interface identical: controllers still see total & composition tasks.
        """
        # Gains tuned for the toy model
        Kp_z, Ki_z, Kd_z = 3.0, 0.8, 1.6
        Kp_psi, Ki_psi, Kd_psi = 1.2, 0.0, 0.3
        mix_gain = 0.030  # scale tau->ratio mix

        # Altitude PID -> total thrust
        ez  = float(z_sp_m - self.z)
        ezd = float(0.0 - self.zd)
        self.z_int = np.clip(self.z_int + ez * self.Ts, -2.0, 2.0)
        u_z = Kp_z * ez + Ki_z * self.z_int + Kd_z * ezd
        Ft  = np.clip(self.m * self.g + u_z, 0.0, float(4.0 * (self.k_coeff.max() * 100.0)))

        # Yaw PID -> differential mix (keep total constant)
        yaw_sp = np.deg2rad(float(yaw_sp_deg))
        epsi  = self._wrap_pi(yaw_sp - self.psi)
        epsid = float(0.0 - self.psid)
        self.psi_int = np.clip(self.psi_int + epsi * self.Ts, -1.0, 1.0)
        tau_cmd = Kp_psi * epsi + Ki_psi * self.psi_int + Kd_psi * epsid

        # Convert tau -> composition differential d (positive -> add to +group)
        d = float(np.clip(mix_gain * tau_cmd, -0.20, 0.20))

        base = torch.ones(self.N, dtype=torch.float64) / self.N
        r = base.clone()
        r[0] += d; r[2] += d  # (+) group
        r[1] -= d; r[3] -= d  # (–) group
        r = torch.clamp(r, min=1e-6)
        r = r / torch.sum(r)

        return r, torch.tensor(Ft, dtype=torch.float64)

    @staticmethod
    def _wrap_pi(a):
        # [-pi, pi)
        return float((a + np.pi) % (2 * np.pi) - np.pi)

class QuadcopterGyroWindProblem:
    """
    Harder 4-rotor: gyro bias+noise, wind/gust vertical disturbance, slight motor asymmetry.
    Returns per-rotor thrust (N) as "measured flows" like other problems.
    """
    def __init__(self, Ts: float):
        self.Ts = float(Ts)
        self.N = 4
        self.labels = ["FL(+)", "FR(-)", "RR(+)", "RL(-)"]
        self.output_name, self.output_unit, self.entity_title = "Thrust", "N", "Rotor"

        # Slightly asymmetric motors
        base = 0.10
        self.k_coeff = torch.tensor([1.00, 0.96, 1.03, 0.98], dtype=torch.float64) * base

        self.speed_min = torch.zeros(self.N, dtype=torch.float64)
        self.speed_max = torch.ones(self.N, dtype=torch.float64) * 100.0
        self.slew_rate = 8.0
        self.alpha = 0.55   # motor response (faster than baseline)

        # Physical parameters
        self.m = 1.1
        self.g = 9.81
        self.Iz = 0.025
        self.c_tau = 0.015
        self.k_drag_z = 0.20
        self.k_drag_yaw = 0.12

        # Wind/gusts (vertical)
        self.wind_bias = 0.0
        self.wind_rw_std = 0.06  # random-walk on wind (N per step)

        # Gyro (yaw rate) noise & bias drift
        self.gyro_bias = 0.0
        self.gyro_rw_std = 0.002   # rad/s bias drift
        self.gyro_noise_std = 0.02 # rad/s white noise

        # Measurement filter
        self.MEAS_EWA = 0.35
        self.noise_std = 0.03

        # States
        self._thrust = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)
        self.z = 0.0; self.zd = 0.0
        self.psi = 0.0; self.psid = 0.0

        # Defaults (hover & equal split)
        self.default_target_ratio = torch.ones(self.N, dtype=torch.float64) / self.N
        self.default_total_flow = torch.tensor(self.m * self.g, dtype=torch.float64)

    def baseline_allocation(self, target_ratio, total_thrust):
        r = torch.clamp(target_ratio.double(), min=1e-12); r = r / torch.sum(r)
        speeds = total_thrust.double() * r / (self.k_coeff + 1e-12)
        return torch.clamp(speeds, self.speed_min, self.speed_max)

    def comp_from_speeds(self, speeds):
        y = (self.k_coeff * speeds.double()).clamp(min=0.0)
        return y / (y.sum() + 1e-12)

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        speeds = speeds_cmd.double()

        # Motor thrust (1st order) + sensor noise
        y_tgt = (self.k_coeff * speeds).clamp(min=0.0)
        self._thrust = self._thrust + self.alpha * (y_tgt - self._thrust)
        if self.noise_std > 0:
            self._thrust = torch.clamp(self._thrust + torch.randn_like(self._thrust) * self.noise_std, min=0.0)

        # Wind random walk (vertical force disturbance)
        self.wind_bias += np.random.randn() * self.wind_rw_std

        # Torques
        T = float(self._thrust.sum().item())
        tau_z = self.c_tau * float((self._thrust[0] + self._thrust[2] - self._thrust[1] - self._thrust[3]).item())

        # Vertical dynamics with wind
        zdd = (T - self.m * self.g + self.wind_bias) / self.m - self.k_drag_z * self.zd
        self.zd += self.Ts * zdd
        self.z  += self.Ts * self.zd

        # Yaw with gyro bias+noise (internal; returned flows don’t include gyro directly)
        self.gyro_bias += np.random.randn() * self.gyro_rw_std
        psidd = (tau_z / self.Iz) - self.k_drag_yaw * self.psid
        self.psid += self.Ts * psidd
        gyro_meas = self.psid + self.gyro_bias + np.random.randn() * self.gyro_noise_std  # available to autopilot if used
        self.psi = float((self.psi + self.Ts * gyro_meas + np.pi) % (2*np.pi) - np.pi)

        # Per-rotor measurement filtering (what app reads)
        self.flows_meas_filt = (1 - self.MEAS_EWA) * self.flows_meas_filt + self.MEAS_EWA * self._thrust
        return self.flows_meas_filt.clone()

class QuadcopterCoupledAxesProblem:
    """
    Adds roll & pitch dynamics; tilt reduces effective vertical thrust.
    Sensor latency on per-rotor measurements; yaw still via (+/-) groups.
    """
    def __init__(self, Ts: float):
        self.Ts = float(Ts)
        self.N = 4
        self.labels = ["FL(+)", "FR(-)", "RR(+)", "RL(-)"]
        self.output_name, self.output_unit, self.entity_title = "Thrust", "N", "Rotor"

        # Slight asymmetry & tighter actuator limits
        self.k_coeff = torch.tensor([0.102, 0.098, 0.105, 0.100], dtype=torch.float64)
        self.speed_min = torch.zeros(self.N, dtype=torch.float64)
        self.speed_max = torch.ones(self.N, dtype=torch.float64) * 100.0
        self.slew_rate = 7.0
        self.alpha = 0.5

        # Mass/inertia/geometry (toy numbers)
        self.m, self.g = 1.2, 9.81
        self.Ix, self.Iy, self.Iz = 0.018, 0.020, 0.022
        self.L = 0.18       # arm length (m)
        self.c_tau = 0.015  # yaw torque per N differential

        # Damping
        self.k_drag_z   = 0.15
        self.k_drag_rp  = 0.20
        self.k_drag_yaw = 0.12

        # Sensor latency (FIFO) on rotor thrusts
        self.lat_steps = 3
        self._fifo = [list([0.0]*self.lat_steps) for _ in range(self.N)]

        self.MEAS_EWA = 0.35
        self.noise_std = 0.03

        # States
        self._thrust = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

        self.z = 0.0; self.zd = 0.0
        self.phi = 0.0; self.phid = 0.0
        self.theta = 0.0; self.thetad = 0.0
        self.psi = 0.0; self.psid = 0.0

        self.default_target_ratio = torch.ones(self.N, dtype=torch.float64) / self.N
        self.default_total_flow = torch.tensor(self.m * self.g, dtype=torch.float64)

    def baseline_allocation(self, tr, Ft):
        tr = torch.clamp(tr.double(), min=1e-12); tr = tr / tr.sum()
        speeds = Ft.double() * tr / (self.k_coeff + 1e-12)
        return torch.clamp(speeds, self.speed_min, self.speed_max)

    def comp_from_speeds(self, s):
        y = (self.k_coeff * s.double()).clamp(min=0.0)
        return y / (y.sum() + 1e-12)

    def _mix_torques(self, thrust):
        # Body torques from rotor thrusts (X config)
        # roll ~ (left - right) * L, pitch ~ (front - rear) * L
        FL, FR, RR, RL = thrust
        tau_roll  = self.L * ((FL + RL) - (FR + RR))
        tau_pitch = self.L * ((FL + FR) - (RR + RL))
        tau_yaw   = self.c_tau * ((FL + RR) - (FR + RL))
        return float(tau_roll), float(tau_pitch), float(tau_yaw)

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        speeds = speeds_cmd.double()
        y_tgt = (self.k_coeff * speeds).clamp(min=0.0)
        self._thrust = self._thrust + self.alpha * (y_tgt - self._thrust)
        if self.noise_std > 0:
            self._thrust = torch.clamp(self._thrust + torch.randn_like(self._thrust) * self.noise_std, min=0.0)

        # Torques
        tau_r, tau_p, tau_y = self._mix_torques(self._thrust)

        # Attitude dynamics
        phidd   = (tau_r / self.Ix) - self.k_drag_rp * self.phid
        thetadd = (tau_p / self.Iy) - self.k_drag_rp * self.thetad
        psidd   = (tau_y / self.Iz) - self.k_drag_yaw * self.psid

        self.phid   += self.Ts * phidd
        self.thetad += self.Ts * thetadd
        self.psid   += self.Ts * psidd

        self.phi   = float((self.phi   + self.Ts * self.phid + np.pi) % (2*np.pi) - np.pi)
        self.theta = float((self.theta + self.Ts * self.thetad + np.pi) % (2*np.pi) - np.pi)
        self.psi   = float((self.psi   + self.Ts * self.psid + np.pi) % (2*np.pi) - np.pi)

        # Vertical dynamics with tilt loss (cos phi * cos theta)
        T = float(self._thrust.sum().item())
        Tz = T * np.cos(self.phi) * np.cos(self.theta)
        zdd = (Tz - self.m * self.g) / self.m - self.k_drag_z * self.zd
        self.zd += self.Ts * zdd
        self.z  += self.Ts * self.zd

        # Sensor latency on rotor thrusts -> then filter
        meas = []
        for i in range(self.N):
            fifo = self._fifo[i]
            fifo.append(float(self._thrust[i].item()))
            out = fifo.pop(0)
            meas.append(out)
        meas = torch.tensor(meas, dtype=torch.float64)
        self.flows_meas_filt = (1 - self.MEAS_EWA)*self.flows_meas_filt + self.MEAS_EWA*meas
        return self.flows_meas_filt.clone()

class QuadcopterLagDeadtimeProblem:
    """
    Motors with per-rotor dead-time and 2nd-order (under-damped) response.
    Gyro (yaw-rate) bias drifts; tougher to control yaw transients.
    """
    def __init__(self, Ts: float):
        self.Ts = float(Ts)
        self.N = 4
        self.labels = ["FL(+)", "FR(-)", "RR(+)", "RL(-)"]
        self.output_name, self.output_unit, self.entity_title = "Thrust", "N", "Rotor"

        self.k_coeff = torch.ones(self.N, dtype=torch.float64) * 0.10
        self.speed_min = torch.zeros(self.N, dtype=torch.float64)
        self.speed_max = torch.ones(self.N, dtype=torch.float64) * 100.0
        self.slew_rate = 6.0
        self.alpha = 0.45

        # 2nd order motor response y'' + 2*zeta*wn*y' + wn^2*y = wn^2*y_cmd
        self.wn  = torch.tensor([10.0, 8.5, 9.0, 10.5], dtype=torch.float64)  # rad/s (scaled by Ts)
        self.zet = torch.tensor([0.35, 0.30, 0.33, 0.28], dtype=torch.float64)

        # Per-rotor dead-steps on the "commanded thrust"
        self.dead_steps = [2, 4, 3, 5]
        self._fifo = [list([0.0]*d) for d in self.dead_steps]

        # State per rotor: thrust and derivative (discrete)
        self.y  = torch.zeros(self.N, dtype=torch.float64)
        self.yd = torch.zeros(self.N, dtype=torch.float64)

        # Body params
        self.m, self.g = 1.1, 9.81
        self.Iz = 0.022
        self.c_tau = 0.015
        self.k_drag_z = 0.16
        self.k_drag_yaw = 0.14

        # Gyro drift/noise
        self.gyro_bias = 0.0
        self.gyro_rw_std = 0.003
        self.gyro_noise_std = 0.025

        # Meas filter
        self.MEAS_EWA = 0.35
        self.noise_std = 0.025
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)

        # Yaw/alt states
        self.z = 0.0; self.zd = 0.0
        self.psi = 0.0; self.psid = 0.0

        self.default_target_ratio = torch.ones(self.N, dtype=torch.float64)/self.N
        self.default_total_flow = torch.tensor(self.m * self.g, dtype=torch.float64)

    def baseline_allocation(self, tr, Ft):
        tr = torch.clamp(tr.double(), min=1e-12); tr = tr / tr.sum()
        s = Ft.double() * tr / (self.k_coeff + 1e-12)
        return torch.clamp(s, self.speed_min, self.speed_max)

    def comp_from_speeds(self, s):
        y_cmd = (self.k_coeff * s.double()).clamp(min=0.0)
        return y_cmd / (y_cmd.sum() + 1e-12)

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        # Dead-time on commanded THRUST (not speed)
        y_cmd = (self.k_coeff * speeds_cmd.double()).clamp(min=0.0)
        delayed = []
        for i in range(self.N):
            fifo = self._fifo[i]
            fifo.append(float(y_cmd[i].item()))
            out = fifo.pop(0)
            delayed.append(out)
        delayed = torch.tensor(delayed, dtype=torch.float64)

        # 2nd-order motor dynamics (discretized via simple Euler)
        # ydd = wn^2*(u - y) - 2*zeta*wn*yd
        ydd = (self.wn**2) * (delayed - self.y) - 2.0*self.zet*self.wn*self.yd
        self.yd += self.Ts * ydd
        self.y  = torch.clamp(self.y + self.Ts * self.yd, min=0.0)

        if self.noise_std > 0:
            self.y = torch.clamp(self.y + torch.randn_like(self.y)*self.noise_std, min=0.0)

        # Rigid-body (z & yaw)
        T = float(self.y.sum().item())
        tau_z = self.c_tau * float((self.y[0]+self.y[2] - self.y[1]-self.y[3]).item())

        zdd = (T - self.m*self.g)/self.m - self.k_drag_z*self.zd
        self.zd += self.Ts*zdd
        self.z  += self.Ts*self.zd

        # Yaw with gyro-bias-driven rate
        self.gyro_bias += np.random.randn()*self.gyro_rw_std
        psidd = (tau_z/self.Iz) - self.k_drag_yaw*self.psid
        self.psid += self.Ts*psidd
        gyro_meas = self.psid + self.gyro_bias + np.random.randn()*self.gyro_noise_std
        self.psi = float((self.psi + self.Ts*gyro_meas + np.pi)%(2*np.pi)-np.pi)

        # Return filtered per-rotor thrust
        self.flows_meas_filt = (1-self.MEAS_EWA)*self.flows_meas_filt + self.MEAS_EWA*self.y
        return self.flows_meas_filt.clone()


class QuadcopterBrownoutFaultsProblem:
    """
    Nasty mix: battery sag (k drops with load), motor stiction & occasional brownout,
    sensor spikes/dropouts, yaw torque asymmetry.
    """
    def __init__(self, Ts: float):
        self.Ts = float(Ts)
        self.N = 4
        self.labels = ["FL(+)", "FR(-)", "RR(+)", "RL(-)"]
        self.output_name, self.output_unit, self.entity_title = "Thrust", "N", "Rotor"

        self.k_nom = torch.tensor([0.10, 0.10, 0.10, 0.10], dtype=torch.float64)
        self.k_coeff = self.k_nom.clone()

        self.speed_min = torch.zeros(self.N, dtype=torch.float64)
        self.speed_max = torch.ones(self.N, dtype=torch.float64) * 100.0
        self.slew_rate = 8.0
        self.alpha = 0.5

        self.m, self.g = 1.25, 9.81
        self.Iz = 0.024
        self.c_tau_nom = 0.015
        self.c_tau_skew = torch.tensor([+1.02, -0.96, +0.98, -1.03], dtype=torch.float64)  # rotor-wise yaw skew

        self.k_drag_z = 0.18
        self.k_drag_yaw = 0.15

        # Battery sag: k := k_nom * (1 - sag_a * clamp(T/ Tmax, 0, 1)) - slow drift
        self.sag_a = 0.22
        self.k_rw_std = 5e-4
        self.Tmax = float((self.k_nom * 100.0).sum().item())  # rough max total thrust

        # Actuator stiction/brownout
        self.p_stick = 0.08
        self.p_brown = 0.015
        self.brown_len = 10
        self._brown_left = torch.zeros(self.N, dtype=torch.int64)
        self._act = torch.zeros(self.N, dtype=torch.float64)  # internal actuator state
        self.act_alpha = 0.35

        # Sensors
        self.MEAS_EWA = 0.34
        self.p_dropout = 0.03
        self.p_spike = 0.015
        self.spike_std = 0.08  # N

        # States
        self._y = torch.zeros(self.N, dtype=torch.float64)
        self.flows_meas_filt = torch.zeros(self.N, dtype=torch.float64)
        self.z = 0.0; self.zd = 0.0
        self.psi = 0.0; self.psid = 0.0

        self.default_target_ratio = torch.ones(self.N, dtype=torch.float64)/self.N
        self.default_total_flow = torch.tensor(self.m * self.g, dtype=torch.float64)

    def baseline_allocation(self, tr, Ft):
        tr = torch.clamp(tr.double(), min=1e-12); tr = tr / tr.sum()
        s = Ft.double() * tr / (self.k_coeff + 1e-12)
        return torch.clamp(s, self.speed_min, self.speed_max)

    def comp_from_speeds(self, s):
        y = (self.k_coeff * s.double()).clamp(min=0.0)
        return y / (y.sum() + 1e-12)

    def _apply_battery_sag(self, y_cmd):
        # Update k with slow random walk and sag vs demanded total
        self.k_coeff = torch.clamp(self.k_coeff + torch.randn_like(self.k_coeff)*self.k_rw_std,
                                   self.k_nom*0.8, self.k_nom*1.05)
        Tdem = float(y_cmd.sum().item())
        sag = self.sag_a * np.clip(Tdem / (self.Tmax + 1e-9), 0.0, 1.0)
        return torch.clamp(self.k_coeff * (1.0 - sag), self.k_nom*0.6, self.k_nom*1.05)

    def _yaw_tau(self, y):
        # Yaw torque with rotor-wise skew
        c_vec = self.c_tau_skew * self.c_tau_nom
        return float((c_vec[0]*y[0] + c_vec[2]*y[2] - c_vec[1]*y[1] - c_vec[3]*y[3]).item())

    def step(self, speeds_cmd: torch.Tensor) -> torch.Tensor:
        # Actuator stiction & smoothing on speeds
        hold = (torch.rand(self.N) < self.p_stick) | (self._brown_left > 0)
        self._act[hold] = self._act[hold]
        self._act[~hold] = self._act[~hold] + self.act_alpha * (speeds_cmd.double()[~hold] - self._act[~hold])

        # Start / decrement brownouts
        start = (self._brown_left == 0) & (torch.rand(self.N) < self.p_brown)
        self._brown_left[start] = self.brown_len
        active = self._brown_left > 0
        self._brown_left[active] -= 1

        # Commanded thrust from (possibly stuck) speeds
        k_eff = self._apply_battery_sag(self.k_nom * self._act)
        y_cmd = (k_eff * self._act).clamp(min=0.0)

        # Brownout forces thrust near zero
        y_cmd[active] = torch.minimum(y_cmd[active], torch.tensor(0.3, dtype=torch.float64))

        # Plant lag
        self._y = self._y + self.alpha * (y_cmd - self._y)

        # Rigid-body z & yaw
        T = float(self._y.sum().item())
        tau_z = self._yaw_tau(self._y)
        zdd = (T - self.m*self.g)/self.m - self.k_drag_z*self.zd
        self.zd += self.Ts*zdd
        self.z  += self.Ts*self.zd

        psidd = (tau_z / self.Iz) - self.k_drag_yaw*self.psid
        self.psid += self.Ts*psidd
        self.psi = float((self.psi + self.Ts*self.psid + np.pi)%(2*np.pi)-np.pi)

        # Sensor path: spikes/dropouts then filter
        meas = self._y.clone()
        drop = (torch.rand(self.N) < self.p_dropout)
        spike = (torch.rand(self.N) < self.p_spike)
        meas[drop]  = self.flows_meas_filt[drop]
        meas[spike] = torch.clamp(meas[spike] + torch.abs(torch.randn(spike.sum(), dtype=torch.float64))*self.spike_std, min=0.0)

        self.flows_meas_filt = (1 - self.MEAS_EWA)*self.flows_meas_filt + self.MEAS_EWA*meas
        return self.flows_meas_filt.clone()


# Registry of problems exposed to the app
AVAILABLE_PROBLEMS: Dict[str, type] = {
    "SingleTankMixer (5-flow)": SingleTankMixerProblem,
    "Deadtime+DriftingGains (5-flow)": DeadtimeVaryingGainsProblem,
    "NonlinearBackpressure (5-flow)": NonlinearBackpressureProblem,
    "TwoTankCascade (5-flow)": TwoTankCascadeProblem,
    "FaultySensorsActuators (5-flow)": FaultySensorsActuatorsProblem,
    "Quadcopter (alt+yaw, 4-rotor)": QuadcopterAltYawProblem,

    # New, harder quad problems:
    "Quadcopter+Gyro+Wind (4-rotor)": QuadcopterGyroWindProblem,
    "Quadcopter+CoupledAxes (4-rotor)": QuadcopterCoupledAxesProblem,
    "Quadcopter+Lag+Deadtime (4-rotor)": QuadcopterLagDeadtimeProblem,
    "Quadcopter+Brownouts+Sag (4-rotor)": QuadcopterBrownoutFaultsProblem,
}

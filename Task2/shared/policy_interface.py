from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class Policy:
    def act(self, obs: np.ndarray) -> float:
        raise NotImplementedError


@dataclass
class ZeroPolicy(Policy):
    def act(self, obs: np.ndarray) -> float:
        return 0.0


@dataclass
class EnergySwingUpPolicy(Policy):
    umax: float = 3.0
    omega0: float = 11.339846957335382
    ku: float = 28.136158407237073
    energy_gain: float = 1.5
    top_region_rad: float = 0.45
    kp: float = 180.0
    kd: float = 18.0
    kick_omega_threshold: float = 0.20

    def act(self, obs: np.ndarray) -> float:
        theta = float(obs[0])
        omega = float(obs[1])
        top_error = wrap_to_pi(theta - np.pi)

        if abs(top_error) < self.top_region_rad:
            u = (-self.kp * top_error - self.kd * omega) / self.ku
        else:
            energy = 0.5 * omega**2 + self.omega0**2 * (1.0 - np.cos(theta))
            target_energy = 2.0 * self.omega0**2
            if abs(omega) < self.kick_omega_threshold:
                u = self.umax
            else:
                u = self.energy_gain * np.sign(omega) * (target_energy - energy) / self.ku

        return float(np.clip(u, -self.umax, self.umax))

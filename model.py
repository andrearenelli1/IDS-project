"""
model.py
========
3D point-mass motion model used by the simulation.

The model is the exact zero-order-hold discretization of a double integrator:

    x = [px, py, pz, vx, vy, vz]^T
    u = [ax, ay, az]^T

    p(k+1) = p(k) + v(k) dt + 1/2 u(k) dt^2
    v(k+1) = v(k) + u(k) dt

The same class provides the dynamics, Jacobians, and process-noise covariance
needed by IMDCL and MPC.
"""

from __future__ import annotations

import numpy as np

Vector = np.ndarray
Matrix = np.ndarray

class PointMass3DModel:
    STATE_DIM = 6
    INPUT_DIM = 3
    ii = np.eye(3)
    zz = np.zeros((3, 3))
    
    def __init__(self, sigma_acc: float | np.ndarray) -> None:
        self._sigma_acc = np.broadcast_to(np.asarray(sigma_acc, dtype=float), (3,)).copy()

    @property
    def state_dim(self) -> int:
        return self.STATE_DIM

    @property
    def input_dim(self) -> int:
        return self.INPUT_DIM

    @staticmethod
    def _F(dt: float) -> Matrix:
        identity = np.eye(3)
        zeros = np.zeros((3, 3))
        return np.block([[identity, dt * identity], [zeros, identity]])

    @staticmethod
    def _G(dt: float) -> Matrix:
        identity = np.eye(3)
        return np.block([[0.5 * dt**2 * identity], [dt * identity]])

    def f(self, x: Vector, u: Vector, dt: float) -> Vector:
        """Deterministic ZOH propagation: x(k+1) = F x(k) + G u(k)."""
        return self._F(dt) @ np.asarray(x, dtype=float).ravel() + self._G(dt) @ np.asarray(u, dtype=float).ravel()

    def F_jacobian(self, x: Vector, u: Vector, dt: float) -> Matrix:
        """Jacobian with respect to the state."""
        return self._F(dt)

    def G_jacobian(self, x: Vector, dt: float) -> Matrix:
        """Jacobian with respect to the acceleration noise."""
        return self._G(dt)

    def Q(self, dt: float) -> Matrix:
        """Process-noise covariance for isotropic acceleration noise."""
        return np.diag(self._sigma_acc**2)
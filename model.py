"""
model.py
========

3D point-mass motion model.

The model is the exact zero-order-hold discretization of a double integrator:

    x = [px, py, pz, vx, vy, vz]^T
    u = [ax, ay, az]^T

    p(k+1) = p(k) + v(k) dt + 1/2 a(k) dt²
    v(k+1) = v(k) + a(k) dt

or, equivalently,

    x(k+1) = F(dt) x(k) + G(dt) u(k)

The class provides:

    - state propagation
    - state Jacobian
    - input/noise Jacobian
    - acceleration-noise covariance

The Jacobian methods are intentionally kept even though the model is
linear, so that estimators can use the same interface for nonlinear
motion models.
"""

from __future__ import annotations

import numpy as np

Vector = np.ndarray
Matrix = np.ndarray


class PointMass3DModel:
    """Exact discrete-time 3D double-integrator model."""

    STATE_DIM = 6
    INPUT_DIM = 3

    _I3 = np.eye(3)
    _Z3 = np.zeros((3, 3))

    def __init__(self, sigma_acc: float | Vector) -> None:
        """
        Parameters
        ----------
        sigma_acc
            Standard deviation of the acceleration noise [m/s²].

            Can be:
                - scalar (same value on x, y, z)
                - array-like of length 3
        """
        sigma_acc = np.asarray(sigma_acc, dtype=float)
        self._sigma_acc = np.broadcast_to(sigma_acc, (3,)).copy()

    @property
    def state_dim(self) -> int:
        return self.STATE_DIM

    @property
    def input_dim(self) -> int:
        return self.INPUT_DIM

    @staticmethod
    def F(dt: float) -> Matrix:
        """
        State-transition matrix.

        x(k+1) = F(dt) x(k) + ...
        """
        I = PointMass3DModel._I3
        Z = PointMass3DModel._Z3

        return np.block([
            [I, dt * I],
            [Z, I],
        ])

    @staticmethod
    def G(dt: float) -> Matrix:
        """
        Input/noise matrix.

        x(k+1) = ... + G(dt) u(k)
        """
        I = PointMass3DModel._I3

        return np.block([
            [0.5 * dt**2 * I],
            [dt * I],
        ])

    def f(self, x: Vector, u: Vector, dt: float) -> Vector:
        """
        Deterministic state propagation.

        Parameters
        ----------
        x : (6,)
            Current state.
        u : (3,)
            Acceleration input.
        dt : float
            Sampling interval.

        Returns
        -------
        (6,)
            Predicted state.
        """
        x = np.asarray(x, dtype=float).reshape(self.STATE_DIM)
        u = np.asarray(u, dtype=float).reshape(self.INPUT_DIM)

        return self.F(dt) @ x + self.G(dt) @ u

    def F_jacobian(self, x: Vector, u: Vector, dt: float) -> Matrix:
        """
        State Jacobian.

            F = ∂f/∂x

        For this linear model, the Jacobian equals the state-transition
        matrix and does not depend on x or u.
        """
        return self.F(dt)

    def G_jacobian(self, x: Vector, dt: float) -> Matrix:
        """
        Input/noise Jacobian.

            G = ∂f/∂u

        For this linear model, the Jacobian equals the input matrix and
        does not depend on x.
        """
        return self.G(dt)

    def Q(self) -> Matrix:
        """
        Acceleration-noise covariance.

        Returns
        -------
        (3,3)
            Covariance of the acceleration noise.

        Used in the prediction covariance update:

            P⁻ = F P⁺ Fᵀ + G Q Gᵀ
        """
        return np.diag(self._sigma_acc**2)
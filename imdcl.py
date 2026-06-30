"""
Interim Master Decentralized Cooperative Localization (IMDCL)
=============================================================
Implementation based on:
  Kia, Rounds, Martínez — "Cooperative Localization for Mobile Agents:
  A Recursive Decentralized Algorithm Based on Kalman-Filter Decoupling"
  IEEE Control Systems Magazine, April 2016.

AgentIMDCL    – agent that runs the IMDCL algorithm (Algorithm 2 of the paper).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from model import PointMass3DModel
from config import IMDCL_PI_MAX_NORM


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vector = np.ndarray   # colonna (n,) o (n,1)
Matrix = np.ndarray   # matrice 2-D


# ===========================================================================
# Funzione di misura relativa 3-D (massa puntiforme)
# ===========================================================================

def relative_position_measurement_3d(
    x_a: Vector, x_b: Vector
) -> Tuple[Vector, Matrix, Matrix]:
    """
    Relative position measurement in 3-D (position only, not velocity).

    z_{ab} = h(x^a, x^b) = p^b − p^a = [pbx−pax, pby−pay, pbz−paz]^T

    The model is linear in x, so the Jacobians are constant:
        Ha = ∂h/∂x^a = [-I₃ | 0₃]   (3×6)
        Hb = ∂h/∂x^b = [ I₃ | 0₃]   (3×6)

    Returns
    -------
    h_val : measurement function value  (3,)
    Ha    : Jacobian w.r.t. x^a        (3×6)
    Hb    : Jacobian w.r.t. x^b        (3×6)
    """
    p_a = x_a.ravel()[:3]
    p_b = x_b.ravel()[:3]
    h_val = p_b - p_a

    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    Ha = np.hstack([-I3, Z3])   # (3×6)
    Hb = np.hstack([ I3, Z3])   # (3×6)

    return h_val, Ha, Hb


# ===========================================================================
# Messages exchanged between agents
# ===========================================================================

@dataclass
class LandmarkMessage:
    """
    Message sent by *landmark* agent b to interim master a
    (Eq. S8 of the paper).
    """
    agent_id: int
    x_hat: Vector    # stima propagata x^{b-}(k+1)
    P: Matrix        # covarianza propagata P^{b-}(k+1)
    Phi: Matrix      # matrice intermedia Φ^b(k+1)


@dataclass
class UpdateMessage:
    """
    Update message broadcast from the interim master to the whole team
    (Eq. S10 of the paper).
    """
    master_id: int
    landmark_id: int
    innovation: Vector              # r^a
    S_inv_sqrt: Matrix              # S_{ab}^{-1/2}
    Gamma_a: Matrix                 # Γ_a  (update term for master)
    Gamma_b: Matrix                 # Γ_b  (update term for landmark)
    Phi_b_T_Hb_T_S_inv_sqrt: Matrix # Φ^{b⊤} H̃_b^⊤ S^{-1/2}  (needed for Eq. S11)
    Phi_a_T_Ha_T_S_inv_sqrt: Matrix # Φ^{a⊤} H̃_a^⊤ S^{-1/2}  (needed for Eq. S11)


# ===========================================================================
# Relative measurement function (example: 2-D relative pose)
# ===========================================================================

def relative_pose_measurement(
    x_a: Vector, x_b: Vector
) -> Tuple[Vector, Matrix, Matrix]:
    """
    2-D relative pose measurement model.

    z_{ab} = h(x^a, x^b) = [Δx·cos θ_a + Δy·sin θ_a,
                             -Δx·sin θ_a + Δy·cos θ_a,
                             θ_b − θ_a]

    Returns
    -------
    h_val : measurement function value
    Ha    : Jacobian w.r.t. x^a
    Hb    : Jacobian w.r.t. x^b
    """
    pax, pay, tha = x_a.ravel()
    pbx, pby, thb = x_b.ravel()
    dx = pbx - pax
    dy = pby - pay
    c, s = np.cos(tha), np.sin(tha)

    h_val = np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
        thb - tha,
    ])

    Ha = np.array([
        [-c, -s,  -s * dx + c * dy],
        [ s, -c,  -c * dx - s * dy],
        [ 0,  0,  -1.0],
    ])
    Hb = np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return h_val, Ha, Hb


# ===========================================================================
# IMDCL Agent
# ===========================================================================

class AgentIMDCL:
    """
    Agent implementing the IMDCL algorithm (Algorithm 2, Kia et al. 2016).

    Each agent maintains locally:
      - x_hat  : state estimate (n_i,)
      - P      : estimation error covariance (n_i × n_i)
      - Phi    : intermediate matrix Φ^i, initialised to I (n_i × n_i)
      - Pi_jl  : dict {(j,l): Π_{jl}^i} — local copies of cross-covariances
                 for all pairs (j < l) in the team, excluding pairs that
                 include the agent's own id (Eq. S5 / S12c).

    Parameters
    ----------
    agent_id    : unique integer identifier for this agent.
    x0          : initial state estimate.
    P0          : initial estimation error covariance.
    team_ids    : list of all team agent ids (including this agent).
    motion_model: MotionModel instance (pluggable).
    """

    def __init__(
        self,
        agent_id: int,
        x0: Vector,
        P0: Matrix,
        team_ids: List[int],
        motion_model: MotionModel,
    ) -> None:
        self.id = agent_id
        self.motion_model = motion_model
        n = motion_model.state_dim

        # State and covariance (Eq. S5)
        self.x_hat: Vector = np.array(x0, dtype=float).ravel()
        self.P: Matrix = np.array(P0, dtype=float)

        # Intermediate matrix Φ^i — initialised to I (Eq. S5)
        self.Phi: Matrix = np.eye(n)

        # Local copies of cross-covariances Π_{jl}^i
        # By paper symmetry only pairs with j < l are stored,
        # j ≠ id, l ≠ id  (Eq. S5: initialised to zero)
        other_ids = sorted(v for v in team_ids if v != agent_id)
        self.Pi_jl: Dict[Tuple[int, int], Matrix] = {}
        for idx_j, j in enumerate(other_ids):
            for l in other_ids[idx_j + 1:]:
                self.Pi_jl[(j, l)] = np.zeros((n, n))

        self._n = n
        self._team_ids = list(team_ids)

    # ------------------------------------------------------------------
    # Read-only properties useful for debugging
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # 1. Propagation (local, no communication)
    # ------------------------------------------------------------------

    def propagate(self, u: Vector, dt: float) -> None:
        """
        Propagation step — Algorithm 2, step 1 (Eq. S6).

        Updates x_hat, P and Phi locally without any communication.

        Parameters
        ----------
        u  : control / odometry input vector u^i(k).
        dt : sampling interval.
        """
        F = self.motion_model.F_jacobian(self.x_hat, u, dt)
        G = self.motion_model.G_jacobian(self.x_hat, dt)
        Q = self.motion_model.Q()

        # x^{i-}(k+1) = f^i(x^{i+}(k), u^i(k))
        self.x_hat = self.motion_model.f(self.x_hat, u, dt)

        # P^{i-}(k+1) = F^i P^{i+}(k) F^{i⊤} + G^i Q^i G^{i⊤}
        self.P = F @ self.P @ F.T + G @ Q @ G.T

        # Φ^i(k+1) = F^i(k) Φ^i(k)
        self.Phi = F @ self.Phi

    # ------------------------------------------------------------------
    # 2a. Update — no team measurement
    # ------------------------------------------------------------------

    def step_no_measurement(self) -> None:
        """
        No relative measurement in the team: propagated variables become
        the updated estimates; Pi_jl remain unchanged (Eq. S7).

        Variables are already updated in-place, so this is an explicit
        no-op kept for clarity.
        """
        pass  # x_hat, P, Phi already propagated; Pi_jl unchanged per (14)

    # ------------------------------------------------------------------
    # 2b. Landmark message generation (landmark agent b)
    # ------------------------------------------------------------------

    def make_landmark_message(self) -> LandmarkMessage:
        """
        Creates the message to send to the interim master (Eq. S8).

        Called by landmark agent b before interim master a performs
        its own computations.
        """
        return LandmarkMessage(
            agent_id=self.id,
            x_hat=self.x_hat.copy(),
            P=self.P.copy(),
            Phi=self.Phi.copy(),
        )

    # ------------------------------------------------------------------
    # 2c. Update message computation (interim master agent a)
    # ------------------------------------------------------------------

    def compute_update_message(
        self,
        lm_msg: LandmarkMessage,
        z_ab: Vector,
        R_ab: Matrix,
        measurement_fn=relative_pose_measurement,
    ) -> UpdateMessage:
        """
        Agent a (interim master) computes the update message to broadcast
        to the whole team (Eq. S9, S10).

        Parameters
        ----------
        lm_msg         : LandmarkMessage received from landmark agent b.
        z_ab           : relative measurement made by a towards b.
        R_ab           : measurement noise covariance.
        measurement_fn : function h(x_a, x_b) → (h_val, Ha, Hb).

        Returns
        -------
        UpdateMessage to pass to all agents via apply_update().
        """
        x_a, P_a, Phi_a = self.x_hat, self.P, self.Phi
        x_b = lm_msg.x_hat
        P_b = lm_msg.P
        Phi_b = lm_msg.Phi
        b = lm_msg.agent_id

        # Measurement function Jacobians evaluated at the propagated state
        h_val, Ha_tilde, Hb_tilde = measurement_fn(x_a, x_b)

        # Innovation  r^a = z_{ab} − h(x^{a-}, x^{b-})  (Eq. S9a)
        r = np.array(z_ab).ravel() - h_val

        # Retrieve Π_{ab}^a (or Π_{ba}^a) from local copy
        Pi_ab = self._get_Pi(self.id, b)   # Π_{ab}^a  →  Π between a and b

        # Innovation covariance  S_{ab}  (Eq. S9b)
        #   S = R + H̃_a P^a H̃_a⊤ + H̃_b P^b H̃_b⊤
        #       − H̃_a Φ^a Π_{ab}^a Φ^{b⊤} H̃_b⊤
        #       − H̃_b Φ^b Π_{ab}^{a⊤} Φ^{a⊤} H̃_a⊤
        S = (
            R_ab
            + Ha_tilde @ P_a @ Ha_tilde.T
            + Hb_tilde @ P_b @ Hb_tilde.T
            - Ha_tilde @ Phi_a @ Pi_ab @ Phi_b.T @ Hb_tilde.T
            - Hb_tilde @ Phi_b @ Pi_ab.T @ Phi_a.T @ Ha_tilde.T
        )

        # Factor S^{-1/2}  (used to normalise Γ and store in message)
        S_inv_sqrt = self._matrix_inv_sqrt(S)

        # Update terms Γ_a, Γ_b  (Eq. S9c of the paper)
        #
        # From the paper (exact signs):
        #   Γ_a = ((Φ^a)^{-1} P^a H̃_a⊤ − (Φ^a)^{-1} Φ^a Π_{ab}^a Φ^{b⊤} H̃_b⊤) S^{-1/2}
        #        = (Φ^a)^{-1} (P^a H̃_a⊤ − Π_{ab}^a Φ^{b⊤} H̃_b⊤) S^{-1/2}
        #
        #   Γ_b = ((Φ^b)^{-1} P^b H̃_b⊤ − (Φ^b)^{-1} Π_{ba}^a Φ^{a⊤} H̃_a⊤) S^{-1/2}
        #        = (Φ^b)^{-1} (P^b H̃_b⊤ − Π_{ab}^{a⊤} Φ^{a⊤} H̃_a⊤) S^{-1/2}
        #
        # Note: Φ^a_inv @ Φ^a = I  →  the Π term simplifies to Pi_ab directly.
        Phi_a_inv = np.linalg.inv(Phi_a)
        Phi_b_inv = np.linalg.inv(Phi_b)

        Gamma_a = (
            Phi_a_inv @ P_a @ Ha_tilde.T
            - Pi_ab @ Phi_b.T @ Hb_tilde.T          # Φ^a_inv @ Φ^a = I
        ) @ S_inv_sqrt

        Gamma_b = (
            Phi_b_inv @ P_b @ Hb_tilde.T
            - Phi_b_inv @ Pi_ab.T @ Phi_a.T @ Ha_tilde.T
        ) @ S_inv_sqrt

        # Auxiliary terms to let every agent i compute Γ_i (Eq. S11)
        Phi_b_T_Hb_T_S_inv = Phi_b.T @ Hb_tilde.T @ S_inv_sqrt
        Phi_a_T_Ha_T_S_inv = Phi_a.T @ Ha_tilde.T @ S_inv_sqrt

        return UpdateMessage(
            master_id=self.id,
            landmark_id=b,
            innovation=r,
            S_inv_sqrt=S_inv_sqrt,
            Gamma_a=Gamma_a,
            Gamma_b=Gamma_b,
            Phi_b_T_Hb_T_S_inv_sqrt=Phi_b_T_Hb_T_S_inv,
            Phi_a_T_Ha_T_S_inv_sqrt=Phi_a_T_Ha_T_S_inv,
        )

    # ------------------------------------------------------------------
    # 2d. Update message application (all agents)
    # ------------------------------------------------------------------

    def apply_update(self, msg: UpdateMessage) -> None:
        """
        Every agent i ∈ V receives the update message and updates its
        local variables (Eq. S11, S12a–S12c).

        Parameters
        ----------
        msg : UpdateMessage broadcast by the interim master.
        """
        a = msg.master_id
        b = msg.landmark_id

        # --- Compute Γ_i for this agent (Eq. S11) ---
        if self.id == a:
            Gamma_i = msg.Gamma_a
        elif self.id == b:
            Gamma_i = msg.Gamma_b
        else:
            # Γ_i = Π_{ib}^i Φ^{b⊤} H̃_b⊤ S^{-1/2} − Π_{ia}^i Φ^{a⊤} H̃_a⊤ S^{-1/2}
            Pi_ib = self._get_Pi(self.id, b)
            Pi_ia = self._get_Pi(self.id, a)
            Gamma_i = (
                Pi_ib @ msg.Phi_b_T_Hb_T_S_inv_sqrt
                - Pi_ia @ msg.Phi_a_T_Ha_T_S_inv_sqrt
            )

        # r̃^a = S^{-1/2} r^a  (normalised innovation)
        r_tilde = msg.S_inv_sqrt @ msg.innovation

        # --- State update (Eq. S12a) ---
        # x^{i+}(k+1) = x^{i-}(k+1) + Φ^i(k+1) Γ_i r̃^a
        self.x_hat = self.x_hat + self.Phi @ Gamma_i @ r_tilde

        # --- Covariance update (Eq. S12b) ---
        # P^{i+}(k+1) = P^{i-}(k+1) − Φ^i Γ_i Γ_i⊤ Φ^{i⊤}
        self.P = self.P - self.Phi @ Gamma_i @ Gamma_i.T @ self.Phi.T

        # --- Local copy update Π_{jl}^i (Eq. S12c) ---
        # Π_{jl}^i(k+1) = Π_{jl}^i(k) − Γ_j Γ_l⊤
        for (j, l) in list(self.Pi_jl.keys()):
            Gamma_j = self._compute_gamma(j, msg)
            Gamma_l = self._compute_gamma(l, msg)
            with np.errstate(over="ignore", invalid="ignore"):
                mat = self.Pi_jl[(j, l)] - Gamma_j @ Gamma_l.T
            # Reset to zero if not finite or norm exceeds threshold (Pi_jl is a cross-covariance:
            # if it diverges, independence is assumed — filter continues without cross-correction).
            if not np.all(np.isfinite(mat)) or np.linalg.norm(mat, "fro") > IMDCL_PI_MAX_NORM:
                mat = np.zeros((self._n, self._n))
            self.Pi_jl[(j, l)] = mat

        # Reset Φ^i after each update (Eq. S5/S6: Φ reset to I after update)
        self.Phi = np.eye(self._n)

    # ------------------------------------------------------------------
    # 2e. Absolute measurement update (local, no communication)
    # ------------------------------------------------------------------

    def apply_absolute_update(
        self,
        z: Vector,
        H: Matrix,
        R: Matrix,
    ) -> None:
        """
        EKF update with a **local** absolute measurement of this agent
        (no communication required).

        In the IMDCL framework an absolute measurement of agent i is
        equivalent to a relative measurement with a = b = i (landmark =
        master = self). The Kalman gain reduces to the standard form:

            K  = P · H^T · (H · P · H^T + R)^{-1}
            x̂  ← x̂ + Φ · K · (z − H · x̂)
            P  ← P  − Φ · K · (H · P · H^T + R) · K^T · Φ^T

        Φ is reset to I after the update (same as apply_update).

        Π_{jl} is **not modified** because the absolute measurement
        concerns only this agent and creates no new cross-covariances
        (cf. paper §"Cooperative Localization via EKF", absolute measurements).

        Parameters
        ----------
        z : measurement vector                   (m,)
        H : observation matrix ∂h/∂x             (m × n)
        R : measurement noise covariance          (m × m)
        """
        z = np.asarray(z, dtype=float).ravel()
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        # Innovation
        innov = z - H @ self.x_hat                         # (m,)

        # Innovation covariance
        S = H @ self.P @ H.T + R                           # (m × m)

        # Standard Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)               # (n × m)

        # State update: x̂ ← x̂ + Φ K (z − H x̂)
        self.x_hat = self.x_hat + self.Phi @ K @ innov

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self._n) - self.Phi @ K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.Phi @ K @ R @ K.T @ self.Phi.T

        # Reset Φ → I after each update
        self.Phi = np.eye(self._n)

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _get_Pi(self, i: int, j: int) -> Matrix:
        """
        Returns Π_{ij}^{self} from the local copy, honouring the
        j > i convention (symmetry: Π_{ji} = Π_{ij}⊤).
        """
        if i == j:
            # Π_{ii} ≡ P^i  (self-pair — never used directly)
            return self.P.copy()
        key = (min(i, j), max(i, j))
        mat = self.Pi_jl.get(key, np.zeros((self._n, self._n)))
        # If i > j the matrix is transposed
        return mat if i < j else mat.T

    def _compute_gamma(self, agent_j: int, msg: UpdateMessage) -> Matrix:
        """
        Computes Γ_j for any team agent j given the update message.
        Used internally during the Pi_jl update.
        """
        a, b = msg.master_id, msg.landmark_id
        if agent_j == a:
            return msg.Gamma_a
        if agent_j == b:
            return msg.Gamma_b
        Pi_jb = self._get_Pi(agent_j, b)
        Pi_ja = self._get_Pi(agent_j, a)
        with np.errstate(over="ignore", invalid="ignore"):
            result = (
                Pi_jb @ msg.Phi_b_T_Hb_T_S_inv_sqrt
                - Pi_ja @ msg.Phi_a_T_Ha_T_S_inv_sqrt
            )
        if not np.all(np.isfinite(result)):
            return np.zeros_like(result)
        return result

    @staticmethod
    def _matrix_inv_sqrt(M: Matrix) -> Matrix:
        """
        Computes M^{-1/2} via eigenvalue decomposition.

        For a symmetric positive-definite matrix M = V Λ V⊤:
            M^{-1/2} = V Λ^{-1/2} V⊤
        """
        eigvals, eigvecs = np.linalg.eigh(M)
        # Numerical clamp to avoid negative eigenvalues from floating-point errors
        eigvals = np.maximum(eigvals, 1e-12)
        inv_sqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
        return eigvecs @ inv_sqrt_diag @ eigvecs.T

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AgentIMDCL(id={self.id}, "
            f"x={np.round(self.x_hat, 3)}, "
            f"P_trace={np.trace(self.P):.4f})"
        )


# ===========================================================================
# Minimal usage example
# ===========================================================================

if __name__ == "__main__":
    # Example: instantiate an IMDCL agent with PointMass3DModel
    # and acceleration noise σ_acc = 0.1 m/s².
    motion_model = PointMass3DModel(sigma_acc=0.1)
    agent = AgentIMDCL(
        agent_id=1,
        x0=np.zeros(6),          # initial state: zero position and velocity
        P0=np.eye(6) * 0.5,     # initial covariance: moderate uncertainty
        team_ids=[1, 2, 3],     # team ids (including this agent)
        motion_model=motion_model,
    )
    print(agent)
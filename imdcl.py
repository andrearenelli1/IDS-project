"""
Interim Master Decentralized Cooperative Localization (IMDCL)
=============================================================
Implementazione basata su:
  Kia, Rounds, Martínez — "Cooperative Localization for Mobile Agents:
  A Recursive Decentralized Algorithm Based on Kalman-Filter Decoupling"
  IEEE Control Systems Magazine, April 2016.

AgentIMDCL    – agente che esegue l'algoritmo IMDCL (Algorithm 2 del paper).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from model import MotionModel, PointMass3DModel


# ---------------------------------------------------------------------------
# Tipi
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
    Misura della posizione relativa in 3-D (solo posizione, non velocità).

    z_{ab} = h(x^a, x^b) = p^b − p^a = [pbx−pax, pby−pay, pbz−paz]^T

    Il modello è lineare in x, quindi i Jacobiani sono costanti:
        Ha = ∂h/∂x^a = [-I₃ | 0₃]   (3×6)
        Hb = ∂h/∂x^b = [ I₃ | 0₃]   (3×6)

    Returns
    -------
    h_val : valore della funzione di misura  (3,)
    Ha    : Jacobiano rispetto a x^a         (3×6)
    Hb    : Jacobiano rispetto a x^b         (3×6)
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
# Messaggi scambiati tra agenti
# ===========================================================================

@dataclass
class LandmarkMessage:
    """
    Messaggio inviato dall'agente *landmark* b all'interim master a
    (Eq. S8 del paper).
    """
    agent_id: int
    x_hat: Vector    # stima propagata x^{b-}(k+1)
    P: Matrix        # covarianza propagata P^{b-}(k+1)
    Phi: Matrix      # matrice intermedia Φ^b(k+1)


@dataclass
class UpdateMessage:
    """
    Messaggio di aggiornamento broadcast dall'interim master a tutto il team
    (Eq. S10 del paper).
    """
    master_id: int
    landmark_id: int
    innovation: Vector              # r^a
    S_inv_sqrt: Matrix              # S_{ab}^{-1/2}
    Gamma_a: Matrix                 # Γ_a  (termine di aggiornamento per master)
    Gamma_b: Matrix                 # Γ_b  (termine di aggiornamento per landmark)
    Phi_b_T_Hb_T_S_inv_sqrt: Matrix # Φ^{b⊤} H̃_b^⊤ S^{-1/2}  (necessario per Eq. S11)
    Phi_a_T_Ha_T_S_inv_sqrt: Matrix # Φ^{a⊤} H̃_a^⊤ S^{-1/2}  (necessario per Eq. S11)


# ===========================================================================
# Funzione di misura relativa (esempio: posa relativa 2-D)
# ===========================================================================

def relative_pose_measurement(
    x_a: Vector, x_b: Vector
) -> Tuple[Vector, Matrix, Matrix]:
    """
    Modello di misura della posa relativa in 2-D.

    z_{ab} = h(x^a, x^b) = [Δx·cos θ_a + Δy·sin θ_a,
                             -Δx·sin θ_a + Δy·cos θ_a,
                             θ_b − θ_a]

    Returns
    -------
    h_val : valore della funzione di misura
    Ha    : Jacobiano rispetto a x^a
    Hb    : Jacobiano rispetto a x^b
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
# Agente IMDCL
# ===========================================================================

class AgentIMDCL:
    """
    Agente che implementa l'algoritmo IMDCL (Algorithm 2, Kia et al. 2016).

    Ogni agente mantiene in locale:
      - x_hat  : stima dello stato (n_i,)
      - P      : covarianza dell'errore (n_i × n_i)
      - Phi    : matrice intermedia Φ^i, inizializzata a I (n_i × n_i)
      - Pi_jl  : dizionario {(j,l): Π_{jl}^i} — copie locali delle
                 covarianze incrociate per tutte le coppie (j < l) nel team
                 eccetto quelle che includono l'id dell'agente stesso
                 (Eq. S5 / S12c).

    Parameters
    ----------
    agent_id    : identificatore univoco intero dell'agente.
    x0          : stima iniziale dello stato.
    P0          : covarianza iniziale dell'errore di stima.
    team_ids    : lista di tutti gli id del team (incluso questo agente).
    motion_model: istanza di MotionModel (sostituibile).
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

        # Stato e covarianza (Eq. S5)
        self.x_hat: Vector = np.array(x0, dtype=float).ravel()
        self.P: Matrix = np.array(P0, dtype=float)

        # Matrice intermedia Φ^i — inizializzata a I (Eq. S5)
        self.Phi: Matrix = np.eye(n)

        # Copie locali delle covarianze incrociate Π_{jl}^i
        # Per simmetria del paper si mantengono solo le coppie con j < l
        # e j ≠ id, l ≠ id  (Eq. S5: inizializzate a zero)
        other_ids = sorted(v for v in team_ids if v != agent_id)
        self.Pi_jl: Dict[Tuple[int, int], Matrix] = {}
        for idx_j, j in enumerate(other_ids):
            for l in other_ids[idx_j + 1:]:
                self.Pi_jl[(j, l)] = np.zeros((n, n))

        self._n = n
        self._team_ids = list(team_ids)

    # ------------------------------------------------------------------
    # Proprietà di sola lettura utili per il debug
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # 1. Propagazione (locale, nessuna comunicazione)
    # ------------------------------------------------------------------

    def propagate(self, u: Vector, dt: float) -> None:
        """
        Fase di propagazione — Algorithm 2, step 1 (Eq. S6).

        Aggiorna localmente x_hat, P e Phi senza alcuna comunicazione.

        Parameters
        ----------
        u  : vettore di controllo/misura odometrica u^i(k).
        dt : intervallo di campionamento.
        """
        F = self.motion_model.F_jacobian(self.x_hat, u, dt)
        G = self.motion_model.G_jacobian(self.x_hat, dt)
        Q = self.motion_model.Q(dt)

        # x^{i-}(k+1) = f^i(x^{i+}(k), u^i(k))
        self.x_hat = self.motion_model.f(self.x_hat, u, dt)

        # P^{i-}(k+1) = F^i P^{i+}(k) F^{i⊤} + G^i Q^i G^{i⊤}
        self.P = F @ self.P @ F.T + G @ Q @ G.T

        # Φ^i(k+1) = F^i(k) Φ^i(k)
        self.Phi = F @ self.Phi

    # ------------------------------------------------------------------
    # 2a. Aggiornamento — nessuna misura nel team
    # ------------------------------------------------------------------

    def step_no_measurement(self) -> None:
        """
        Nessuna misura relativa nel team: le variabili propagate diventano
        le stime aggiornate; Pi_jl rimangono invariate (Eq. S7).

        In questa implementazione le variabili sono già in-place, quindi
        questo metodo è un no-op esplicito utile per chiarezza.
        """
        pass  # x_hat, P, Phi già propagati; Pi_jl invariati per (14)

    # ------------------------------------------------------------------
    # 2b. Generazione del landmark-message (agente landmark b)
    # ------------------------------------------------------------------

    def make_landmark_message(self) -> LandmarkMessage:
        """
        Crea il messaggio da inviare all'interim master (Eq. S8).

        Chiamato dall'agente landmark b prima che l'interim master a
        esegua i propri calcoli.
        """
        return LandmarkMessage(
            agent_id=self.id,
            x_hat=self.x_hat.copy(),
            P=self.P.copy(),
            Phi=self.Phi.copy(),
        )

    # ------------------------------------------------------------------
    # 2c. Calcolo update-message (agente interim master a)
    # ------------------------------------------------------------------

    def compute_update_message(
        self,
        lm_msg: LandmarkMessage,
        z_ab: Vector,
        R_ab: Matrix,
        measurement_fn=relative_pose_measurement,
    ) -> UpdateMessage:
        """
        L'agente a (interim master) calcola il messaggio di aggiornamento
        da broadcast al team intero (Eq. S9, S10).

        Parameters
        ----------
        lm_msg         : LandmarkMessage ricevuto dall'agente landmark b.
        z_ab           : misura relativa effettuata da a verso b.
        R_ab           : covarianza del rumore di misura.
        measurement_fn : funzione h(x_a, x_b) → (h_val, Ha, Hb).

        Returns
        -------
        UpdateMessage da passare a tutti gli agenti tramite apply_update().
        """
        x_a, P_a, Phi_a = self.x_hat, self.P, self.Phi
        x_b = lm_msg.x_hat
        P_b = lm_msg.P
        Phi_b = lm_msg.Phi
        b = lm_msg.agent_id

        # Jacobiani della funzione di misura valutati nello stato propagato
        h_val, Ha_tilde, Hb_tilde = measurement_fn(x_a, x_b)

        # Innovazione  r^a = z_{ab} − h(x^{a-}, x^{b-})  (Eq. S9a)
        r = np.array(z_ab).ravel() - h_val

        # Recupera Π_{ab}^a (o Π_{ba}^a) dalla copia locale
        Pi_ab = self._get_Pi(self.id, b)   # Π_{ab}^a  →  Π tra a e b

        # Covarianza dell'innovazione  S_{ab}  (Eq. S9b)
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

        # Fattore S^{-1/2}  (usato per normalizzare Γ e conservare in msg)
        S_inv_sqrt = self._matrix_inv_sqrt(S)

        # Termini di aggiornamento Γ_a, Γ_b  (Eq. S9c del paper)
        #
        # Dal paper (segni esatti):
        #   Γ_a = ((Φ^a)^{-1} P^a H̃_a⊤ − (Φ^a)^{-1} Φ^a Π_{ab}^a Φ^{b⊤} H̃_b⊤) S^{-1/2}
        #        = (Φ^a)^{-1} (P^a H̃_a⊤ − Π_{ab}^a Φ^{b⊤} H̃_b⊤) S^{-1/2}
        #
        #   Γ_b = ((Φ^b)^{-1} P^b H̃_b⊤ − (Φ^b)^{-1} Π_{ba}^a Φ^{a⊤} H̃_a⊤) S^{-1/2}
        #        = (Φ^b)^{-1} (P^b H̃_b⊤ − Π_{ab}^{a⊤} Φ^{a⊤} H̃_a⊤) S^{-1/2}
        #
        # Nota: Φ^a_inv @ Φ^a = I  →  il termine Π si semplifica a Pi_ab direttamente.
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

        # Termini ausiliari per consentire ad ogni agente i di calcolare Γ_i (Eq. S11)
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
    # 2d. Applicazione dell'update-message (tutti gli agenti)
    # ------------------------------------------------------------------

    def apply_update(self, msg: UpdateMessage) -> None:
        """
        Ogni agente i ∈ V riceve l'update-message e aggiorna le proprie
        variabili locali (Eq. S11, S12a–S12c).

        Parameters
        ----------
        msg : UpdateMessage broadcast dall'interim master.
        """
        a = msg.master_id
        b = msg.landmark_id

        # --- Calcolo di Γ_i per questo agente (Eq. S11) ---
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

        # r̃^a = S^{-1/2} r^a  (innovazione normalizzata)
        r_tilde = msg.S_inv_sqrt @ msg.innovation

        # --- Aggiornamento stato (Eq. S12a) ---
        # x^{i+}(k+1) = x^{i-}(k+1) + Φ^i(k+1) Γ_i r̃^a
        self.x_hat = self.x_hat + self.Phi @ Gamma_i @ r_tilde

        # --- Aggiornamento covarianza (Eq. S12b) ---
        # P^{i+}(k+1) = P^{i-}(k+1) − Φ^i Γ_i Γ_i⊤ Φ^{i⊤}
        self.P = self.P - self.Phi @ Gamma_i @ Gamma_i.T @ self.Phi.T

        # --- Aggiornamento copie locali Π_{jl}^i (Eq. S12c) ---
        # Π_{jl}^i(k+1) = Π_{jl}^i(k) − Γ_j Γ_l⊤
        for (j, l) in list(self.Pi_jl.keys()):
            Gamma_j = self._compute_gamma(j, msg)
            Gamma_l = self._compute_gamma(l, msg)
            self.Pi_jl[(j, l)] = self.Pi_jl[(j, l)] - Gamma_j @ Gamma_l.T

        # Azzera Φ^i dopo ogni aggiornamento (Eq. S5/S6: Φ reset a I dopo update)
        self.Phi = np.eye(self._n)

    # ------------------------------------------------------------------
    # 2e. Aggiornamento con misura assoluta (locale, nessuna comunicazione)
    # ------------------------------------------------------------------

    def apply_absolute_update(
        self,
        z: Vector,
        H: Matrix,
        R: Matrix,
    ) -> None:
        """
        Aggiornamento EKF con una misura assoluta **locale** dell'agente stesso
        (nessuna comunicazione richiesta).

        Nel framework IMDCL una misura assoluta di un agente i è equivalente
        a una misura relativa con a = b = i (landmark = master = stesso agente).
        Il guadagno di Kalman si riduce alla forma standard:

            K  = P · H^T · (H · P · H^T + R)^{-1}
            x̂  ← x̂ + Φ · K · (z − H · x̂)
            P  ← P  − Φ · K · (H · P · H^T + R) · K^T · Φ^T

        Dopo l'aggiornamento Φ viene resettato a I (come per apply_update).

        Il termine Π_{jl} **non viene modificato** perché la misura assoluta
        riguarda solo questo agente: non crea correlazioni incrociate nuove
        (cfr. paper §"Cooperative Localization via EKF", misure assolute).

        Parameters
        ----------
        z : vettore di misura                  (m,)
        H : matrice di osservazione ∂h/∂x      (m × n)
        R : covarianza rumore di misura        (m × m)
        """
        z = np.asarray(z, dtype=float).ravel()
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        # Innovazione
        innov = z - H @ self.x_hat                         # (m,)

        # Covarianza dell'innovazione
        S = H @ self.P @ H.T + R                           # (m × m)

        # Guadagno di Kalman standard
        K = self.P @ H.T @ np.linalg.inv(S)               # (n × m)

        # Aggiornamento stato: x̂ ← x̂ + Φ K (z − H x̂)
        self.x_hat = self.x_hat + self.Phi @ K @ innov

        # Aggiornamento covarianza (Joseph form per stabilità numerica)
        I_KH = np.eye(self._n) - self.Phi @ K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.Phi @ K @ R @ K.T @ self.Phi.T

        # Reset Φ → I dopo ogni aggiornamento
        self.Phi = np.eye(self._n)

    # ------------------------------------------------------------------
    # Metodi ausiliari privati
    # ------------------------------------------------------------------

    def _get_Pi(self, i: int, j: int) -> Matrix:
        """
        Restituisce Π_{ij}^{self} dalla copia locale, rispettando la
        convenzione j > i (simmetria: Π_{ji} = Π_{ij}⊤).
        """
        if i == j:
            # Π_{ii} ≡ P^i  (autocoppia — non viene mai usata direttamente)
            return self.P.copy()
        key = (min(i, j), max(i, j))
        mat = self.Pi_jl.get(key, np.zeros((self._n, self._n)))
        # Se i > j la matrice è trasposta
        return mat if i < j else mat.T

    def _compute_gamma(self, agent_j: int, msg: UpdateMessage) -> Matrix:
        """
        Calcola Γ_j per un agente j qualsiasi del team, dato l'update-message.
        Usato internamente durante l'aggiornamento di Pi_jl.
        """
        a, b = msg.master_id, msg.landmark_id
        if agent_j == a:
            return msg.Gamma_a
        if agent_j == b:
            return msg.Gamma_b
        Pi_jb = self._get_Pi(agent_j, b)
        Pi_ja = self._get_Pi(agent_j, a)
        return (
            Pi_jb @ msg.Phi_b_T_Hb_T_S_inv_sqrt
            - Pi_ja @ msg.Phi_a_T_Ha_T_S_inv_sqrt
        )

    @staticmethod
    def _matrix_inv_sqrt(M: Matrix) -> Matrix:
        """
        Calcola M^{-1/2} tramite decomposizione agli autovalori.

        Per una matrice simmetrica definita positiva M = V Λ V⊤:
            M^{-1/2} = V Λ^{-1/2} V⊤
        """
        eigvals, eigvecs = np.linalg.eigh(M)
        # Clamp numerico per evitare autovalori negativi per errori di floating point
        eigvals = np.maximum(eigvals, 1e-12)
        inv_sqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
        return eigvecs @ inv_sqrt_diag @ eigvecs.T

    # ------------------------------------------------------------------
    # Rappresentazione testuale
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AgentIMDCL(id={self.id}, "
            f"x={np.round(self.x_hat, 3)}, "
            f"P_trace={np.trace(self.P):.4f})"
        )


# ===========================================================================
# Esempio d'uso minimale
# ===========================================================================

if __name__ == "__main__":
    # Esempio di istanziazione di un agente IMDCL con modello di moto
    # PointMass3DModel e rumore di accelerazione σ_acc = 0.1 m/s².
    motion_model = PointMass3DModel(sigma_acc=0.1)
    agent = AgentIMDCL(
        agent_id=1,
        x0=np.zeros(6),          # stato iniziale: posizione e velocità nulle
        P0=np.eye(6) * 0.5,     # covarianza iniziale: incertezza moderata
        team_ids=[1, 2, 3],     # id del team (incluso questo agente)
        motion_model=motion_model,
    )
    print(agent)
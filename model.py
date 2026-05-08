import numpy as np
import abc

Vector = np.ndarray   # colonna (n,) o (n,1)
Matrix = np.ndarray   # matrice 2
# ===========================================================================
# Interfaccia del modello di moto
# ===========================================================================

class MotionModel(abc.ABC):
    """
    Interfaccia astratta per il modello di moto di un agente.

    Sottoclassifica questa classe per supportare cinematiche diverse
    (unicycle, quadrotor, AUV, …).  L'unico requisito è implementare
    i quattro metodi astratti qui sotto.
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Dimensione del vettore di stato n_i."""

    @property
    @abc.abstractmethod
    def noise_dim(self) -> int:
        """Dimensione del vettore di rumore di processo p_i."""

    @abc.abstractmethod
    def f(self, x: Vector, u: Vector, dt: float) -> Vector:
        """
        Propagazione deterministica dello stato.

        Restituisce  f^i(x^i, u^i)  (Eq. 1 del paper).
        """

    @abc.abstractmethod
    def F_jacobian(self, x: Vector, u: Vector, dt: float) -> Matrix:
        """
        Jacobiano di f rispetto allo stato: ∂f/∂x  (matrice F^i).
        """

    @abc.abstractmethod
    def G_jacobian(self, x: Vector, dt: float) -> Matrix:
        """
        Jacobiano di f rispetto al rumore: ∂f/∂η  (matrice G^i).
        """

    @abc.abstractmethod
    def Q(self, dt: float) -> Matrix:
        """Covarianza del rumore di processo Q^i(k)."""


# ===========================================================================
# Modello di moto unicycle (2-D)
# ===========================================================================

class UnicycleModel(MotionModel):
    """
    Modello unicycle discreto su piano 2-D.

    Stato  : x = [px, py, θ]^T
    Ingresso: u = [v, ω]^T   (velocità lineare e angolare *misurate*)
    Rumore  : η = [η_v, η_ω]^T  (rumore additivo sulle misure odometriche)

    Equazione di moto (Eulero):
        px(k+1) = px(k) + v·cos(θ)·dt
        py(k+1) = py(k) + v·sin(θ)·dt
        θ(k+1)  = θ(k)  + ω·dt

    Il rumore entra tramite la stessa struttura:
        G·η  con  G = [cos(θ)·dt, 0; sin(θ)·dt, 0; 0, dt]
    """

    def __init__(self, sigma_v: float, sigma_omega: float) -> None:
        """
        Parameters
        ----------
        sigma_v     : deviazione standard del rumore sulla velocità lineare [m/s].
        sigma_omega : deviazione standard del rumore sulla velocità angolare [rad/s].
        """
        self._sigma_v = sigma_v
        self._sigma_omega = sigma_omega

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def noise_dim(self) -> int:
        return 2

    def f(self, x: Vector, u: Vector, dt: float) -> Vector:
        px, py, th = x.ravel()
        v, w = u.ravel()
        return np.array([
            px + v * np.cos(th) * dt,
            py + v * np.sin(th) * dt,
            th + w * dt,
        ])

    def F_jacobian(self, x: Vector, u: Vector, dt: float) -> Matrix:
        _, _, th = x.ravel()
        v, _ = u.ravel()
        return np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0,  v * np.cos(th) * dt],
            [0.0, 0.0,  1.0],
        ])

    def G_jacobian(self, x: Vector, dt: float) -> Matrix:
        _, _, th = x.ravel()
        return np.array([
            [np.cos(th) * dt, 0.0],
            [np.sin(th) * dt, 0.0],
            [0.0,             dt],
        ])

    def Q(self, dt: float) -> Matrix:
        return np.diag([self._sigma_v**2, self._sigma_omega**2])


# ===========================================================================
# Modello di moto massa puntiforme 3-D (controllo in accelerazione)
# ===========================================================================

class PointMass3DModel(MotionModel):
    """
    Massa puntiforme in 3-D controllata in accelerazione, con dinamica
    discreta di tipo doppio integratore (zero-order-hold esatto).

    Stato   : x = [px, py, pz, vx, vy, vz]^T   (posizione + velocità)
    Ingresso: u = [ax, ay, az]^T                 (accelerazione comandata,
                                                  già compensata dalla gravità
                                                  se necessario)
    Rumore  : η = [ηax, ηay, ηaz]^T              (rumore additivo sull'accel.)

    Discretizzazione ZOH (esatta per sistemi lineari):
        p(k+1) = p(k) + v(k)·dt + ½·(u(k) + η(k))·dt²
        v(k+1) = v(k) + (u(k) + η(k))·dt

    In forma matriciale:
        x(k+1) = F·x(k) + B·u(k) + G·η(k)

    con:
        F = [[I₃, dt·I₃],   B = G = [[½dt²·I₃],
             [0₃,    I₃]]             [   dt·I₃]]

    Il modello è lineare → F_jacobian = F, G_jacobian = G (costanti).

    Parameters
    ----------
    sigma_acc : deviazione standard del rumore di accelerazione [m/s²]
                (uguale per i tre assi; passare un array (3,) per valori
                 diversi per asse).
    """

    def __init__(self, sigma_acc: float | np.ndarray) -> None:
        sigma = np.broadcast_to(np.asarray(sigma_acc, dtype=float), (3,)).copy()
        self._sigma_acc = sigma   # (3,)

    @property
    def state_dim(self) -> int:
        return 6   # [px, py, pz, vx, vy, vz]

    @property
    def noise_dim(self) -> int:
        return 3   # [ηax, ηay, ηaz]

    def _F(self, dt: float) -> Matrix:
        I3 = np.eye(3)
        return np.block([[I3, dt * I3],
                         [np.zeros((3, 3)), I3]])

    def _G(self, dt: float) -> Matrix:
        I3 = np.eye(3)
        return np.block([[0.5 * dt**2 * I3],
                         [dt * I3]])

    def f(self, x: Vector, u: Vector, dt: float) -> Vector:
        """x(k+1) = F·x(k) + G·u(k)  (il rumore è separato in G·η)."""
        return self._F(dt) @ x.ravel() + self._G(dt) @ u.ravel()

    def F_jacobian(self, x: Vector, u: Vector, dt: float) -> Matrix:
        """Jacobiano di f rispetto a x — costante (modello lineare)."""
        return self._F(dt)

    def G_jacobian(self, x: Vector, dt: float) -> Matrix:
        """Jacobiano di f rispetto a η — costante (modello lineare)."""
        return self._G(dt)

    def Q(self, dt: float) -> Matrix:
        """Covarianza del rumore di processo: diag(σ_ax², σ_ay², σ_az²)."""
        return np.diag(self._sigma_acc**2)

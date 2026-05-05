"""
MPC per droni modellati come massa puntiforme 3-D
=================================================
Usa CasADi + IPOPT con l'interfaccia cs.Opti(), stesso stile del progetto
double-pendulum.

Modello di moto (da imdcl.py — doppio integratore ZOH):

    x = [px, py, pz, vx, vy, vz]^T
    u = [ax, ay, az]^T   (accelerazione — gravità già compensata)

    x(k+1) = F·x(k) + B·u(k)
        F = [[I3, dt·I3], [0, I3]]
        B = [[0.5·dt²·I3], [dt·I3]]

Funzione costo quadratica sull'orizzonte N:

    J = Σ_{k=0}^{N-1} [ (x_k − x_ref)^T Q (x_k − x_ref) + u_k^T R u_k ]
        + (x_N − x_ref)^T P_f (x_N − x_ref)

Vincoli:
    x(k+1) = F·x(k) + B·u(k)       dinamica
    |u(k)|_∞ ≤ a_max                saturazione accelerazione
    |v(k)|_∞ ≤ v_max                saturazione velocità

Utilizzo
--------
    python mpc_drone.py               # lancia la simulazione con plot
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from time import perf_counter

try:
    import casadi as cs
except ImportError:
    raise ImportError(
        "CasADi non trovato. Installalo con:  pip install casadi"
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from imdcl import PointMass3DModel


# ===========================================================================
# Parametri globali (modificabili)
# ===========================================================================

# Orizzonte MPC e passo temporale
N_MPC   = 20
DT_MPC  = 0.1    # [s]

# Pesi costo
W_POS     = 1e2   # peso sulla posizione
W_VEL     = 1e0   # peso sulla velocità
W_ACC     = 1e-1  # peso sull'accelerazione (ingresso)
W_FINAL_P = 1e3   # peso terminale sulla posizione
W_FINAL_V = 1e1   # peso terminale sulla velocità

# Saturazioni
A_MAX = 6.0   # [m/s²]  per asse
V_MAX = 5.0   # [m/s]   per asse

# Tolleranza convergenza IPOPT
SOLVER_TOLERANCE = 1e-4
SOLVER_MAX_ITER  = 3      # iterazioni per i passi MPC (dopo warm-start)
SOLVER_MAX_ITER_INIT = 1000  # iterazioni per il primo solve (warm-start)

# Simulazione
N_SIM       = 200         # passi massimi
DT_SIM      = DT_MPC     # passo simulazione = passo MPC
SIGMA_ACC   = 0.05        # [m/s²] rumore additivo sull'accelerazione
STOP_THRESH = 0.05        # [m]  soglia di arrivo


# ===========================================================================
# Controllore MPC
# ===========================================================================

class DroneMPC:
    """
    Controllore MPC per drone (massa puntiforme 3-D) con CasADi Opti.

    Parameters
    ----------
    dt    : passo di campionamento [s]
    N     : orizzonte di predizione
    a_max : accelerazione massima per asse [m/s²]
    v_max : velocità massima per asse [m/s]
    """

    def __init__(
        self,
        dt:    float = DT_MPC,
        N:     int   = N_MPC,
        a_max: float = A_MAX,
        v_max: float = V_MAX,
    ) -> None:
        self.dt    = dt
        self.N     = N
        self.a_max = a_max
        self.v_max = v_max

        nx, nu = 6, 3
        self.nx = nx
        self.nu = nu

        # Matrici di sistema (costanti — modello lineare)
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        self.F_np = np.block([[I3, dt * I3], [Z3, I3]])          # 6×6
        self.B_np = np.block([[0.5 * dt**2 * I3], [dt * I3]])    # 6×3

        # Matrici peso
        self.Q   = np.diag([W_POS, W_POS, W_POS, W_VEL, W_VEL, W_VEL])
        self.R   = np.diag([W_ACC, W_ACC, W_ACC])
        self.P_f = np.diag([W_FINAL_P, W_FINAL_P, W_FINAL_P,
                            W_FINAL_V, W_FINAL_V, W_FINAL_V])

        # Costruisce il problema Opti (una sola volta)
        self._build_opti()
        self._sol = None   # terrà l'ultima soluzione per warm-start

    # ------------------------------------------------------------------
    def _build_opti(self):
        """Costruisce il problema di ottimizzazione con cs.Opti."""
        nx, nu, N = self.nx, self.nu, self.N

        opti = cs.Opti()

        # — Parametri (aggiornati a ogni passo MPC) —
        self.param_x0    = opti.parameter(nx)   # stato corrente
        self.param_x_ref = opti.parameter(nx)   # stato di riferimento

        # — Variabili di decisione —
        X = [opti.variable(nx) for _ in range(N + 1)]
        U = [opti.variable(nu) for _ in range(N)]

        # Bounds sugli stati (solo velocità)
        for k in range(N + 1):
            opti.subject_to(opti.bounded(-v_max_vec(self.v_max),
                                         X[k][3:], v_max_vec(self.v_max)))

        # Bounds sugli ingressi (accelerazione)
        for k in range(N):
            opti.subject_to(opti.bounded(
                -self.a_max * np.ones(nu),
                 U[k],
                  self.a_max * np.ones(nu),
            ))

        # — Funzione costo —
        F_cs = cs.DM(self.F_np)
        B_cs = cs.DM(self.B_np)
        Q_cs = cs.DM(self.Q)
        R_cs = cs.DM(self.R)
        Pf_cs= cs.DM(self.P_f)

        cost = 0
        for k in range(N):
            dx    = X[k] - self.param_x_ref
            cost += dx.T @ Q_cs @ dx + U[k].T @ R_cs @ U[k]
        dx_N  = X[N] - self.param_x_ref
        cost += dx_N.T @ Pf_cs @ dx_N

        opti.minimize(cost)

        # — Vincoli di dinamica —
        opti.subject_to(X[0] == self.param_x0)
        for k in range(N):
            opti.subject_to(X[k + 1] == F_cs @ X[k] + B_cs @ U[k])

        # — Opzioni solver —
        opts_init = {
            "ipopt.print_level":      0,
            "ipopt.tol":              SOLVER_TOLERANCE,
            "ipopt.constr_viol_tol":  SOLVER_TOLERANCE,
            "ipopt.compl_inf_tol":    SOLVER_TOLERANCE,
            "print_time":             False,
            "detect_simple_bounds":   True,
            "ipopt.max_iter":         SOLVER_MAX_ITER_INIT,
        }
        opti.solver("ipopt", opts_init)

        self._opti = opti
        self._X    = X
        self._U    = U

        # Opzioni ridotte per i passi a regime (sostituite dopo il warm-start)
        self._opts_mpc = {
            **opts_init,
            "ipopt.max_iter": SOLVER_MAX_ITER,
        }

    # ------------------------------------------------------------------
    def _x_ref_from_target(self, target: np.ndarray) -> np.ndarray:
        """Costruisce lo stato di riferimento: target pos + velocità zero."""
        x_ref = np.zeros(self.nx)
        x_ref[:3] = np.asarray(target, dtype=float).ravel()[:3]
        return x_ref

    # ------------------------------------------------------------------
    def warm_start(self, x0: np.ndarray, target: np.ndarray) -> None:
        """
        Primo solve a convergenza completa (SOLVER_MAX_ITER_INIT iter).
        Va chiamato una volta prima del loop MPC.
        """
        x_ref = self._x_ref_from_target(target)
        self._opti.set_value(self.param_x0,    x0)
        self._opti.set_value(self.param_x_ref, x_ref)
        self._sol = self._opti.solve()

        # Dopo il warm-start, abbassa il limite di iterazioni
        self._opti.solver("ipopt", self._opts_mpc)

    # ------------------------------------------------------------------
    def step(self, x0: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Un passo MPC: aggiorna parametri, shifta la soluzione precedente
        come initial guess, risolve, restituisce u*(0).

        Parameters
        ----------
        x0     : stato corrente (6,)
        target : punto target 3-D (3,)

        Returns
        -------
        u_opt : accelerazione ottima (3,)
        """
        N   = self.N
        sol = self._sol

        # — Shift dell'orizzonte (warm-start per il passo successivo) —
        for k in range(N):
            self._opti.set_initial(self._X[k], sol.value(self._X[k + 1]))
        for k in range(N - 1):
            self._opti.set_initial(self._U[k], sol.value(self._U[k + 1]))
        # Ultimi valori: replica l'ultimo elemento
        self._opti.set_initial(self._X[N], sol.value(self._X[N]))
        self._opti.set_initial(self._U[N - 1], sol.value(self._U[N - 1]))

        # Warm-start sui moltiplicatori di Lagrange
        lam_g0 = sol.value(self._opti.lam_g)
        self._opti.set_initial(self._opti.lam_g, lam_g0)

        # — Aggiorna parametri —
        x_ref = self._x_ref_from_target(target)
        self._opti.set_value(self.param_x0,    x0)
        self._opti.set_value(self.param_x_ref, x_ref)

        # — Risolvi —
        try:
            self._sol = self._opti.solve()
        except Exception:
            # Se IPOPT non converge entro max_iter usa la soluzione debug
            self._sol = self._opti.debug

        return np.array(self._sol.value(self._U[0])).ravel()

    # ------------------------------------------------------------------
    def predicted_trajectory(self) -> np.ndarray:
        """Restituisce la traiettoria predetta (N+1)×6 dall'ultima soluzione."""
        return np.array([
            self._sol.value(self._X[k]).ravel()
            for k in range(self.N + 1)
        ])


def v_max_vec(v_max: float) -> np.ndarray:
    return v_max * np.ones(3)


# ===========================================================================
# Simulazione
# ===========================================================================

def simulate(
    starts:   dict,
    targets:  dict,
    dt:       float = DT_SIM,
    n_steps:  int   = N_SIM,
    sigma:    float = SIGMA_ACC,
    rng_seed: int   = 42,
) -> tuple[dict, dict, dict, dict]:
    """
    Simula più droni con MPC verso i rispettivi target (waypoint sequenziali).

    Parameters
    ----------
    starts  : {id: x0 (6,)}
    targets : {id: array-like}
              Accetta due formati:
                - singolo target  →  np.array([x, y, z])
                - lista waypoint  →  [np.array([x,y,z]), np.array([...]), ...]
              I droni non devono avere lo stesso numero di waypoint.

    Returns
    -------
    history   : {id: list of x (6,)}
    inputs    : {id: list of u (3,)}
    solve_t   : {id: list of float}    tempi di solve [s]
    waypoints : {id: list of np.array} sequenza waypoint normalizzata
    """
    rng       = np.random.default_rng(rng_seed)
    model     = PointMass3DModel(sigma_acc=sigma)
    drone_ids = list(starts.keys())

    # — Normalizza targets: ogni drone ha sempre una lista di waypoint —
    waypoints = {}
    for i in drone_ids:
        t = targets[i]
        # singolo array 1-D → lista con un elemento
        if isinstance(t, np.ndarray) and t.ndim == 1:
            waypoints[i] = [t]
        else:
            waypoints[i] = [np.asarray(w, dtype=float) for w in t]

    # — Indice waypoint corrente per ogni drone —
    wp_idx = {i: 0 for i in drone_ids}

    def current_target(i):
        return waypoints[i][wp_idx[i]]

    # — Crea controllori e warm-start sul primo waypoint —
    ctrls = {i: DroneMPC(dt=dt) for i in drone_ids}
    x_cur = {i: starts[i].copy() for i in drone_ids}

    history = {i: [x_cur[i].copy()] for i in drone_ids}
    inputs  = {i: []                 for i in drone_ids}
    solve_t = {i: []                 for i in drone_ids}

    print("Warm-start iniziale...")
    for i in drone_ids:
        t0 = perf_counter()
        ctrls[i].warm_start(x_cur[i], current_target(i))
        n_wp = len(waypoints[i])
        print(f"  Drone {i}: warm-start in {perf_counter()-t0:.3f} s  "
              f"({n_wp} waypoint)")

    # — Header log —
    print(f"\n{'Step':>5}  {'Time':>7}  " +
          "  ".join(f"D{i}:wp/dist/t_solve" for i in drone_ids))

    # — Loop MPC —
    for step in range(n_steps):
        for i in drone_ids:
            tgt = current_target(i)

            t0    = perf_counter()
            u_opt = ctrls[i].step(x_cur[i], tgt)
            dt_s  = perf_counter() - t0

            noise    = rng.multivariate_normal(
                           np.zeros(3), np.diag([sigma**2] * 3))
            x_cur[i] = model.f(x_cur[i], u_opt + noise, dt)

            history[i].append(x_cur[i].copy())
            inputs[i].append(u_opt.copy())
            solve_t[i].append(dt_s)

            # — Controlla se il drone ha raggiunto il waypoint corrente —
            dist = np.linalg.norm(x_cur[i][:3] - tgt)
            if dist < STOP_THRESH and wp_idx[i] < len(waypoints[i]) - 1:
                wp_idx[i] += 1
                new_tgt = current_target(i)
                print(f"  ► Drone {i} raggiunto wp {wp_idx[i]-1} "
                      f"al passo {step+1} (t={step*dt:.2f}s)  "
                      f"→ wp {wp_idx[i]}: {new_tgt}")
                # Aggiorna solo il riferimento — la soluzione corrente
                # rimane come warm-start per il nuovo waypoint
                x_ref = ctrls[i]._x_ref_from_target(new_tgt)
                ctrls[i]._opti.set_value(ctrls[i].param_x_ref, x_ref)

        # Log ogni 10 passi
        if (step + 1) % 10 == 0:
            row = f"{step+1:>5}  {(step+1)*dt:>6.2f}s  "
            for i in drone_ids:
                dist = np.linalg.norm(x_cur[i][:3] - current_target(i))
                row += (f"  {wp_idx[i]}/{len(waypoints[i])-1} "
                        f"{dist:>6.2f}m  {solve_t[i][-1]*1e3:>5.1f}ms")
            print(row)

        # Stop globale: tutti i droni hanno completato tutti i waypoint
        all_done = all(
            wp_idx[i] == len(waypoints[i]) - 1 and
            np.linalg.norm(x_cur[i][:3] - current_target(i)) < STOP_THRESH
            for i in drone_ids
        )
        if all_done:
            print(f"\nTutti i droni hanno completato la missione "
                  f"al passo {step+1} (t = {(step+1)*dt:.2f} s).")
            break

    return history, inputs, solve_t, waypoints


# ===========================================================================
# Plot
# ===========================================================================

COLORS = {0: "#e63946", 1: "#2a9d8f", 2: "#e9c46a"}


def plot_results(starts, targets, history, inputs, solve_t, dt,
                 waypoints=None):
    drone_ids = list(starts.keys())
    time_u    = {i: np.arange(len(inputs[i])) * dt for i in drone_ids}
    time_x    = {i: np.arange(len(history[i])) * dt for i in drone_ids}

    # Normalizza waypoints se non passati esplicitamente
    if waypoints is None:
        waypoints = {}
        for i in drone_ids:
            t = targets[i]
            waypoints[i] = ([t] if isinstance(t, np.ndarray) and t.ndim == 1
                            else [np.asarray(w) for w in t])

    # ── Figura 1: traiettorie 3-D + proiezioni ──────────────────────────────
    fig1 = plt.figure(figsize=(17, 9))
    fig1.patch.set_facecolor("#ffffff")

    ax3d  = fig1.add_subplot(2, 3, (1, 4), projection="3d")
    ax_xy = fig1.add_subplot(2, 3, 2)
    ax_xz = fig1.add_subplot(2, 3, 3)
    ax_yz = fig1.add_subplot(2, 3, 5)
    ax_u  = fig1.add_subplot(2, 3, 6)

    proj_cfg = [
        (ax_xy, 0, 1, "x [m]", "y [m]", "XY"),
        (ax_xz, 0, 2, "x [m]", "z [m]", "XZ"),
        (ax_yz, 1, 2, "y [m]", "z [m]", "YZ"),
    ]

    for i in drone_ids:
        c    = COLORS.get(i, "#888888")
        traj = np.array(history[i])
        us   = np.array(inputs[i])
        wps  = waypoints[i]
        wp_arr = np.array(wps)

        # Vista 3-D — traiettoria
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color=c, lw=1.8, alpha=0.9, label=f"Drone {i}")
        ax3d.scatter(*traj[0, :3], marker="o", color=c, s=60,
                     edgecolors="white", linewidths=1.2, zorder=5)
        # Waypoint collegati da linea tratteggiata
        ax3d.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                  color=c, lw=0.8, ls=":", alpha=0.5)
        for k, wp in enumerate(wps):
            ax3d.scatter(*wp, marker="*", color=c,
                         s=200 if k == len(wps)-1 else 120,
                         edgecolors="white", linewidths=0.8, zorder=6)
            ax3d.text(wp[0], wp[1], wp[2]+0.15, f" wp{k}",
                      color=c, fontsize=6.5, alpha=0.85)

        # Proiezioni
        for ax2, ix, iy, xl, yl, name in proj_cfg:
            ax2.plot(traj[:, ix], traj[:, iy],
                     color=c, lw=1.6, alpha=0.85)
            ax2.plot(traj[0, ix], traj[0, iy], "o",
                     color=c, ms=6, mec="white", mew=1.2, zorder=5)
            ax2.plot(wp_arr[:, ix], wp_arr[:, iy],
                     color=c, lw=0.7, ls=":", alpha=0.5)
            for k, wp in enumerate(wps):
                ax2.plot(wp[ix], wp[iy], "*", color=c,
                         ms=10 if k == len(wps)-1 else 7,
                         mec="white", mew=0.6, zorder=6)
            ax2.set_xlabel(xl, fontsize=8)
            ax2.set_ylabel(yl, fontsize=8)
            ax2.set_title(f"Proiezione {name}", fontsize=9,
                          fontweight="bold", pad=3)
            ax2.set_aspect("equal")
            ax2.grid(True, ls=":", alpha=0.45)
            ax2.tick_params(labelsize=7)
            ax2.set_facecolor("#f8f8f8")

        # Norma accelerazione nel tempo
        ax_u.plot(time_u[i], np.linalg.norm(us, axis=1),
                  color=c, lw=1.5, label=f"Drone {i}")

    ax_u.axhline(A_MAX * np.sqrt(3), color="grey", lw=1.0, ls="--",
                 alpha=0.6, label=f"|u|₂ max = {A_MAX*np.sqrt(3):.2f}")
    ax_u.set_xlabel("Tempo [s]", fontsize=8)
    ax_u.set_ylabel("|u|₂  [m/s²]", fontsize=8)
    ax_u.set_title("Norma accelerazione", fontsize=9, fontweight="bold", pad=3)
    ax_u.legend(fontsize=7.5); ax_u.grid(True, ls=":", alpha=0.45)
    ax_u.tick_params(labelsize=7); ax_u.set_facecolor("#f8f8f8")

    ax3d.set_xlabel("x [m]", fontsize=9, labelpad=5)
    ax3d.set_ylabel("y [m]", fontsize=9, labelpad=5)
    ax3d.set_zlabel("z [m]", fontsize=9, labelpad=5)
    ax3d.set_title("Traiettorie 3-D", fontsize=10, fontweight="bold")
    ax3d.tick_params(labelsize=7)
    handles = []
    for i in drone_ids:
        c = COLORS.get(i, "#888888")
        n_wp = len(waypoints[i])
        handles.append(plt.Line2D([0],[0], color=c, lw=2,
                                  label=f"Drone {i}  ({n_wp} wp)"))
    handles += [
        plt.Line2D([0],[0], marker="*", color="w", mfc="grey",
                   ms=10, linestyle="None", label="Waypoint"),
        plt.Line2D([0],[0], marker="o", color="w", mfc="grey",
                   ms=7,  linestyle="None", label="Inizio"),
    ]
    ax3d.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.85)
    fig1.suptitle(f"MPC Droni 3-D — N={N_MPC}, dt={DT_MPC} s, "
                  f"|a|≤{A_MAX} m/s², |v|≤{V_MAX} m/s",
                  fontsize=11, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Figura 2: posizione e velocità nel tempo ─────────────────────────────
    fig2, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    fig2.patch.set_facecolor("#ffffff")
    comp_labels = ["x", "y", "z"]

    for row, comp in enumerate(range(3)):
        ax_p, ax_v = axes[row, 0], axes[row, 1]
        for i in drone_ids:
            c    = COLORS.get(i, "#888888")
            traj = np.array(history[i])
            ax_p.plot(time_x[i], traj[:, comp], color=c, lw=1.6,
                      label=f"Drone {i}")
            # Linea tratteggiata per ogni waypoint
            for k, wp in enumerate(waypoints[i]):
                ax_p.axhline(wp[comp], color=c, lw=0.8, ls="--", alpha=0.4)
            ax_v.plot(time_x[i], traj[:, comp+3], color=c, lw=1.6,
                      label=f"Drone {i}")
        ax_v.axhline( V_MAX, color="grey", lw=0.9, ls="--", alpha=0.5)
        ax_v.axhline(-V_MAX, color="grey", lw=0.9, ls="--", alpha=0.5)
        ax_p.set_ylabel(f"p{comp_labels[comp]} [m]", fontsize=9)
        ax_v.set_ylabel(f"v{comp_labels[comp]} [m/s]", fontsize=9)
        for ax in (ax_p, ax_v):
            ax.grid(True, ls=":", alpha=0.4)
            ax.tick_params(labelsize=8)
            ax.set_facecolor("#f8f8f8")
        if row == 0:
            ax_p.set_title("Posizione", fontsize=10, fontweight="bold")
            ax_v.set_title("Velocità",  fontsize=10, fontweight="bold")
            ax_p.legend(fontsize=8, loc="upper right")
        if row == 2:
            ax_p.set_xlabel("Tempo [s]", fontsize=9)
            ax_v.set_xlabel("Tempo [s]", fontsize=9)
    fig2.suptitle("MPC Droni 3-D — Stati nel tempo",
                  fontsize=11, fontweight="bold")
    fig2.tight_layout()

    # ── Figura 3: tempi di solve ──────────────────────────────────────────────
    fig3, ax_t = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor("#ffffff")
    for i in drone_ids:
        c    = COLORS.get(i, "#888888")
        t_ms = np.array(solve_t[i]) * 1e3
        ax_t.plot(time_u[i], t_ms, color=c, lw=1.4, alpha=0.85,
                  label=f"Drone {i}  (mean={np.mean(t_ms):.1f} ms)")
        ax_t.axhline(np.mean(t_ms), color=c, lw=1.0, ls="--", alpha=0.6)
    ax_t.set_xlabel("Tempo [s]", fontsize=10)
    ax_t.set_ylabel("Tempo di solve [ms]", fontsize=10)
    ax_t.set_title(f"Tempo di solve MPC  (max_iter={SOLVER_MAX_ITER})",
                   fontsize=10, fontweight="bold")
    ax_t.legend(fontsize=9); ax_t.grid(True, ls=":", alpha=0.45)
    ax_t.set_facecolor("#f8f8f8")
    fig3.tight_layout()

    return fig1, fig2, fig3


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    # — Scenario: droni con numero diverso di waypoint —
    starts = {
        0: np.array([0.0,  0.0,  0.0,  0.0, 0.0, 0.0]),
        1: np.array([5.0,  0.0,  1.0,  0.0, 0.0, 0.0]),
        2: np.array([2.5,  4.0,  3.0,  0.0, 0.0, 0.0]),
    }
    targets = {
        # Drone 0: tre waypoint
        0: [np.array([4.0,  3.0,  2.0]),
            np.array([2.0,  5.0,  1.0]),
            np.array([0.0,  0.0,  0.5])],
        # Drone 1: due waypoint
        1: [np.array([0.0,  4.0,  0.5]),
            np.array([3.0,  2.0,  3.0])],
        # Drone 2: singolo target (formato vecchio — compatibile)
        2: np.array([5.0, -1.0,  4.0]),
    }

    print("=" * 60)
    print("  MPC — Droni 3-D con waypoint sequenziali")
    print(f"  N={N_MPC}, dt={DT_MPC} s, max_iter={SOLVER_MAX_ITER}")
    print(f"  |a|≤{A_MAX} m/s²,  |v|≤{V_MAX} m/s")
    print("=" * 60)

    history, inputs, solve_t, waypoints = simulate(
        starts=starts,
        targets=targets,
        dt=DT_SIM,
        n_steps=N_SIM,
        sigma=SIGMA_ACC,
    )

    fig1, fig2, fig3 = plot_results(
        starts, targets, history, inputs, solve_t, DT_SIM,
        waypoints=waypoints,
    )
    plt.show()

    from animate_drone import animate
    anim = animate(starts, targets, history, inputs, dt=DT_SIM,
                   waypoints=waypoints, speed=0.5, save=True)
    plt.show()
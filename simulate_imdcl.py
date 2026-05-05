"""
Simulazione IMDCL — due scenari selezionabili
==============================================
Importa AgentIMDCL da imdcl.py.

Imposta MODEL = "unicycle"    →  3 robot unicycle su piano 2-D
        MODEL = "pointmass3d" →  3 masse puntiformi in 3-D controllate
                                  in accelerazione

Per entrambi i modelli vengono prodotte due figure:
  1. Traiettorie reali vs stimate + ellissi di covarianza
     (per il 3-D: vista 3-D + proiezioni XY / XZ / YZ)
  2. Traccia di P nel tempo con le singole varianze per componente
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from imdcl import (
    AgentIMDCL,
    UnicycleModel,
    PointMass3DModel,
    relative_pose_measurement,
    relative_position_measurement_3d,
)

# ===========================================================================
# ── SCEGLI IL MODELLO QUI ──────────────────────────────────────────────────
MODEL = "pointmass3d"   # "unicycle"  oppure  "pointmass3d"
# ===========================================================================

# ---------------------------------------------------------------------------
# Parametri comuni
# ---------------------------------------------------------------------------
RNG_SEED      = 7
DT            = 0.1
N_STEPS       = 300
MEAS_EVERY    = 10
ELLIPSE_EVERY = 50
SIGMA_3       = 3

COLORS = {0: "#e63946", 1: "#2a9d8f", 2: "#e9c46a"}
LABELS = {0: "Agent 0", 1: "Agent 1", 2: "Agent 2"}
TEAM   = [0, 1, 2]

# ---------------------------------------------------------------------------
# Parametri specifici per modello
# ---------------------------------------------------------------------------
if MODEL == "unicycle":
    SIGMA_V     = 0.08
    SIGMA_OMEGA = 0.03
    SIGMA_POS   = 0.05
    SIGMA_ANG   = np.deg2rad(1.5)
    R_MEAS      = np.diag([SIGMA_POS**2, SIGMA_POS**2, SIGMA_ANG**2])

    motion_model   = UnicycleModel(sigma_v=SIGMA_V, sigma_omega=SIGMA_OMEGA)
    measurement_fn = relative_pose_measurement

    CONTROLS = {
        0: np.array([0.6,  0.08]),
        1: np.array([0.5, -0.06]),
        2: np.array([0.4,  0.15]),
    }
    X0 = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([2.0, 0.0, 0.1]),
        2: np.array([1.0, 1.8, np.pi / 2]),
    }
    P0 = np.diag([0.05**2, 0.05**2, np.deg2rad(2)**2])

else:
    # Stato: [px, py, pz, vx, vy, vz]   Ingresso: [ax, ay, az]
    SIGMA_ACC   = 0.15          # m/s²
    SIGMA_POS3D = 0.08          # m
    R_MEAS      = np.diag([SIGMA_POS3D**2] * 3)

    motion_model   = PointMass3DModel(sigma_acc=SIGMA_ACC)
    measurement_fn = relative_position_measurement_3d

    CONTROLS = {
        0: np.array([ 0.0,  0.3,  0.05]),
        1: np.array([ 0.3,  0.0,  0.08]),
        2: np.array([-0.2,  0.2,  0.10]),
    }
    X0 = {
        0: np.array([0.0, 0.0, 0.0,  1.0,  0.0,  0.2]),
        1: np.array([3.0, 0.0, 1.0,  0.0,  0.8,  0.1]),
        2: np.array([1.5, 3.0, 2.0, -0.5,  0.5,  0.3]),
    }
    P0 = np.diag([0.05**2, 0.05**2, 0.05**2,
                  0.10**2, 0.10**2, 0.10**2])


# ===========================================================================
# Simulazione
# ===========================================================================

def run_simulation():
    rng   = np.random.default_rng(RNG_SEED)
    n_u   = motion_model.noise_dim

    x_true = {i: X0[i].copy() for i in TEAM}
    agents = {
        i: AgentIMDCL(
            agent_id=i, x0=X0[i].copy(), P0=P0.copy(),
            team_ids=TEAM, motion_model=motion_model,
        )
        for i in TEAM
    }

    history_true  = {i: [x_true[i].copy()] for i in TEAM}
    history_est   = {i: [agents[i].x_hat.copy()] for i in TEAM}
    history_cov   = {i: [agents[i].P.copy()]     for i in TEAM}
    ellipse_steps = []

    for step in range(N_STEPS):
        # 1. Ground truth
        for i in TEAM:
            noise  = rng.multivariate_normal(np.zeros(n_u), motion_model.Q(DT))
            x_true[i] = motion_model.f(x_true[i], CONTROLS[i] + noise, DT)

        # 2. Propagazione EKF locale
        for i in TEAM:
            noise  = rng.multivariate_normal(np.zeros(n_u), motion_model.Q(DT))
            agents[i].propagate(CONTROLS[i] + noise, DT)

        # 3. Aggiornamento cooperativo
        if (step + 1) % MEAS_EVERY == 0:
            _cooperative_update(agents, 0, 1, x_true, rng)
            _cooperative_update(agents, 1, 2, x_true, rng)
        else:
            for ag in agents.values():
                ag.step_no_measurement()

        # 4. Logging
        for i in TEAM:
            history_true[i].append(x_true[i].copy())
            history_est[i].append(agents[i].x_hat.copy())
            history_cov[i].append(agents[i].P.copy())

        if (step + 1) % ELLIPSE_EVERY == 0 or step == N_STEPS - 1:
            ellipse_steps.append(step + 1)

    return history_true, history_est, history_cov, ellipse_steps


def _cooperative_update(agents, master_id, landmark_id, x_true, rng):
    n_z    = R_MEAS.shape[0]
    h_val, _, _ = measurement_fn(x_true[master_id], x_true[landmark_id])
    z_ab   = h_val + rng.multivariate_normal(np.zeros(n_z), R_MEAS)
    lm_msg = agents[landmark_id].make_landmark_message()
    upd    = agents[master_id].compute_update_message(
                 lm_msg, z_ab, R_MEAS, measurement_fn=measurement_fn)
    for ag in agents.values():
        ag.apply_update(upd)


# ===========================================================================
# Helper: ellisse 2-D
# ===========================================================================

def _cov_ellipse(ax, mean2d, cov2d, n_sigma=3.0, **kw):
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    eigvals = np.maximum(eigvals, 1e-14)
    order   = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ax.add_patch(Ellipse(xy=mean2d,
                         width=2 * n_sigma * np.sqrt(eigvals[0]),
                         height=2 * n_sigma * np.sqrt(eigvals[1]),
                         angle=angle, **kw))


# ===========================================================================
# Plot traiettorie — Unicycle 2-D
# ===========================================================================

def plot_trajectories_2d(history_true, history_est, history_cov, ellipse_steps):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("#f8f8f8")

    for i in TEAM:
        c  = COLORS[i]
        tr = np.array(history_true[i])
        es = np.array(history_est[i])

        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=2.2, alpha=0.9,
                label=f"{LABELS[i]} — reale", zorder=3)
        ax.plot(es[:, 0], es[:, 1], color=c, lw=1.6, ls="--", alpha=0.75,
                label=f"{LABELS[i]} — stimata", zorder=4)
        for k in ellipse_steps:
            _cov_ellipse(ax, es[k, :2], history_cov[i][k][:2, :2],
                         n_sigma=SIGMA_3, facecolor=c, alpha=0.15,
                         edgecolor=c, lw=1.2, zorder=2)
        ax.plot(tr[0,  0], tr[0,  1], "o", color=c, ms=9, zorder=6,
                mec="white", mew=1.5)
        ax.plot(tr[-1, 0], tr[-1, 1], "*", color=c, ms=14, zorder=6,
                mec="white", mew=0.8)

    handles = []
    for i in TEAM:
        c = COLORS[i]
        handles += [
            mpatches.Patch(facecolor=c, edgecolor="none",
                           label=f"{LABELS[i]} — reale"),
            mpatches.Patch(facecolor=c, edgecolor="none", alpha=0.5,
                           label=f"{LABELS[i]} — stimata"),
        ]
    handles += [
        mpatches.Patch(facecolor="grey", alpha=0.3,
                       label=f"Ellisse cov. {SIGMA_3}σ"),
        plt.Line2D([0],[0], marker="o", color="w", mfc="grey", ms=8,
                   label="Inizio"),
        plt.Line2D([0],[0], marker="*", color="w", mfc="grey", ms=11,
                   label="Fine"),
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9,
              loc="upper left", ncol=2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("IMDCL Unicycle 2-D — traiettorie e covarianza (3σ)",
                 fontweight="bold")
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


# ===========================================================================
# Plot traiettorie — Massa puntiforme 3-D
# ===========================================================================

def plot_trajectories_3d(history_true, history_est, history_cov, ellipse_steps):
    """
    Layout: vista 3-D a sinistra + tre proiezioni (XY, XZ, YZ) a destra.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#ffffff")

    ax3d  = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(3, 2, 2)
    ax_xz = fig.add_subplot(3, 2, 4)
    ax_yz = fig.add_subplot(3, 2, 6)

    proj_cfg = [
        (ax_xy, 0, 1, "x [m]", "y [m]", "XY"),
        (ax_xz, 0, 2, "x [m]", "z [m]", "XZ"),
        (ax_yz, 1, 2, "y [m]", "z [m]", "YZ"),
    ]

    for i in TEAM:
        c  = COLORS[i]
        tr = np.array(history_true[i])   # (N+1, 6)
        es = np.array(history_est[i])

        # ── Vista 3-D ────────────────────────────────────────────────────
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                  color=c, lw=1.8, alpha=0.85,
                  label=f"{LABELS[i]} — reale")
        ax3d.plot(es[:, 0], es[:, 1], es[:, 2],
                  color=c, lw=1.2, ls="--", alpha=0.65,
                  label=f"{LABELS[i]} — stimata")
        ax3d.scatter(*tr[0,  :3], marker="o", color=c, s=55, zorder=5,
                     edgecolors="white", linewidths=1.2)
        ax3d.scatter(*tr[-1, :3], marker="*", color=c, s=110, zorder=5,
                     edgecolors="white", linewidths=0.8)

        # ── Proiezioni 2-D ───────────────────────────────────────────────
        for ax2, ix, iy, xl, yl, name in proj_cfg:
            ax2.plot(tr[:, ix], tr[:, iy], color=c, lw=1.8, alpha=0.85)
            ax2.plot(es[:, ix], es[:, iy], color=c, lw=1.2, ls="--",
                     alpha=0.65)
            ax2.plot(tr[0,  ix], tr[0,  iy], "o", color=c, ms=6,
                     mec="white", mew=1.2, zorder=5)
            ax2.plot(tr[-1, ix], tr[-1, iy], "*", color=c, ms=10,
                     mec="white", mew=0.6, zorder=5)
            for k in ellipse_steps:
                cov2d = history_cov[i][k][np.ix_([ix, iy], [ix, iy])]
                _cov_ellipse(ax2, es[k, [ix, iy]], cov2d,
                             n_sigma=SIGMA_3, facecolor=c, alpha=0.15,
                             edgecolor=c, lw=1.0)
            ax2.set_xlabel(xl, fontsize=8); ax2.set_ylabel(yl, fontsize=8)
            ax2.set_title(f"Proiezione {name}", fontsize=9,
                          fontweight="bold", pad=3)
            ax2.set_aspect("equal")
            ax2.grid(True, ls=":", alpha=0.45)
            ax2.tick_params(labelsize=7)
            ax2.set_facecolor("#f8f8f8")

    ax3d.set_xlabel("x [m]", fontsize=9, labelpad=5)
    ax3d.set_ylabel("y [m]", fontsize=9, labelpad=5)
    ax3d.set_zlabel("z [m]", fontsize=9, labelpad=5)
    ax3d.set_title("Vista 3-D", fontsize=10, fontweight="bold")
    ax3d.tick_params(labelsize=7)

    handles = []
    for i in TEAM:
        c = COLORS[i]
        handles += [
            plt.Line2D([0],[0], color=c, lw=2,
                       label=f"{LABELS[i]} — reale"),
            plt.Line2D([0],[0], color=c, lw=1.5, ls="--",
                       label=f"{LABELS[i]} — stimata"),
        ]
    handles += [
        mpatches.Patch(facecolor="grey", alpha=0.3,
                       label=f"Ellisse cov. {SIGMA_3}σ (proiezioni)"),
        plt.Line2D([0],[0], marker="o", color="w", mfc="grey",
                   ms=7, label="Inizio"),
        plt.Line2D([0],[0], marker="*", color="w", mfc="grey",
                   ms=10, label="Fine"),
    ]
    ax3d.legend(handles=handles, fontsize=7.5, loc="upper left",
                framealpha=0.85)

    fig.suptitle(
        "IMDCL — Massa puntiforme 3-D: traiettorie reali vs stimate\n"
        f"Misura relativa ogni {MEAS_EVERY} passi  |  "
        f"T = {N_STEPS * DT:.0f} s  |  Δt = {DT} s",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ===========================================================================
# Plot tracce di covarianza (comune a entrambi i modelli)
# ===========================================================================

def plot_covariance_trace(history_cov):
    """
    3 pannelli (uno per agente).
    Asse sinistro : tr(P) + varianze di posizione
    Asse destro   : varianze di velocità (3-D) oppure varianza θ (unicycle)
    """
    is_3d   = (motion_model.state_dim == 6)
    time    = np.arange(len(history_cov[0])) * DT
    meas_t  = np.arange(MEAS_EVERY, N_STEPS + 1, MEAS_EVERY) * DT

    if is_3d:
        pos_idx  = [0, 1, 2];  vel_idx  = [3, 4, 5]
        pos_lbls = ["P$_{xx}$", "P$_{yy}$", "P$_{zz}$"]
        vel_lbls = ["P$_{vxvx}$", "P$_{vyvy}$", "P$_{vzvz}$"]
        r_lbl    = "Varianza velocità  [m²/s²]"
    else:
        pos_idx  = [0, 1];     vel_idx  = [2]
        pos_lbls = ["P$_{xx}$", "P$_{yy}$"]
        vel_lbls = ["P$_{θθ}$"]
        r_lbl    = "P$_{θθ}$  [rad²]"

    traces  = {i: np.array([np.trace(P) for P in history_cov[i]]) for i in TEAM}
    pos_v   = {i: {k: np.array([P[k,k] for P in history_cov[i]])
                   for k in pos_idx} for i in TEAM}
    vel_v   = {i: {k: np.array([P[k,k] for P in history_cov[i]])
                   for k in vel_idx} for i in TEAM}

    ls_cyc = ["--", "-.", (0, (3, 1, 1, 1)), ":", "--", "-."]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.patch.set_facecolor("#ffffff")

    for idx, i in enumerate(TEAM):
        ax_l = axes[idx]
        ax_r = ax_l.twinx()
        c    = COLORS[i]

        for mt in meas_t:
            ax_l.axvline(mt, color="#bbbbbb", lw=0.6, ls="--",
                         alpha=0.5, zorder=1)

        ax_l.plot(time, traces[i], color=c, lw=2.3, alpha=0.95,
                  label="tr(P)", zorder=4)
        for j, k in enumerate(pos_idx):
            ax_l.plot(time, pos_v[i][k], color=c, lw=1.2,
                      ls=ls_cyc[j], alpha=0.60,
                      label=pos_lbls[j], zorder=3)
        for j, k in enumerate(vel_idx):
            ax_r.plot(time, vel_v[i][k], color=c, lw=1.1,
                      ls=ls_cyc[len(pos_idx) + j], alpha=0.45, zorder=2)

        ax_r.set_ylabel(r_lbl, fontsize=8.5, color=c, alpha=0.75)
        ax_r.tick_params(axis="y", labelcolor=c, labelsize=8)
        ax_l.set_facecolor("#f8f8f8")
        ax_l.set_ylabel("Varianza posizione  [m²]", fontsize=9)
        ax_l.set_title(LABELS[i], fontsize=11, fontweight="bold",
                       color=c, loc="left", pad=3)
        ax_l.grid(True, ls=":", alpha=0.4)
        ax_l.tick_params(labelsize=9)

        if idx == 0:
            ll, la = ax_l.get_legend_handles_labels()
            r_prox = [plt.Line2D([0],[0], color=COLORS[0], lw=1.1,
                                 ls=ls_cyc[len(pos_idx)+j], alpha=0.6,
                                 label=vel_lbls[j])
                      for j in range(len(vel_idx))]
            ax_l.legend(ll + r_prox, la + [h.get_label() for h in r_prox],
                        fontsize=8.5, loc="upper right",
                        framealpha=0.9, ncol=2)
            ax_l.text(0.01, 0.96,
                      f"linee grigie = update cooperativo ogni {MEAS_EVERY} passi",
                      transform=ax_l.transAxes, ha="left", va="top",
                      fontsize=7.5, color="#555555",
                      bbox=dict(boxstyle="round,pad=0.25",
                                facecolor="white", alpha=0.7))

    axes[-1].set_xlabel("Tempo  [s]", fontsize=11)
    mname = "Massa puntiforme 3-D" if is_3d else "Unicycle 2-D"
    fig.suptitle(
        f"IMDCL [{mname}] — Traccia matrice di covarianza nel tempo\n"
        "tr(P): cresce durante propagazione  ↓  decresce ad ogni misura cooperativa",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"Avvio simulazione IMDCL  [modello: {MODEL}] ...")
    hist_true, hist_est, hist_cov, ell_steps = run_simulation()
    print(f"Completata: {N_STEPS} passi, {N_STEPS * DT:.0f} s.")

    if MODEL == "unicycle":
        fig_traj = plot_trajectories_2d(hist_true, hist_est, hist_cov, ell_steps)
    else:
        fig_traj = plot_trajectories_3d(hist_true, hist_est, hist_cov, ell_steps)

    path_traj = os.path.join(SCRIPT_DIR, f"imdcl_{MODEL}_trajectories.png")
    fig_traj.savefig(path_traj, dpi=150, bbox_inches="tight")
    print(f"Traiettorie → {path_traj}")

    fig_cov  = plot_covariance_trace(hist_cov)
    path_cov = os.path.join(SCRIPT_DIR, f"imdcl_{MODEL}_covariance.png")
    fig_cov.savefig(path_cov, dpi=150, bbox_inches="tight")
    print(f"Covarianza  → {path_cov}")

    plt.show()
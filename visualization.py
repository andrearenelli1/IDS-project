"""
visualization.py
================
Visualizzazione e animazione per la simulazione di ricerca valanga multi-drone.

Consolida visualization.py + animate_drone.py (eliminando ridondanze).

Funzioni pubbliche
------------------
  plot_mission          — figura statica 6 pannelli missione
  animate_mission       — animazione 3-D reale + stima IMDCL (usata da main.py)
  animate_mpc_standalone— animazione MPC standalone (ex animate_drone.py)

Utilizzo standalone (ex animate_drone.py)
-----------------------------------------
    python visualization.py [--save] [--fps 30] [--speed 2.0] [--out path]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch

from artva import ARTVASource
from terrain import Terrain
from drone_agent import DroneAgent, DroneState
from pf import (weighted_mean_cov_xy, ellipse_axes_angle,
               run_ellipse_metrics)
import config as _cfg
from config import (
    AGL_HEIGHT, DT_SIM, N_MPC, DT_MPC, A_MAX, V_MAX,
    ARTVA_DETECT_THR, TRACK_STOP_THR, ARTVA_MOMENT,
    IMDCL_R_MEAS_STD,
    COLORS, BG_DARK,
)

# ---------------------------------------------------------------------------
# Palette condivisa (usata anche in animate_mpc_standalone)
# ---------------------------------------------------------------------------
_BG_COLOR   = BG_DARK
_GRID_COLOR = "#1e2730"
_TEXT_COLOR = "#c9d1d9"
_TRAIL_LEN  = 50

# RC params per i plot MPC — Computer Modern via LaTeX
_LATEX_RC: dict = {
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman"],
    "font.size":       13,
    "axes.labelsize":  13,
    "axes.titlesize":  15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

_LATEX_RC_2X: dict = {
    **_LATEX_RC,
    "font.size":       22,
    "axes.labelsize":  22,
    "axes.titlesize":  26,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}


# ============================================================================
# Helper privato
# ============================================================================

def _reconstruct_state_sequence(ag: DroneAgent) -> List[DroneState]:
    """
    Ricostruisce la sequenza FSM dal log segnali.
    SEARCH fino al primo rilevamento, poi TRACK fino alla soglia STOP,
    poi STOP/SUPPORT (approssimato come TRACK per il plot della traiettoria).
    """
    from config import TRACK_STOP_THR
    n = len(ag.history)
    if not ag.detected:
        return [DroneState.SEARCH] * n
    detect_k = None
    stop_k   = None
    for k, (_, s) in enumerate(ag.signal_log):
        if detect_k is None and s >= ARTVA_DETECT_THR:
            detect_k = k
        if detect_k is not None and stop_k is None and s >= TRACK_STOP_THR:
            stop_k = k
    if detect_k is None:
        return [DroneState.SEARCH] * n
    seq = [DroneState.SEARCH] * (detect_k + 1)
    if stop_k is not None:
        seq += [DroneState.TRACK]  * max(0, stop_k - detect_k)
        seq += [DroneState.STOP]   * max(0, n - stop_k)
    else:
        seq += [DroneState.TRACK] * max(0, n - detect_k - 1)
    return seq


def _save_animation(anim, save_path, fps, bg_color=_BG_COLOR):
    """Tenta il salvataggio mp4 (ffmpeg), poi gif (Pillow)."""
    try:
        out = save_path + ".mp4"
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=1800,
                                           metadata={"title": "Drone Sim"}),
                  dpi=150, savefig_kwargs={"facecolor": bg_color})
        print(f"Animazione salvata in: {out}")
    except Exception as e_mp4:
        print(f"ffmpeg non disponibile ({e_mp4}), provo con Pillow...")
        try:
            out = save_path + ".gif"
            anim.save(out, writer=PillowWriter(fps=fps), dpi=100,
                      savefig_kwargs={"facecolor": bg_color})
            print(f"Animazione salvata in: {out}")
        except Exception as e_gif:
            print(f"Salvataggio fallito: {e_gif}")


# ============================================================================
# Plot statico missione
# ============================================================================

def plot_mission(
    terrain:  Terrain,
    artva:    ARTVASource,
    agents:   Dict[int, DroneAgent],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    sub_dem:  np.ndarray,
    TRACK_STOP_THR: float = TRACK_STOP_THR,
    ARTVA_DETECT_THR: float = ARTVA_DETECT_THR,
    
) -> plt.Figure:
    """
    Figura missione — 6 pannelli:
      A — Vista 2-D: traiettorie reali (—/--) + stimate IMDCL (:)
      B — Vista 3-D: reale e IMDCL sul terreno
      C — Segnale ARTVA nel tempo
      D — Altezza AGL nel tempo
      E — Errore stima IMDCL |x_reale − x̂|
      F — Errore per asse x/y/z
    """
    drone_ids = list(agents.keys())

    nx, ny = 80, 80
    xs = np.linspace(terrain.x_min, terrain.x_max, nx)
    ys = np.linspace(terrain.y_min, terrain.y_max, ny)
    XS, YS = np.meshgrid(xs, ys)
    ZS = terrain.z(XS.ravel(), YS.ravel()).reshape(ny, nx)
    ARTVA_MAP = np.array([
        [artva.signal([xs[j], ys[i], ZS[i, j] + AGL_HEIGHT], noisy=False)
         for j in range(nx)]
        for i in range(ny)
    ])
    ext = [terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#ffffff")

    # ── A: Vista dall'alto ──────────────────────────────────────────────
    ax_a = fig.add_subplot(2, 3, 1)
    ax_a.set_facecolor("#1a1a2e")
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(sub_dem, nan=0.0), vert_exag=3)
    yext = [float(np.min(y_coords)), float(np.max(y_coords))]
    ax_a.imshow(hs, cmap="gray", alpha=0.5,
                extent=[float(x_coords.min()), float(x_coords.max()), yext[0], yext[1]],
                origin="upper" if y_coords[0] > y_coords[-1] else "lower",
                interpolation="bilinear", zorder=1)
    im_artva = ax_a.imshow(np.log1p(ARTVA_MAP), cmap="inferno", alpha=0.55,
                           extent=ext, origin="lower", zorder=2)
    fig.colorbar(im_artva, ax=ax_a, fraction=0.046, pad=0.04,
                 label="log(1+ARTVA) [a.u.]")

    for i in drone_ids:
        ag        = agents[i]
        traj      = np.array(ag.history)
        est       = np.array(ag.est_history) if ag.est_history else traj
        c         = COLORS.get(i, "#aaaaaa")
        state_seq = _reconstruct_state_sequence(ag)
        n         = min(len(traj), len(state_seq))
        s_idx = [k for k in range(n) if state_seq[k] == DroneState.SEARCH]
        t_idx = [k for k in range(n) if state_seq[k] in (
            DroneState.TRACK, DroneState.SUPPORT, DroneState.FINAL_ORBIT)]
        p_idx = [k for k in range(n) if state_seq[k] == DroneState.STOP]
        if s_idx:
            ax_a.plot(traj[s_idx, 0], traj[s_idx, 1], color=c, lw=1.0, alpha=0.45, ls="--")
        if t_idx:
            ax_a.plot(traj[t_idx, 0], traj[t_idx, 1], color=c, lw=2.0, alpha=0.9,  ls="-")
        if p_idx:
            ax_a.plot(traj[p_idx, 0], traj[p_idx, 1], color=c, lw=1.5, alpha=0.7,  ls=":")
        ax_a.plot(est[:min(len(est), n), 0], est[:min(len(est), n), 1],
                  color=c, lw=1.0, alpha=0.65, ls=":")
        ax_a.plot(*traj[0, :2],  "o", color=c, ms=7, mec="white", mew=1.0, zorder=6)
        ax_a.plot(*traj[-1, :2], "^", color=c, ms=9, mec="white", mew=0.8, zorder=6)

    ax_a.plot(*artva._theta[:2], "*", color="white", ms=18, zorder=10,
              mec="yellow", mew=1.5)
    # Waypoint markers (pallini semi-trasparenti per ogni drone)
    for i in drone_ids:
        c  = COLORS.get(i, "#aaaaaa")
        ag = agents[i]
        if ag.wp_target_log:
            wps_xy = np.array([w[:2] for w in ag.wp_target_log])
            # deduplica mantenendo l'ordine
            seen, unique_wps = set(), []
            for w in wps_xy:
                key = (round(w[0], 1), round(w[1], 1))
                if key not in seen:
                    seen.add(key)
                    unique_wps.append(w)
            unique_wps = np.array(unique_wps)
            ax_a.scatter(unique_wps[:, 0], unique_wps[:, 1],
                         color=c, s=18, alpha=0.30, zorder=4, linewidths=0)

    ax_a.set_xlabel("x [m]", fontsize=9)
    ax_a.set_ylabel("y [m]", fontsize=9)
    ax_a.set_title("Piano XY — traiettorie · stima IMDCL", fontweight="bold", fontsize=10)
    ax_a.tick_params(labelsize=8)
    handles = [mpatches.Patch(color=COLORS.get(i, "#aaa"), label=f"Drone {i}") for i in drone_ids]
    handles += [
        plt.Line2D([0],[0], color="w", lw=2.0,  label="TRACK/SUPPORT"),
        plt.Line2D([0],[0], color="w", lw=1.0, ls="--", label="SEARCH"),
        plt.Line2D([0],[0], color="w", lw=1.0, ls=":",  label="Stima IMDCL"),
        plt.Line2D([0],[0], marker="o", color="w", mfc="w", ms=6, alpha=0.35,
                   label="Waypoint", linestyle="None"),
        plt.Line2D([0],[0], marker="*", color="w", mfc="yellow", ms=12,
                   label="Vittima", linestyle="None"),
    ]
    ax_a.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.75)

    # ── B: Vista 3-D ────────────────────────────────────────────────────
    ax_b = fig.add_subplot(2, 3, 2, projection="3d")
    ax_b.set_facecolor(_BG_COLOR)
    xs_3d = np.linspace(terrain.x_min, terrain.x_max, 40)
    ys_3d = np.linspace(terrain.y_min, terrain.y_max, 40)
    X3, Y3 = np.meshgrid(xs_3d, ys_3d)
    Z3 = terrain.z(X3.ravel(), Y3.ravel()).reshape(X3.shape)
    ax_b.plot_surface(X3, Y3, Z3, cmap="terrain", alpha=0.45, rcount=40, ccount=40, linewidth=0)
    for i in drone_ids:
        ag   = agents[i]
        traj = np.array(ag.history)
        est  = np.array(ag.est_history) if ag.est_history else traj
        c    = COLORS.get(i, "#aaaaaa")
        ax_b.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color=c, lw=1.5, alpha=0.85, label=f"D{i} reale")
        ax_b.scatter(*traj[0, :3], color=c, s=40, zorder=6)
        ax_b.scatter(*traj[-1, :3], marker="^", color=c, s=60, zorder=6)
        n_est = min(len(est), len(traj))
        ax_b.plot(est[:n_est, 0], est[:n_est, 1], est[:n_est, 2],
                  color=c, lw=1.0, alpha=0.5, ls="--", label=f"D{i} IMDCL")
    ax_b.scatter(*artva._theta, marker="*", color="yellow", s=250,
                 zorder=10, edgecolors="red", linewidths=1)
    ax_b.set_xlabel("x [m]", fontsize=8, labelpad=3)
    ax_b.set_ylabel("y [m]", fontsize=8, labelpad=3)
    ax_b.set_zlabel("z [m]", fontsize=8, labelpad=3)
    ax_b.set_title("Vista 3D — reale (—) · IMDCL (--)", fontweight="bold", fontsize=10)
    ax_b.tick_params(labelsize=7)
    ax_b.legend(fontsize=6.5, loc="upper left")

    # ── C: Segnale ARTVA ────────────────────────────────────────────────
    ax_c = fig.add_subplot(2, 3, 3)
    ax_c.set_facecolor("#f8f8f8")
    for i in drone_ids:
        ag   = agents[i]
        c    = COLORS.get(i, "#aaaaaa")
        sigs = [s for _, s in ag.signal_log]
        time = np.arange(len(sigs)) * DT_SIM
        ax_c.semilogy(time, np.maximum(sigs, 1e-8), color=c, lw=1.3, alpha=0.85, label=f"Drone {i}")
    ax_c.axhline(ARTVA_DETECT_THR, color="red", lw=1.2, ls="--",
                 label=f"Detect ({ARTVA_DETECT_THR:.0e})")
    ax_c.axhline(TRACK_STOP_THR, color="orange", lw=1.2, ls="--",
                 label=f"Stop ({TRACK_STOP_THR:.0e})")
    ax_c.set_xlabel("Tempo [s]", fontsize=9)
    ax_c.set_ylabel("Segnale ARTVA [a.u.]", fontsize=9)
    ax_c.set_title("Segnale ARTVA", fontweight="bold", fontsize=10)
    ax_c.legend(fontsize=8); ax_c.grid(True, ls=":", alpha=0.5); ax_c.tick_params(labelsize=8)

    # ── D: Altezza AGL ──────────────────────────────────────────────────
    ax_d = fig.add_subplot(2, 3, 4)
    ax_d.set_facecolor("#f8f8f8")
    for i in drone_ids:
        ag   = agents[i]
        c    = COLORS.get(i, "#aaaaaa")
        traj = np.array(ag.history)
        time = np.arange(len(traj)) * DT_SIM
        z_t  = np.array([terrain.z(traj[k, 0], traj[k, 1]) for k in range(len(traj))])
        ax_d.plot(time, traj[:, 2] - z_t, color=c, lw=1.2, alpha=0.85, label=f"Drone {i}")
    ax_d.axhline(AGL_HEIGHT, color="green", lw=1.2, ls="--",
                 label=f"AGL target ({AGL_HEIGHT} m)")
    ax_d.axhline(0, color="red", lw=1.0, ls=":", alpha=0.7, label="Terreno")
    ax_d.set_xlabel("Tempo [s]", fontsize=9)
    ax_d.set_ylabel("Altezza sopra terreno [m]", fontsize=9)
    ax_d.set_title("Quota AGL", fontweight="bold", fontsize=10)
    ax_d.legend(fontsize=8); ax_d.grid(True, ls=":", alpha=0.5)
    ax_d.tick_params(labelsize=8); ax_d.set_ylim(bottom=-0.5)

    # ── E: Errore stima IMDCL ───────────────────────────────────────────
    ax_e = fig.add_subplot(2, 3, 5)
    ax_e.set_facecolor("#f8f8f8")
    for i in drone_ids:
        ag   = agents[i]
        c    = COLORS.get(i, "#aaaaaa")
        traj = np.array(ag.history)
        est  = np.array(ag.est_history) if ag.est_history else traj
        n    = min(len(traj), len(est))
        time = np.arange(n) * DT_SIM
        ax_e.plot(time, np.linalg.norm(traj[:n, :3] - est[:n, :3], axis=1),
                  color=c, lw=1.2, alpha=0.85, label=f"Drone {i}")
    ax_e.set_xlabel("Tempo [s]", fontsize=9)
    ax_e.set_ylabel("Errore posizione [m]", fontsize=9)
    ax_e.set_title("Errore localizzazione IMDCL", fontweight="bold", fontsize=10)
    ax_e.legend(fontsize=8); ax_e.grid(True, ls=":", alpha=0.5)
    ax_e.tick_params(labelsize=8); ax_e.set_ylim(bottom=0)

    # ── F: Errore per asse ──────────────────────────────────────────────
    ax_f = fig.add_subplot(2, 3, 6)
    ax_f.set_facecolor("#f8f8f8")
    ls_map   = {0: "-", 1: "--", 2: ":"}
    ax_names = {0: "x", 1: "y", 2: "z"}
    for i in drone_ids:
        ag   = agents[i]
        c    = COLORS.get(i, "#aaaaaa")
        traj = np.array(ag.history)
        est  = np.array(ag.est_history) if ag.est_history else traj
        n    = min(len(traj), len(est))
        time = np.arange(n) * DT_SIM
        for ax_idx in range(3):
            ax_f.plot(time, np.abs(traj[:n, ax_idx] - est[:n, ax_idx]),
                      color=c, lw=1.0, alpha=0.7, ls=ls_map[ax_idx],
                      label=f"D{i} {ax_names[ax_idx]}" if i == drone_ids[0] else None)
    for ax_idx in range(3):
        ax_f.plot([], [], color="gray", lw=1.0, ls=ls_map[ax_idx],
                  label=f"asse {ax_names[ax_idx]}")
    ax_f.set_xlabel("Tempo [s]", fontsize=9)
    ax_f.set_ylabel("|errore| per asse [m]", fontsize=9)
    ax_f.set_title("Errore IMDCL per asse", fontweight="bold", fontsize=10)
    ax_f.legend(fontsize=7.5); ax_f.grid(True, ls=":", alpha=0.5)
    ax_f.tick_params(labelsize=8); ax_f.set_ylim(bottom=0)

    fig.suptitle(
        f"Missione valanga · {len(agents)} droni · "
        f"AGL={AGL_HEIGHT} m · N_MPC={N_MPC} · dt={DT_SIM} s · "
        f"IMDCL (Rc={_cfg.IMDCL_COMM_RADIUS:.0f} m, σ={IMDCL_R_MEAS_STD} m)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ============================================================================
# MPC path-following performance — 3 figure separate (main.py)
# ============================================================================

def _mpc_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, ls=":", alpha=0.5, color="#cccccc")


def plot_mpc_trajectories(
    terrain: Terrain,
    agents:  Dict[int, DroneAgent],
    artva:   ARTVASource,
) -> plt.Figure:
    """Traiettorie 2D (piano XY) viste dall'alto: reale (piena) ed IMDCL stimata (tratteggiata)."""
    drone_ids = list(agents.keys())
    with plt.rc_context(_LATEX_RC_2X):
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#ffffff")
        _mpc_style(ax)

        for i in drone_ids:
            ag        = agents[i]
            c         = COLORS.get(i, "#aaaaaa")
            traj      = np.array(ag.history)       # (T,6) reale
            est       = np.array(ag.est_history)   # (T,6) stimata IMDCL
            state_seq = _reconstruct_state_sequence(ag)
            n         = min(len(traj), len(state_seq))
            n_est     = min(len(est),  n)

            s_idx = [k for k in range(n) if state_seq[k] == DroneState.SEARCH]
            t_idx = [k for k in range(n) if state_seq[k] in (DroneState.TRACK, DroneState.SUPPORT)]
            p_idx = [k for k in range(n) if state_seq[k] == DroneState.STOP]

            # ── traiettoria reale ──
            if s_idx:
                ax.plot(traj[s_idx, 0], traj[s_idx, 1],
                        color=c, lw=1.5, alpha=0.85, ls="-")
            if t_idx:
                ax.plot(traj[t_idx, 0], traj[t_idx, 1],
                        color=c, lw=2.0, alpha=0.90, ls="-")
            if p_idx:
                ax.plot(traj[p_idx, 0], traj[p_idx, 1],
                        color=c, lw=1.5, alpha=0.60, ls="-")

            # ── traiettoria stimata IMDCL ──
            if n_est > 0:
                ax.plot(est[:n_est, 0], est[:n_est, 1],
                        color=c, lw=1.0, alpha=0.55, ls="--")

            ax.plot(*traj[0,  :2], "o", color=c, ms=7, mec="white", mew=1.0, zorder=7)
            ax.plot(*traj[-1, :2], "^", color=c, ms=8, mec="white", mew=0.8, zorder=7)

        ax.plot(*artva._theta[:2], "*", color="crimson", ms=14,
                mec="black", mew=0.5, zorder=9)

        cx = (terrain.x_min + terrain.x_max) / 2.0
        cy = (terrain.y_min + terrain.y_max) / 2.0
        ax.set_xlim(cx - 100, cx + 100)
        ax.set_ylim(cy - 100, cy + 100)
        ax.set_xlabel(r"$x$ [m]", fontsize=17)
        ax.set_ylabel(r"$y$ [m]", fontsize=17)
        ax.set_title(r"Trajectories - Top view", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

        handles = [mpatches.Patch(color=COLORS.get(i, "#aaa"), label=f"Drone {i}")
                   for i in drone_ids]
        handles += [
            plt.Line2D([0], [0], color="gray", lw=1.5, ls="-",  label="True trajectory"),
            plt.Line2D([0], [0], color="gray", lw=1.0, ls="--", label="Estimated trajectory"),
            plt.Line2D([0], [0], marker="*",   color="crimson", ms=10, lw=0, label="Victim"),
        ]
        ax.legend(handles=handles, loc="best", framealpha=0.85, fontsize=10)
        fig.tight_layout()
    return fig


def plot_mpc_trajectories_3d(
    terrain: Terrain,
    agents:  Dict[int, DroneAgent],
    artva:   ARTVASource,
    res:     int = 60,
) -> plt.Figure:
    """Vista 3D con superficie del terreno, traiettorie reali e stimate IMDCL."""
    drone_ids = list(agents.keys())

    xs_3d = np.linspace(terrain.x_min, terrain.x_max, res)
    ys_3d = np.linspace(terrain.y_min, terrain.y_max, res)
    X3, Y3 = np.meshgrid(xs_3d, ys_3d)
    Z3 = terrain.z(X3.ravel(), Y3.ravel()).reshape(X3.shape)

    with plt.rc_context(_LATEX_RC_2X):
        fig = plt.figure(figsize=(7, 6))
        fig.patch.set_facecolor("#ffffff")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f8f8f8")
        ax.tick_params(axis="both", labelsize=20)

        ax.plot_surface(X3, Y3, Z3,
                        cmap="copper", alpha=0.75,
                        rcount=res, ccount=res, linewidth=0,
                        antialiased=True)

        for i in drone_ids:
            ag    = agents[i]
            c     = COLORS.get(i, "#aaaaaa")
            traj  = np.array(ag.history)
            est   = np.array(ag.est_history)
            n_est = min(len(est), len(traj))

            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color="black", lw=3.5, alpha=1.0, ls="-", zorder=5)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=c, lw=1.8, alpha=1.0, ls="-", zorder=6, label=f"Drone {i}")
            if n_est > 0:
                ax.plot(est[:n_est, 0], est[:n_est, 1], est[:n_est, 2],
                        color="black", lw=2.5, alpha=1.0, ls="--", zorder=5)
                ax.plot(est[:n_est, 0], est[:n_est, 1], est[:n_est, 2],
                        color=c, lw=1.0, alpha=1.0, ls="--", zorder=6)

            ax.scatter(*traj[0,  :3], color=c, s=35, zorder=6,
                       edgecolors="white", linewidths=0.5)
            ax.scatter(*traj[-1, :3], marker="^", color=c, s=55, zorder=6,
                       edgecolors="white", linewidths=0.5)

        ax.scatter(*artva._theta, marker="*", color="crimson", s=220,
                   zorder=9, edgecolors="black", linewidths=0.4, label="Victim")

        ax.view_init(elev=25, azim=225)
        ax.set_xlabel(r"$x$ [m]", labelpad=6, fontsize=17)
        ax.set_ylabel(r"$y$ [m]", labelpad=6, fontsize=17)
        ax.set_zlabel(r"$z$ [m]", labelpad=6, fontsize=17)
        ax.set_xlim(terrain.x_min, terrain.x_max)
        ax.set_ylim(terrain.y_min, terrain.y_max)
        ax.set_title(r"Trajectories - 3D view", fontweight="bold")
        fig.tight_layout()
    return fig


def plot_mpc_inputs(
    agents: Dict[int, DroneAgent],
) -> plt.Figure:
    """Accelerazioni (top) e velocità (bottom) nel tempo con rispettivi limiti."""
    drone_ids = list(agents.keys())
    comp_ls   = ["-", "--", ":"]
    comp_col  = ["#e63946", "#2a9d8f", "#f4a261"]

    with plt.rc_context(_LATEX_RC):
        fig, (ax_a, ax_v) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        fig.patch.set_facecolor("#ffffff")
        for ax in (ax_a, ax_v):
            _mpc_style(ax)

        for i in drone_ids:
            ag = agents[i]
            if not ag.input_log:
                continue
            inputs = np.array(ag.input_log)
            traj   = np.array(ag.history)
            T      = min(len(inputs), len(traj))
            time   = np.arange(T) * DT_SIM

            for j, lbl in enumerate([r"$a_x$", r"$a_y$", r"$a_z$"]):
                ax_a.plot(time, inputs[:T, j], color=comp_col[j],
                          lw=0.9, alpha=0.70, ls=comp_ls[j],
                          label=lbl if i == drone_ids[0] else None)

            for j, lbl in enumerate([r"$v_x$", r"$v_y$", r"$v_z$"]):
                ax_v.plot(time, traj[:T, 3 + j], color=comp_col[j],
                          lw=0.9, alpha=0.70, ls=comp_ls[j],
                          label=lbl if i == drone_ids[0] else None)

        ax_a.axhline( A_MAX, color="red", lw=1.2, ls="--",
                      label=rf"$\pm a_{{\rm max}} = \pm{A_MAX}$~m/s$^2$")
        ax_a.axhline(-A_MAX, color="red", lw=1.2, ls="--")
        ax_v.axhline( V_MAX, color="red", lw=1.2, ls="--",
                      label=r"$\pm v_{\rm max}$")
        ax_v.axhline(-V_MAX, color="red", lw=1.2, ls="--")

        drone_handles = [mpatches.Patch(color=COLORS.get(i, "#aaa"), label=f"Drone {i}")
                         for i in drone_ids]
        _leg_fs = 9
        ax_a.set_ylabel(r"Acceleration [m/s$^2$]", fontsize=15)
        ax_a.set_title(r"MPC Control Inputs and Velocity Profiles", fontweight="bold", fontsize=18)
        ax_a.legend(loc="upper right", framealpha=0.80, fontsize=_leg_fs)
        ax_a.add_artist(ax_a.legend(handles=drone_handles, loc="upper left",
                                    framealpha=0.80, fontsize=_leg_fs))

        ax_v.set_xlabel(r"Time [s]", fontsize=15)
        ax_v.set_ylabel(r"Velocity [m/s]", fontsize=15)
        ax_v.legend(loc="upper right", framealpha=0.80, fontsize=_leg_fs)

        fig.tight_layout()
    return fig


def plot_mpc_altitude(
    terrain: Terrain,
    agents:  Dict[int, DroneAgent],
) -> plt.Figure:
    """Altitudine sopra il terreno (AGL) nel tempo per ogni drone."""
    drone_ids = list(agents.keys())
    with plt.rc_context(_LATEX_RC):
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#ffffff")
        _mpc_style(ax)

        for i in drone_ids:
            ag   = agents[i]
            c    = COLORS.get(i, "#aaaaaa")
            traj = np.array(ag.history)
            time = np.arange(len(traj)) * DT_SIM
            z_ground = np.array([terrain.z(traj[k, 0], traj[k, 1])
                                  for k in range(len(traj))])
            ax.plot(time, traj[:, 2] - z_ground,
                    color=c, lw=1.3, alpha=0.85, label=f"Drone {i}")

        ax.axhline(AGL_HEIGHT, color="green", lw=1.3, ls="--",
                   label=rf"AGL target ({AGL_HEIGHT}~m)")
        ax.axhline(0, color="saddlebrown", lw=1.0, ls=":", alpha=0.7, label="Ground")
        ax.set_xlabel(r"Time [s]", fontsize=15)
        ax.set_ylabel(r"Height above ground [m]", fontsize=15)
        ax.set_title(r"Altitude above ground level", fontweight="bold", fontsize=18)
        ax.set_ylim(bottom=-0.3)
        ax.legend(loc="upper right", framealpha=0.80)
        fig.tight_layout()
    return fig


def plot_imdcl_error(
    agents: Dict[int, DroneAgent],
) -> plt.Figure:
    """Errore di stima IMDCL: norma 3D (top) e per-asse (bottom) nel tempo."""
    drone_ids = list(agents.keys())

    with plt.rc_context(_LATEX_RC):
        fig, ax_n = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#ffffff")
        _mpc_style(ax_n)

        for i in drone_ids:
            ag   = agents[i]
            c    = COLORS.get(i, "#aaaaaa")
            traj = np.array(ag.history)
            est  = np.array(ag.est_history)
            n    = min(len(traj), len(est))
            time = np.arange(n) * DT_SIM
            ax_n.plot(time, np.linalg.norm(traj[:n, :3] - est[:n, :3], axis=1),
                      color=c, lw=1.4, alpha=0.90, label=f"Drone {i}")

        ax_n.set_xlabel(r"Time [s]", fontsize=15)
        ax_n.set_ylabel(r"Position error [m]", fontsize=15)
        ax_n.set_title(r"IMDCL Estimation Error", fontweight="bold", fontsize=18)
        ax_n.set_ylim(bottom=0)
        ax_n.legend(loc="upper left", framealpha=0.85)

        fig.tight_layout()
    return fig


def plot_artva_signal(
    agents:           Dict[int, DroneAgent],
    artva_detect_thr: float = ARTVA_DETECT_THR,
    track_stop_thr:   float = TRACK_STOP_THR,
) -> plt.Figure:
    """Segnale ARTVA filtrato nel tempo per ogni drone, con soglie DETECT e STOP."""
    drone_ids = list(agents.keys())

    with plt.rc_context(_LATEX_RC):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#ffffff")
        _mpc_style(ax)

        for i in drone_ids:
            ag  = agents[i]
            sig = np.array([s for _, s in ag.signal_log])
            t   = np.arange(len(sig)) * DT_SIM
            ax.plot(t, sig, color=COLORS.get(i, "#aaaaaa"),
                    lw=1.4, alpha=0.90, label=f"Drone {i}")

        ax.axhline(artva_detect_thr, color="#2a9d8f", lw=1.2, ls="--",
                   label=r"Detect threshold")
        ax.axhline(track_stop_thr,   color="#e63946", lw=1.2, ls="--",
                   label=r"Stop threshold")

        ax.set_yscale("log")
        ax.set_xlabel(r"Time [s]", fontsize=15)
        ax.set_ylabel(r"ARTVA signal (filtered) [a.u.]", fontsize=15)
        ax.set_title(r"ARTVA Signal over Time", fontweight="bold", fontsize=18)
        ax.legend(loc="upper left", framealpha=0.85)
        fig.tight_layout()
    return fig


# ============================================================================
# Plot posizioni finali
# ============================================================================

def plot_final_positions(
    terrain: Terrain,
    artva:   ARTVASource,
    agents:  Dict[int, DroneAgent],
) -> plt.Figure:
    """
    Mappa 2D — posizioni finali della missione:
      ▲ (pieno)    — posizione reale del drone
      ▲ (bordo)    — posizione stimata IMDCL del drone
      ◆            — stima PF della sorgente, depurata dal drift IMDCL
                     (riportata nel frame reale per il confronto con la vittima)
      ellisse 95%  — regione di confidenza PF (covarianza 2×2 delle particelle)
      ★            — posizione reale vittima

    In alto a sinistra: area media dell'ellisse di confidenza e IoU media a
    coppie tra droni (consenso inter-drone), coerenti con lo sweep parametrico.
    """
    all_ids   = list(agents.keys())
    drone_ids = [i for i in all_ids if agents[i].pf is not None]
    if not drone_ids:
        drone_ids = all_ids   # fallback: nessun PF attivo
    victim_xy   = artva._theta[:2]
    finals_real = {i: agents[i].history[-1][:2]     for i in drone_ids}
    finals_est  = {i: agents[i].est_history[-1][:2] for i in drone_ids}

    # Il PF lavora nel frame stimato del drone: per confrontare la stima con la
    # vittima vera si rimuove il drift IMDCL  center = m_xy − (x_est − x).
    source_ests: Dict[int, np.ndarray] = {}
    source_covs: Dict[int, np.ndarray] = {}
    for i in drone_ids:
        ag = agents[i]
        if ag.source_est is not None and ag.pf is not None:
            m_xy, cov_xy = weighted_mean_cov_xy(ag.pf.particles, ag.pf.weights)
            drift = ag.x_est[:2] - ag.x[:2]
            source_ests[i] = m_xy - drift
            source_covs[i] = cov_xy
        elif ag.source_est is not None:
            source_ests[i] = ag.source_est[:2] - (ag.x_est[:2] - ag.x[:2])

    # Metriche aggregate coerenti con lo sweep (area media ellisse 95% + IoU media)
    _ids_cov  = [i for i in drone_ids if i in source_covs]
    mean_area, mean_iou = run_ellipse_metrics(
        [source_ests[i] for i in _ids_cov],
        [source_covs[i] for i in _ids_cov],
    )

    # estensione delle ellissi per non tagliarle dai limiti degli assi
    ell_extents = []
    for i in _ids_cov:
        a, b, ang = ellipse_axes_angle(source_covs[i], conf=0.95)
        # semi-bounding-box dell'ellisse ruotata
        hx = np.hypot(a * np.cos(ang), b * np.sin(ang))
        hy = np.hypot(a * np.sin(ang), b * np.cos(ang))
        c0 = source_ests[i]
        ell_extents += [c0 + [hx, hy], c0 - [hx, hy]]

    all_pts_list = (
        list(finals_real.values())
        + list(finals_est.values())
        + [victim_xy]
        + list(source_ests.values())
        + ell_extents
    )
    all_pts = np.vstack(all_pts_list)
    span   = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
                 all_pts[:, 1].max() - all_pts[:, 1].min())
    margin = max(8.0, span * 0.30)
    xlim   = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim   = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

    with plt.rc_context(_LATEX_RC):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        _mpc_style(ax)

        for i in drone_ids:
            c  = COLORS.get(i, "#aaaaaa")

            # ── posizione reale e stimata IMDCL ───────────────────────────
            ax.plot(*finals_real[i], "^", color=c, ms=11, mec="white", mew=0.9,
                    zorder=7, label=f"$p_{{{i}}}$ reale")
            ax.plot(*finals_est[i], "^", color=c, ms=11, mec="black", mew=1.0,
                    alpha=0.55, zorder=6, label=f"$\\hat{{p}}_{{{i}}}$ IMDCL")
            ax.plot(
                [finals_real[i][0], finals_est[i][0]],
                [finals_real[i][1], finals_est[i][1]],
                color=c, lw=0.8, ls=":", alpha=0.6, zorder=5,
            )

            # ── stima PF sorgente (senza correzione drift) ────────────────
            if i in source_ests:
                src = source_ests[i]
                ax.plot(*src, "D", color=c, ms=10, mec=c, mew=1.5,
                        zorder=8, label=f"$\\hat{{\\theta}}^{{\\mathrm{{PF}}}}_{{{i}}}$")

                # ellisse di confidenza 95% (covarianza 2×2 delle particelle)
                if i in source_covs:
                    a, b, ang = ellipse_axes_angle(source_covs[i], conf=0.95)
                    ell = mpatches.Ellipse(
                        (src[0], src[1]), width=2 * a, height=2 * b,
                        angle=np.degrees(ang),
                        linewidth=1.5, edgecolor=c, facecolor=c,
                        alpha=0.18, zorder=6,
                    )
                    ax.add_patch(ell)
                    ell_edge = mpatches.Ellipse(
                        (src[0], src[1]), width=2 * a, height=2 * b,
                        angle=np.degrees(ang),
                        linewidth=1.5, edgecolor=c, facecolor="none",
                        alpha=0.75, zorder=7,
                    )
                    ax.add_patch(ell_edge)

        # ── posizione reale vittima ────────────────────────────────────────
        ax.plot(*victim_xy, "*", color="crimson", ms=16,
                mec="black", mew=0.6, zorder=9, label=r"$\theta$ (vittima)")

        # ── annotazione metriche aggregate (coerenti con lo sweep) ──────────
        if not np.isnan(mean_area):
            txt = rf"$\bar A_{{95\%}}$ = {mean_area:.1f} m$^2$"
            if not np.isnan(mean_iou):
                txt += "\n" + rf"$\overline{{\mathrm{{IoU}}}}$ = {100*mean_iou:.0f}\%"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#999999", alpha=0.85), zorder=10)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$y$ [m]")
        ax.set_title(r"Final positions — PF source estimate (95\% confidence ellipse)",
                     fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", framealpha=0.85, fontsize=9)
        fig.tight_layout()
    return fig


# ============================================================================
# Evoluzione nuvola di particelle PF (per il report)
# ============================================================================

def plot_pf_evolution(
    terrain:  Terrain,
    artva:    ARTVASource,
    agents:   Dict[int, DroneAgent],
    n_snaps:  int = 3,
    drone_id: int = None,
) -> plt.Figure:
    """
    Griglia n_snaps pannelli 3D con range assi fissi e patch di terreno,
    per mostrare la concentrazione delle particelle nel tempo.
    """
    drone_ids = list(agents.keys())
    if drone_id is None:
        drone_id = max(
            drone_ids,
            key=lambda i: sum(1 for e in agents[i].pf_log if e is not None),
        )

    ag     = agents[drone_id]
    victim = artva._theta

    active = [k for k, e in enumerate(ag.pf_log) if e is not None]
    if len(active) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Nessun dato PF disponibile",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    if len(active) <= n_snaps:
        snap_idx = active
    else:
        indices = np.round(np.linspace(0, len(active) - 1, n_snaps)).astype(int)
        snap_idx = [active[i] for i in indices]

    # Le particelle vivono nel frame stimato del drone (il PF usa x_est): per
    # confrontarle con vittima e drone reali si rimuove il drift IMDCL del passo.
    def _world_particles(step):
        parts, w = ag.pf_log[step]
        drift = np.zeros(3)
        if step < len(ag.history) and step < len(ag.est_history):
            drift = ag.est_history[step][:3] - ag.history[step][:3]
        return parts[:, :3] - drift, w

    # ── Range fissi: bounding box di tutte le particelle + vittima ──────────
    all_parts = np.vstack([_world_particles(k)[0] for k in snap_idx])
    margin = 5.0
    xl = (all_parts[:, 0].min() - margin, all_parts[:, 0].max() + margin)
    yl = (all_parts[:, 1].min() - margin, all_parts[:, 1].max() + margin)
    zl = (min(all_parts[:, 2].min(), victim[2]) - margin,
          max(all_parts[:, 2].max(), victim[2]) + margin)

    # ── Superficie del terreno nella patch ──────────────────────────────────
    res = 20
    xs_t = np.linspace(max(xl[0], terrain.x_min), min(xl[1], terrain.x_max), res)
    ys_t = np.linspace(max(yl[0], terrain.y_min), min(yl[1], terrain.y_max), res)
    XT, YT = np.meshgrid(xs_t, ys_t)
    ZT = terrain.z(XT.ravel(), YT.ravel()).reshape(res, res)

    ncols = len(snap_idx)
    with plt.rc_context(_LATEX_RC):
        fig = plt.figure(figsize=(4.5 * ncols, 4.6))
        fig.patch.set_facecolor("#ffffff")

        for col, step in enumerate(snap_idx):
            ax = fig.add_subplot(1, ncols, col + 1, projection="3d")
            ax.set_facecolor("#f8f8f8")

            # Terreno
            ax.plot_surface(XT, YT, ZT, cmap="copper", alpha=0.35,
                            rcount=res, ccount=res, linewidth=0, zorder=1)

            particles, weights = _world_particles(step)
            w_norm = weights / (weights.max() + 1e-30)

            sc = ax.scatter(
                particles[:, 0], particles[:, 1], particles[:, 2],
                c=w_norm, cmap="viridis", s=6, alpha=0.7,
                vmin=0, vmax=1, zorder=4,
            )

            mean_est = np.average(particles, weights=weights, axis=0)
            ax.scatter(*mean_est, color="orange", s=55, marker="D",
                       edgecolors="black", linewidths=0.5, zorder=7,
                       label=r"$\hat{\theta}$")

            ax.scatter(*victim, color="crimson", s=70, marker="*",
                       edgecolors="black", linewidths=0.4, zorder=8,
                       label=r"$\theta$ (victim)")

            if step < len(ag.history):
                p_drone = ag.history[step][:3]
                ax.scatter(*p_drone, color="royalblue", s=35, marker="^",
                           edgecolors="white", linewidths=0.5, zorder=6,
                           label="Drone")

            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
            ax.set_title(rf"step {step - active[0]}",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("x [m]", fontsize=7, labelpad=2)
            ax.set_ylabel("y [m]", fontsize=7, labelpad=2)
            ax.set_zlabel("z [m]", fontsize=7, labelpad=2)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=20, azim=225)

        # ── Colorbar orizzontale in basso ────────────────────────────────────
        cax = fig.add_axes([0.15, 0.04, 0.50, 0.025])
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.set_label("Normalized weight", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # ── Legenda a destra della colorbar ─────────────────────────────────
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", ncol=1,
                   fontsize=8, framealpha=0.85,
                   bbox_to_anchor=(0.98, 0.01))

        fig.suptitle(rf"PF particle cloud — Drone {drone_id}",
                     fontsize=10, fontweight="bold")
        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
    return fig


# ============================================================================
# Animazione missione (main.py)
# ============================================================================

def animate_mission(
    terrain:   Terrain,
    artva:     ARTVASource,
    agents:    Dict[int, DroneAgent],
    dt:        float = DT_SIM,
    fps:       int   = 30,
    speed:     float = 2.0,
    interval:  int   = None,
    save:      bool  = False,
    save_path: str   = "mission_animation",
) -> FuncAnimation:
    """
    Animazione 3-D (sinistra) + vista overhead 2-D (destra).
    Nel pannello 2-D: cerchi di distanza stimata ARTVA per ogni drone in TRACK;
    al momento del raffinamento: cerchio passante per le intersezioni dei 3 cerchi.
    """
    drone_ids = list(agents.keys())
    T = max(len(ag.history) for ag in agents.values())
    N_THETA = 90
    _th  = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    _cos = np.cos(_th)
    _sin = np.sin(_th)

    # ── Pre-calcola step di stop (usato per il testo stato) ─────────────
    stop_steps: Dict[int, int] = {}
    for i in drone_ids:
        for k, (_, s) in enumerate(agents[i].signal_log):
            if s >= TRACK_STOP_THR:
                stop_steps[i] = k
                break

    # ── Figura: 3-D a sinistra, 2-D a destra ────────────────────────────
    xs_3d = np.linspace(terrain.x_min, terrain.x_max, 35)
    ys_3d = np.linspace(terrain.y_min, terrain.y_max, 35)
    X3, Y3 = np.meshgrid(xs_3d, ys_3d)
    Z3 = terrain.z(X3.ravel(), Y3.ravel()).reshape(X3.shape)

    fig = plt.figure(figsize=(18, 8), facecolor=_BG_COLOR)
    ax  = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # — 3-D setup —
    ax.set_facecolor(_BG_COLOR)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.plot_surface(X3, Y3, Z3, cmap="terrain", alpha=0.35,
                    rcount=35, ccount=35, linewidth=0, zorder=1)
    ax.scatter(*artva._theta, marker="*", color="yellow",
               s=300, zorder=10, edgecolors="red", linewidths=1.5)
    ax.set_xlabel("x [m]", fontsize=8, labelpad=4)
    ax.set_ylabel("y [m]", fontsize=8, labelpad=4)
    ax.set_zlabel("z [m]", fontsize=8, labelpad=4)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=7)

    # — 2-D setup: sfondo statico (terreno + sorgente) —
    ax2.set_facecolor(_BG_COLOR)
    ax2.tick_params(colors=_TEXT_COLOR, labelsize=7)
    ax2.set_xlabel("x [m]", color=_TEXT_COLOR, fontsize=8)
    ax2.set_ylabel("y [m]", color=_TEXT_COLOR, fontsize=8)
    ax2.set_title("Piano XY", color=_TEXT_COLOR, fontsize=9, fontweight="bold")
    for sp in ax2.spines.values():
        sp.set_edgecolor(_GRID_COLOR)
    ax2.grid(True, color=_GRID_COLOR, lw=0.4, alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(terrain.x_min, terrain.x_max)
    ax2.set_ylim(terrain.y_min, terrain.y_max)

    nx2 = 80
    xs2 = np.linspace(terrain.x_min, terrain.x_max, nx2)
    ys2 = np.linspace(terrain.y_min, terrain.y_max, nx2)
    XS2, YS2 = np.meshgrid(xs2, ys2)
    ZS2 = terrain.z(XS2.ravel(), YS2.ravel()).reshape(nx2, nx2)
    ls_  = LightSource(azdeg=315, altdeg=45)
    hs2  = ls_.hillshade(np.nan_to_num(ZS2, nan=0.0), vert_exag=3)
    ext  = [terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max]
    ax2.imshow(hs2, cmap="gray", alpha=0.4, extent=ext, origin="lower", zorder=1)
    # Posizione reale vittima
    ax2.scatter(*artva._theta[:2], marker="*", color="yellow",
                s=200, zorder=10, edgecolors="red", linewidths=1.2)

    # — Artists dinamici: 3-D —
    trails_r, dots_r = {}, {}
    trails_e, dots_e = {}, {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        trails_r[i], = ax.plot([], [], [], color=c, lw=1.5, alpha=0.85)
        dots_r[i],   = ax.plot([], [], [], "o", color=c, ms=7, mec="white", mew=0.8, zorder=8)
        trails_e[i], = ax.plot([], [], [], color=c, lw=1.0, alpha=0.5, ls="--")
        dots_e[i],   = ax.plot([], [], [], "o", color=c, ms=4, mfc="none", mec=c, mew=1.0, zorder=7)

    # — Artists dinamici: 2-D trail, dot reale, dot stimato, dot waypoint —
    trails2, dots2, dots2_e, wp_dots2 = {}, {}, {}, {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        trails2[i],  = ax2.plot([], [], color=c, lw=1.2, alpha=0.65, zorder=3)
        dots2[i],    = ax2.plot([], [], "o", color=c, ms=6, mec="white", mew=0.8, zorder=5)
        dots2_e[i],  = ax2.plot([], [], "o", color=c, ms=4, mfc="none", mec=c, mew=1.2, zorder=6)
        wp_dots2[i], = ax2.plot([], [], "o", color=c, ms=6, mec="white", mew=0.6,
                                alpha=0.35, zorder=4)

    # — Frecce direzione ES (visibili solo in TRACK) ───────────────────────
    es_arrows2: Dict[int, FancyArrowPatch] = {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        arrow = FancyArrowPatch(
            (0, 0), (1, 1),
            arrowstyle="-|>",
            color=c, lw=1.8,
            mutation_scale=14,
            alpha=0.0, zorder=9,
        )
        ax2.add_patch(arrow)
        es_arrows2[i] = arrow

    # — Pre-calcola pf_log per step (forward-fill: None prima dell'attivazione) ──
    _pf_at: Dict[int, list] = {}
    for i in drone_ids:
        log  = agents[i].pf_log
        arr  = [None] * T
        for k in range(min(len(log), T)):
            arr[k] = log[k]
        last = None
        for k in range(T):
            if arr[k] is not None:
                last = arr[k]
            else:
                arr[k] = last
        _pf_at[i] = arr

    # — Particelle PF: scatter 2-D (overhead) e 3-D —————————————————————────
    pf_scatters:    dict = {}
    pf_scatters_3d: dict = {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        pf_scatters[i] = ax2.scatter(
            [], [], s=5, c=c, alpha=0.50, zorder=2, linewidths=0,
        )
        pf_scatters_3d[i] = ax.scatter(
            [], [], [], s=5, c=c, alpha=0.35, zorder=6, linewidths=0,
            depthshade=False,
        )

    # — Sfera di comunicazione 3-D: 3 cerchi ortogonali per drone ────────
    N_SP   = 60
    _sp_t  = np.linspace(0, 2 * np.pi, N_SP, endpoint=False)
    _sp_c  = np.cos(_sp_t)
    _sp_s  = np.sin(_sp_t)
    R_comm = _cfg.IMDCL_COMM_RADIUS
    sph_xy, sph_xz, sph_yz = {}, {}, {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        sph_xy[i], = ax.plot([], [], [], color=c, lw=0.6, alpha=0.25)
        sph_xz[i], = ax.plot([], [], [], color=c, lw=0.6, alpha=0.25)
        sph_yz[i], = ax.plot([], [], [], color=c, lw=0.6, alpha=0.25)

    info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                     color=_TEXT_COLOR, fontsize=8, va="top", fontfamily="monospace")

    fig.suptitle(
        "Ricerca valanga multi-agente — MPC + IMDCL + PF\n"
        "reale (—)  ·  IMDCL (--)  ·  particelle PF (coordinate mondo)  ·  →: ES",
        color=_TEXT_COLOR, fontsize=10, fontweight="bold",
    )

    step_skip = max(1, int(round(1.0 / (dt * fps) * speed)))
    frame_idx = list(range(0, T, step_skip))

    _empty2 = np.empty((0, 2))

    all_artists = (
        list(trails_r.values()) + list(dots_r.values())
        + list(trails_e.values()) + list(dots_e.values())
        + list(sph_xy.values()) + list(sph_xz.values()) + list(sph_yz.values())
        + list(trails2.values()) + list(dots2.values())
        + list(dots2_e.values())
        + list(wp_dots2.values())
        + list(pf_scatters.values())
        + list(pf_scatters_3d.values())
        + [info]
    )

    _empty3 = (np.array([]), np.array([]), np.array([]))

    def init():
        for i in drone_ids:
            for obj in (trails_r[i], trails_e[i], dots_r[i], dots_e[i],
                        sph_xy[i], sph_xz[i], sph_yz[i]):
                obj.set_data([], [])
                obj.set_3d_properties([])
            for obj in (trails2[i], dots2[i], dots2_e[i], wp_dots2[i]):
                obj.set_data([], [])
            es_arrows2[i].set_alpha(0.0)
            pf_scatters[i].set_offsets(_empty2)
            pf_scatters_3d[i]._offsets3d = _empty3
        info.set_text("")
        return all_artists

    def update(f):
        t_step = frame_idx[f]
        lines  = [f"t = {t_step * dt:.2f} s  step {t_step}/{T-1}"]

        for i in drone_ids:
            ag   = agents[i]
            traj = np.array(ag.history)
            est  = np.array(ag.est_history) if ag.est_history else traj
            ti   = min(t_step, len(traj) - 1)
            ts   = max(0, ti - _TRAIL_LEN)
            ti_e = min(t_step, len(est) - 1)
            ts_e = max(0, ti_e - _TRAIL_LEN)

            # 3-D trails
            trails_r[i].set_data(traj[ts:ti+1, 0], traj[ts:ti+1, 1])
            trails_r[i].set_3d_properties(traj[ts:ti+1, 2])
            dots_r[i].set_data([traj[ti, 0]], [traj[ti, 1]])
            dots_r[i].set_3d_properties([traj[ti, 2]])
            trails_e[i].set_data(est[ts_e:ti_e+1, 0], est[ts_e:ti_e+1, 1])
            trails_e[i].set_3d_properties(est[ts_e:ti_e+1, 2])
            dots_e[i].set_data([est[ti_e, 0]], [est[ti_e, 1]])
            dots_e[i].set_3d_properties([est[ti_e, 2]])

            # Sfera comunicazione 3-D (3 cerchi ortogonali centrati sul drone reale)
            cx, cy, cz = traj[ti, 0], traj[ti, 1], traj[ti, 2]
            sph_xy[i].set_data(cx + R_comm * _sp_c, cy + R_comm * _sp_s)
            sph_xy[i].set_3d_properties(np.full(N_SP, cz))
            sph_xz[i].set_data(cx + R_comm * _sp_c, np.full(N_SP, cy))
            sph_xz[i].set_3d_properties(cz + R_comm * _sp_s)
            sph_yz[i].set_data(np.full(N_SP, cx), cy + R_comm * _sp_c)
            sph_yz[i].set_3d_properties(cz + R_comm * _sp_s)

            # 2-D trail + dot reale + dot stimato IMDCL
            trails2[i].set_data(traj[ts:ti+1, 0], traj[ts:ti+1, 1])
            dots2[i].set_data([traj[ti, 0]], [traj[ti, 1]])
            dots2_e[i].set_data([est[ti_e, 0]], [est[ti_e, 1]])

            # Waypoint corrente (pallino semi-trasparente)
            wp_log = ag.wp_target_log
            if wp_log:
                wp = wp_log[min(t_step, len(wp_log) - 1)]
                wp_dots2[i].set_data([wp[0]], [wp[1]])
            else:
                wp_dots2[i].set_data([], [])

            # Particelle PF — il PF lavora nel frame stimato del drone (x_est):
            # per il rendering nel mondo reale si rimuove il drift IMDCL del
            # passo corrente (est − reale), così la nuvola converge alla vittima.
            pf_entry = _pf_at[i][t_step]
            if pf_entry is not None:
                parts, w = pf_entry
                drift    = est[ti_e, :3] - traj[ti, :3]
                parts_w  = parts[:, :3] - drift
                w_norm   = w / (w.max() + 1e-15)
                sizes    = 3 + w_norm * 22
                pf_scatters[i].set_offsets(parts_w[:, :2])
                pf_scatters[i].set_sizes(sizes)
                pf_scatters_3d[i]._offsets3d = (parts_w[:, 0], parts_w[:, 1], parts_w[:, 2])
                pf_scatters_3d[i].set_sizes(sizes)
            else:
                pf_scatters[i].set_offsets(_empty2)
                pf_scatters_3d[i]._offsets3d = _empty3

            # Direzione ES: freccia dal drone verso il riferimento ES
            wp_log = ag.wp_target_log
            in_track = (ag.state_log and ti < len(ag.state_log)
                        and ag.state_log[ti] == DroneState.TRACK)
            if in_track and wp_log:
                es_ref    = wp_log[min(ti, len(wp_log) - 1)][:2]
                drone_xy  = traj[ti, :2]
                direction = es_ref - drone_xy
                norm      = np.linalg.norm(direction)
                if norm > 0.1:
                    es_tip = drone_xy + direction / norm * max(norm, 25.0)
                else:
                    es_tip = es_ref
                es_arrows2[i].set_positions(tuple(drone_xy), tuple(es_tip))
                es_arrows2[i].set_alpha(0.85)
            else:
                es_arrows2[i].set_alpha(0.0)

            # Testo stato — usa lo state_log reale se disponibile
            if ag.state_log and ti < len(ag.state_log):
                _s = ag.state_log[ti]
                if _s == DroneState.SEARCH:
                    st = "SRCH"
                elif _s == DroneState.TRACK:
                    st = "TRCK"
                elif _s == DroneState.STOP:
                    st = "STOP"
                elif _s == DroneState.FINAL_ORBIT:
                    st = "ORBT"
                else:
                    st = "SUPP"
            elif sig < ARTVA_DETECT_THR:
                st = "SRCH"
            elif ti >= stop_steps.get(i, T):
                st = "STOP"
            else:
                st = "TRCK"
            err = np.linalg.norm(traj[ti, :3] - est[ti_e, :3])
            lines.append(f"D{i}: {st}  z={traj[ti, 2]:.1f}m  Δ={err:.2f}m")

        info.set_text("\n".join(lines))
        return all_artists

    # ── Slider temporale ─────────────────────────────────────────────────
    fig.subplots_adjust(bottom=0.09)
    ax_sl = fig.add_axes([0.10, 0.025, 0.80, 0.025], facecolor="#1e2730")
    t_max = (T - 1) * dt
    slider = Slider(ax_sl, "t [s]", 0.0, t_max,
                    valinit=0.0, color="#3a7fc1")
    slider.label.set_color(_TEXT_COLOR)
    slider.valtext.set_color(_TEXT_COLOR)
    slider.label.set_fontsize(8)
    slider.valtext.set_fontsize(8)

    # _cur_f è il frame corrente — lo gestiamo noi, FuncAnimation passa f ma lo ignoriamo
    _sl_lock = [False]
    _cur_f   = [0]

    def _on_slider(val):
        if _sl_lock[0]:
            return
        f = int(val / (step_skip * dt))
        _cur_f[0] = max(0, min(f, len(frame_idx) - 1))
        update(_cur_f[0])
        fig.canvas.draw_idle()

    slider.on_changed(_on_slider)

    _base_update = update

    def update(f):                          # noqa: F811  — ignora f di FuncAnimation
        fi     = _cur_f[0]
        result = _base_update(fi)
        _cur_f[0] = (fi + 1) % len(frame_idx)
        _sl_lock[0] = True
        slider.set_val(frame_idx[fi] * dt)
        _sl_lock[0] = False
        return result

    anim = FuncAnimation(fig, update, frames=len(frame_idx),
                         init_func=init, blit=False,
                         interval=interval if interval is not None else 1000 // fps)

    if save:
        _save_animation(anim, save_path, fps)
    return anim


# ============================================================================
# Animazione MPC standalone (ex animate_drone.py)
# ============================================================================

def _build_mpc_figure(drone_ids, waypoints):
    """Costruisce la figura per animate_mpc_standalone."""
    fig = plt.figure(figsize=(16, 9), facecolor=_BG_COLOR)
    ax3d  = fig.add_subplot(1, 3, (1, 2), projection="3d")
    ax_xy = fig.add_subplot(3, 3, 3)
    ax_xz = fig.add_subplot(3, 3, 6)
    ax_yz = fig.add_subplot(3, 3, 9)

    for ax in (ax3d, ax_xy, ax_xz, ax_yz):
        ax.set_facecolor(_BG_COLOR)

    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_GRID_COLOR)
    ax3d.grid(True, color=_GRID_COLOR, linewidth=0.5)
    ax3d.tick_params(colors=_TEXT_COLOR, labelsize=7)
    for lbl in (ax3d.xaxis.label, ax3d.yaxis.label, ax3d.zaxis.label):
        lbl.set_color(_TEXT_COLOR)
    ax3d.set_xlabel("x [m]", fontsize=8, labelpad=4)
    ax3d.set_ylabel("y [m]", fontsize=8, labelpad=4)
    ax3d.set_zlabel("z [m]", fontsize=8, labelpad=4)

    proj_axs = {
        "XY": (ax_xy, 0, 1, "x [m]", "y [m]"),
        "XZ": (ax_xz, 0, 2, "x [m]", "z [m]"),
        "YZ": (ax_yz, 1, 2, "y [m]", "z [m]"),
    }
    for name, (ax, _, _, xl, yl) in proj_axs.items():
        ax.set_title(name, color=_TEXT_COLOR, fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel(xl, color=_TEXT_COLOR, fontsize=7)
        ax.set_ylabel(yl, color=_TEXT_COLOR, fontsize=7)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=6)
        ax.grid(True, color=_GRID_COLOR, linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COLOR)

    for i in drone_ids:
        c   = COLORS.get(i, "#aaaaaa")
        wps = waypoints[i]
        for k, wp in enumerate(wps):
            size = 200 if k == len(wps) - 1 else 100
            ax3d.scatter(*wp, marker="*", color=c, s=size,
                         edgecolors="white", linewidths=0.8, zorder=10, alpha=0.9)
            ax3d.text(wp[0], wp[1], wp[2] + 0.1, f" wp{k}",
                      color=c, fontsize=6, alpha=0.7)
            for name, (ax2, ix, iy, _, _) in proj_axs.items():
                ax2.plot(wp[ix], wp[iy], "*", color=c,
                         ms=9 if k == len(wps) - 1 else 6,
                         mec="white", mew=0.5, zorder=10, alpha=0.9)
        if len(wps) > 1:
            wp_arr = np.array(wps)
            ax3d.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                      color=c, lw=0.7, ls=":", alpha=0.35)
            for name, (ax2, ix, iy, _, _) in proj_axs.items():
                ax2.plot(wp_arr[:, ix], wp_arr[:, iy], color=c, lw=0.7, ls=":", alpha=0.35)

    fig.tight_layout(pad=1.2)
    return fig, ax3d, proj_axs


def animate_mpc_standalone(
    starts:    dict,
    targets:   dict,
    history:   dict,
    inputs:    dict,
    dt:        float = DT_SIM,
    save:      bool  = False,
    save_path: str   = "drone_animation",
    fps:       int   = 30,
    speed:     float = 1.0,
    waypoints: dict  = None,
) -> FuncAnimation:
    """
    Animazione MPC 3-D + proiezioni.

    Parameters
    ----------
    starts, targets : {id: array}
    history         : {id: list of x (6,)}  — output di mpc_drone.simulate()
    inputs          : {id: list of u (3,)}
    waypoints       : {id: list of np.array(3,)}; se None derivato da targets
    """
    from mpc_drone import N_MPC as _N_MPC, DT_MPC as _DT_MPC, A_MAX as _A_MAX, V_MAX as _V_MAX

    drone_ids = list(starts.keys())
    T         = max(len(history[i]) for i in drone_ids)

    if waypoints is None:
        waypoints = {}
        for i in drone_ids:
            t = targets[i]
            waypoints[i] = ([t] if isinstance(t, np.ndarray) and t.ndim == 1
                            else [np.asarray(w) for w in t])

    all_pos = np.vstack([np.array(history[i])[:, :3] for i in drone_ids])
    all_wps = np.vstack([np.array(wps) for wps in waypoints.values()])
    all_pts = np.vstack([all_pos, all_wps])
    margin  = 0.5
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    fig, ax3d, proj_axs = _build_mpc_figure(drone_ids, waypoints)

    ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
    for name, (ax, ix, iy, _, _) in proj_axs.items():
        lims = [xlim, ylim, zlim]
        ax.set_xlim(lims[ix]); ax.set_ylim(lims[iy])

    trails_3d, dots_3d = {}, {}
    trails_2d = {name: {} for name in proj_axs}
    dots_2d   = {name: {} for name in proj_axs}

    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        trails_3d[i], = ax3d.plot([], [], [], color=c, lw=1.4, alpha=0.7)
        dots_3d[i],   = ax3d.plot([], [], [], "o", color=c, ms=7,
                                   mec="white", mew=0.8, zorder=8)
        for name, (ax, ix, iy, _, _) in proj_axs.items():
            trails_2d[name][i], = ax.plot([], [], color=c, lw=1.2, alpha=0.7)
            dots_2d[name][i],   = ax.plot([], [], "o", color=c, ms=5,
                                           mec="white", mew=0.6, zorder=8)

    info_text = ax3d.text2D(
        0.02, 0.97, "", transform=ax3d.transAxes,
        color=_TEXT_COLOR, fontsize=8, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG_COLOR,
                  alpha=0.6, edgecolor=_GRID_COLOR),
    )
    fig.suptitle(
        f"MPC Droni 3-D  ·  N={_N_MPC}, dt={_DT_MPC} s  ·  "
        f"|a|≤{_A_MAX} m/s²  |v|≤{_V_MAX} m/s",
        color=_TEXT_COLOR, fontsize=10, fontweight="bold", y=0.99,
    )
    legend_handles = [
        plt.Line2D([0],[0], color=COLORS.get(i, "#aaa"), lw=2,
                   label=f"Drone {i}  ({len(waypoints[i])} wp)")
        for i in drone_ids
    ]
    legend_handles.append(
        plt.Line2D([0],[0], marker="*", color="w", mfc=_TEXT_COLOR,
                   ms=9, label="Waypoint", linestyle="None")
    )
    ax3d.legend(handles=legend_handles, fontsize=7.5, loc="lower right",
                framealpha=0.4, facecolor=_BG_COLOR, edgecolor=_GRID_COLOR,
                labelcolor=_TEXT_COLOR)

    step_skip     = max(1, int((1.0 / dt) / fps * speed))
    frame_indices = list(range(0, T, step_skip))
    TRAIL = 40

    def init():
        for i in drone_ids:
            trails_3d[i].set_data([], []); trails_3d[i].set_3d_properties([])
            dots_3d[i].set_data([], []);   dots_3d[i].set_3d_properties([])
            for name in proj_axs:
                trails_2d[name][i].set_data([], [])
                dots_2d[name][i].set_data([], [])
        info_text.set_text("")
        return (list(trails_3d.values()) + list(dots_3d.values()) +
                [v for d in trails_2d.values() for v in d.values()] +
                [v for d in dots_2d.values() for v in d.values()] + [info_text])

    def update(frame_idx):
        t = frame_indices[frame_idx]
        info_lines = [f"t = {t * dt:5.2f} s   step {t:4d}/{T-1}"]
        for i in drone_ids:
            traj = np.array(history[i])
            t_i  = min(t, len(traj) - 1)
            t_s  = max(0, t_i - TRAIL)
            trails_3d[i].set_data(traj[t_s:t_i+1, 0], traj[t_s:t_i+1, 1])
            trails_3d[i].set_3d_properties(traj[t_s:t_i+1, 2])
            dots_3d[i].set_data([traj[t_i, 0]], [traj[t_i, 1]])
            dots_3d[i].set_3d_properties([traj[t_i, 2]])
            for name, (ax, ix, iy, _, _) in proj_axs.items():
                trails_2d[name][i].set_data(traj[t_s:t_i+1, ix], traj[t_s:t_i+1, iy])
                dots_2d[name][i].set_data([traj[t_i, ix]], [traj[t_i, iy]])
            wps   = waypoints[i]
            dists = [np.linalg.norm(traj[t_i, :3] - wp) for wp in wps]
            cur_wp = min(range(len(wps)), key=lambda k: dists[k])
            info_lines.append(
                f"D{i}: wp{cur_wp}/{len(wps)-1}  "
                f"dist={dists[cur_wp]:5.2f}m  |v|={np.linalg.norm(traj[t_i, 3:6]):4.2f}m/s"
            )
        info_text.set_text("\n".join(info_lines))
        return (list(trails_3d.values()) + list(dots_3d.values()) +
                [v for d in trails_2d.values() for v in d.values()] +
                [v for d in dots_2d.values() for v in d.values()] + [info_text])

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         init_func=init, blit=True, interval=1000 / fps)
    if save:
        _save_animation(anim, save_path, fps)
    return anim


# ============================================================================
# Main standalone (ex animate_drone.py)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animazione standalone simulazione MPC droni 3-D"
    )
    parser.add_argument("--save",  action="store_true")
    parser.add_argument("--fps",   type=int,   default=30)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--out",   type=str,   default="drone_animation")
    args = parser.parse_args()

    from mpc_drone import simulate as mpc_simulate, DT_SIM as _DT, N_SIM as _N

    starts = {
        0: np.array([0.0,  0.0,  0.0,  0.0, 0.0, 0.0]),
        1: np.array([5.0,  0.0,  1.0,  0.0, 0.0, 0.0]),
        2: np.array([2.5,  4.0,  3.0,  0.0, 0.0, 0.0]),
    }
    targets = {
        0: [np.array([4.0,  3.0,  2.0]),
            np.array([2.0,  5.0,  1.0]),
            np.array([0.0,  0.0,  0.5])],
        1: [np.array([0.0,  4.0,  0.5]),
            np.array([3.0,  2.0,  3.0])],
        2: np.array([5.0, -1.0,  4.0]),
    }

    print("Esecuzione simulazione MPC...")
    history, inputs, solve_t, waypoints = mpc_simulate(
        starts=starts, targets=targets, dt=_DT, n_steps=_N,
    )
    print("Simulazione completata. Avvio animazione...")

    anim = animate_mpc_standalone(
        starts=starts, targets=targets,
        history=history, inputs=inputs,
        dt=_DT, waypoints=waypoints,
        save=args.save, save_path=args.out,
        fps=args.fps, speed=args.speed,
    )
    plt.show()
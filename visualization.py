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

from artva import ARTVASource
from terrain import Terrain
from drone_agent import DroneAgent, DroneState
from config import (
    AGL_HEIGHT, DT_SIM, N_MPC, DT_MPC, A_MAX, V_MAX,
    ARTVA_DETECT_THR, TRACK_STOP_THR, ARTVA_MOMENT,
    IMDCL_COMM_RADIUS, IMDCL_R_MEAS_STD,
    TRACK_STEP_M,
    COLORS, BG_DARK,
)

# ---------------------------------------------------------------------------
# Palette condivisa (usata anche in animate_mpc_standalone)
# ---------------------------------------------------------------------------
_BG_COLOR   = BG_DARK
_GRID_COLOR = "#1e2730"
_TEXT_COLOR = "#c9d1d9"
_TRAIL_LEN  = 50


# ============================================================================
# Helper privato
# ============================================================================

def _circle_intersections_2d(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float,
) -> tuple | None:
    """Restituisce i 2 punti di intersezione di due cerchi XY, o None."""
    d = np.linalg.norm(c2 - c1)
    if d < 1e-9 or d > r1 + r2 + 1e-6 or d < abs(r1 - r2) - 1e-6:
        return None
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = np.sqrt(max(0.0, r1**2 - a**2))
    direction = (c2 - c1) / d
    perp = np.array([-direction[1], direction[0]])
    mid  = c1 + a * direction
    return mid + h * perp, mid - h * perp


def _circumcircle_2d(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
) -> tuple:
    """Circonferenza passante per 3 punti 2D. Restituisce (center, radius) o (None, None)."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    D = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(D) < 1e-9:
        return None, None
    a2 = ax**2 + ay**2
    b2 = bx**2 + by**2
    c2 = cx**2 + cy**2
    ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / D
    uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / D
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(p1 - center))


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
        t_idx = [k for k in range(n) if state_seq[k] in (DroneState.TRACK, DroneState.SUPPORT)]
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

    ax_a.plot(*artva.position[:2], "*", color="white", ms=18, zorder=10,
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
    ax_b.scatter(*artva.position, marker="*", color="yellow", s=250,
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
        f"IMDCL (Rc={IMDCL_COMM_RADIUS:.0f} m, σ={IMDCL_R_MEAS_STD} m) · "
        f"TRACK step={TRACK_STEP_M} m",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ============================================================================
# Animazione missione (main.py)
# ============================================================================

def animate_mission(
    terrain:          Terrain,
    artva:            ARTVASource,
    agents:           Dict[int, DroneAgent],
    dt:               float = DT_SIM,
    fps:              int   = 30,
    speed:            float = 2.0,
    interval:         int   = None,
    save:             bool  = False,
    save_path:        str   = "mission_animation",
    consensus_events: list  = None,
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
    ax.scatter(*artva.position, marker="*", color="yellow",
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
    ax2.scatter(*artva.position[:2], marker="*", color="yellow",
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

    # — Artists dinamici: 2-D trail, dot reale, dot stimato, cerchio, dot waypoint —
    trails2, dots2, dots2_e, circles2, wp_dots2 = {}, {}, {}, {}, {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        trails2[i],  = ax2.plot([], [], color=c, lw=1.2, alpha=0.65, zorder=3)
        dots2[i],    = ax2.plot([], [], "o", color=c, ms=6, mec="white", mew=0.8, zorder=5)
        dots2_e[i],  = ax2.plot([], [], "o", color=c, ms=4, mfc="none", mec=c, mew=1.2, zorder=6)
        circles2[i], = ax2.plot([], [], color=c, lw=1.0, alpha=0.7, ls="--", zorder=4)
        wp_dots2[i], = ax2.plot([], [], "o", color=c, ms=6, mec="white", mew=0.6,
                                alpha=0.35, zorder=4)  # waypoint corrente

    # — Sfera di comunicazione 3-D: 3 cerchi ortogonali per drone ────────
    N_SP   = 60
    _sp_t  = np.linspace(0, 2 * np.pi, N_SP, endpoint=False)
    _sp_c  = np.cos(_sp_t)
    _sp_s  = np.sin(_sp_t)
    R_comm = IMDCL_COMM_RADIUS
    sph_xy, sph_xz, sph_yz = {}, {}, {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        sph_xy[i], = ax.plot([], [], [], color=c, lw=0.6, alpha=0.25)
        sph_xz[i], = ax.plot([], [], [], color=c, lw=0.6, alpha=0.25)
        sph_yz[i], = ax.plot([], [], [], color=c, lw=0.6, alpha=0.25)

    info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                     color=_TEXT_COLOR, fontsize=8, va="top", fontfamily="monospace")

    fig.suptitle(
        "Ricerca valanga multi-agente — MPC + IMDCL + FSM 4 stati\n"
        "reale (—)  ·  stima IMDCL (--)  ·  cerchi: distanza stimata ARTVA",
        color=_TEXT_COLOR, fontsize=10, fontweight="bold",
    )

    step_skip = max(1, int(round(1.0 / (dt * fps) * speed)))
    frame_idx = list(range(0, T, step_skip))

    # ── Consensus overlay artists (2-D panel) ────────────────────────────
    # Colours
    _C_LINK    = "#ff3333"   # red  — active communication link
    _C_HOVER   = "#ff0055"   # rose — hover drone ring
    _C_INFORM  = "#ff9900"   # orange — newly-informed drone ring
    _C_PARTNER = "#00ff88"   # green — selected partner ring

    _FRAMES_PER_ROUND  = 6   # animation frames to show each consensus round
    _PARTNER_LINGER    = 20  # extra frames to keep green partner rings

    # Pre-compute which animation frames map to which consensus round
    # Each entry: {'f_start', 'f_end', 'round', 'round_idx', 'n_rounds',
    #              'hover_id', 'partners', 'show_partners',
    #              'hover_pos', 'drone_pos'}
    _csn_windows: List[dict] = []
    if consensus_events:
        for ev in consensus_events:
            base_f     = ev['step'] // step_skip
            n_rounds   = len(ev['rounds'])
            hover_id   = ev.get('stop_id', ev.get('hover_id'))
            # Snapshot positions at the event step (fixed reference)
            ev_step    = min(ev['step'], T - 1)
            hover_pos2 = np.array(agents[hover_id].history[ev_step][:2])
            drone_pos2 = {
                did: np.array(agents[did].history[ev_step][:2])
                for did in drone_ids
            }
            for r_idx, rnd in enumerate(ev['rounds']):
                f_s = base_f + r_idx * _FRAMES_PER_ROUND
                f_e = f_s + _FRAMES_PER_ROUND
                is_last = (r_idx == n_rounds - 1)
                _csn_windows.append({
                    'f_start':       f_s,
                    'f_end':         f_e + (_PARTNER_LINGER if is_last else 0),
                    'round_end':     f_e,
                    'round':         rnd,
                    'round_idx':     r_idx,
                    'n_rounds':      n_rounds,
                    'hover_id':      hover_id,
                    'partners':      ev['partners'],
                    'show_partners': is_last,
                    'hover_pos':     hover_pos2,
                    'drone_pos':     drone_pos2,
                })

    # One comm-link Line2D per undirected pair (shared across all consensus events)
    n_dr   = len(drone_ids)
    _pairs = [
        (drone_ids[a], drone_ids[b])
        for a in range(n_dr) for b in range(a + 1, n_dr)
    ]
    comm_links = {}
    for pair in _pairs:
        comm_links[pair], = ax2.plot(
            [], [], color=_C_LINK, lw=2.0, ls="--",
            alpha=0.0, zorder=8, solid_capstyle="round",
        )

    # Per-drone rings (orange = informed, green = partner, red = hover)
    informed_rings = {}
    partner_rings  = {}
    hover_rings    = {}
    for i in drone_ids:
        informed_rings[i], = ax2.plot(
            [], [], "o", ms=20, mfc="none",
            mec=_C_INFORM, mew=2.5, alpha=0.0, zorder=9,
        )
        partner_rings[i], = ax2.plot(
            [], [], "o", ms=26, mfc="none",
            mec=_C_PARTNER, mew=2.5, alpha=0.0, zorder=10,
        )
        hover_rings[i], = ax2.plot(
            [], [], "*", ms=22, mfc="none",
            mec=_C_HOVER, mew=2.5, alpha=0.0, zorder=11,
        )

    consensus_label = ax2.text(
        terrain.x_max - 5, terrain.y_max - 8, "",
        color=_C_LINK, fontsize=8, fontweight="bold",
        ha="right", va="top", alpha=0.0, zorder=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG_COLOR,
                  alpha=0.7, edgecolor=_C_LINK, lw=1.0),
    )

    _csn_artists = (
        list(comm_links.values())
        + list(informed_rings.values())
        + list(partner_rings.values())
        + list(hover_rings.values())
        + [consensus_label]
    )

    all_artists = (
        list(trails_r.values()) + list(dots_r.values())
        + list(trails_e.values()) + list(dots_e.values())
        + list(sph_xy.values()) + list(sph_xz.values()) + list(sph_yz.values())
        + list(trails2.values()) + list(dots2.values())
        + list(dots2_e.values()) + list(circles2.values())
        + list(wp_dots2.values())
        + _csn_artists + [info]
    )

    def _reset_consensus_artists():
        for ln in comm_links.values():
            ln.set_data([], [])
            ln.set_alpha(0.0)
        for i in drone_ids:
            for ring in (informed_rings[i], partner_rings[i], hover_rings[i]):
                ring.set_data([], [])
                ring.set_alpha(0.0)
        consensus_label.set_text("")
        consensus_label.set_alpha(0.0)

    def init():
        for i in drone_ids:
            for obj in (trails_r[i], trails_e[i], dots_r[i], dots_e[i],
                        sph_xy[i], sph_xz[i], sph_yz[i]):
                obj.set_data([], [])
                obj.set_3d_properties([])
            for obj in (trails2[i], dots2[i], dots2_e[i], circles2[i], wp_dots2[i]):
                obj.set_data([], [])
        info.set_text("")
        _reset_consensus_artists()
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

            # Cerchio: raggio = (ARTVA_MOMENT / segnale)^(1/3), solo in TRACK
            sig = ag.signal_log[ti][1] if ti < len(ag.signal_log) else 0.0
            if sig >= ARTVA_DETECT_THR:
                r  = (ARTVA_MOMENT / max(sig, 1e-12)) ** (1.0 / 3.0)
                px, py = traj[ti, 0], traj[ti, 1]
                circles2[i].set_data(px + r * _cos, py + r * _sin)
            else:
                circles2[i].set_data([], [])

            # Waypoint corrente (pallino semi-trasparente)
            wp_log = ag.wp_target_log
            if wp_log:
                wp = wp_log[min(t_step, len(wp_log) - 1)]
                wp_dots2[i].set_data([wp[0]], [wp[1]])
            else:
                wp_dots2[i].set_data([], [])

            # Testo stato — usa lo state_log reale se disponibile
            if ag.state_log and ti < len(ag.state_log):
                _s = ag.state_log[ti]
                if _s == DroneState.SEARCH:
                    st = "SRCH"
                elif _s == DroneState.TRACK:
                    st = "TRCK"
                elif _s == DroneState.STOP:
                    st = "STOP"
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

        # ── Consensus overlay ─────────────────────────────────────────────
        _reset_consensus_artists()
        active_win = None
        for win in _csn_windows:
            if win['f_start'] <= f < win['f_end']:
                active_win = win
                break

        if active_win is not None:
            dp    = active_win['drone_pos']
            hp    = active_win['hover_pos']
            rnd   = active_win['round']
            r_idx = active_win['round_idx']
            n_r   = active_win['n_rounds']
            h_id  = active_win['hover_id']

            # Blink: high alpha on even frames within the window, low on odd
            blink_hi = ((f - active_win['f_start']) % 4) < 2
            a_link   = 0.85 if blink_hi else 0.20
            a_ring   = 0.90 if blink_hi else 0.25

            still_in_round = f < active_win['round_end']

            if still_in_round:
                # Draw all in-range communication links
                for (id_a, id_b) in rnd['links']:
                    pair_key = (min(id_a, id_b), max(id_a, id_b))
                    if pair_key in comm_links:
                        pa, pb = dp[id_a], dp[id_b]
                        comm_links[pair_key].set_data([pa[0], pb[0]], [pa[1], pb[1]])
                        comm_links[pair_key].set_alpha(a_link)

                # Orange rings on newly-informed drones
                for did in rnd['newly_informed']:
                    p = dp[did]
                    informed_rings[did].set_data([p[0]], [p[1]])
                    informed_rings[did].set_alpha(a_ring)

                # Red star on hover drone
                hover_rings[h_id].set_data([hp[0]], [hp[1]])
                hover_rings[h_id].set_alpha(a_ring)

                label = f"MIN-CONSENSUS  round {r_idx + 1}/{n_r}  (Rc={IMDCL_COMM_RADIUS:.0f} m)"
            else:
                # Linger phase: only show partner rings (no links, no label round)
                label = f"CONSENSO — partner selezionati"

            # Green rings on selected partners (last round + linger)
            # Usa la posizione corrente del drone (non lo snapshot al momento del consenso)
            if active_win['show_partners']:
                for did in active_win['partners']:
                    hist = agents[did].history
                    p = hist[min(t_step, len(hist) - 1)][:2]
                    partner_rings[did].set_data([p[0]], [p[1]])
                    partner_rings[did].set_alpha(a_ring)

            consensus_label.set_text(label)
            consensus_label.set_alpha(a_ring)

        info.set_text("\n".join(lines))
        return all_artists

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
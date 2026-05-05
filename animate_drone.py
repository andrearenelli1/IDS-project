"""
Animazione della simulazione MPC droni 3-D
==========================================
Importa i risultati da mpc_drone.py e produce:
  - Finestra interattiva matplotlib con animazione 3-D + proiezioni
  - File  drone_animation.mp4  (se ffmpeg è disponibile)
    oppure drone_animation.gif (fallback con Pillow)

Utilizzo standalone
-------------------
    python animate_drone.py          # esegue la simulazione e anima
    python animate_drone.py --save   # salva anche il file video/gif

Utilizzo come modulo (dopo aver chiamato simulate())
----------------------------------------------------
    from animate_drone import animate
    history, inputs, solve_t, waypoints = simulate(...)
    anim = animate(starts, targets, history, inputs,
                   dt=DT_SIM, waypoints=waypoints, save=True)
    plt.show()
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mpc_drone import simulate, N_MPC, DT_MPC, DT_SIM, N_SIM, A_MAX, V_MAX

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
COLORS     = {0: "#e63946", 1: "#2a9d8f", 2: "#e9c46a"}
BG_COLOR   = "#0d1117"
GRID_COLOR = "#1e2730"
TEXT_COLOR = "#c9d1d9"
TRAIL_LEN  = 40


# ===========================================================================
# Costruzione figura (waypoints già normalizzati)
# ===========================================================================

def _build_figure(drone_ids, waypoints):
    """
    Disegna la figura statica (sfondo, assi, stelle waypoint fisse).

    Parameters
    ----------
    waypoints : {id: list of np.array(3,)}  — già normalizzato
    """
    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)

    ax3d  = fig.add_subplot(1, 3, (1, 2), projection="3d")
    ax_xy = fig.add_subplot(3, 3, 3)
    ax_xz = fig.add_subplot(3, 3, 6)
    ax_yz = fig.add_subplot(3, 3, 9)

    for ax in (ax3d, ax_xy, ax_xz, ax_yz):
        ax.set_facecolor(BG_COLOR)

    # Stile assi 3-D
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax3d.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax3d.tick_params(colors=TEXT_COLOR, labelsize=7)
    for lbl in (ax3d.xaxis.label, ax3d.yaxis.label, ax3d.zaxis.label):
        lbl.set_color(TEXT_COLOR)
    ax3d.set_xlabel("x [m]", fontsize=8, labelpad=4)
    ax3d.set_ylabel("y [m]", fontsize=8, labelpad=4)
    ax3d.set_zlabel("z [m]", fontsize=8, labelpad=4)

    proj_axs = {
        "XY": (ax_xy, 0, 1, "x [m]", "y [m]"),
        "XZ": (ax_xz, 0, 2, "x [m]", "z [m]"),
        "YZ": (ax_yz, 1, 2, "y [m]", "z [m]"),
    }
    for name, (ax, _, _, xl, yl) in proj_axs.items():
        ax.set_title(name, color=TEXT_COLOR, fontsize=8,
                     fontweight="bold", pad=3)
        ax.set_xlabel(xl, color=TEXT_COLOR, fontsize=7)
        ax.set_ylabel(yl, color=TEXT_COLOR, fontsize=7)
        ax.tick_params(colors=TEXT_COLOR, labelsize=6)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    # Stelle waypoint fisse + linea tratteggiata del percorso pianificato
    for i in drone_ids:
        c   = COLORS.get(i, "#aaaaaa")
        wps = waypoints[i]
        for k, wp in enumerate(wps):
            size = 200 if k == len(wps) - 1 else 100
            ax3d.scatter(*wp, marker="*", color=c, s=size,
                         edgecolors="white", linewidths=0.8,
                         zorder=10, alpha=0.9)
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
                ax2.plot(wp_arr[:, ix], wp_arr[:, iy],
                         color=c, lw=0.7, ls=":", alpha=0.35)

    fig.tight_layout(pad=1.2)
    return fig, ax3d, proj_axs


# ===========================================================================
# Funzione principale di animazione
# ===========================================================================

def animate(
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
    Crea e (opzionalmente) salva l'animazione.

    Parameters
    ----------
    starts, targets : dizionari {id: array}
    history         : {id: list of x (6,)}  — output di simulate()
    inputs          : {id: list of u (3,)}
    dt              : passo temporale della simulazione
    save            : se True salva il file
    save_path       : percorso senza estensione
    fps             : frame per secondo del video salvato
    speed           : moltiplicatore velocità (2.0 = doppia velocità)
    waypoints       : {id: list of np.array(3,)} — quarto return di simulate();
                      se None viene ricavato da targets (retrocompatibile)

    Returns
    -------
    anim : FuncAnimation (tienilo in vita assegnandolo a una variabile)
    """
    drone_ids = list(starts.keys())
    T         = max(len(history[i]) for i in drone_ids)

    # — Normalizza waypoints (retrocompatibile con singolo target) —
    if waypoints is None:
        waypoints = {}
        for i in drone_ids:
            t = targets[i]
            waypoints[i] = ([t] if isinstance(t, np.ndarray) and t.ndim == 1
                            else [np.asarray(w) for w in t])

    # — Bounds globali (traiettorie + tutti i waypoint) —
    all_pos = np.vstack([np.array(history[i])[:, :3] for i in drone_ids])
    all_wps = np.vstack([np.array(wps) for wps in waypoints.values()])
    all_pts = np.vstack([all_pos, all_wps])
    margin  = 0.5
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    fig, ax3d, proj_axs = _build_figure(drone_ids, waypoints)

    # Applica bounds
    ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
    for name, (ax, ix, iy, _, _) in proj_axs.items():
        lims = [xlim, ylim, zlim]
        ax.set_xlim(lims[ix]); ax.set_ylim(lims[iy])

    # — Oggetti grafici animati —
    trails_3d = {}
    dots_3d   = {}
    trails_2d = {name: {} for name in proj_axs}
    dots_2d   = {name: {} for name in proj_axs}

    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        trail_3d, = ax3d.plot([], [], [], color=c, lw=1.4, alpha=0.7)
        dot_3d,   = ax3d.plot([], [], [], "o", color=c, ms=7,
                               mec="white", mew=0.8, zorder=8)
        trails_3d[i] = trail_3d
        dots_3d[i]   = dot_3d

        for name, (ax, ix, iy, _, _) in proj_axs.items():
            tr,  = ax.plot([], [], color=c, lw=1.2, alpha=0.7)
            dot, = ax.plot([], [], "o", color=c, ms=5,
                           mec="white", mew=0.6, zorder=8)
            trails_2d[name][i] = tr
            dots_2d[name][i]   = dot

    # Testo info
    info_text = ax3d.text2D(
        0.02, 0.97, "", transform=ax3d.transAxes,
        color=TEXT_COLOR, fontsize=8, va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR,
                  alpha=0.6, edgecolor=GRID_COLOR),
    )

    fig.suptitle(
        f"MPC Droni 3-D  ·  N={N_MPC}, dt={DT_MPC} s  ·  "
        f"|a|≤{A_MAX} m/s²  |v|≤{V_MAX} m/s",
        color=TEXT_COLOR, fontsize=10, fontweight="bold", y=0.99,
    )

    # Legenda
    legend_handles = []
    for i in drone_ids:
        c   = COLORS.get(i, "#aaaaaa")
        n_wp = len(waypoints[i])
        legend_handles.append(
            plt.Line2D([0],[0], color=c, lw=2,
                       label=f"Drone {i}  ({n_wp} wp)")
        )
    legend_handles.append(
        plt.Line2D([0],[0], marker="*", color="w", mfc=TEXT_COLOR,
                   ms=9, label="Waypoint", linestyle="None")
    )
    ax3d.legend(handles=legend_handles, fontsize=7.5,
                loc="lower right", framealpha=0.4,
                facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR)

    # — Frame scheduling —
    step_skip     = max(1, int((1.0 / dt) / fps * speed))
    frame_indices = list(range(0, T, step_skip))
    n_frames      = len(frame_indices)

    # — Init —
    def init():
        for i in drone_ids:
            trails_3d[i].set_data([], [])
            trails_3d[i].set_3d_properties([])
            dots_3d[i].set_data([], [])
            dots_3d[i].set_3d_properties([])
            for name in proj_axs:
                trails_2d[name][i].set_data([], [])
                dots_2d[name][i].set_data([], [])
        info_text.set_text("")
        return (list(trails_3d.values()) + list(dots_3d.values()) +
                [v for d in trails_2d.values() for v in d.values()] +
                [v for d in dots_2d.values() for v in d.values()] +
                [info_text])

    # — Update —
    def update(frame_idx):
        t = frame_indices[frame_idx]
        info_lines = [f"t = {t * dt:5.2f} s   step {t:4d}/{T-1}"]

        for i in drone_ids:
            traj = np.array(history[i])
            t_i  = min(t, len(traj) - 1)
            t_s  = max(0, t_i - TRAIL_LEN)

            trails_3d[i].set_data(traj[t_s:t_i+1, 0], traj[t_s:t_i+1, 1])
            trails_3d[i].set_3d_properties(traj[t_s:t_i+1, 2])
            dots_3d[i].set_data([traj[t_i, 0]], [traj[t_i, 1]])
            dots_3d[i].set_3d_properties([traj[t_i, 2]])

            for name, (ax, ix, iy, _, _) in proj_axs.items():
                trails_2d[name][i].set_data(traj[t_s:t_i+1, ix],
                                             traj[t_s:t_i+1, iy])
                dots_2d[name][i].set_data([traj[t_i, ix]], [traj[t_i, iy]])

            # Waypoint corrente (il più vicino non ancora raggiunto)
            wps   = waypoints[i]
            dists = [np.linalg.norm(traj[t_i, :3] - wp) for wp in wps]
            cur_wp = min(range(len(wps)), key=lambda k: dists[k])
            dist   = dists[cur_wp]
            spd    = np.linalg.norm(traj[t_i, 3:6])
            info_lines.append(
                f"D{i}: wp{cur_wp}/{len(wps)-1}  "
                f"dist={dist:5.2f}m  |v|={spd:4.2f}m/s"
            )

        info_text.set_text("\n".join(info_lines))

        return (list(trails_3d.values()) + list(dots_3d.values()) +
                [v for d in trails_2d.values() for v in d.values()] +
                [v for d in dots_2d.values() for v in d.values()] +
                [info_text])

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=1000 / fps,
    )

    # — Salvataggio —
    if save:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800,
                                  metadata={"title": "MPC Drones"})
            out = save_path + ".mp4"
            anim.save(out, writer=writer, dpi=150,
                      savefig_kwargs={"facecolor": BG_COLOR})
            print(f"Animazione salvata in: {out}")
        except Exception as e_mp4:
            print(f"ffmpeg non disponibile ({e_mp4}), provo con Pillow...")
            try:
                out = save_path + ".gif"
                anim.save(out, writer=PillowWriter(fps=fps), dpi=100,
                          savefig_kwargs={"facecolor": BG_COLOR})
                print(f"Animazione salvata in: {out}")
            except Exception as e_gif:
                print(f"Salvataggio fallito: {e_gif}")

    return anim


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animazione simulazione MPC droni 3-D"
    )
    parser.add_argument("--save",  action="store_true",
                        help="Salva il video/gif su disco")
    parser.add_argument("--fps",   type=int,   default=30)
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Velocità riproduzione (default 2x)")
    parser.add_argument("--out",   type=str,   default="drone_animation")
    args = parser.parse_args()

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
    history, inputs, solve_t, waypoints = simulate(
        starts=starts, targets=targets,
        dt=DT_SIM, n_steps=N_SIM,
    )
    print("Simulazione completata. Avvio animazione...")

    anim = animate(
        starts=starts,
        targets=targets,
        history=history,
        inputs=inputs,
        dt=DT_SIM,
        waypoints=waypoints,
        save=args.save,
        save_path=args.out,
        fps=args.fps,
        speed=args.speed,
    )
    plt.show()
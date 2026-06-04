"""
main.py
=======
Entry point della simulazione di ricerca in valanga multi-agente.

Parametri di esecuzione  → config.py
Modello terreno          → terrain.py
Sorgente ARTVA           → artva.py
Agente drone             → drone_agent.py
Loop simulazione         → simulation.py
Visualizzazione          → visualization.py
Filtro IMDCL             → imdcl.py
Controllore MPC          → mpc_drone.py
DEM TINItaly             → dem_tinitaly.py

Utilizzo
--------
    python main.py --n 3 --steps 600 --animate --save
    python main.py --n 2 --agl 2.0 --seed 7
"""

from __future__ import annotations

import argparse
import numpy as np

import config
import artva   as artva_mod
import simulation as sim_mod
import terrain as terrain_mod

from config import (
    N_DRONES, AGL_HEIGHT, N_SIM, DT_SIM, SIGMA_ACC_SIM,
    ARTVA_MOMENT, VICTIM_XY, VICTIM_DEPTH,
    ANIM_SPEED,
)
from terrain import build_terrain
from artva import ARTVASource
from simulation import build_agents, simulate
from visualization import plot_mission, animate_mission

import matplotlib.pyplot as plt


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulazione ricerca valanga multi-drone MPC + IMDCL"
    )
    parser.add_argument("--n",            type=int,   default=N_DRONES,
                        help="Numero di droni")
    parser.add_argument("--agl",          type=float, default=AGL_HEIGHT,
                        help="Altezza sopra terreno [m]")
    parser.add_argument("--steps",        type=int,   default=N_SIM,
                        help="Passi simulazione")
    parser.add_argument("--animate",      action="store_true",
                        help="Mostra animazione 3-D")
    parser.add_argument("--fps",          type=int,   default=30,
                        help="Frame per secondo animazione (default: 30)")
    parser.add_argument("--save",         action="store_true",
                        help="Salva animazione su disco")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--noise",        type=float, default=None,
                        help="ARTVA noise std (override config, es: 1e-8)")
    parser.add_argument("--rc",           type=float, default=None,
                        help="Raggio comunicazione [m] (override config)")
    parser.add_argument("--area",         type=float, default=None,
                        help="Lato workspace [m] (override config)")
    parser.add_argument("--victim-x",     type=float, default=None,
                        help="Posizione x vittima [m] nel workspace locale")
    parser.add_argument("--victim-y",     type=float, default=None,
                        help="Posizione y vittima [m] nel workspace locale")
    parser.add_argument("--victim-depth", type=float, default=None,
                        help="Profondità sepoltura [m] (override config)")
    parser.add_argument("--ws",           type=str,   default=None,
                        help="Workspace center: 'center' oppure 'r,c' (es: 0.60,0.45)")
    args = parser.parse_args()

    # — Patch parametri sui moduli (prima di qualsiasi import lazy) —
    if args.noise is not None:
        artva_mod.ARTVA_NOISE_STD  = args.noise
        config.ARTVA_NOISE_STD     = args.noise
    if args.rc is not None:
        sim_mod.IMDCL_COMM_RADIUS  = args.rc
        config.IMDCL_COMM_RADIUS   = args.rc
    if args.area is not None:
        terrain_mod.AREA_SIZE_M    = args.area
        config.AREA_SIZE_M         = args.area

    victim_depth = args.victim_depth if args.victim_depth is not None else VICTIM_DEPTH

    if args.ws is None:
        center_frac = (0.60, 0.58)   # default originale main.py
    elif args.ws == "center":
        center_frac = None           # centro DEM (come run_experiments)
    else:
        r_s, c_s = args.ws.split(",")
        center_frac = (float(r_s), float(c_s))

    print("=" * 62)
    print("  ARTVA Search & Rescue — simulazione multi-drone MPC")
    print(f"  Droni: {args.n}   AGL: {args.agl} m   Passi: {args.steps}")
    if args.noise is not None:
        print(f"  noise={args.noise:.0e}  rc={args.rc}m  area={args.area}m")
    print("=" * 62)

    # — Costruzione ambiente —
    print("\nLettura e interpolazione DEM...")
    terrain_obj, x_coords, y_coords, sub_dem, transform = build_terrain(center_frac)
    print(f"  Workspace: x=[{terrain_obj.x_min:.0f}, {terrain_obj.x_max:.0f}]  "
          f"y=[{terrain_obj.y_min:.0f}, {terrain_obj.y_max:.0f}] m  "
          f"(UTM origine: E≈{terrain_obj.utm_origin[0]:.0f}, N≈{terrain_obj.utm_origin[1]:.0f})")

    # — Deployment nell'angolo SW del workspace —
    deploy_xy = np.array([
        terrain_obj.x_min + 5.0,
        terrain_obj.y_min + 5.0,
    ])

    # — Posizione vittima —
    if args.victim_x is not None and args.victim_y is not None:
        victim_x = terrain_obj.x_min + args.victim_x
        victim_y = terrain_obj.y_min + args.victim_y
    elif VICTIM_XY is not None:
        victim_x, victim_y = float(VICTIM_XY[0]), float(VICTIM_XY[1])
    else:
        rng_main = np.random.default_rng(args.seed)
        victim_x = rng_main.uniform(terrain_obj.x_min + 30, terrain_obj.x_max - 30)
        victim_y = rng_main.uniform(terrain_obj.y_min + 30, terrain_obj.y_max - 30)
    victim_z = terrain_obj.z(victim_x, victim_y) - victim_depth

    artva = ARTVASource(
        position=np.array([victim_x, victim_y, victim_z]),
        moment=ARTVA_MOMENT,
        rng_seed=args.seed + 1,
    )
    print(f"\n  Vittima: x={victim_x:.1f}  y={victim_y:.1f}  z={victim_z:.1f} m (locale)")

    # — Costruzione agenti —
    print(f"\nCostruzione {args.n} droni dal deployment ({deploy_xy.round(1)})...")
    agents = build_agents(
        deploy_xy=deploy_xy,
        terrain=terrain_obj,
        n_drones=args.n,
        agl=args.agl,
    )

    # — Simulazione —
    print("\nAvvio simulazione...\n")
    agents, consensus_events, artva_detect_thr, track_stop_thr = simulate(
        terrain=terrain_obj,
        artva=artva,
        agents=agents,
        n_steps=args.steps,
        dt=DT_SIM,
        sigma=SIGMA_ACC_SIM,
        agl=args.agl,
        rng_seed=args.seed,
    )

    # — Plot risultati —
    plot_mission(terrain_obj, artva, agents, x_coords, y_coords, sub_dem, TRACK_STOP_THR=track_stop_thr, ARTVA_DETECT_THR=artva_detect_thr)

    # — Animazione (opzionale) —
    anim = None
    if args.animate:
        anim = animate_mission(
            terrain_obj, artva, agents,
            dt=DT_SIM, speed=ANIM_SPEED,
            fps=args.fps,
            save=args.save,
            consensus_events=consensus_events,
        )

    plt.show()


if __name__ == "__main__":
    main()

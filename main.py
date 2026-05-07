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

from config import (
    N_DRONES, AGL_HEIGHT, N_SIM, DT_SIM, SIGMA_ACC_SIM,
    ARTVA_MOMENT, ARTVA_DETECT_THR, ARTVA_NOISE_STD, VICTIM_XY, VICTIM_DEPTH,
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
    parser.add_argument("--n",       type=int,   default=N_DRONES,
                        help="Numero di droni")
    parser.add_argument("--agl",     type=float, default=AGL_HEIGHT,
                        help="Altezza sopra terreno [m]")
    parser.add_argument("--steps",   type=int,   default=N_SIM,
                        help="Passi simulazione")
    parser.add_argument("--animate", action="store_true",
                        help="Mostra animazione 3-D")
    parser.add_argument("--save",    action="store_true",
                        help="Salva animazione su disco")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    print("=" * 62)
    print("  ARTVA Search & Rescue — simulazione multi-drone MPC")
    print(f"  Droni: {args.n}   AGL: {args.agl} m   Passi: {args.steps}")
    print("=" * 62)

    # — Costruzione ambiente —
    print("\nLettura e interpolazione DEM...")
    terrain_obj, x_coords, y_coords, sub_dem, transform = build_terrain()
    print(f"  Workspace: x=[{terrain_obj.x_min:.0f}, {terrain_obj.x_max:.0f}]  "
          f"y=[{terrain_obj.y_min:.0f}, {terrain_obj.y_max:.0f}] m  "
          f"(UTM origine: E≈{terrain_obj.utm_origin[0]:.0f}, N≈{terrain_obj.utm_origin[1]:.0f})")

    # — Deployment nell'angolo SW del workspace —
    deploy_xy = np.array([
        terrain_obj.x_min + 5.0,
        terrain_obj.y_min + 5.0,
    ])

    # — Posizione vittima —
    if VICTIM_XY is not None:
        victim_x, victim_y = float(VICTIM_XY[0]), float(VICTIM_XY[1])
    else:
        rng_main = np.random.default_rng(args.seed)
        victim_x = rng_main.uniform(terrain_obj.x_min + 30, terrain_obj.x_max - 30)
        victim_y = rng_main.uniform(terrain_obj.y_min + 30, terrain_obj.y_max - 30)
    victim_z = terrain_obj.z(victim_x, victim_y) - VICTIM_DEPTH

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
        rng_seed=args.seed,
    )

    # — Simulazione —
    print("\nAvvio simulazione...\n")
    agents = simulate(
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
    plot_mission(terrain_obj, artva, agents, x_coords, y_coords, sub_dem)

    # — Animazione (opzionale) —
    anim = None
    if args.animate:
        anim = animate_mission(
            terrain_obj, artva, agents,
            dt=DT_SIM, speed=2.0,
            save=args.save,
        )

    plt.show()


if __name__ == "__main__":
    main()

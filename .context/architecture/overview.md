# Architettura del sistema

## Pipeline di esecuzione

```
main.py
  │
  ├── build_terrain()          ← terrain.py
  │     read_geotiff → extract_area → RegularGridInterpolator → Terrain
  │
  ├── ARTVASource(position)    ← artva.py
  │     modello dipolo verticale S(r) = moment · sqrt(1+3cos²θ) / r³
  │
  ├── build_agents()           ← simulation.py
  │     per ogni drone:
  │       lawnmower_waypoints() → DroneAgent(MPC + IMDCL + FSM)
  │
  └── simulate()               ← simulation.py
        loop per step t=0..N_SIM:
          1. misura ARTVA reale
          2. FSM: SEARCH→TRACK se segnale ≥ ARTVA_DETECT_THR
             TRACK stopping: segnale ≥ TRACK_STOP_THR → hovering
             hill-climbing reattivo solo se non fermato
          3. MPC step → u_opt   (usa stima IMDCL, non posizione reale)
          4. propagazione dinamica reale + rumore
          5. IMDCL: propagazione + update LiDAR + update cooperativo
          6. DCGD: Adapt+Combine per stima distribuita posizione sorgente
          7. avanza waypoint se raggiunto
          8. se tutti TRACK fermati → raffinamento DCGD → break
```

## Layers

```
┌─────────────────────────────────────────┐
│  main.py  (CLI, orchestrazione)         │
├─────────────────────────────────────────┤
│  simulation.py  (loop temporale)        │
├────────────────┬────────────────────────┤
│  drone_agent   │  visualization.py      │
│  (FSM+nav)     │  (plot + animazione)   │
├────────┬───────┴────────────────────────┤
│ MPC    │  IMDCL   │  ARTVA  │  Terrain  │
│ (ctrl) │  (stima) │  (sens) │  (env)    │
├────────┴──────────┴─────────┴───────────┤
│  config.py  (parametri globali)         │
└─────────────────────────────────────────┘
```

## Flusso dati per drone

```
x_real (6,) ─────────────────────────────► history[]
     │                                          │
     │  rumore processo                         │
     ▼                                          │
PointMass3D.f(x, u, dt)                        │
     │                                          │
     └──► IMDCL.propagate(u, dt)               │
               │                               │
     LiDAR ───► IMDCL.apply_absolute_update    │
     rel.pos. ► IMDCL.apply_update             │
               │                               │
               ▼                               │
          x_hat (6,) ──► DroneMPC.step(x_hat, wp) ──► u_opt
               │
               └──► est_history[]
```

## Stato del drone (6,)

`[px, py, pz, vx, vy, vz]`  — posizione + velocità in coordinate locali workspace (m, m/s)

## Controllo MPC

- Modello interno: punto-massa 3-D (doppio integratore per asse)
- Variabile di controllo: accelerazione `u ∈ ℝ³`  con `|u| ≤ A_MAX`
- Vincolo velocità: `|v| ≤ V_MAX`
- Orizzonte: `N_MPC` passi da `DT_MPC` s
- Solver: CasADi IPOPT (NLP)
- Target: posizione waypoint corrente (no tracking velocità)
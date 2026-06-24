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
        while step < N_SIM (o finché l'orbita finale non è completa):
          1. misura ARTVA reale
          2. Transizioni FSM (sospese durante FINAL_ORBIT):
               SEARCH → TRACK se segnale ≥ ARTVA_DETECT_THR
               TRACK/SUPPORT → STOP se segnale ≥ TRACK_STOP_THR
               STOP (primo): consenso → 2 droni SUPPORT con circonferenza
          3. MPC step → u_opt   (usa stima IMDCL, non posizione reale)
          4. propagazione dinamica reale + rumore
          5. IMDCL: propagazione + update LiDAR + update cooperativo
          6. Particle Filter: update pesi (proprie + vicini, su x_est) + resample
          7. avanza waypoint (arrival-gated, dispatch per stato FSM)
          8. Terminazione: N_STOP droni in STOP, o chiamata SUPPORT scaduta
             (timeout sempre atteso), o timeout globale → avvia FINAL_ORBIT
             (orbita di raffinamento dei droni con PF attivo); break quando
             tutti raggiungono l'ultimo waypoint del cerchio
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
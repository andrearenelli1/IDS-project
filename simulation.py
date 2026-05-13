"""
simulation.py
=============
Loop principale di simulazione multi-agente.

Funzioni pubbliche
------------------
  build_agents              — costruisce N DroneAgent con MPC + IMDCL + warm-start
  triangulate_victim        — stima posizione vittima dalla media delle source_est DCGD
  simulate                  — loop temporale: MPC + dinamica + IMDCL + hill-climbing +
                              stopping a soglia ARTVA + DCGD distribuito

Algoritmo stima sorgente
------------------------
  DCGD — Distributed Consensus Gradient Descent.
  Ogni drone in TRACK mantiene una stima locale source_est della posizione
  della sorgente. Ad ogni passo:
    1. Adapt  : passo di discesa del gradiente normalizzato sul batch locale
    2. Combine: media pesata con le stime dei vicini TRACK (consensus)
  Quando tutti i droni TRACK si fermano (segnale ≥ TRACK_STOP_THR) si eseguono
  DIST_EST_REFINE iterazioni extra di raffinamento, poi si termina.
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List

import numpy as np

from imdcl import (
    AgentIMDCL,
    PointMass3DModel as IMDCLPointMass3DModel,
    relative_position_measurement_3d,
)
from mpc_drone import DroneMPC, PointMass3DModel

from artva import ARTVASource
from terrain import Terrain
from drone_agent import DroneAgent, DroneState, lawnmower_waypoints, track_next_waypoint
from config import (
    N_DRONES, DEPLOY_OFFSET,
    AGL_HEIGHT, LIDAR_SIGMA,
    DT_MPC, N_MPC, A_MAX, V_MAX,
    N_SIM, DT_SIM, SIGMA_ACC_SIM, STOP_THRESH,
    IMDCL_SIGMA_ACC, IMDCL_P0_POS, IMDCL_P0_VEL,
    IMDCL_COMM_RADIUS, IMDCL_R_MEAS_STD,
    IMDCL_R_LIDAR_STD, IMDCL_H_LIDAR,
    TRACK_STEP_M, TRACK_TURN_DEG,
    TRACK_STOP_THR, SUPPORT_STEP_M, N_SIGNAL_SAMPLES,
    DIST_EST_ALPHA, DIST_EST_BETA, DIST_EST_H, DIST_EST_REFINE, DIST_EST_BATCH,
    TRIANGULATE_N_PARTNERS,
    ARTVA_DETECT_THR, ARTVA_MOMENT,
)


# ============================================================================
# Helper ARTVA model (senza rumore) — usato da DCGD
# ============================================================================

def _artva_model(pos: np.ndarray, theta: np.ndarray, moment: float) -> float:
    """Segnale ARTVA (senza rumore) per sorgente in theta, sensore in pos."""
    delta = np.asarray(pos, dtype=float) - np.asarray(theta, dtype=float)
    r = max(np.linalg.norm(delta), 1e-2)
    cos_th = delta[2] / r
    return moment * np.sqrt(1.0 + 3.0 * cos_th**2) / r**3


def _grad_S(pos: np.ndarray, theta: np.ndarray, moment: float) -> np.ndarray:
    """Gradiente numerico del segnale ARTVA rispetto alla posizione sorgente theta."""
    grad = np.zeros(3)
    for k in range(3):
        dp = np.zeros(3); dp[k] = DIST_EST_H
        grad[k] = (
            _artva_model(pos, theta + dp, moment)
            - _artva_model(pos, theta - dp, moment)
        ) / (2.0 * DIST_EST_H)
    return grad


# ============================================================================
# DCGD — un passo di aggiornamento online
# ============================================================================

def _dcgd_step(
    agents:    Dict[int, DroneAgent],
    drone_ids: list,
) -> None:
    """
    Un passo DCGD (Adapt + Combine) per tutti i droni TRACK con source_est.

    Adapt  : theta_i -= alpha * grad_J_i / ||grad_J_i||  (batch locale)
    Combine: theta_i  = (1-beta)*theta_i + beta * mean(theta_j, j∈N_i TRACK)
    """
    track_ids = [
        i for i in drone_ids
        if agents[i].state == DroneState.TRACK and agents[i].source_est is not None
    ]
    if not track_ids:
        return

    # Snapshot per il consensus (evita che l'ordine di aggiornamento influenzi il risultato)
    snap = {i: agents[i].source_est.copy() for i in track_ids}

    for i in track_ids:
        ag      = agents[i]
        theta_i = snap[i].copy()

        # ── Adapt: gradiente sul batch più recente ─────────────────────────
        batch   = ag.signal_log[-DIST_EST_BATCH:]
        grad_J  = np.zeros(3)
        for pos, s_meas in batch:
            s_pred  = _artva_model(pos, theta_i, ARTVA_MOMENT)
            grad_s  = _grad_S(pos, theta_i, ARTVA_MOMENT)
            grad_J += -2.0 * (s_meas - s_pred) * grad_s
        if batch:
            grad_J /= len(batch)
        norm_g = np.linalg.norm(grad_J)
        if norm_g > 1e-12:
            theta_i -= DIST_EST_ALPHA * grad_J / norm_g

        # ── Combine: consensus con vicini TRACK ────────────────────────────
        neighbors = [
            j for j in track_ids
            if j != i
            and np.linalg.norm(agents[i].x[:3] - agents[j].x[:3]) < IMDCL_COMM_RADIUS
        ]
        if neighbors:
            avg_nbr  = np.mean([snap[j] for j in neighbors], axis=0)
            theta_i  = (1.0 - DIST_EST_BETA) * theta_i + DIST_EST_BETA * avg_nbr

        ag.source_est = theta_i


def _dcgd_refine(
    agents:    Dict[int, DroneAgent],
    drone_ids: list,
    n_iters:   int = DIST_EST_REFINE,
) -> None:
    """
    Fase di raffinamento DCGD post-blocco: itera n_iters volte usando
    le ultime DIST_EST_BATCH misure di ogni drone (posizione fissa).
    """
    for _ in range(n_iters):
        _dcgd_step(agents, drone_ids)


def _support_waypoint_from_direction(
    source_pos: np.ndarray,
    direction_2d: np.ndarray | None,
    terrain: Terrain,
    step_m: float = TRACK_STEP_M,
    agl: float = AGL_HEIGHT,
) -> np.ndarray:
    """Genera un waypoint di supporto lungo la direzione finale del drone."""
    direction = np.asarray(direction_2d if direction_2d is not None else [1.0, 0.0], dtype=float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])

    target_xy = np.asarray(source_pos, dtype=float)[:2] + step_m * direction
    target_z  = terrain.agl_z(target_xy[0], target_xy[1], agl)
    return np.array([target_xy[0], target_xy[1], target_z])


# ============================================================================
# Costruzione agenti
# ============================================================================

def build_agents(
    deploy_xy: np.ndarray,
    terrain:   Terrain,
    n_drones:  int   = N_DRONES,
    agl:       float = AGL_HEIGHT,
    rng_seed:  int   = 42,
) -> Dict[int, DroneAgent]:
    """
    Crea N droni dal punto di deployment condiviso, ognuno con:
      - posizione iniziale lateralmente distanziata
      - waypoint lawnmower precompilati (in coordinate locali workspace)
      - controllore MPC con warm-start
      - filtro IMDCL inizializzato

    Returns
    -------
    agents : {id: DroneAgent}
    """
    agents: Dict[int, DroneAgent] = {}

    x_min = terrain.x_min;  x_max = terrain.x_max
    y_min = terrain.y_min;  y_max = terrain.y_max

    team_ids     = list(range(n_drones))
    imdcl_motion = IMDCLPointMass3DModel(sigma_acc=IMDCL_SIGMA_ACC)
    P0_imdcl     = np.diag([IMDCL_P0_POS**2] * 3 + [IMDCL_P0_VEL**2] * 3)

    for i in range(n_drones):
        offset = (i - (n_drones - 1) / 2) * DEPLOY_OFFSET
        px0    = deploy_xy[0]
        py0    = deploy_xy[1] + offset
        pz0    = terrain.agl_z(px0, py0, agl)
        x0     = np.array([px0, py0, pz0, 0.0, 0.0, 0.0])

        wps  = lawnmower_waypoints(
            i, n_drones, x_min, x_max, y_min, y_max, terrain, agl=agl
        )
        ctrl = DroneMPC(dt=DT_MPC, N=N_MPC, ax_max=A_MAX, ay_max=A_MAX, az_max=A_MAX, vx_max=V_MAX, vy_max=V_MAX, vz_max=V_MAX)
        imdcl_agent = AgentIMDCL(
            agent_id=i,
            x0=x0.copy(),
            P0=P0_imdcl.copy(),
            team_ids=team_ids,
            motion_model=imdcl_motion,
        )

        agent = DroneAgent(
            id=i, x=x0.copy(), waypoints=wps,
            ctrl=ctrl, imdcl=imdcl_agent,
            state=DroneState.SEARCH,
        )
        agent.history.append(x0.copy())
        agent.est_history.append(imdcl_agent.x_hat.copy())
        agents[i] = agent

    print("Warm-start MPC droni...")
    for i, ag in agents.items():
        ag.ctrl.warm_start(ag.x_est, ag.current_target())
        print(f"  Drone {i}: wp[0]={ag.current_target().round(1)}")

    return agents


# ============================================================================
# Stima finale posizione vittima
# ============================================================================

def triangulate_victim(
    agents: Dict[int, DroneAgent],
    artva:  ARTVASource,
) -> np.ndarray:
    """
    Stima posizione vittima: media delle stime source_est DCGD dei droni in TRACK.
    Fallback su artva.position se nessun drone ha una stima inizializzata.
    """
    ests = [
        ag.source_est for ag in agents.values()
        if ag.state == DroneState.TRACK and ag.source_est is not None
    ]
    if not ests:
        return artva.position.copy()
    return np.mean(ests, axis=0)


# ============================================================================
# Loop di simulazione
# ============================================================================

def simulate(
    terrain:  Terrain,
    artva:    ARTVASource,
    agents:   Dict[int, DroneAgent],
    n_steps:  int   = N_SIM,
    dt:       float = DT_SIM,
    sigma:    float = SIGMA_ACC_SIM,
    agl:      float = AGL_HEIGHT,
    rng_seed: int   = 42,
) -> Dict[int, DroneAgent]:
    """
    Esegue la simulazione multi-agente.

    Per ogni passo temporale:
      1.  Misura ARTVA reale nella posizione corrente
      2.  FSM: transizioni SEARCH→TRACK + init source_est DCGD
          Stopping TRACK: se segnale ≥ TRACK_STOP_THR → drone hovering
      3.  Hill-climbing reattivo (solo droni TRACK non fermati)
      4.  MPC step → u_opt  (usa stima IMDCL, non posizione reale)
      5.  Propagazione dinamica reale con rumore
      6.  Aggiornamento filtro IMDCL:
            a) propagazione locale
            b) update assoluto LiDAR (quota pz)
            c) update relativo cooperativo inter-drone
      7.  DCGD: un passo Adapt+Combine per tutti i droni TRACK
      8.  Avanza waypoint se raggiunto (controlla stima IMDCL)
      9.  Stop: tutti i droni TRACK fermati → raffinamento DCGD → break
    """
    rng       = np.random.default_rng(rng_seed)
    model     = PointMass3DModel(sigma_acc=sigma)
    drone_ids = list(agents.keys())

    R_rel   = np.eye(3) * IMDCL_R_MEAS_STD**2
    R_lidar = np.array([[IMDCL_R_LIDAR_STD**2]])

    # Conserva i waypoint lawnmower originali per init_track_dir
    lawnmower_orig = {i: list(agents[i].waypoints) for i in drone_ids}

    # Centro workspace — punto di inizializzazione source_est per DCGD
    cx_ws = (terrain.x_min + terrain.x_max) / 2.0
    cy_ws = (terrain.y_min + terrain.y_max) / 2.0

    print(
        f"\n{'Step':>5}  {'t[s]':>6}  "
        + "  ".join(f"D{i}:st/wp/dist/Δest" for i in drone_ids)
    )

    for step in range(n_steps):
        t = step * dt

        # ── 1. Misura ARTVA ──────────────────────────────────────────────
        for i in drone_ids:
            ag       = agents[i]
            prev_pos = ag.history[-1][:3] if ag.history else ag.x[:3]
            alphas   = np.linspace(0.0, 1.0, N_SIGNAL_SAMPLES)
            sig      = float(np.mean([
                artva.signal(prev_pos * (1.0 - a) + ag.x[:3] * a, noisy=True)
                for a in alphas
            ]))
            ag.signal_log.append((ag.x[:3].copy(), sig))

            # Transizione SEARCH → TRACK
            if ag.state == DroneState.SEARCH and sig >= ARTVA_DETECT_THR:
                ag.state    = DroneState.TRACK
                ag.detected = True
                ag.init_track_dir(lawnmower_orig[i])
                ag.track_signal_prev = sig
                # Inizializza DCGD al centro del workspace
                ag.source_est = np.array([cx_ws, cy_ws, terrain.z(cx_ws, cy_ws)])
                print(
                    f"\n  ★ Drone {i} RILEVATO (S={sig:.2e}) "
                    f"al passo {step} (t={t:.2f}s)"
                )
                print(f"    Posizione reale   : {ag.x[:3].round(2)}")
                print(f"    Posizione stimata : {ag.x_est[:3].round(2)}")
                print(f"    Direzione iniziale: {ag.track_dir.round(3)}")

            # ── Stopping: segnale ≥ TRACK_STOP_THR → hovering ───────────
            if ag.state == DroneState.TRACK and not ag.track_stopped and sig >= TRACK_STOP_THR:
                ag.track_stopped = True
                hover_wp = ag.x_est[:3].copy()
                ag.waypoints = [hover_wp]
                ag.wp_idx    = 0
                print(
                    f"\n  ⬛ Drone {i} FERMATO (S={sig:.2e}, ≥{TRACK_STOP_THR:.0e}) "
                    f"al passo {step} (t={t:.2f}s)  pos={ag.x[:3].round(1)}"
                )

                # Chiama droni vicini in supporto sul prolungamento dell'ultima direzione
                dists = {
                    j: np.linalg.norm(agents[j].x[:3] - ag.x[:3])
                    for j in drone_ids if j != i
                }
                partners = sorted(dists, key=dists.get)[:TRIANGULATE_N_PARTNERS]
                support_wp = _support_waypoint_from_direction(
                    ag.x_est[:3], ag.track_dir, terrain, step_m=SUPPORT_STEP_M, agl=agl
                )
                for j in partners:
                    if agents[j].state == DroneState.SEARCH:
                        # Partner rimane in SEARCH: ha waypoint verso il punto lungo la direzione finale
                        # del drone che si è appena fermato.
                        # Passerà a TRACK quando rileverà il segnale ARTVA ≥ TRACK_STOP_THR
                        agents[j].waypoints = [support_wp.copy()]
                        agents[j].wp_idx = 0
                        print(
                            f"    → Drone {j} in supporto verso punto {support_wp[:2].round(1)} "
                            f"lungo la direzione finale di drone {i} (dist={dists[j]:.1f} m)"
                        )

            # ── Hill-climbing reattivo (solo droni TRACK non fermati) ────
            if ag.state == DroneState.TRACK and not ag.track_stopped and ag.track_dir is not None:
                ag.track_time += 1
                ag.update_track_dir(sig)
                wp = track_next_waypoint(
                    ag.x_est, ag.track_dir, terrain,
                    step_m=TRACK_STEP_M, agl=agl,
                )
                ag.waypoints = [wp]
                ag.wp_idx    = 0

        # ── 2. MPC step (usa stima IMDCL) ────────────────────────────────
        u_commands: Dict[int, np.ndarray] = {}
        for i in drone_ids:
            ag     = agents[i]
            target = ag.current_target().copy()
            target[2] = terrain.agl_z(ag.x_est[0], ag.x_est[1], agl)
            t0    = perf_counter()
            u_opt = ag.ctrl.step(ag.x_est, target)
            ag.solve_t_log.append(perf_counter() - t0)
            u_commands[i] = u_opt

        # ── 3. Propagazione dinamica reale con rumore ─────────────────────
        for i in drone_ids:
            ag    = agents[i]
            noise = rng.multivariate_normal(np.zeros(3), np.diag([sigma**2] * 3))
            ag.x  = model.f(ag.x, u_commands[i] + noise, dt)

            z_floor = terrain.agl_z(ag.x[0], ag.x[1], agl * 0.5)
            if ag.x[2] < z_floor:
                ag.x[2] = z_floor
                ag.x[5] = max(0.0, ag.x[5])

            ag.history.append(ag.x.copy())
            ag.input_log.append(u_commands[i].copy())

        # ── 4a. IMDCL — propagazione locale ──────────────────────────────
        for i in drone_ids:
            agents[i].imdcl.propagate(u_commands[i], dt)

        # ── 4b. IMDCL — update assoluto LiDAR (quota pz) ─────────────────
        for i in drone_ids:
            ag          = agents[i]
            z_terr_real = terrain.z(ag.x[0], ag.x[1])
            agl_lidar   = (ag.x[2] - z_terr_real) + rng.normal(0, IMDCL_R_LIDAR_STD)
            z_terr_est  = terrain.z(ag.imdcl.x_hat[0], ag.imdcl.x_hat[1])
            z_obs       = np.array([z_terr_est + agl_lidar])
            ag.imdcl.apply_absolute_update(z_obs, IMDCL_H_LIDAR, R_lidar)

        # ── 4c. IMDCL — update relativo cooperativo ───────────────────────
        processed: set = set()
        for i in drone_ids:
            ag_i = agents[i]
            best_j, best_d = None, float("inf")
            for j in drone_ids:
                if j == i:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in processed:
                    continue
                d = np.linalg.norm(ag_i.x[:3] - agents[j].x[:3])
                if d < IMDCL_COMM_RADIUS and d < best_d:
                    best_j, best_d = j, d
            if best_j is None:
                continue
            processed.add((min(i, best_j), max(i, best_j)))

            ag_j     = agents[best_j]
            z_true, _, _ = relative_position_measurement_3d(ag_i.x, ag_j.x)
            z_noisy  = z_true + rng.multivariate_normal(np.zeros(3), R_rel)
            lm_msg   = ag_j.imdcl.make_landmark_message()
            upd_msg  = ag_i.imdcl.compute_update_message(
                lm_msg, z_noisy, R_rel,
                measurement_fn=relative_position_measurement_3d,
            )
            for k in drone_ids:
                agents[k].imdcl.apply_update(upd_msg)

        for i in drone_ids:
            agents[i].imdcl.step_no_measurement()

        for i in drone_ids:
            agents[i].est_history.append(agents[i].imdcl.x_hat.copy())

        # ── 5. DCGD — un passo online Adapt+Combine ──────────────────────
        _dcgd_step(agents, drone_ids)

        # ── 6. Avanza waypoint ────────────────────────────────────────────
        for i in drone_ids:
            ag = agents[i]
            if np.linalg.norm(ag.x_est[:3] - ag.current_target()) < STOP_THRESH:
                ag.advance_waypoint()

        # ── 7. Log periodico ──────────────────────────────────────────────
        if (step + 1) % 20 == 0:
            row = f"{step+1:>5}  {(step+1)*dt:>5.1f}s  "
            for i in drone_ids:
                ag      = agents[i]
                dist    = np.linalg.norm(ag.x_est[:3] - ag.current_target())
                est_err = np.linalg.norm(ag.x[:3] - ag.x_est[:3])
                st      = "SRCH" if ag.state == DroneState.SEARCH else (
                    "STOP" if ag.track_stopped else "TRCK"
                )
                row    += f"  {st}/{ag.wp_idx:02d}/{dist:5.2f}m/Δ{est_err:.2f}m/"
            print(row)

        # ── 8. Stop: almeno 3 droni in stop mode ─────────────────────────
        stopped_agents = [ag for ag in agents.values() if ag.track_stopped]
        if len(stopped_agents) >= 3:
            print(
                f"\n  ✔ {len(stopped_agents)} droni in stop mode al passo {step+1} "
                f"(t={(step+1)*dt:.2f}s) — raffinamento DCGD ({DIST_EST_REFINE} iter)..."
            )
            _dcgd_refine(agents, drone_ids, DIST_EST_REFINE)
            break

    # ── Report finale ─────────────────────────────────────────────────────
    est = triangulate_victim(agents, artva)
    err = np.linalg.norm(est[:2] - artva.position[:2])
    print(f"\n  Posizione vittima reale  : {artva.position.round(2)}")
    print(f"  Stima DCGD distribuita   : {est.round(2)}")
    print(f"  Errore planimetrico      : {err:.2f} m")
    print("\n  Stime source_est per drone:")
    valid_ests = [agents[i].source_est for i in drone_ids if agents[i].source_est is not None]
    for i in drone_ids:
        ag = agents[i]
        if ag.source_est is not None:
            e_i = np.linalg.norm(ag.source_est[:2] - artva.position[:2])
            print(f"    Drone {i}: {ag.source_est.round(2)}  (err={e_i:.2f} m)")
    if len(valid_ests) >= 2:
        ests_xy = np.array([e[:2] for e in valid_ests])
        var_xy  = np.var(ests_xy, axis=0)
        print(f"    Varianza stime [σ²x={var_xy[0]:.3f}, σ²y={var_xy[1]:.3f}]  "
              f"(σ_planimetrica={np.sqrt(var_xy.sum()):.3f} m)")
    print("\n  Errore stima IMDCL finale:")
    for i in drone_ids:
        ag = agents[i]
        print(f"    Drone {i}: |x_real - x_est| = "
              f"{np.linalg.norm(ag.x[:3] - ag.x_est[:3]):.3f} m")

    return agents

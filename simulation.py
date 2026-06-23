"""
simulation.py
=============
Loop principale di simulazione multi-agente.

Funzioni pubbliche
------------------
  build_agents — costruisce N DroneAgent con MPC + IMDCL + warm-start
  simulate     — loop temporale multi-agente

FSM a 4 stati
-------------
  SEARCH  → lawnmower arrival-gated
  TRACK   → ES (Extremum Seeking) verso la sorgente; → STOP a soglia
  STOP    → hovering sulla posizione raggiunta
  SUPPORT → naviga verso posizione di supporto cooperativo

Algoritmo stima sorgente: Particle Filter distribuito.
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
from drone_agent import (
    DroneAgent, DroneState,
    lawnmower_waypoints, single_lane_waypoints, circle_waypoints,
)
from pf import ParticleFilter
from config import (
    N_DRONES, DEPLOY_OFFSET,
    AGL_HEIGHT, LIDAR_SIGMA,
    DT_MPC, N_MPC, A_MAX, V_MAX,
    N_SIM, DT_SIM, SIGMA_ACC_SIM, STOP_THRESH,
    IMDCL_SIGMA_ACC, IMDCL_P0_POS, IMDCL_P0_VEL,
    IMDCL_COMM_RADIUS, IMDCL_R_MEAS_STD,
    IMDCL_R_LIDAR_STD, IMDCL_H_LIDAR,
    N_SIGNAL_SAMPLES,
    N_NOISE_CALIB_SAMPLES, NOISE_CONSENSUS_ITERS,
    ES_DETECT_MAX_R, FOUND_RADIUS,
    ES_ALPHA_MAX, ES_OMEGA, ES_KAPPA, ES_LAMBDA, ES_EPS,
    ARTVA_MOMENT,
    PF_N_PARTICLES,
    SUPPORT_CIRCLE_N, TRIANGULATE_N_PARTNERS,
    SUPPORT_SEARCH_TIMEOUT, CONSENSUS_K_MAX,
    N_STOP,
)


# ============================================================================
# Average consensus distribuito — helper generico
# ============================================================================

def _average_consensus(
    agents:      Dict[int, DroneAgent],
    drone_ids:   list,
    values:      dict,
    comm_radius: float = IMDCL_COMM_RADIUS,
    iters:       int   = 1,
    beta:        float = None,
) -> dict:
    """
    Average consensus distribuito su grandezze scalari o vettoriali.

    beta=None  → ogni nodo fa media uguale con se stesso e i vicini.
    beta=float → (1-beta)*self + beta*mean(neighbors); invariante se no vicini.
    """
    v = {i: np.asarray(values[i], dtype=float) for i in drone_ids}
    for _ in range(iters):
        v_new: dict = {}
        for i in drone_ids:
            nbrs = [
                j for j in drone_ids
                if j != i
                and np.linalg.norm(agents[i].x[:3] - agents[j].x[:3]) < comm_radius
            ]
            if beta is None:
                v_new[i] = np.mean([v[i]] + [v[j] for j in nbrs], axis=0)
            elif nbrs:
                avg_nbr  = np.mean([v[j] for j in nbrs], axis=0)
                v_new[i] = (1.0 - beta) * v[i] + beta * avg_nbr
            else:
                v_new[i] = v[i].copy()
        v = v_new
    return v


# ============================================================================
# Min-consensus SUPPORT — selezione partner cooperativi
# ============================================================================

def _reachable_from(
    drone_ids:   list,
    agents:      Dict[int, DroneAgent],
    start_id:    int,
    comm_radius: float = IMDCL_COMM_RADIUS,
) -> list:
    """BFS sulla rete di comunicazione: ritorna la componente connessa di start_id."""
    pos       = {did: agents[did].x[:3] for did in drone_ids}
    reachable = {start_id}
    frontier  = {start_id}
    while frontier:
        new_frontier: set = set()
        for fid in frontier:
            for did in drone_ids:
                if did not in reachable and np.linalg.norm(pos[fid] - pos[did]) <= comm_radius:
                    reachable.add(did)
                    new_frontier.add(did)
        frontier = new_frontier
    return [did for did in drone_ids if did in reachable]


def _consensus_select_partners(
    agents:      Dict[int, DroneAgent],
    drone_ids:   list,
    stop_id:     int,
    n_partners:  int   = TRIANGULATE_N_PARTNERS,
    k_max:       int   = CONSENSUS_K_MAX,
    comm_radius: float = IMDCL_COMM_RADIUS,
) -> List[int]:
    """
    Min-consensus distribuito: seleziona i n_partners droni disponibili
    più vicini al drone STOP (stop_id) raggiungibili via rete.

    Usa posizioni stimate (x_est) per le distanze; iterazioni k_max ≥ diametro grafo.
    """
    reachable  = set(_reachable_from(drone_ids, agents, stop_id, comm_radius))
    candidates = [
        did for did in drone_ids
        if did != stop_id
        and did in reachable
        and agents[did].state not in (DroneState.STOP, DroneState.SUPPORT)
    ]
    if not candidates:
        return []

    pos  = {did: agents[did].x_est[:3] for did in drone_ids}
    dist = {did: float("inf") for did in drone_ids}
    dist[stop_id] = 0.0

    for _ in range(k_max):
        dist_new = dist.copy()
        for did in drone_ids:
            for jid in drone_ids:
                if did == jid:
                    continue
                if np.linalg.norm(pos[did] - pos[jid]) <= comm_radius:
                    d_via = dist[jid] + np.linalg.norm(pos[jid] - pos[did])
                    if d_via < dist_new[did]:
                        dist_new[did] = d_via
        dist = dist_new

    candidates.sort(key=lambda did: dist[did])
    return candidates[:n_partners]


def _transition_to_support(
    ag:          DroneAgent,
    drone_id:    int,
    stop_id:     int,
    wps:         list,
    center:      np.ndarray,
    orbit_r:     float,
    cw:          bool,
    sigma_noise: float,
    step:        int,
) -> None:
    """SEARCH/TRACK → SUPPORT: assegna orbita cooperativa e inizializza il PF se mancante."""
    ag.state                = DroneState.SUPPORT
    ag.waypoints            = wps
    ag.wp_idx               = 0
    ag.support_center       = center.copy()
    ag.support_orbit_radius = orbit_r
    ag.support_cw           = cw

    if ag.pf is None:
        ag.artva_sigma_noise = sigma_noise
        sig0 = ag.signal_log[-1][1] if ag.signal_log else 1e-9
        ag.pf = ParticleFilter(PF_N_PARTICLES, state_dim=3, measurement_dim=1)
        ag.pf.initialize_particles(ag.x[:3], sig0)

    print(
        f"    SUPPORT: Drone {drone_id} → orbita Drone {stop_id}"
        f"  r={orbit_r:.1f} m  {'CW' if cw else 'CCW'}  (step {step})"
    )


def _assign_support_partners(
    agents:         Dict[int, DroneAgent],
    drone_ids:      list,
    stop_id:        int,
    sig:            float,
    sigma_noise:    float,
    step:           int,
    terrain:        Terrain,
    agl:            float,
    consensus_done: list,      # [bool] flag mutable
) -> None:
    """
    Lancia il min-consensus e assegna l'orbita cooperativa ai droni selezionati.
    Imposta support_pending=True sul drone STOP se mancano partner.
    """
    consensus_done[0] = True
    partners = _consensus_select_partners(agents, drone_ids, stop_id)
    stop_ag  = agents[stop_id]
    orbit_r  = (ARTVA_MOMENT / sig) ** (1.0 / 3.0)
    center   = stop_ag.x_est[:3].copy()
    stop_ag.support_orbit_radius = orbit_r

    n_assigned = 0
    for k, pid in enumerate(partners):
        cw          = (k % 2 == 0)
        start_angle = float(np.pi * k)
        wps = circle_waypoints(center, orbit_r, start_angle, cw, terrain, agl=agl)
        _transition_to_support(agents[pid], pid, stop_id, wps, center, orbit_r, cw,
                                sigma_noise, step)
        n_assigned += 1

    n_missing = TRIANGULATE_N_PARTNERS - n_assigned
    if n_missing > 0:
        stop_ag.support_pending  = True
        stop_ag.support_deadline = step + SUPPORT_SEARCH_TIMEOUT
        stop_ag.support_n_needed = n_missing
        print(
            f"    SUPPORT: {n_assigned}/{TRIANGULATE_N_PARTNERS} assegnati; "
            f"cerco ancora {n_missing} (deadline step {stop_ag.support_deadline})"
        )
    else:
        stop_ag.support_pending = False


def _retry_support_search(
    agents:      Dict[int, DroneAgent],
    drone_ids:   list,
    stop_id:     int,
    sigma_noise: float,
    step:        int,
    terrain:     Terrain,
    agl:         float,
) -> None:
    """Riprova ogni step ad assegnare i partner SUPPORT mancanti per un drone STOP."""
    stop_ag = agents[stop_id]
    if not stop_ag.support_pending:
        return
    if step >= stop_ag.support_deadline:
        stop_ag.support_pending = False
        print(f"    SUPPORT timeout: Drone {stop_id} rinuncia (step {step})")
        return

    partners = _consensus_select_partners(
        agents, drone_ids, stop_id, n_partners=stop_ag.support_n_needed,
    )
    if not partners:
        return

    center  = stop_ag.waypoints[0][:3].copy()
    orbit_r = stop_ag.support_orbit_radius if stop_ag.support_orbit_radius > 0 else 5.0

    existing = sum(
        1 for did in drone_ids
        if agents[did].state == DroneState.SUPPORT
        and agents[did].support_center is not None
        and np.linalg.norm(agents[did].support_center[:2] - center[:2]) < 2.0
    )

    for k, pid in enumerate(partners):
        cw          = ((existing + k) % 2 == 0)
        start_angle = float(np.pi * (existing + k))
        wps = circle_waypoints(center, orbit_r, start_angle, cw, terrain, agl=agl)
        _transition_to_support(agents[pid], pid, stop_id, wps, center, orbit_r, cw,
                                sigma_noise, step)
        stop_ag.support_n_needed -= 1

    if stop_ag.support_n_needed <= 0:
        stop_ag.support_pending = False


# ============================================================================
# Extremum Seeking — TRACK mode  [Azzollini et al. 2021]
# ============================================================================

def _es_condition_signal(sig: float) -> float:
    """yt = 1 / ∛S — convesso con minimo in sorgente."""
    return 1.0 / np.cbrt(max(sig, ES_EPS))


def _es_update(
    ag:      "DroneAgent",
    yt:      float,
    dt:      float,
    terrain: Terrain,
    agl:     float,
) -> None:
    """Un passo ES (Azzollini et al. 2021, eq. 11-13), Euler in avanti."""
    ag.es_alpha += dt * (ES_ALPHA_MAX - ag.es_alpha) / ES_LAMBDA
    speed = np.sqrt(ag.es_alpha * ES_OMEGA)
    phase = ES_OMEGA * ag.es_time + ES_KAPPA * yt
    ag.es_x_ref += dt * speed * np.cos(phase)
    ag.es_y_ref += dt * speed * np.sin(phase)
    ag.es_time  += dt
    x = float(np.clip(ag.es_x_ref, terrain.x_min, terrain.x_max))
    y = float(np.clip(ag.es_y_ref, terrain.y_min, terrain.y_max))
    z = terrain.agl_z(x, y, agl)
    ag.waypoints = [np.array([x, y, z])]
    ag.wp_idx    = 0


# ============================================================================
# Calibrazione rumore pre-volo
# ============================================================================

def _calibrate_noise(
    agents:          Dict[int, DroneAgent],
    artva:           ARTVASource,
    n_samples:       int = N_NOISE_CALIB_SAMPLES,
    consensus_iters: int = NOISE_CONSENSUS_ITERS,
) -> tuple[float, float]:
    """
    Ogni drone stima σ_noise da misure ripetute alla propria posizione iniziale.
    Average-consensus distribuito converge a σ̂ comune.
    """
    drone_ids = list(agents.keys())

    mu_local:    Dict[int, float] = {}
    sigma_local: Dict[int, float] = {}
    for i in drone_ids:
        pos     = agents[i].x[:3]
        samples = [artva.signal(pos, noisy=True) for _ in range(n_samples)]
        mu_local[i]    = float(np.mean(samples))
        sigma_local[i] = float(np.std(samples))

    print("  [Calibrazione] σ_noise locale per drone:")
    for i in drone_ids:
        print(f"    Drone {i}: σ̂={sigma_local[i]:.2e}")

    sigma_agreed = _average_consensus(agents, drone_ids, sigma_local, iters=consensus_iters)
    mu_agreed    = _average_consensus(agents, drone_ids, mu_local,    iters=consensus_iters)
    mu_agreed    = float(np.mean([mu_agreed[i]    for i in drone_ids]))
    sigma_agreed = float(np.mean([sigma_agreed[i] for i in drone_ids]))
    print(f"  [Calibrazione] σ̂ consensus = {sigma_agreed:.2e}")
    return mu_agreed, sigma_agreed


# ============================================================================
# Waypoint dispatch — arrival-gated per stato
# ============================================================================

def _on_wp_reached(
    ag:      DroneAgent,
    sig:     float,
    terrain: Terrain,
    agl:     float,
) -> None:
    """Chiamato quando il drone raggiunge il waypoint corrente."""
    if ag.state == DroneState.SEARCH:
        if not ag.advance_waypoint():
            next_idx = ag.current_lane_idx + ag.n_drones_total
            if next_idx >= len(ag.lane_xs):
                next_idx = ag.id % len(ag.lane_xs)
            ag.current_lane_idx = next_idx
            ag.lane_go_up = not ag.lane_go_up
            ag.waypoints = single_lane_waypoints(
                ag.lane_xs[next_idx], ag.lane_go_up,
                terrain.y_min, terrain.y_max, terrain, agl,
            )
            ag.wp_idx = 0

    elif ag.state == DroneState.SUPPORT:
        if not ag.advance_waypoint():
            ag.wp_idx = 0   # cicla la circonferenza

    elif ag.state == DroneState.STOP:
        pass  # hovering


# ============================================================================
# Transizioni di stato
# ============================================================================

def _transition_search_to_track(
    ag:          DroneAgent,
    drone_id:    int,
    sig:         float,
    sigma_noise: float,
    step:        int,
    t:           float,
    terrain:     Terrain,
    agl:         float,
) -> None:
    """SEARCH → TRACK: avvia l'Extremum Seeking e inizializza il Particle Filter."""
    ag.state    = DroneState.TRACK
    ag.detected = True
    px, py      = float(ag.x_est[0]), float(ag.x_est[1])
    ag.source_est   = np.array([px, py, terrain.z(px, py)])
    ag.init_es(terrain, agl)
    ag.es_x_hist.clear()
    ag.es_y_hist.clear()

    ag.artva_sigma_noise = sigma_noise
    ag.pf = ParticleFilter(PF_N_PARTICLES, state_dim=3, measurement_dim=1)
    ag.pf.initialize_particles(ag.x[:3], sig)

    print(
        f"\n    Drone {drone_id} TRACK-ES (S={sig:.2e}) al passo {step} (t={t:.2f}s)"
        f"\n    pos reale={ag.x[:3].round(2)}  stimata={ag.x_est[:3].round(2)}"
        f"\n    ES ref init=({ag.es_x_ref:.1f}, {ag.es_y_ref:.1f})"
        f"\n    PF inizializzato con {PF_N_PARTICLES} particelle"
    )


def _transition_to_stop(
    ag:      DroneAgent,
    drone_id: int,
    sig:     float,
    step:    int,
    t:       float,
) -> None:
    """TRACK → STOP quando il segnale supera track_stop_thr."""
    ag.state     = DroneState.STOP
    hover_wp     = ag.x_est[:3].copy()
    ag.waypoints = [hover_wp]
    ag.wp_idx    = 0
    print(
        f"\n  ⬛ Drone {drone_id} STOP (S={sig:.2e}) al passo {step} (t={t:.2f}s)"
        f"  pos={ag.x[:3].round(1)}"
    )


# ============================================================================
# Costruzione agenti
# ============================================================================

def build_agents(
    deploy_xy: np.ndarray,
    terrain:   Terrain,
    n_drones:  int   = N_DRONES,
    agl:       float = AGL_HEIGHT,
) -> Dict[int, DroneAgent]:
    """Crea N droni con posizione iniziale, lawnmower, MPC e IMDCL."""
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

        wps  = lawnmower_waypoints(i, n_drones, x_min, x_max, y_min, y_max, terrain, agl=agl)
        ctrl = DroneMPC(
            dt=DT_MPC, N=N_MPC,
            ax_max=A_MAX, ay_max=A_MAX, az_max=A_MAX,
            vx_max=V_MAX, vy_max=V_MAX, vz_max=V_MAX,
        )
        imdcl_agent = AgentIMDCL(
            agent_id=i, x0=x0.copy(), P0=P0_imdcl.copy(),
            team_ids=team_ids, motion_model=imdcl_motion,
        )
        agent = DroneAgent(
            id=i, x=x0.copy(), waypoints=wps,
            ctrl=ctrl, imdcl=imdcl_agent,
            state=DroneState.SEARCH,
        )
        agent.history.append(x0.copy())
        agent.state_log.append(DroneState.SEARCH)
        agent.est_history.append(imdcl_agent.x_hat.copy())
        agents[i] = agent

    print("Warm-start MPC droni...")
    for i, ag in agents.items():
        ag.ctrl.first_step(ag.x_est, ag.current_target())
        print(f"  Drone {i}: wp[0]={ag.current_target().round(1)}")

    return agents


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
) -> tuple:
    """
    Esegue la simulazione multi-agente.

    Per ogni passo temporale:
      1. Misura ARTVA reale (filtrata)
      2. Transizioni di stato (SEARCH→TRACK, TRACK→STOP, SUPPORT→STOP)
      3. ES update per droni TRACK
      4. MPC step → u_opt
      5. Propagazione dinamica reale con rumore
      5a-c. IMDCL: propagazione, update LiDAR, update cooperativo
      6. Particle Filter: update pesi + resample + stima sorgente
      7. Avanza waypoint se raggiunto (arrival-gated)
      8. Terminazione
    """
    rng       = np.random.default_rng(rng_seed)
    model     = PointMass3DModel(sigma_acc=sigma)
    drone_ids = list(agents.keys())

    # ── Calibrazione rumore e soglie dinamiche ───────────────────────────────
    print("Calibrazione rumore ARTVA...")
    mu_noise, sigma_noise = _calibrate_noise(agents, artva)
    artva_detect_thr = max(mu_noise + 5*sigma_noise, ARTVA_MOMENT / ES_DETECT_MAX_R**3)
    track_stop_thr   = ARTVA_MOMENT / FOUND_RADIUS**3
    r_detect     = (ARTVA_MOMENT / artva_detect_thr) ** (1.0 / 3.0)
    workspace_w  = terrain.x_max - terrain.x_min
    n_agents     = len(agents)
    n_cov_lanes  = max(1, int(np.ceil(workspace_w / (2.0 * r_detect))))
    n_passes     = max(2, int(np.ceil(n_cov_lanes / n_agents)))
    n_lanes_tot  = n_passes * n_agents
    lane_spacing = workspace_w / n_lanes_tot
    all_lane_xs  = np.arange(
        terrain.x_min + lane_spacing / 2,
        terrain.x_max,
        lane_spacing,
    )
    print(
        f"  Soglie dinamiche: DETECT={artva_detect_thr:.2e}  STOP={track_stop_thr:.2e}"
        f"  r_detect={r_detect:.1f} m  →  {n_cov_lanes} corsie di copertura"
        f"  →  {n_passes} passate di pettine  ×  {n_agents} droni"
        f"  =  {n_lanes_tot} corsie totali  (lane_spacing={lane_spacing:.1f} m)\n"
    )

    for i, ag in agents.items():
        ag.lane_xs          = all_lane_xs
        ag.n_drones_total   = n_agents
        ag.current_lane_idx = i % len(all_lane_xs)
        ag.lane_go_up       = True
        ag.waypoints = single_lane_waypoints(
            all_lane_xs[ag.current_lane_idx], True,
            terrain.y_min, terrain.y_max, terrain, agl,
        )
        ag.wp_idx = 0

    R_rel         = np.eye(3) * IMDCL_R_MEAS_STD**2
    R_lidar       = np.array([[IMDCL_R_LIDAR_STD**2]])
    consensus_done = [False]   # True dopo il primo STOP con partner assegnati

    _STATE_LABEL = {
        DroneState.SEARCH:  "SRCH",
        DroneState.TRACK:   "TRCK",
        DroneState.STOP:    "STOP",
        DroneState.SUPPORT: "SUPP",
    }

    print(
        f"\n{'Step':>5}  {'t[s]':>6}  "
        + "  ".join(f"D{i}:st/wp/dist/Δest" for i in drone_ids)
    )

    for step in range(n_steps):
        t = step * dt

        # ── 1. Misura ARTVA ──────────────────────────────────────────────
        signals: Dict[int, float] = {}
        for i in drone_ids:
            ag       = agents[i]
            sig      = float(artva.signal(ag.x[:3] , noisy=True))
            sig_filt = ag.update_signal_filter_batch(sig)
            ag.sig_batch.append(sig_filt)
            ag.r     = (ARTVA_MOMENT / sig_filt) ** (1.0 / 3.0) if sig_filt > 0.0 else None
            ag.r_log.append(ag.r)
            ag.signal_log.append((ag.x[:3].copy(), sig_filt))
            signals[i] = sig_filt

        # ── 2. Transizioni di stato ──────────────────────────────────────
        for i in drone_ids:
            ag  = agents[i]
            sig = signals[i]

            if ag.state == DroneState.SEARCH and sig >= artva_detect_thr:
                _transition_search_to_track(ag, i, sig, sigma_noise, step, t, terrain, agl)

            elif ag.state == DroneState.TRACK and sig >= track_stop_thr:
                _transition_to_stop(ag, i, sig, step, t)
                if not consensus_done[0]:
                    _assign_support_partners(
                        agents, drone_ids, i, sig, sigma_noise, step, terrain, agl, consensus_done,
                    )

            elif ag.state == DroneState.SUPPORT and sig >= track_stop_thr:
                ag.state     = DroneState.STOP
                hover_wp     = ag.x_est[:3].copy()
                ag.waypoints = [hover_wp]
                ag.wp_idx    = 0
                print(
                    f"\n  ⬛ Drone {i} STOP (SUPPORT→STOP, S={sig:.2e}) "
                    f"al passo {step} (t={t:.2f}s)"
                )

        # ── 2b. Retry SUPPORT search per droni STOP in attesa ────────────
        for i in drone_ids:
            if agents[i].state == DroneState.STOP and agents[i].support_pending:
                _retry_support_search(agents, drone_ids, i, sigma_noise, step, terrain, agl)

        # ── 2c. ES update per droni in TRACK ────────────────────────────
        for i in drone_ids:
            ag = agents[i]
            if ag.state == DroneState.TRACK:
                yt = _es_condition_signal(signals[i])
                _es_update(ag, yt, dt, terrain, agl)

        # Log waypoint corrente
        for i in drone_ids:
            agents[i].wp_target_log.append(agents[i].current_target().copy())

        # ── 3. MPC step ──────────────────────────────────────────────────
        u_commands: Dict[int, np.ndarray] = {}
        for i in drone_ids:
            ag     = agents[i]
            target = ag.current_target().copy()
            target[2] = terrain.agl_z(ag.x_est[0], ag.x_est[1], agl)
            t0    = perf_counter()
            u_opt = ag.ctrl.step(ag.x_est, target)
            ag.solve_t_log.append(perf_counter() - t0)
            u_commands[i] = u_opt

        # ── 4. Propagazione dinamica reale ───────────────────────────────
        for i in drone_ids:
            ag    = agents[i]
            noise = rng.multivariate_normal(np.zeros(3), np.diag([sigma**2] * 3))
            ag.x  = model.f(ag.x, u_commands[i] + noise, dt)
            z_floor = terrain.agl_z(ag.x[0], ag.x[1], agl * 0.5)
            if ag.x[2] < z_floor:
                ag.x[2] = z_floor
                ag.x[5] = max(0.0, ag.x[5])
            ag.history.append(ag.x.copy())
            ag.state_log.append(ag.state)
            ag.input_log.append(u_commands[i].copy())

        # ── 5a. IMDCL — propagazione ─────────────────────────────────────
        for i in drone_ids:
            agents[i].imdcl.propagate(u_commands[i], dt)

        # ── 5b. IMDCL — update LiDAR ────────────────────────────────────
        for i in drone_ids:
            ag          = agents[i]
            z_terr_real = terrain.z(ag.x[0], ag.x[1])
            agl_lidar   = (ag.x[2] - z_terr_real) + rng.normal(0, IMDCL_R_LIDAR_STD)
            z_terr_est  = terrain.z(ag.imdcl.x_hat[0], ag.imdcl.x_hat[1])
            z_obs       = np.array([z_terr_est + agl_lidar])
            ag.imdcl.apply_absolute_update(z_obs, IMDCL_H_LIDAR, R_lidar)

        # ── 5c. IMDCL — update cooperativo ──────────────────────────────
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

        # ── 6. Source estimation — Particle Filter update ────────────────
        for i in drone_ids:
            ag = agents[i]
            if ag.pf is None:
                ag.pf_log.append(None)
                continue

            # update con la misura del drone stesso
            ag.pf.update_weights(ag.x[:3], signals[i], ARTVA_MOMENT, ag.artva_sigma_noise)

            # update con le misure dei droni vicini che hanno già il PF inizializzato
            for j in drone_ids:
                if j == i or agents[j].pf is None:
                    continue
                if np.linalg.norm(ag.x[:3] - agents[j].x[:3]) < IMDCL_COMM_RADIUS:
                    ag.pf.update_weights(agents[j].x[:3], signals[j], ARTVA_MOMENT, ag.artva_sigma_noise)

            ag.pf.resample_particles()
            ag.source_est = np.average(ag.pf.particles, weights=ag.pf.weights, axis=0)
            ag.pf_log.append((ag.pf.particles.copy(), ag.pf.weights.copy()))

        # Log source_est
        for i in drone_ids:
            ag_i = agents[i]
            if ag_i.source_est is not None and ag_i.state != DroneState.SEARCH:
                ag_i.source_est_log.append((step, ag_i.source_est.copy()))

        # ── 7. Avanza waypoint (arrival-gated) ───────────────────────────
        for i in drone_ids:
            ag = agents[i]
            if ag.state == DroneState.TRACK:
                continue  # guidato dall'ES
            if np.linalg.norm(ag.x_est[:3] - ag.current_target()) < STOP_THRESH:
                _on_wp_reached(ag, signals[i], terrain, agl)

        # ── 8. Log periodico ─────────────────────────────────────────────
        if (step + 1) % 20 == 0:
            row = f"{step+1:>5}  {(step+1)*dt:>5.1f}s  "
            for i in drone_ids:
                ag      = agents[i]
                dist    = np.linalg.norm(ag.x_est[:3] - ag.current_target())
                est_err = np.linalg.norm(ag.x[:3] - ag.x_est[:3])
                row    += f"  {_STATE_LABEL[ag.state]}/{ag.wp_idx:02d}/{dist:5.2f}m/Δ{est_err:.2f}m"
            print(row)

        # ── 9. Terminazione ──────────────────────────────────────────────
        n_stopped = sum(1 for i in drone_ids if agents[i].state == DroneState.STOP)
        if n_stopped >= N_STOP:
            print(
                f"\n  {N_STOP} droni in STOP al passo {step} (t={t:.2f}s) — "
                "simulazione terminata"
            )
            break


    # ── Stima finale PF: media pesata + deviazione standard ───────────────
    for i in drone_ids:
        ag = agents[i]
        if ag.pf is not None and ag.source_est is not None:
            diff = ag.pf.particles - ag.source_est
            var  = np.average(diff ** 2, weights=ag.pf.weights, axis=0)
            ag.source_est_std = np.sqrt(var)

    # ── Report finale ──────────────────────────────────────────────────────
    valid_ests = [agents[i].source_est for i in drone_ids if agents[i].source_est is not None]

    print("\n  Stime PF source_est per drone:")
    for i in drone_ids:
        ag = agents[i]
        if ag.source_est is not None:
            e_i = np.linalg.norm(ag.source_est[:2] - artva._theta[:2])
            std_str = ""
            if ag.source_est_std is not None:
                s = ag.source_est_std
                std_str = f"  σ=({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}) m"
            print(f"    Drone {i}: {ag.source_est.round(2)}  (err_xy={e_i:.2f} m){std_str}")
        else:
            print(f"    Drone {i}: n/a (PF non attivato)")

    if valid_ests:
        est    = np.mean(valid_ests, axis=0)
        err_xy = np.linalg.norm(est[:2] - artva._theta[:2])
        print(f"\n  Stima PF media          : {est.round(2)}")
        print(f"  Errore planimetrico     : {err_xy:.2f} m")
        if len(valid_ests) >= 2:
            ests_xy = np.array([e[:2] for e in valid_ests])
            var_xy  = np.var(ests_xy, axis=0)
            print(
                f"  Varianza stime [σ²x={var_xy[0]:.3f}, σ²y={var_xy[1]:.3f}]  "
                f"(σ_planimetrica={np.sqrt(var_xy.sum()):.3f} m)"
            )

    print("\n  Errore stima IMDCL finale:")
    for i in drone_ids:
        ag = agents[i]
        print(f"    Drone {i}: |x_real - x_est| = {np.linalg.norm(ag.x[:3] - ag.x_est[:3]):.3f} m")

    print("\n  Errore di landing per drone:")
    for i in drone_ids:
        ag = agents[i]
        if ag.source_est is None:
            print(f"    Drone {i}: n/a (nessuna stima PF)")
            continue
        landing_point = ag.source_est - (ag.x_est[:3] - ag.x[:3])
        landing_err   = landing_point[:2] - artva._theta[:2]
        print(
            f"    Drone {i}: {np.linalg.norm(landing_err):.3f} m  "
            f"(Δx={landing_err[0]:+.2f}  Δy={landing_err[1]:+.2f})"
        )

    return agents, artva_detect_thr, track_stop_thr

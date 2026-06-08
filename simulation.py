"""
simulation.py
=============
Loop principale di simulazione multi-agente.

Funzioni pubbliche
------------------
  build_agents          — costruisce N DroneAgent con MPC + IMDCL + warm-start
  triangulate_victim    — stima posizione vittima dalla media delle source_est DCGD
  simulate              — loop temporale multi-agente

FSM a 4 stati
-------------
  SEARCH  → lawnmower arrival-gated
  TRACK   → esplorazione 3 candidati (avanti, ±60°); arrival-gated; → STOP a soglia
  STOP    → hovering; seleziona 2 droni SUPPORT per triangolazione
  SUPPORT → percorre cerchio attorno al drone STOP; → STOP a soglia

Waypoint feed
-------------
Tutti gli stati sono arrival-gated: _on_wp_reached() viene chiamato solo quando
il drone raggiunge il waypoint corrente (distanza < STOP_THRESH). Eccezione:
transizioni di stato, che resettano i waypoint immediatamente.

Algoritmo stima sorgente: DCGD (Distributed Consensus Gradient Descent).
"""

from __future__ import annotations

from itertools import permutations as _iterperms
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
    lawnmower_waypoints, single_lane_waypoints,
)
from config import (
    N_DRONES, DEPLOY_OFFSET,
    AGL_HEIGHT, LIDAR_SIGMA,
    DT_MPC, N_MPC, A_MAX, V_MAX,
    N_SIM, DT_SIM, SIGMA_ACC_SIM, STOP_THRESH,
    IMDCL_SIGMA_ACC, IMDCL_P0_POS, IMDCL_P0_VEL,
    IMDCL_COMM_RADIUS, IMDCL_R_MEAS_STD,
    IMDCL_R_LIDAR_STD, IMDCL_H_LIDAR,
    N_SIGNAL_SAMPLES,
    DIST_EST_ALPHA, DIST_EST_BETA, DIST_EST_H, DIST_EST_REFINE, DIST_EST_BATCH,
    TRIANGULATE_N_PARTNERS,
    SUPPORT_SEARCH_TIMEOUT,
    ARTVA_MOMENT,
    CONSENSUS_K_MAX,
    N_NOISE_CALIB_SAMPLES, NOISE_CONSENSUS_ITERS,
    ES_DETECT_MAX_R, FOUND_RADIUS,
    ES_ALPHA_MAX, ES_OMEGA, ES_KAPPA, ES_LAMBDA, ES_EPS,
    CONVERGE_RADIUS, ES_DCGD_SKIP_THR,
)


# ============================================================================
# Helper ARTVA model (senza rumore) — usato da DCGD
# ============================================================================

def _artva_model(pos: np.ndarray, theta: np.ndarray, moment: float) -> float:
    delta  = np.asarray(pos, dtype=float) - np.asarray(theta, dtype=float)
    r      = max(np.linalg.norm(delta), 1e-2)
    cos_th = delta[2] / r
    return moment * np.sqrt(1.0 + 3.0 * cos_th**2) / r**3


def _grad_S(pos: np.ndarray, theta: np.ndarray, moment: float) -> np.ndarray:
    grad = np.zeros(3)
    for k in range(3):
        dp = np.zeros(3); dp[k] = DIST_EST_H
        grad[k] = (
            _artva_model(pos, theta + dp, moment)
            - _artva_model(pos, theta - dp, moment)
        ) / (2.0 * DIST_EST_H)
    return grad


# ============================================================================
# DCGD
# ============================================================================

def _dcgd_step(agents: Dict[int, DroneAgent], drone_ids: list) -> None:
    """Un passo DCGD (Adapt + Combine) per droni TRACK/SUPPORT/STOP con source_est."""
    active_ids = [
        i for i in drone_ids
        if agents[i].state in (DroneState.TRACK, DroneState.SUPPORT, DroneState.STOP)
        and agents[i].source_est is not None
    ]
    if not active_ids:
        return

    snap = {i: agents[i].source_est.copy() for i in active_ids}

    # Adapt: gradient descent locale
    for i in active_ids:
        ag      = agents[i]
        theta_i = snap[i].copy()

        batch  = ag.signal_log[-DIST_EST_BATCH:]
        grad_J = np.zeros(3)
        for pos, s_meas in batch:
            s_pred  = _artva_model(pos, theta_i, ARTVA_MOMENT)
            grad_s  = _grad_S(pos, theta_i, ARTVA_MOMENT)
            grad_J += -2.0 * (s_meas - s_pred) * grad_s
        if batch:
            grad_J /= len(batch)
        norm_g = np.linalg.norm(grad_J)
        if norm_g > 1e-12:
            theta_i -= DIST_EST_ALPHA * grad_J / norm_g
        ag.source_est = theta_i

    # Combine: average consensus pesato sui valori post-adapt
    adapted  = {i: agents[i].source_est.copy() for i in active_ids}
    combined = _average_consensus(agents, active_ids, adapted, iters=1, beta=DIST_EST_BETA)
    for i in active_ids:
        agents[i].source_est = combined[i]


# ============================================================================
# Stima distribuita della profondità di sepoltura
# ============================================================================



# ============================================================================
# Extremum Seeking — TRACK mode  [Azzollini et al. 2021]
# ============================================================================

def _es_condition_signal(sig: float) -> float:
    """
    Condiziona il segnale ARTVA secondo eq. 5 del paper:
        yt = 1 / ∛S
    La mappa risultante è continua, limitata, con minimo globale in 0
    corrispondente al massimo del segnale grezzo (= posizione sorgente).
    """
    return 1.0 / np.cbrt(max(sig, ES_EPS))


def _es_update(
    ag:      "DroneAgent",
    yt:      float,
    dt:      float,
    terrain: Terrain,
    agl:     float,
) -> None:
    """
    Un passo dell'algoritmo ES a bounded update rate (Proposition 1,
    Azzollini et al. 2021, eq. 11-13), discretizzato con Euler in avanti.

    Aggiorna ag.es_alpha, ag.es_time, ag.es_x_ref, ag.es_y_ref
    e scrive ag.waypoints[0] con il nuovo riferimento per l'MPC.

    La velocità istantanea del riferimento è √(α·ω) ≤ √(α_max·ω) = V_MAX.
    Il centro del cerchio converge verso il minimo di yt (= sorgente ARTVA)
    seguendo la dinamica media di discesa del gradiente (eq. 12).
    """
    # α-filter ramp: α̇ = (α_max − α) / λ  →  α → α_max esponenzialmente
    ag.es_alpha += dt * (ES_ALPHA_MAX - ag.es_alpha) / ES_LAMBDA

    # Velocità istantanea del riferimento (costante in modulo)
    speed = np.sqrt(ag.es_alpha * ES_OMEGA)

    # Fase: ωt + κ·yt  (il termine κ·yt sposta il centro verso il minimo)
    phase = ES_OMEGA * ag.es_time + ES_KAPPA * yt

    # Euler forward integration of eq. 11
    ag.es_x_ref += dt * speed * np.cos(phase)
    ag.es_y_ref += dt * speed * np.sin(phase)
    ag.es_time  += dt

    # Il riferimento è clampato ai bordi del workspace (non modifica l'integrazione
    # interna per non interrompere la dinamica del cerchio)
    x = float(np.clip(ag.es_x_ref, terrain.x_min, terrain.x_max))
    y = float(np.clip(ag.es_y_ref, terrain.y_min, terrain.y_max))
    z = terrain.agl_z(x, y, agl)

    ag.waypoints = [np.array([x, y, z])]
    ag.wp_idx    = 0


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
# Consensus distribuito — selezione partner per drone in STOP
# ============================================================================

def _reachable_from(
    drone_ids:   list,
    agents:      Dict[int, DroneAgent],
    start_id:    int,
    comm_radius: float,
) -> list:
    """BFS: restituisce i drone_ids raggiungibili da start_id nel grafo di comunicazione."""
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
    comm_radius: float = IMDCL_COMM_RADIUS,
    k_max:       int   = CONSENSUS_K_MAX,
) -> tuple:
    """
    Selezione distribuita dei TRIANGULATE_N_PARTNERS più vicini al drone STOP
    tramite min-consensus su grafo limitato a comm_radius.

    Solo i droni nella componente connessa di stop_id partecipano: la richiesta
    non può propagarsi a cluster isolati, quindi non devono contribuire al consensus.
    """
    drone_ids = _reachable_from(drone_ids, agents, stop_id, comm_radius)
    if len(drone_ids) > 1:
        isolated = len(list(agents.keys())) - len(drone_ids)
        if isolated:
            print(f"    [consensus] componente connessa: {drone_ids} ({isolated} droni isolati esclusi)")

    n      = len(drone_ids)
    id2idx = {did: k for k, did in enumerate(drone_ids)}
    h_idx  = id2idx[stop_id]

    stop_pos        = agents[stop_id].x_est[:3].copy()
    pos             = np.array([agents[did].x_est[:3] for did in drone_ids])
    pairwise        = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    in_range        = pairwise <= comm_radius
    np.fill_diagonal(in_range, False)
    dist_from_stop  = np.linalg.norm(pos - stop_pos, axis=-1)

    all_links = [
        (drone_ids[i], drone_ids[j])
        for i in range(n) for j in range(i + 1, n)
        if in_range[i, j]
    ]

    D              = np.full((n, n), np.inf)
    D[h_idx, h_idx] = 0.0
    knows_target   = np.zeros(n, dtype=bool)
    knows_target[h_idx] = True
    rounds_log: list = []

    for _ in range(k_max):
        D_new     = D.copy()
        knows_new = knows_target.copy()
        newly_informed: list = []

        for i in range(n):
            for j in range(n):
                if not in_range[i, j]:
                    continue
                if knows_target[j] and not knows_new[i]:
                    knows_new[i] = True
                    D_new[i, i]  = dist_from_stop[i]
                    newly_informed.append(drone_ids[i])
                D_new[i] = np.minimum(D_new[i], D[j])

        rounds_log.append({'links': all_links, 'newly_informed': newly_informed})
        converged    = (len(newly_informed) == 0 and np.array_equal(D_new, D))
        D            = D_new
        knows_target = knows_new
        if converged:
            break

    D_agreed   = np.min(D, axis=0)
    candidates = sorted(
        [(drone_ids[k], D_agreed[k]) for k in range(n) if k != h_idx and D_agreed[k] < np.inf],
        key=lambda x: x[1],
    )
    partners = [did for did, _ in candidates[:TRIANGULATE_N_PARTNERS]]
    return partners, rounds_log


# ============================================================================
# Waypoint dispatch — arrival-gated per stato
# ============================================================================

def _on_wp_reached(
    ag:      DroneAgent,
    sig:     float,
    terrain: Terrain,
    agl:     float,
) -> None:
    """
    Chiamato quando il drone raggiunge il waypoint corrente.
    Calcola il prossimo waypoint in base allo stato FSM.
    """
    if ag.state == DroneState.SEARCH:
        if not ag.advance_waypoint():
            next_idx = ag.current_lane_idx + ag.n_drones_total
            if next_idx >= len(ag.lane_xs):
                next_idx = ag.id % len(ag.lane_xs)  # wrap: ricomincia dalla corsia iniziale
            ag.current_lane_idx = next_idx
            ag.lane_go_up = not ag.lane_go_up
            ag.waypoints = single_lane_waypoints(
                ag.lane_xs[next_idx], ag.lane_go_up,
                terrain.y_min, terrain.y_max, terrain, agl,
            )
            ag.wp_idx = 0

    elif ag.state == DroneState.STOP:
        pass  # hovering

    elif ag.state == DroneState.SUPPORT:
        if not ag.advance_waypoint():
            if ag.source_est is not None:
                mid_xy = (ag.x_est[:2] + ag.source_est[:2]) / 2.0
                ag.waypoints = [np.array([mid_xy[0], mid_xy[1], terrain.agl_z(mid_xy[0], mid_xy[1], agl)])]
                ag.wp_idx    = 0



# ============================================================================
# Transizioni di stato
# ============================================================================

def _transition_search_to_track(
    ag:            DroneAgent,
    drone_id:      int,
    sig:           float,
    step:          int,
    t:             float,
    terrain:       Terrain,
    agl:           float,
    lawnmower_orig: dict,
    cx_ws:         float,
    cy_ws:         float,
) -> None:
    """SEARCH → TRACK: avvia l'Extremum Seeking dalla posizione corrente."""
    ag.state    = DroneState.TRACK
    ag.detected = True
    ag.init_es(terrain, agl)
    ag.es_x_hist.clear()
    ag.es_y_hist.clear()

    # source_est inizializzata alla posizione del drone al rilevamento.
    # Verrà aggiornata ogni ciclo ES con la media del ciclo (step 2d).
    px, py = float(ag.x_est[0]), float(ag.x_est[1])
    ag.source_est = np.array([px, py, terrain.z(px, py)])
    print(
        f"\n  ★ Drone {drone_id} TRACK-ES (S={sig:.2e}) al passo {step} (t={t:.2f}s)"
        f"\n    pos reale={ag.x[:3].round(2)}  stimata={ag.x_est[:3].round(2)}"
        f"\n    ES ref init=({ag.es_x_ref:.1f}, {ag.es_y_ref:.1f})"
    )


def _transition_to_stop(
    ag:              DroneAgent,
    drone_id:        int,
    sig:             float,
    step:            int,
    t:               float,
    agents:          Dict[int, DroneAgent],
    drone_ids:       list,
    terrain:         Terrain,
    agl:             float,
    consensus_done:  bool,
    consensus_events: list,
) -> bool:
    """
    TRACK/SUPPORT → STOP quando segnale ≥ TRACK_STOP_THR.
    Se il drone era TRACK e il consenso non è ancora avvenuto, seleziona
    i partner SUPPORT e genera le loro circonferenze.
    Restituisce il nuovo valore di consensus_done.
    """
    ag.state = DroneState.STOP
    hover_wp = ag.x_est[:3].copy()
    ag.waypoints = [hover_wp]
    ag.wp_idx    = 0
    print(
        f"\n  ⬛ Drone {drone_id} STOP (S={sig:.2e}) al passo {step} (t={t:.2f}s)"
        f"  pos={ag.x[:3].round(1)}"
    )

    # Consenso e assegnazione SUPPORT solo al primo drone TRACK che si ferma
    if not consensus_done:
        consensus_done = True
        partners, rounds_log = _consensus_select_partners(agents, drone_ids, drone_id)
        consensus_events.append({
            'step': step, 'stop_id': drone_id,
            'rounds': rounds_log, 'partners': partners,
        })
        # 3 posizioni a 120° attorno alla stima della sorgente (identico a converge_triggered)
        center = ag.source_est[:3].copy() if ag.source_est is not None else ag.x_est[:3].copy()
        all_ids = [drone_id] + list(partners)
        n_assign = len(all_ids)
        targets = [
            np.array([
                center[0] + CONVERGE_RADIUS * np.cos(k * 2.0 * np.pi / 3.0),
                center[1] + CONVERGE_RADIUS * np.sin(k * 2.0 * np.pi / 3.0),
                terrain.agl_z(
                    center[0] + CONVERGE_RADIUS * np.cos(k * 2.0 * np.pi / 3.0),
                    center[1] + CONVERGE_RADIUS * np.sin(k * 2.0 * np.pi / 3.0),
                    agl,
                ),
            ])
            for k in range(3)
        ]
        best_perm, best_cost = None, float('inf')
        for perm in _iterperms(range(3)):
            cost = sum(
                np.linalg.norm(agents[all_ids[k]].x_est[:2] - targets[perm[k]][:2])
                for k in range(n_assign)
            )
            if cost < best_cost:
                best_cost, best_perm = cost, perm

        ag.waypoints = [targets[best_perm[0]]]
        ag.wp_idx    = 0

        for k, partner_id in enumerate(partners):
            partner = agents[partner_id]
            t = targets[best_perm[k + 1]]
            partner.state          = DroneState.SUPPORT
            partner.support_center = ag.x_est[:3].copy()
            partner.waypoints      = [t]
            partner.wp_idx         = 0
            if partner.source_est is None and ag.source_est is not None:
                partner.source_est = ag.source_est.copy()
            print(
                f"    → Drone {partner_id} SUPPORT "
                f"target=({t[0]:.1f}, {t[1]:.1f}) a 120° attorno a ({center[0]:.1f}, {center[1]:.1f})"
            )

        n_missing = TRIANGULATE_N_PARTNERS - len(partners)
        if n_missing > 0:
            print(
                f"    ⚠ Consenso: solo {len(partners)}/{TRIANGULATE_N_PARTNERS} partner "
                f"(Rc={IMDCL_COMM_RADIUS} m) — ricerca aperta per {SUPPORT_SEARCH_TIMEOUT} step"
            )
            ag.support_pending  = True
            ag.support_deadline = step + SUPPORT_SEARCH_TIMEOUT
            ag.support_n_needed = n_missing
    else:
        print(f"    (consenso già eseguito — nessuna nuova selezione partner)")

    return consensus_done


# ============================================================================
# Retry ricerca partner SUPPORT
# ============================================================================

def _retry_support_search(
    agents:    Dict[int, DroneAgent],
    drone_ids: list,
    terrain:   Terrain,
    agl:       float,
    step:      int,
) -> None:
    """
    Chiamata ogni step per ogni drone STOP con ricerca partner aperta.
    Ritenta l'assegnazione SUPPORT oppure chiude per timeout.
    """
    for drone_id in drone_ids:
        ag = agents[drone_id]
        if ag.state != DroneState.STOP or not ag.support_pending:
            continue

        if step >= ag.support_deadline:
            ag.support_pending = False
            refine_ids = [
                j for j in drone_ids
                if agents[j].state in (DroneState.STOP, DroneState.SUPPORT)
                and agents[j].source_est is not None
            ]
            n_avail = len(refine_ids)
            print(
                f"\n  ⏱ Drone {drone_id}: timeout supporto al passo {step} — "
                f"{ag.support_n_needed} partner non trovati. "
                f"Raffinamento DCGD con {n_avail} droni disponibili..."
            )
            if n_avail >= 2:
                for _ in range(DIST_EST_REFINE):
                    _dcgd_step(agents, refine_ids)
            continue

        # Cerca droni non STOP e non già in SUPPORT nel raggio di comunicazione
        candidates = sorted(
            [
                j for j in drone_ids
                if j != drone_id
                and agents[j].state not in (DroneState.STOP, DroneState.SUPPORT)
                and np.linalg.norm(agents[drone_id].x[:3] - agents[j].x[:3]) < IMDCL_COMM_RADIUS
            ],
            key=lambda j: np.linalg.norm(agents[drone_id].x[:3] - agents[j].x[:3]),
        )
        if not candidates:
            continue

        to_assign = candidates[:ag.support_n_needed]

        # Ricalcola 3 posizioni a 120° e riassegna ottimalmente stop + tutti i support
        center = ag.source_est[:3].copy() if ag.source_est is not None else ag.x_est[:3].copy()
        targets = [
            np.array([
                center[0] + CONVERGE_RADIUS * np.cos(k * 2.0 * np.pi / 3.0),
                center[1] + CONVERGE_RADIUS * np.sin(k * 2.0 * np.pi / 3.0),
                terrain.agl_z(
                    center[0] + CONVERGE_RADIUS * np.cos(k * 2.0 * np.pi / 3.0),
                    center[1] + CONVERGE_RADIUS * np.sin(k * 2.0 * np.pi / 3.0),
                    agl,
                ),
            ])
            for k in range(3)
        ]
        existing_support = [j for j in drone_ids if j != drone_id and agents[j].state == DroneState.SUPPORT]
        all_assign_ids = ([drone_id] + existing_support + list(to_assign))[:3]
        n_assign = len(all_assign_ids)

        best_perm, best_cost = None, float('inf')
        for perm in _iterperms(range(3)):
            cost = sum(
                np.linalg.norm(agents[all_assign_ids[k]].x_est[:2] - targets[perm[k]][:2])
                for k in range(n_assign)
            )
            if cost < best_cost:
                best_cost, best_perm = cost, perm

        ag.waypoints = [targets[best_perm[0]]]
        ag.wp_idx    = 0

        for k, sup_id in enumerate(all_assign_ids[1:]):
            partner = agents[sup_id]
            t = targets[best_perm[k + 1]]
            partner.state          = DroneState.SUPPORT
            partner.support_center = ag.x_est[:3].copy()
            partner.waypoints      = [t]
            partner.wp_idx         = 0
            if partner.source_est is None and ag.source_est is not None:
                partner.source_est = ag.source_est.copy()
            if sup_id in to_assign:
                print(
                    f"    → Drone {sup_id} SUPPORT (retry, passo {step}) "
                    f"target=({t[0]:.1f}, {t[1]:.1f}) a 120° attorno a ({center[0]:.1f}, {center[1]:.1f})"
                )

        ag.support_n_needed -= len(to_assign)
        if ag.support_n_needed <= 0:
            ag.support_pending = False


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
    Ogni drone stima σ_noise da misure ripetute alla propria posizione iniziale
    (std delle misure è indipendente dal segnale medio → stima robusta).
    Poi average-consensus distribuito converge a una σ̂ comune.
    """
    drone_ids = list(agents.keys())
    
    mu_local: Dict[int, float] = {}
    sigma_local: Dict[int, float] = {}
    for i in drone_ids:
        pos     = agents[i].x[:3]
        samples = [artva.signal(pos, noisy=True) for _ in range(n_samples)]
        mu_local[i] = np.mean(samples)
        sigma_local[i] = float(np.std(samples))

    print("  [Calibrazione] σ_noise locale per drone:")
    for i in drone_ids:
        print(f"    Drone {i}: σ̂={sigma_local[i]:.2e}")

    sigma_agreed = _average_consensus(agents, drone_ids, sigma_local, iters=consensus_iters)
    mu_agreed = _average_consensus(agents, drone_ids, mu_local, iters=consensus_iters)
    mu_agreed = float(np.mean([mu_agreed[i] for i in drone_ids]))
    sigma_agreed = float(np.mean([sigma_agreed[i] for i in drone_ids]))
    print(f"  [Calibrazione] σ̂ consensus = {sigma_agreed:.2e}")
    return mu_agreed, sigma_agreed


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
# Stima finale posizione vittima
# ============================================================================

def triangulate_victim(
    agents: Dict[int, DroneAgent],
    artva:  ARTVASource,
) -> np.ndarray:
    """Stima posizione vittima: media delle stime source_est di droni TRACK/STOP."""
    ests = [
        ag.source_est for ag in agents.values()
        if ag.state in (DroneState.TRACK, DroneState.STOP)
        and ag.source_est is not None
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
) -> tuple:
    """
    Esegue la simulazione multi-agente.

    Per ogni passo temporale:
      1. Misura ARTVA reale
      2. Transizioni di stato (SEARCH→TRACK, TRACK/SUPPORT→STOP)
      3. MPC step → u_opt  (usa stima IMDCL)
      4. Propagazione dinamica reale con rumore
      5. Aggiornamento filtro IMDCL (propagazione + LiDAR + cooperativo)
      6. DCGD: un passo Adapt+Combine per droni TRACK
      7. Avanza waypoint se raggiunto (arrival-gated, dispatch per stato)
      8. Stop: ≥3 droni in STOP → raffinamento DCGD → break
    """
    rng              = np.random.default_rng(rng_seed)
    model            = PointMass3DModel(sigma_acc=sigma)
    drone_ids        = list(agents.keys())
    consensus_events: list = []   # log eventi consenso per analisi post-simulazione
    consensus_done:   bool = False
    converge_triggered: bool = False   # True dopo aver assegnato i target a 120°
    converge_ids:       list = []      # i 3 droni che devono convergere
    refine_phase:       int  = 0       # 0=attesa, 1=DCGD refine
    refine_ctr:         int  = 0       # step trascorsi nella fase corrente

    # ── Calibrazione rumore e soglie dinamiche ───────────────────────────────
    # DETECT: max(μ + 5σ, moment/R_ES³) — gate rumore + floor ES (R_ES=50m).
    #   Il 5σ screena falsi positivi prima di entrare in TRACK.
    # STOP: puro floor geometrico moment/R_found³ (R_found=10m).
    #   Il rumore è già stato screened da detect_thr; la stop è solo prossimità.
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

    R_rel   = np.eye(3) * IMDCL_R_MEAS_STD**2
    R_lidar = np.array([[IMDCL_R_LIDAR_STD**2]])

    lawnmower_orig = {i: list(agents[i].waypoints) for i in drone_ids}

    cx_ws = (terrain.x_min + terrain.x_max) / 2.0
    cy_ws = (terrain.y_min + terrain.y_max) / 2.0

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
            prev_pos = ag.history[-1][:3] if ag.history else ag.x[:3]
            alphas   = np.linspace(0.0, 1.0, N_SIGNAL_SAMPLES)
            sig      = float(np.mean([
                artva.signal(prev_pos * (1.0 - a) + ag.x[:3] * a, noisy=True)
                for a in alphas
            ]))
            
            sig_filt = ag.update_signal_filter(sig)
            ag.signal_log.append((ag.x[:3].copy(), sig_filt))
            signals[i] = sig_filt

        # ── 2. Transizioni di stato ──────────────────────────────────────
        for i in drone_ids:
            ag  = agents[i]
            sig = signals[i]

            if ag.state == DroneState.SEARCH and sig >= artva_detect_thr:
                _transition_search_to_track(
                    ag, i, sig, step, t, terrain, agl,
                    lawnmower_orig, cx_ws, cy_ws,
                )

            elif ag.state == DroneState.TRACK and sig >= track_stop_thr:
                consensus_done = _transition_to_stop(
                    ag, i, sig, step, t,
                    agents, drone_ids, terrain, agl,
                    consensus_done, consensus_events,
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

        # ── 2b. Retry partner SUPPORT pendenti ──────────────────────────
        _retry_support_search(agents, drone_ids, terrain, agl, step)

        # ── 2c. ES update per droni in TRACK ────────────────────────────
        # Aggiorna il riferimento ES e scrive waypoints[0] prima che l'MPC
        # calcoli il controllo, garantendo la separazione temporale ES↔MPC.
        for i in drone_ids:
            ag = agents[i]
            if ag.state == DroneState.TRACK:
                yt = _es_condition_signal(signals[i])
                _es_update(ag, yt, dt, terrain, agl)
                ag.es_x_hist.append(ag.es_x_ref)
                ag.es_y_hist.append(ag.es_y_ref)

        # ── 2d. ES-guided DCGD initialization ───────────────────────────
        # Ogni ciclo ES completo (T = 2π/ω), la media di (es_x_ref, es_y_ref)
        # sull'ultimo ciclo converge alla posizione della sorgente (Azzollini
        # et al. 2021, eq. 12 — media della dinamica ES = gradiente zero).
        # Aggiornare source_est con questa media evita l'ambiguità di direzione
        # del drift istantaneo (che nei primi passi dipende dalla fase iniziale).
        _ES_CYCLE = int(2.0 * np.pi / (ES_OMEGA * dt))  # ~139 step a dt=0.1 s
        for i in drone_ids:
            ag = agents[i]
            if ag.state != DroneState.TRACK or len(ag.es_x_hist) != _ES_CYCLE:
                continue
            # Criterio distribuito: se source_est si è già spostato dalla posizione
            # di rilevamento (es_x_hist[0], es_y_hist[0]) di più di ES_DCGD_SKIP_THR,
            # il consensus ha già propagato una stima migliore → non reinizializzare.
            detect_pos = np.array([ag.es_x_hist[0], ag.es_y_hist[0]])
            moved = np.linalg.norm(ag.source_est[:2] - detect_pos)
            if moved > ES_DCGD_SKIP_THR:
                print(
                    f"    [ES→DCGD] Drone {i}: skip reinit"
                    f"  (source_est già spostato di {moved:.1f} m dal consensus)"
                )
                continue
            sx = float(np.mean(ag.es_x_hist))
            sy = float(np.mean(ag.es_y_hist))
            ag.source_est = np.array([sx, sy, terrain.z(sx, sy)])
            print(
                f"    [ES→DCGD] Drone {i}: source_est → ({sx:.1f}, {sy:.1f})"
                f"  (media primo ciclo ES, consensus non ancora attivo)"
            )

        # Log waypoint corrente (dopo transizioni, prima di muoversi)
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

        # ── 6. DCGD ──────────────────────────────────────────────────────
        if refine_phase <= 1:   # sospeso durante depth consensus (fase 2)
            _dcgd_step(agents, drone_ids)
        for i in drone_ids:
            ag_i = agents[i]
            if ag_i.source_est is not None and ag_i.state in (
                DroneState.TRACK, DroneState.SUPPORT, DroneState.STOP
            ):
                ag_i.source_est_log.append((step, ag_i.source_est.copy()))

        # ── 7. Avanza waypoint (arrival-gated) ───────────────────────────
        # I droni in TRACK sono guidati dall'ES (aggiornato al passo 2c)
        # e non usano l'arrival gate — il loro target cambia ogni step.
        for i in drone_ids:
            ag = agents[i]
            if ag.state == DroneState.TRACK:
                continue
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

        # ── 9. Terminazione: ≥3 droni in STOP ───────────────────────────
        n_stopped = sum(1 for ag in agents.values() if ag.state == DroneState.STOP)

        if n_stopped >= 3 and not converge_triggered:
            converge_ids = [i for i in drone_ids if agents[i].state == DroneState.STOP][:3]
            ests = [agents[i].source_est for i in converge_ids
                    if agents[i].source_est is not None]
            center = np.mean(ests, axis=0) if ests else np.array(
                [cx_ws, cy_ws, terrain.z(cx_ws, cy_ws)]
            )

            # 3 posizioni equidistanti (120°) attorno alla stima della sorgente
            targets = [
                np.array([
                    center[0] + CONVERGE_RADIUS * np.cos(k * 2.0 * np.pi / 3.0),
                    center[1] + CONVERGE_RADIUS * np.sin(k * 2.0 * np.pi / 3.0),
                    terrain.agl_z(
                        center[0] + CONVERGE_RADIUS * np.cos(k * 2.0 * np.pi / 3.0),
                        center[1] + CONVERGE_RADIUS * np.sin(k * 2.0 * np.pi / 3.0),
                        agl,
                    ),
                ])
                for k in range(3)
            ]

            # Assegnazione ottimale: minimizza spostamento totale (6 permutazioni)
            best_perm, best_cost = None, float('inf')
            for perm in _iterperms(range(3)):
                cost = sum(
                    np.linalg.norm(agents[converge_ids[k]].x_est[:2] - targets[perm[k]][:2])
                    for k in range(3)
                )
                if cost < best_cost:
                    best_cost, best_perm = cost, perm

            for k, did in enumerate(converge_ids):
                agents[did].waypoints = [targets[best_perm[k]]]
                agents[did].wp_idx    = 0

            converge_triggered = True
            print(
                f"\n  ↗ {n_stopped} droni STOP al passo {step+1} (t={(step+1)*dt:.2f}s)"
                f" — posizionamento a 120° attorno a {center[:2].round(1)} (r={CONVERGE_RADIUS} m)"
            )

        # Attendi arrivo a 120° (solo finché non è ancora iniziato il raffinamento)
        if converge_triggered and refine_phase == 0:
            all_in_place = all(
                np.linalg.norm(agents[i].x_est[:3] - agents[i].current_target()) < STOP_THRESH
                for i in converge_ids
            )
            if all_in_place:
                print(
                    f"\n  ✔ Posizionamento completato al passo {step+1}"
                    f" (t={(step+1)*dt:.2f}s)"
                    f" — DCGD refine ({DIST_EST_REFINE} step nel loop)..."
                )
                for i in drone_ids:
                    agents[i].dcgd_refine_step = step + 1
                refine_phase = 1
                refine_ctr   = 0

        # Fase 1: DCGD refinement — _dcgd_step già chiamato al passo 6
        if refine_phase == 1:
            refine_ctr += 1
            if refine_ctr >= DIST_EST_REFINE:
                print(
                    f"\n  ✔ DCGD refine completato al passo {step+1}"
                    f" (t={(step+1)*dt:.2f}s)"
                )
                break

    # ── Report finale ─────────────────────────────────────────────────────
    est      = triangulate_victim(agents, artva)
    err_xy   = np.linalg.norm(est[:2] - artva.position[:2])
    z_true   = artva.position[2]
    depth_true  = terrain.z(artva.position[0], artva.position[1]) - z_true
    depth_est   = terrain.z(est[0], est[1]) - est[2]
    err_depth   = abs(depth_est - depth_true)
    print(f"\n  Posizione vittima reale  : {artva.position.round(2)}")
    print(f"  Stima DCGD distribuita   : {est.round(2)}")
    print(f"  Errore planimetrico      : {err_xy:.2f} m")
    print(f"  Profondità reale         : {depth_true:.2f} m")
    print(f"  Profondità stimata       : {depth_est:.2f} m  (errore {err_depth:.2f} m)")
    err = err_xy  # compatibilità con il resto del codice
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
        print(
            f"    Varianza stime [σ²x={var_xy[0]:.3f}, σ²y={var_xy[1]:.3f}]  "
            f"(σ_planimetrica={np.sqrt(var_xy.sum()):.3f} m)"
        )
    print("\n  Errore stima IMDCL finale:")
    for i in drone_ids:
        ag = agents[i]
        print(f"    Drone {i}: |x_real - x_est| = {np.linalg.norm(ag.x[:3] - ag.x_est[:3]):.3f} m")

    return agents, consensus_events, artva_detect_thr, track_stop_thr


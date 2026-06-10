"""
simulation.py
=============
Loop principale di simulazione multi-agente.

Funzioni pubbliche
------------------
  build_agents          — costruisce N DroneAgent con MPC + IMDCL + warm-start
  triangulate_victim    — stima posizione vittima dalla media delle source_est
  simulate              — loop temporale multi-agente

FSM a 4 stati
-------------
  SEARCH  → lawnmower arrival-gated
  TRACK   → ES (Extremum Seeking) verso la sorgente; → STOP a soglia
  STOP    → hovering; trigger disambiguazione DICT
  SUPPORT → naviga verso la posizione target (depth circle o candidato 2-drone)

Algoritmo stima sorgente: DICT (Distributed Iterative Consensus Triangulation).
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
    N_NOISE_CALIB_SAMPLES, NOISE_CONSENSUS_ITERS,
    ES_DETECT_MAX_R, FOUND_RADIUS,
    ES_ALPHA_MAX, ES_OMEGA, ES_KAPPA, ES_LAMBDA, ES_EPS,
    ARTVA_MOMENT,
    DICT_BETA, DICT_XY_ITERS, DICT_DEPTH_ITERS, CONVERGE_RADIUS,
    DCGD_ITERS, DCGD_STEP_XY,
)


# ============================================================================
# Circle intersections helper
# ============================================================================

def _circle_intersections_2d(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float,
) -> tuple | None:
    """Restituisce i 2 punti di intersezione di due cerchi XY, o None se non si intersecano."""
    d = np.linalg.norm(c2 - c1)
    if d < 1e-9 or d > r1 + r2 + 1e-6 or d < abs(r1 - r2) - 1e-6:
        return None
    a    = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h    = np.sqrt(max(0.0, r1**2 - a**2))
    dir_ = (c2 - c1) / d
    perp = np.array([-dir_[1], dir_[0]])
    mid  = c1 + a * dir_
    return mid + h * perp, mid - h * perp


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
# DICT — Distributed Iterative Consensus Triangulation
# ============================================================================

def _dict_step(
    agents:   Dict[int, DroneAgent],
    dict_ids: list,
    terrain:  Terrain,
) -> None:
    """
    Un passo DICT Adapt+Combine per i due droni in TRACK.

    Adapt: ogni drone calcola i due punti di intersezione tra le proprie
    due circonferenze usando la propria stima di posizione e raggio.
    Combine: [c1,c2]_i^{k+1} = (1-β)*half_i + β*half_j
    I candidati sono ordinati per x per garantire abbinamento coerente.
    """
    d0, d1 = dict_ids
    if not agents[d0].signal_log or not agents[d1].signal_log:
        return

    s0 = agents[d0].signal_log[-1][1]
    s1 = agents[d1].signal_log[-1][1]
    if s0 < 1e-12 or s1 < 1e-12:
        return

    p0 = agents[d0].x_est[:2].copy()
    p1 = agents[d1].x_est[:2].copy()
    r0 = (ARTVA_MOMENT / s0) ** (1.0 / 3.0)
    r1 = (ARTVA_MOMENT / s1) ** (1.0 / 3.0)

    def _intersect_sorted(pa, ra, pb, rb):
        result = _circle_intersections_2d(pa, ra, pb, rb)
        if result is None:
            w = ra / (ra + rb)
            m = pa + w * (pb - pa)
            return [m.copy(), m.copy()]
        return sorted([result[0].copy(), result[1].copy()], key=lambda c: (c[0], c[1]))

    # Adapt
    half_d0 = _intersect_sorted(p0, r0, p1, r1)
    half_d1 = _intersect_sorted(p1, r1, p0, r0)

    # Combine
    beta = DICT_BETA
    new_d0 = [(1 - beta) * half_d0[k] + beta * half_d1[k] for k in range(2)]
    new_d1 = [(1 - beta) * half_d1[k] + beta * half_d0[k] for k in range(2)]

    def _to_3d(pts, terr):
        return [np.array([c[0], c[1], terr.z(float(c[0]), float(c[1]))]) for c in pts]

    agents[d0].source_cands = _to_3d(new_d0, terrain)
    agents[d1].source_cands = _to_3d(new_d1, terrain)


def _dict_step_3drone(
    agents:     Dict[int, DroneAgent],
    active_ids: list,
    terrain:    Terrain,
) -> None:
    """
    Un passo DICT Adapt+Combine a N≥2 droni sulla stima XY (fase post-disambiguazione).

    Adapt : per ogni coppia (i,j) calcola l'intersezione dei cerchi e seleziona
            il punto più vicino alla source_est corrente (già disambiguata).
    Combine: source_est_i = (1−β)·media_locale_coppie_i + β·media_globale
    """
    ids = [
        i for i in active_ids
        if agents[i].signal_log and agents[i].source_est is not None
    ]
    if len(ids) < 2:
        return

    positions = {i: agents[i].x_est[:2].copy() for i in ids}
    radii: dict = {}
    for i in ids:
        s = agents[i].signal_log[-1][1]
        if s < 1e-12:
            return
        radii[i] = (ARTVA_MOMENT / s) ** (1.0 / 3.0)

    ref = np.mean([agents[i].source_est[:2] for i in ids], axis=0)

    # Adapt: una stima per ogni coppia
    pair_ests: dict = {}
    for k in range(len(ids)):
        for l in range(k + 1, len(ids)):
            ia, ib = ids[k], ids[l]
            result = _circle_intersections_2d(
                positions[ia], radii[ia], positions[ib], radii[ib]
            )
            if result is None:
                w  = radii[ia] / (radii[ia] + radii[ib])
                pt = positions[ia] + w * (positions[ib] - positions[ia])
            else:
                c0, c1 = result
                pt = c0 if np.linalg.norm(c0 - ref) <= np.linalg.norm(c1 - ref) else c1
            pair_ests[(ia, ib)] = pt

    global_mean = np.mean(list(pair_ests.values()), axis=0)

    # Combine
    beta = DICT_BETA
    for i in ids:
        my_pts    = [pt for (a, b), pt in pair_ests.items() if a == i or b == i]
        local_avg = np.mean(my_pts, axis=0)
        new_xy    = (1.0 - beta) * local_avg + beta * global_mean
        agents[i].source_est = np.array([
            new_xy[0], new_xy[1],
            terrain.z(float(new_xy[0]), float(new_xy[1]))
        ])


def _dict_disambiguate_3drone(
    agents:   Dict[int, DroneAgent],
    dict_ids: list,
    third_id: int,
) -> tuple:
    """
    Seleziona c1 o c2 in base al residuo sul cerchio del terzo drone.
    Restituisce (punto_2D_scelto, indice_scelto) — indice 0 o 1 in source_cands[d0].
    """
    d0 = dict_ids[0]
    cands = agents[d0].source_cands
    if len(cands) < 2:
        pt = np.asarray(cands[0][:2]) if cands else agents[d0].x_est[:2].copy()
        return pt, 0

    c1_2d = np.asarray(cands[0][:2], dtype=float)
    c2_2d = np.asarray(cands[1][:2], dtype=float)

    s3 = agents[third_id].signal_log[-1][1] if agents[third_id].signal_log else 0.0
    if s3 > 1e-12:
        p3 = agents[third_id].x_est[:2].copy()
        r3 = (ARTVA_MOMENT / s3) ** (1.0 / 3.0)
        res1 = abs(np.linalg.norm(c1_2d - p3) - r3)
        res2 = abs(np.linalg.norm(c2_2d - p3) - r3)
        return (c1_2d, 0) if res1 <= res2 else (c2_2d, 1)

    # Fallback: candidato più vicino a d0
    d0_pos = agents[d0].x_est[:2]
    if np.linalg.norm(c1_2d - d0_pos) <= np.linalg.norm(c2_2d - d0_pos):
        return c1_2d, 0
    return c2_2d, 1


def _dict_setup_depth_circle(
    agents:     Dict[int, DroneAgent],
    active_ids: list,
    source_est: np.ndarray,
    terrain:    Terrain,
    agl:        float,
) -> None:
    """
    Invia i 3 droni non-SEARCH a posizioni equidistanziate a 120° sulla
    circonferenza di raggio CONVERGE_RADIUS attorno a source_est.
    Tutti transitano in SUPPORT.
    """
    cx, cy = float(source_est[0]), float(source_est[1])
    for k, drone_id in enumerate(active_ids):
        angle = np.radians(k * 120.0)
        x = cx + CONVERGE_RADIUS * np.cos(angle)
        y = cy + CONVERGE_RADIUS * np.sin(angle)
        wp = np.array([x, y, terrain.agl_z(x, y, agl)])
        ag = agents[drone_id]
        ag.state     = DroneState.SUPPORT
        ag.waypoints = [wp]
        ag.wp_idx    = 0
        print(f"    → Drone {drone_id} SUPPORT depth-circle (120°·{k}) → ({x:.1f}, {y:.1f})")


def _dict_setup_2drone_disam(
    agents:   Dict[int, DroneAgent],
    dict_ids: list,
    terrain:  Terrain,
    agl:      float,
) -> None:
    """
    Disambiguazione 2 droni: d0 → c1, d1 → c2.
    Entrambi transitano in SUPPORT verso il proprio candidato.
    """
    d0, d1 = dict_ids
    cands = agents[d0].source_cands
    if len(cands) < 2:
        return

    c1_2d = np.asarray(cands[0][:2], dtype=float)
    c2_2d = np.asarray(cands[1][:2], dtype=float)

    # Assegna a d0 il candidato più vicino alla sua posizione corrente
    d0_pos = agents[d0].x_est[:2]
    if np.linalg.norm(c1_2d - d0_pos) > np.linalg.norm(c2_2d - d0_pos):
        c1_2d, c2_2d = c2_2d, c1_2d

    for drone_id, cand_2d in [(d0, c1_2d), (d1, c2_2d)]:
        ag = agents[drone_id]
        wp = np.array([cand_2d[0], cand_2d[1], terrain.agl_z(cand_2d[0], cand_2d[1], agl)])
        ag.state     = DroneState.SUPPORT
        ag.waypoints = [wp]
        ag.wp_idx    = 0
        ag.source_est = np.array([cand_2d[0], cand_2d[1], terrain.z(float(cand_2d[0]), float(cand_2d[1]))])
        print(f"    → Drone {drone_id} SUPPORT disambig-2drone → ({cand_2d[0]:.1f}, {cand_2d[1]:.1f})")


def _dict_depth_step(
    agents:     Dict[int, DroneAgent],
    active_ids: list,
    agl:        float,
) -> None:
    """
    Un passo DICT Adapt+Combine sulla stima di profondità.

    Adapt: h_depth_i^{k+1/2} = sqrt(r_i² - Δxy²) - z_agl
    Combine: h_depth_i^{k+1} = (1-β)*half_i + β*mean(half_j, j≠i)
    """
    halfs: dict = {}
    for i in active_ids:
        ag = agents[i]
        if not ag.signal_log or ag.source_est is None:
            continue
        s_i = ag.signal_log[-1][1]
        if s_i < 1e-12:
            continue
        r_i      = (ARTVA_MOMENT / s_i) ** (1.0 / 3.0)
        delta_xy = float(np.linalg.norm(ag.x_est[:2] - ag.source_est[:2]))
        under    = max(0.0, r_i**2 - delta_xy**2)
        halfs[i] = np.sqrt(under) - agl

    if len(halfs) < 2:
        for i in halfs:
            agents[i].depth_est = halfs[i]
        return

    beta = DICT_BETA
    ids  = list(halfs.keys())
    combined: dict = {}
    for i in ids:
        others        = [halfs[j] for j in ids if j != i]
        avg_others    = float(np.mean(others))
        combined[i]   = (1 - beta) * halfs[i] + beta * avg_others

    for i in ids:
        agents[i].depth_est = combined[i]


# ============================================================================
# DCGD — helpers orbit update
# ============================================================================

def _update_orbit_center(
    agents:     Dict[int, DroneAgent],
    active_ids: list,
    center:     np.ndarray,
    terrain:    Terrain,
    agl:        float,
) -> None:
    """
    Aggiorna i waypoint dei droni SUPPORT sulla circonferenza a 120°
    attorno a 'center', mantenendo l'assegnazione angolare k*120°.
    Non cambia lo stato FSM né stampa nulla.
    """
    cx, cy = float(center[0]), float(center[1])
    for k, drone_id in enumerate(active_ids):
        angle = np.radians(k * 120.0)
        x  = cx + CONVERGE_RADIUS * np.cos(angle)
        y  = cy + CONVERGE_RADIUS * np.sin(angle)
        wp = np.array([x, y, terrain.agl_z(x, y, agl)])
        agents[drone_id].waypoints = [wp]
        agents[drone_id].wp_idx    = 0


# ============================================================================
# DCGD — Distributed Consensus Gradient Descent (XY, fase SUPPORT)
# ============================================================================

def _artva_xy(xy: np.ndarray, p_drone: np.ndarray, z_v: float) -> float:
    """Modello ARTVA valutato in xy = [x_v, y_v], con vittima a quota z_v."""
    pv  = np.array([xy[0], xy[1], z_v])
    dv  = p_drone - pv
    r   = max(np.linalg.norm(dv), 1e-3)
    ct  = dv[2] / r
    return ARTVA_MOMENT * np.sqrt(1.0 + 3.0 * ct**2) / r**3


def _dcgd_step(
    agents:    Dict[int, DroneAgent],
    active_ids: list,
    terrain:   Terrain,
    agl:       float,
    step_size: float = DCGD_STEP_XY,
    beta:      float = DICT_BETA,
) -> np.ndarray:
    """
    Un passo DCGD Adapt+Combine sulla stima XY della sorgente.

    Adapt : ogni drone calcola il gradiente numerico di (S_model - S_meas)²
            rispetto a (x_v, y_v), normalizza e fa un passo di ampiezza step_size.
    Combine: average consensus pesato con i vicini (stesso schema di _dict_step_3drone).

    Restituisce lo spostamento medio assoluto per drone (utile per log).
    """
    ids = [
        i for i in active_ids
        if agents[i].signal_log and agents[i].source_est is not None
    ]
    if len(ids) < 2:
        return np.zeros(2)

    eps = 0.05   # [m] passo differenza finita

    # ── Adapt ────────────────────────────────────────────────────────────────
    half_xy: Dict[int, np.ndarray] = {}
    for i in ids:
        ag     = agents[i]
        s_meas = ag.signal_log[-1][1]
        if s_meas < 1e-12:
            half_xy[i] = ag.source_est[:2].copy()
            continue

        p_d = ag.x_est[:3].copy()
        xy0 = ag.source_est[:2].copy()
        # z_v: approssimazione con superficie del terreno (depth ignota qui)
        z_v = float(terrain.z(float(xy0[0]), float(xy0[1])))

        s0   = _artva_xy(xy0, p_d, z_v)
        grad = np.zeros(2)
        for k in range(2):
            xp = xy0.copy(); xp[k] += eps
            xm = xy0.copy(); xm[k] -= eps
            ds      = (_artva_xy(xp, p_d, z_v) - _artva_xy(xm, p_d, z_v)) / (2.0 * eps)
            grad[k] = 2.0 * (s0 - s_meas) * ds   # ∂L/∂x_v = 2·(ŝ - s)·∂ŝ/∂x_v

        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-15:
            half_xy[i] = xy0 - step_size * (grad / g_norm)
        else:
            half_xy[i] = xy0.copy()

    # ── Combine ───────────────────────────────────────────────────────────────
    combined: Dict[int, np.ndarray] = {}
    for i in ids:
        nbrs = [
            j for j in ids if j != i
            and np.linalg.norm(agents[i].x[:3] - agents[j].x[:3]) < IMDCL_COMM_RADIUS
        ]
        if nbrs:
            nbr_avg     = np.mean([half_xy[j] for j in nbrs], axis=0)
            combined[i] = (1.0 - beta) * half_xy[i] + beta * nbr_avg
        else:
            combined[i] = half_xy[i].copy()

    # ── Aggiorna source_est ───────────────────────────────────────────────────
    shifts = []
    for i in ids:
        old_xy = agents[i].source_est[:2].copy()
        new_xy = combined[i]
        new_z  = terrain.z(float(new_xy[0]), float(new_xy[1]))
        agents[i].source_est = np.array([new_xy[0], new_xy[1], new_z])
        shifts.append(np.abs(new_xy - old_xy))

    return np.mean(shifts, axis=0) if shifts else np.zeros(2)


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

    elif ag.state in (DroneState.STOP, DroneState.SUPPORT):
        pass  # hovering al target


# ============================================================================
# Transizioni di stato
# ============================================================================

def _transition_search_to_track(
    ag:      DroneAgent,
    drone_id: int,
    sig:     float,
    step:    int,
    t:       float,
    terrain: Terrain,
    agl:     float,
) -> None:
    """SEARCH → TRACK: avvia l'Extremum Seeking dalla posizione corrente."""
    ag.state        = DroneState.TRACK
    ag.detected     = True
    ag.source_cands = []
    px, py          = float(ag.x_est[0]), float(ag.x_est[1])
    ag.source_est   = np.array([px, py, terrain.z(px, py)])
    ag.init_es(terrain, agl)
    ag.es_x_hist.clear()
    ag.es_y_hist.clear()
    print(
        f"\n  ★ Drone {drone_id} TRACK-ES (S={sig:.2e}) al passo {step} (t={t:.2f}s)"
        f"\n    pos reale={ag.x[:3].round(2)}  stimata={ag.x_est[:3].round(2)}"
        f"\n    ES ref init=({ag.es_x_ref:.1f}, {ag.es_y_ref:.1f})"
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
# Stima finale posizione vittima
# ============================================================================

def triangulate_victim(
    agents: Dict[int, DroneAgent],
    artva:  ARTVASource,
) -> np.ndarray:
    """Stima posizione vittima: media delle stime source_est (droni non SEARCH)."""
    ests = [
        ag.source_est for ag in agents.values()
        if ag.state != DroneState.SEARCH
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
      1. Misura ARTVA reale (filtrata)
      2. Transizioni di stato (SEARCH→TRACK, TRACK→STOP, SUPPORT→STOP)
      3. ES update per droni TRACK
      4. MPC step → u_opt
      5. Propagazione dinamica reale con rumore
      6. Aggiornamento filtro IMDCL
      7. DICT: un passo Adapt+Combine (o depth step)
      8. Avanza waypoint se raggiunto (arrival-gated)
      9. Terminazione
    """
    rng              = np.random.default_rng(rng_seed)
    model            = PointMass3DModel(sigma_acc=sigma)
    drone_ids        = list(agents.keys())
    consensus_events: list = []   # mantenuto per compatibilità interfaccia (vuoto)

    # DICT state
    dict_ids:           list = []    # i primi due droni non-SEARCH
    dict_disam_done:    bool = False
    dict_disam_mode:    str  = ''    # '3drone_xy'|'3drone'|'2drone'|'2drone_done'|'all_stop'
    dict_xy_active_ids: list = []    # droni per il consenso XY a 3
    dict_xy_ctr:        int  = 0
    dict_depth_ctr:     int  = 0

    # DCGD state
    dcgd_ctr:  int  = 0
    dcgd_done: bool = False

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

    R_rel   = np.eye(3) * IMDCL_R_MEAS_STD**2
    R_lidar = np.array([[IMDCL_R_LIDAR_STD**2]])

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
                _transition_search_to_track(ag, i, sig, step, t, terrain, agl)

            elif ag.state == DroneState.TRACK and sig >= track_stop_thr:
                _transition_to_stop(ag, i, sig, step, t)

            elif (ag.state == DroneState.SUPPORT and sig >= track_stop_thr
                  and not dict_disam_done):
                ag.state     = DroneState.STOP
                hover_wp     = ag.x_est[:3].copy()
                ag.waypoints = [hover_wp]
                ag.wp_idx    = 0
                print(
                    f"\n  ⬛ Drone {i} STOP (SUPPORT→STOP, S={sig:.2e}) "
                    f"al passo {step} (t={t:.2f}s)"
                )

        # ── 2b. Aggiorna insieme dict_ids ────────────────────────────────
        not_search = [i for i in drone_ids if agents[i].state != DroneState.SEARCH]
        if len(dict_ids) < 2 and len(not_search) >= 2:
            dict_ids = not_search[:2]
            print(f"\n  ◆ DICT attivato: droni {dict_ids[0]}, {dict_ids[1]} al passo {step}")

        # ── 2b2. Trigger disambiguazione DICT (prima dell'MPC) ──────────────
        if len(dict_ids) == 2 and not dict_disam_done:
            first_stopped_early = next(
                (i for i in dict_ids if agents[i].state == DroneState.STOP), None
            )
            if first_stopped_early is not None:
                dict_disam_done = True
                not_search_now  = [i for i in drone_ids if agents[i].state != DroneState.SEARCH]
                third_ids = [i for i in not_search_now if i not in dict_ids]
                third_id  = third_ids[0] if third_ids else None

                if third_id is not None and agents[third_id].signal_log:
                    dict_disam_mode    = '3drone_xy'
                    dict_xy_active_ids = not_search_now[:]
                    sel_2d, chosen_idx = _dict_disambiguate_3drone(agents, dict_ids, third_id)

                    # Ogni drone DICT mantiene il proprio punto di intersezione
                    for i in dict_ids:
                        cands_i = agents[i].source_cands
                        if len(cands_i) > chosen_idx:
                            c = cands_i[chosen_idx]
                            agents[i].source_est = np.array([
                                c[0], c[1], terrain.z(float(c[0]), float(c[1]))
                            ])
                        else:
                            agents[i].source_est = np.array([
                                sel_2d[0], sel_2d[1],
                                terrain.z(float(sel_2d[0]), float(sel_2d[1]))
                            ])
                        agents[i].source_cands = [agents[i].source_est.copy()]
                        agents[i].source_cands_log.append(
                            (step, [agents[i].source_est.copy()])
                        )

                    # Terzo drone: media delle stime dei due DICT (inizializzazione XY consensus)
                    src_mean = np.mean([agents[i].source_est[:2] for i in dict_ids], axis=0)
                    src = np.array([
                        src_mean[0], src_mean[1],
                        terrain.z(float(src_mean[0]), float(src_mean[1]))
                    ])
                    agents[third_id].source_est = src.copy()

                    print(
                        f"\n  ◆ DICT disambiguazione (3 droni) al passo {step}:"
                        f" cand_idx={chosen_idx}"
                        f"  d{dict_ids[0]}→({agents[dict_ids[0]].source_est[0]:.1f},{agents[dict_ids[0]].source_est[1]:.1f})"
                        f"  d{dict_ids[1]}→({agents[dict_ids[1]].source_est[0]:.1f},{agents[dict_ids[1]].source_est[1]:.1f})"
                        f" — avvio XY consensus ({DICT_XY_ITERS} iter)"
                    )

                    # Tutti in hover per il consenso XY
                    for i in dict_xy_active_ids:
                        if agents[i].state != DroneState.STOP:
                            agents[i].state     = DroneState.STOP
                            agents[i].waypoints = [agents[i].x_est[:3].copy()]
                            agents[i].wp_idx    = 0

                else:
                    dict_disam_mode = '2drone'
                    d0_, d1_ = dict_ids
                    cands_ = agents[d0_].source_cands
                    print(f"\n  ◆ DICT disambiguazione (2 droni) al passo {step}: droni → c1, c2")
                    print(f"    source_cands di d{d0_}: {[f'({c[0]:.1f},{c[1]:.1f})' for c in cands_]}")
                    print(f"    pos d{d0_}={agents[d0_].x_est[:2].round(1)}  stato={agents[d0_].state.name}")
                    print(f"    pos d{d1_}={agents[d1_].x_est[:2].round(1)}  stato={agents[d1_].state.name}")
                    if len(cands_) >= 2:
                        dist_cands = np.linalg.norm(np.array(cands_[0][:2]) - np.array(cands_[1][:2]))
                        print(f"    distanza tra candidati: {dist_cands:.2f} m")
                    _dict_setup_2drone_disam(agents, dict_ids, terrain, agl)

        # ── 2b3. All-stop: tutti i droni in STOP → depth-circle ─────────────
        if (
            dict_disam_mode not in ('3drone_xy', '3drone', 'all_stop')
            and all(agents[i].state == DroneState.STOP for i in drone_ids)
        ):
            src_estimates = [
                agents[i].source_est if agents[i].source_est is not None
                else agents[i].x_est[:3]
                for i in drone_ids
            ]
            src = np.mean(src_estimates, axis=0).copy()
            src[2] = terrain.z(float(src[0]), float(src[1]))
            for i in drone_ids:
                agents[i].source_est = src.copy()
            print(
                f"\n  ◆ Tutti i droni in STOP al passo {step}"
                f" (t={t:.2f}s): avvio depth-circle (r={CONVERGE_RADIUS}m)"
            )
            _dict_setup_depth_circle(agents, list(drone_ids), src, terrain, agl)
            dict_disam_mode = 'all_stop'
            dict_disam_done = True

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

        # ── 6. DICT ──────────────────────────────────────────────────────
        if len(dict_ids) == 2 and not dict_disam_done:
            # Adapt+Combine mentre entrambi sono ancora in TRACK
            both_track = (
                agents[dict_ids[0]].state == DroneState.TRACK and
                agents[dict_ids[1]].state == DroneState.TRACK
            )
            if both_track:
                _dict_step(agents, dict_ids, terrain)
                for i in dict_ids:
                    if agents[i].source_cands:
                        agents[i].source_cands_log.append(
                            (step, [c.copy() for c in agents[i].source_cands])
                        )

        # Disambiguazione 2 droni: confronto segnali a destinazione
        if dict_disam_mode == '2drone' and dict_disam_done:
            d0, d1 = dict_ids
            d0_dist = np.linalg.norm(agents[d0].x_est[:3] - agents[d0].current_target())
            d1_dist = np.linalg.norm(agents[d1].x_est[:3] - agents[d1].current_target())
            d0_at = agents[d0].state == DroneState.SUPPORT and d0_dist < STOP_THRESH
            d1_at = agents[d1].state == DroneState.SUPPORT and d1_dist < STOP_THRESH
            if step % 20 == 0:
                print(
                    f"  [2drone step={step}]"
                    f"  d{d0}: {agents[d0].state.name} dist={d0_dist:.2f}m tgt={agents[d0].current_target()[:2].round(1)} {'✓' if d0_at else '✗'}"
                    f"  |  d{d1}: {agents[d1].state.name} dist={d1_dist:.2f}m tgt={agents[d1].current_target()[:2].round(1)} {'✓' if d1_at else '✗'}"
                )
            if d0_at and d1_at:
                s0 = agents[d0].signal_log[-1][1] if agents[d0].signal_log else 0.0
                s1 = agents[d1].signal_log[-1][1] if agents[d1].signal_log else 0.0
                winner = d0 if s0 >= s1 else d1
                loser  = d1 if winner == d0 else d0

                # Winner: mantiene il proprio source_est (già il candidato corretto)
                # Loser: prende il proprio source_cands più vicino al winner
                winner_pos = agents[winner].source_est[:2] \
                    if agents[winner].source_est is not None \
                    else agents[winner].x_est[:2]
                loser_cands = agents[loser].source_cands
                if len(loser_cands) >= 2:
                    d0c = np.linalg.norm(np.asarray(loser_cands[0][:2]) - winner_pos)
                    d1c = np.linalg.norm(np.asarray(loser_cands[1][:2]) - winner_pos)
                    c = loser_cands[0] if d0c <= d1c else loser_cands[1]
                    agents[loser].source_est = np.array([
                        c[0], c[1], terrain.z(float(c[0]), float(c[1]))
                    ])
                elif loser_cands:
                    c = loser_cands[0]
                    agents[loser].source_est = np.array([
                        c[0], c[1], terrain.z(float(c[0]), float(c[1]))
                    ])

                # Log 2→1 candidato per entrambi (animazione)
                for i in dict_ids:
                    agents[i].source_cands = [agents[i].source_est.copy()]
                    agents[i].source_cands_log.append((step, [agents[i].source_est.copy()]))
                print(
                    f"\n  ◆ DICT 2-drone: drone {winner} segnale più forte."
                    f"  d{winner}→({agents[winner].source_est[0]:.1f},{agents[winner].source_est[1]:.1f})"
                    f"  d{loser}→({agents[loser].source_est[0]:.1f},{agents[loser].source_est[1]:.1f})"
                )
                dict_disam_mode = '2drone_done'

        # XY consensus a 3 droni (dopo disambiguazione, prima del depth-circle)
        if dict_disam_mode == '3drone_xy':
            _dict_step_3drone(agents, dict_xy_active_ids, terrain)
            dict_xy_ctr += 1
            if dict_xy_ctr >= DICT_XY_ITERS:
                # Centro del depth-circle = media delle stime individuali
                src_xy = np.mean(
                    [agents[i].source_est[:2] for i in dict_xy_active_ids
                     if agents[i].source_est is not None], axis=0
                )
                circle_center = np.array([
                    src_xy[0], src_xy[1],
                    terrain.z(float(src_xy[0]), float(src_xy[1]))
                ])
                # Ogni drone mantiene il proprio source_est raffinato dal consenso XY
                ids_with_est = [i for i in dict_xy_active_ids if agents[i].source_est is not None]
                print(
                    f"\n  ◆ DICT 3-drone XY consensus completato al passo {step}:"
                    f" centro depth-circle → ({circle_center[0]:.1f}, {circle_center[1]:.1f})"
                )
                for i in ids_with_est:
                    e = agents[i].source_est
                    print(f"    Drone {i}: source_est=({e[0]:.1f},{e[1]:.1f})")
                _dict_setup_depth_circle(agents, dict_xy_active_ids, circle_center, terrain, agl)
                dict_disam_mode = '3drone'

        # DCGD + Depth estimation — droni in cerchio a 120°
        if dict_disam_mode in ('3drone', 'all_stop'):
            support_ids = [i for i in drone_ids if agents[i].state == DroneState.SUPPORT]
            if support_ids:
                all_in_pos = all(
                    np.linalg.norm(agents[i].x_est[:3] - agents[i].current_target()) < STOP_THRESH
                    for i in support_ids
                )
                if all_in_pos:
                    if not dcgd_done:
                        # Fase 1: DCGD — Adapt+Combine + spostamento fisico dei droni
                        _dcgd_step(agents, support_ids, terrain, agl, DCGD_STEP_XY)
                        dcgd_ctr += 1
                        # I droni seguono subito la correzione: aggiorna i waypoint
                        src_xy = np.mean(
                            [agents[i].source_est[:2] for i in support_ids
                             if agents[i].source_est is not None], axis=0
                        )
                        new_center = np.array([
                            src_xy[0], src_xy[1],
                            terrain.z(float(src_xy[0]), float(src_xy[1]))
                        ])
                        _update_orbit_center(agents, support_ids, new_center, terrain, agl)
                        if dcgd_ctr % 20 == 0:
                            print(
                                f"  [DCGD iter={dcgd_ctr}/{DCGD_ITERS}]"
                                f" est=({src_xy[0]:.1f}, {src_xy[1]:.1f})"
                            )
                        if dcgd_ctr >= DCGD_ITERS:
                            dcgd_done = True
                            print(
                                f"\n  ◆ DCGD completato ({DCGD_ITERS} iter) al passo {step}:"
                                f" centro finale → ({src_xy[0]:.1f}, {src_xy[1]:.1f})"
                            )
                    else:
                        # Fase 2: stima profondità dalla posizione raffinata dal DCGD
                        _dict_depth_step(agents, support_ids, agl)
                        dict_depth_ctr += 1
                        for i in support_ids:
                            if agents[i].depth_est is not None:
                                agents[i].depth_est_log.append((step, agents[i].depth_est))

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
        if dict_disam_mode in ('3drone', 'all_stop') and dict_depth_ctr >= DICT_DEPTH_ITERS:
            print(
                f"\n  ✔ Depth consensus completato al passo {step+1}"
                f" (t={(step+1)*dt:.2f}s)"
            )
            break

        if dict_disam_mode == '2drone_done':
            print(
                f"\n  ✔ DICT 2-drone disambiguazione completata al passo {step+1}"
                f" (t={(step+1)*dt:.2f}s)"
            )
            break

    # ── Report finale ──────────────────────────────────────────────────────
    est      = triangulate_victim(agents, artva)
    err_xy   = np.linalg.norm(est[:2] - artva.position[:2])
    z_true   = artva.position[2]
    depth_true  = terrain.z(artva.position[0], artva.position[1]) - z_true

    # Depth estimate: usa depth_est se disponibile, altrimenti stima da source_est[2]
    depth_ests = [ag.depth_est for ag in agents.values() if ag.depth_est is not None]
    if depth_ests:
        depth_est_val = float(np.mean(depth_ests))
    else:
        depth_est_val = terrain.z(est[0], est[1]) - est[2]
    err_depth = abs(depth_est_val - depth_true)

    print(f"\n  Posizione vittima reale  : {artva.position.round(2)}")
    print(f"  Stima DICT distribuita   : {est.round(2)}")
    print(f"  Errore planimetrico      : {err_xy:.2f} m")
    print(f"  Profondità reale         : {depth_true:.2f} m")
    print(f"  Profondità stimata DICT  : {depth_est_val:.2f} m  (errore {err_depth:.2f} m)")

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

    print("\n  Errore di landing per drone:")
    for i in drone_ids:
        ag = agents[i]
        if ag.source_est is None:
            print(f"    Drone {i}: n/a (nessuna stima)")
            continue
        landing_point = ag.source_est - (ag.x_est[:3] - ag.x[:3])
        landing_err   = landing_point[:2] - artva.position[:2]
        print(
            f"    Drone {i}: {np.linalg.norm(landing_err):.3f} m  "
            f"(Δx={landing_err[0]:+.2f}  Δy={landing_err[1]:+.2f})"
        )

    return agents, consensus_events, artva_detect_thr, track_stop_thr, dict_ids, dict_disam_mode

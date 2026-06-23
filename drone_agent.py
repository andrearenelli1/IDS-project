"""
drone_agent.py
==============
Definizioni dell'agente drone e delle funzioni di navigazione locale.

Contenuto
---------
  DroneState           — FSM a 4 stati: SEARCH / TRACK / STOP / SUPPORT
  DroneAgent           — dataclass con stato reale, filtro IMDCL, navigazione
  lawnmower_waypoints  — pattern a greca per la fase SEARCH
  rotate_2d            — utility: ruota vettore 2D
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from imdcl import AgentIMDCL
from mpc_drone import DroneMPC
from terrain import Terrain
from config import (
    AGL_HEIGHT, LANE_SPACING, STOP_THRESH,
    TAU_FILTER_ARTVA, DT_MPC, N_SIGNAL_SAMPLES,
    SUPPORT_CIRCLE_N,
)

from pf import ParticleFilter

# ============================================================================
# FSM
# ============================================================================

class DroneState(IntEnum):
    SEARCH  = 0   # lawnmower, ricerca attiva
    TRACK   = 1   # esplorazione 3 punti, convergenza sulla sorgente
    STOP    = 2   # hovering, ha raggiunto la soglia TRACK_STOP_THR
    SUPPORT = 3   # si posiziona a 120° attorno alla stima della sorgente per triangolazione


# ============================================================================
# Waypoint generators
# ============================================================================

def lawnmower_waypoints(
    drone_id: int,
    n_drones: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    terrain: Terrain,
    lane_spacing: float = LANE_SPACING,
    agl: float = AGL_HEIGHT,
) -> List[np.ndarray]:
    """
    Genera la sequenza di waypoint della greca per il drone 'drone_id'.

    Il workspace è diviso in N_DRONES strisce verticali (lungo x).
    Ogni drone percorre le corsie orizzontali (lungo y) alternando il verso.
    I waypoint includono la quota z = terrain(x, y) + agl.
    """
    width      = (x_max - x_min) / n_drones
    x0_s       = x_min + drone_id * width
    x1_s       = x0_s + width
    x_positions = np.arange(x0_s + lane_spacing / 2, x1_s, lane_spacing)
    if len(x_positions) == 0:
        x_positions = np.array([(x0_s + x1_s) / 2])

    waypoints: List[np.ndarray] = []
    go_up = True
    for x in x_positions:
        y_start, y_end = (y_min, y_max) if go_up else (y_max, y_min)
        for y in (y_start, y_end):
            waypoints.append(np.array([x, y, terrain.agl_z(x, y, agl)]))
        go_up = not go_up
    return waypoints


def single_lane_waypoints(
    x:       float,
    go_up:   bool,
    y_min:   float,
    y_max:   float,
    terrain: Terrain,
    agl:     float = AGL_HEIGHT,
) -> List[np.ndarray]:
    """Genera i 2 waypoint di una singola corsia verticale (y_min↔y_max a x fisso)."""
    y_start, y_end = (y_min, y_max) if go_up else (y_max, y_min)
    return [
        np.array([x, y_start, terrain.agl_z(x, y_start, agl)]),
        np.array([x, y_end,   terrain.agl_z(x, y_end,   agl)]),
    ]


# ============================================================================
# Navigation utilities
# ============================================================================

def rotate_2d(v: np.ndarray, deg: float) -> np.ndarray:
    """Ruota il vettore 2D v di 'deg' gradi in senso antiorario."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def circle_waypoints(
    center:      np.ndarray,
    radius:      float,
    start_angle: float,
    clockwise:   bool,
    terrain:     "Terrain",
    n_pts:       int   = SUPPORT_CIRCLE_N,
    agl:         float = AGL_HEIGHT,
) -> List[np.ndarray]:
    """
    Genera n_pts waypoint equidistanti su una circonferenza di dato raggio
    attorno a center, a quota AGL costante.
    """
    sign   = -1.0 if clockwise else +1.0
    angles = start_angle + sign * np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    wps    = []
    for a in angles:
        x = center[0] + radius * np.cos(a)
        y = center[1] + radius * np.sin(a)
        wps.append(np.array([x, y, terrain.agl_z(x, y, agl)]))
    return wps


# ============================================================================
# DroneAgent
# ============================================================================

@dataclass
class DroneAgent:
    """
    Drone con FSM a 4 stati, controllore MPC e filtro IMDCL.

    Waypoint management
    -------------------
    Tutti gli stati sono arrival-gated: il prossimo waypoint viene calcolato
    solo quando quello corrente è raggiunto (o su transizione di stato).
    Il dispatch avviene in simulation.py tramite _on_wp_reached().

    Stima sorgente: Particle Filter
    --------------------------------
    source_est : stima corrente della posizione della sorgente (media pesata PF)
    """

    id:           int
    x:            np.ndarray          # stato reale (6,)
    waypoints:    List[np.ndarray]    # waypoint correnti
    ctrl:         DroneMPC
    imdcl:        AgentIMDCL

    state:        DroneState = DroneState.SEARCH
    wp_idx:       int        = 0
    signal_log:   list       = field(default_factory=list)
    history:      list       = field(default_factory=list)
    state_log:    list       = field(default_factory=list)   # DroneState ad ogni step (parallelo a history)
    est_history:  list       = field(default_factory=list)
    input_log:    list       = field(default_factory=list)
    solve_t_log:  list       = field(default_factory=list)
    wp_target_log: list      = field(default_factory=list)   # current_target() ad ogni step
    detected:     bool       = False

    # SOURCE ESTIMATION (particle filter)
    pf:                Optional[ParticleFilter] = None
    artva_sigma_noise: float                    = 0.0
    source_est:        Optional[np.ndarray]     = None
    source_est_std:    Optional[np.ndarray]     = None  # deviazione std pesata su (x,y,z) alla fine
    source_est_log:    list                     = field(default_factory=list)
    pf_log:            list                     = field(default_factory=list)  # (particles, weights) per step

    # ARTVA distance estimate
    r:     Optional[float] = None
    r_log: list            = field(default_factory=list)

    # SEARCH — stato lawnmower a corsie globali
    lane_xs:          np.ndarray = field(default_factory=lambda: np.array([]))
    current_lane_idx: int        = 0
    n_drones_total:   int        = 1
    lane_go_up:       bool       = True

    # TRACK-ES — stato interno Extremum Seeking  [Azzollini et al. 2021, eq. 11-13]
    es_x_ref: float = 0.0   # riferimento x corrente generato dall'ES [m]
    es_y_ref: float = 0.0   # riferimento y corrente generato dall'ES [m]
    es_alpha: float = 0.0   # valore corrente di α (rampa 0 → ES_ALPHA_MAX)
    es_time:  float = 0.0   # tempo interno ES [s]

    # ES history — per calcolare la media su un ciclo completo (usata da step 2d)
    es_x_hist: list = field(default_factory=list)
    es_y_hist: list = field(default_factory=list)
    es_active: bool = False   # True dopo init_es: ES gestisce i waypoint del drone

    # SUPPORT — orbita cooperativa
    support_center:       Optional[np.ndarray] = None
    support_orbit_radius: float = 0.0
    support_cw:           bool  = False
    support_pending:      bool  = False   # True se aspetta ancora partner
    support_deadline:     int   = 0       # step oltre cui si rinuncia
    support_n_needed:     int   = 0       # partner ancora mancanti


    # ARTVA signal
    sig_filt: Optional[float] = None
    sig_raw_last: float = 0.0
    sig_batch:  deque = field(default_factory=lambda: deque(maxlen=N_SIGNAL_SAMPLES))

    # ── Proprietà ──────────────────────────────────────────────────────────

    @property
    def x_est(self) -> np.ndarray:
        """Stima corrente dello stato da IMDCL (usata dal controllore MPC)."""
        return self.imdcl.x_hat

    # ── Waypoint management ────────────────────────────────────────────────

    def current_target(self) -> np.ndarray:
        idx = min(self.wp_idx, len(self.waypoints) - 1)
        return self.waypoints[idx]

    def advance_waypoint(self) -> bool:
        """Avanza al prossimo waypoint. Restituisce True se ce ne sono altri."""
        if self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            return True
        return False

    def all_waypoints_done(self) -> bool:
        return (
            self.wp_idx >= len(self.waypoints) - 1
            and np.linalg.norm(self.x_est[:3] - self.current_target()) < STOP_THRESH
        )

    def update_signal_filter_batch(self, sig: float) -> float:
        """
        Aggiorna sig_batch con il nuovo campione grezzo e restituisce
        la media mobile sugli ultimi N_SIGNAL_SAMPLES valori.
        """
        self.sig_batch.append(sig)
        return float(np.mean(self.sig_batch))

    def init_es(self, terrain: Terrain, agl: float) -> None:
        """
        Inizializza lo stato ES (TRACK o SUPPORT).
        Il riferimento parte dalla posizione stimata corrente;
        α parte da 0 e rampa verso ES_ALPHA_MAX secondo eq. 13.
        """
        self.es_x_ref  = float(self.x_est[0])
        self.es_y_ref  = float(self.x_est[1])
        self.es_alpha  = 0.0
        self.es_time   = 0.0
        self.es_active = True
        z = terrain.agl_z(self.es_x_ref, self.es_y_ref, agl)
        self.waypoints = [np.array([self.es_x_ref, self.es_y_ref, z])]
        self.wp_idx    = 0

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
  track_next_waypoint  — calcola waypoint nella direzione data (usato da TRACK)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from imdcl import AgentIMDCL
from mpc_drone import DroneMPC
from terrain import Terrain
from config import (
    AGL_HEIGHT, LANE_SPACING, STOP_THRESH,
    TRACK_STEP_M, SUPPORT_CIRCLE_N,
)


# ============================================================================
# FSM
# ============================================================================

class DroneState(IntEnum):
    SEARCH  = 0   # lawnmower, ricerca attiva
    TRACK   = 1   # esplorazione 3 punti, convergenza sulla sorgente
    STOP    = 2   # hovering, ha raggiunto la soglia TRACK_STOP_THR
    SUPPORT = 3   # percorre cerchio intorno al drone STOP per triangolazione


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


def circle_waypoints(
    center:    np.ndarray,
    radius:    float,
    start_angle: float,
    clockwise: bool,
    terrain:   Terrain,
    n_pts:     int   = SUPPORT_CIRCLE_N,
    agl:       float = AGL_HEIGHT,
) -> List[np.ndarray]:
    """
    Genera n_pts waypoint equidistribuiti su una circonferenza.
    clockwise=True → senso orario (angoli decrescenti).
    Il primo punto è all'angolo start_angle.
    """
    sign   = -1.0 if clockwise else +1.0
    angles = start_angle + sign * np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    wps: List[np.ndarray] = []
    for a in angles:
        x = center[0] + radius * np.cos(a)
        y = center[1] + radius * np.sin(a)
        wps.append(np.array([x, y, terrain.agl_z(x, y, agl)]))
    return wps


# ============================================================================
# Navigation utilities
# ============================================================================

def rotate_2d(v: np.ndarray, deg: float) -> np.ndarray:
    """Ruota il vettore 2D v di 'deg' gradi in senso antiorario."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def track_next_waypoint(
    current_pos_est: np.ndarray,
    direction:       np.ndarray,
    terrain:         Terrain,
    step_m:          float = TRACK_STEP_M,
    agl:             float = AGL_HEIGHT,
) -> np.ndarray:
    """Calcola il prossimo waypoint TRACK nella direzione data."""
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
    next_xy = current_pos_est[:2] + step_m * direction
    return np.array([next_xy[0], next_xy[1], terrain.agl_z(next_xy[0], next_xy[1], agl)])


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

    Campi TRACK
    -----------
    track_dir          : direzione di ricerca corrente (2,), unitaria
    track_start_pos    : posizione (3,) all'inizio del round corrente
    track_start_signal : segnale ARTVA al punto di partenza del round
    track_candidates   : lista di 3 waypoint candidati del round corrente
    track_cand_signals : segnale misurato a ciascun candidato (NaN = non visitato)
    track_cand_idx     : 0-2 = visitando candidato i; 3 = tornando al migliore

    Campi SUPPORT
    -------------
    support_center   : posizione (3,) del drone STOP che ha chiamato il supporto
    support_radius   : raggio della circonferenza da percorrere [m]
    support_cw       : True = senso orario, False = antiorario
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
    est_history:  list       = field(default_factory=list)
    input_log:    list       = field(default_factory=list)
    solve_t_log:  list       = field(default_factory=list)
    wp_target_log: list      = field(default_factory=list)   # current_target() ad ogni step
    detected:     bool       = False

    # DCGD
    source_est:   Optional[np.ndarray] = None

    # TRACK
    track_dir:          Optional[np.ndarray] = None
    track_start_pos:    Optional[np.ndarray] = None
    track_start_signal: float                = 0.0
    track_candidates:   List[np.ndarray]     = field(default_factory=list)
    track_cand_signals: List[float]          = field(default_factory=list)
    track_cand_idx:     int                  = 0
    track_time:         int                  = 0

    # SUPPORT
    support_center:  Optional[np.ndarray] = None
    support_radius:  float                = 0.0
    support_cw:      bool                 = False

    # SUPPORT — ricerca partner pendente
    support_orbit_radius: float = 0.0   # raggio cerchio memorizzato per retry
    support_pending:      bool  = False  # True se si attendono ancora partner
    support_deadline:     int   = 0      # step entro cui chiudere la ricerca
    support_n_needed:     int   = 0      # partner SUPPORT ancora mancanti

    # TRACK-ES — stato interno Extremum Seeking  [Azzollini et al. 2021, eq. 11-13]
    es_x_ref: float = 0.0   # riferimento x corrente generato dall'ES [m]
    es_y_ref: float = 0.0   # riferimento y corrente generato dall'ES [m]
    es_alpha: float = 0.0   # valore corrente di α (rampa 0 → ES_ALPHA_MAX)
    es_time:  float = 0.0   # tempo interno ES [s]

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

    # ── TRACK helpers ─────────────────────────────────────────────────────

    def init_track_dir(self, lawnmower_wps: List[np.ndarray]) -> None:
        """
        Inizializza track_dir al primo rilevamento ARTVA.
        Usa il momentum del lawnmower (o velocità stimata come fallback).
        """
        next_idx = min(self.wp_idx + 1, len(lawnmower_wps) - 1)
        delta    = lawnmower_wps[next_idx][:2] - self.x_est[:2]
        norm     = np.linalg.norm(delta)
        if norm > 1e-3:
            self.track_dir = delta / norm
            return
        v  = self.x_est[3:5]
        nv = np.linalg.norm(v)
        self.track_dir = v / nv if nv > 1e-3 else np.array([1.0, 0.0])

    def init_track_round(self, signal: float, terrain: Terrain, agl: float) -> None:
        """
        Inizia un nuovo round di esplorazione 3 punti dalla posizione stimata corrente.
        Calcola i 3 candidati (avanti, +60°, -60°) e imposta il primo waypoint.
        """
        d = self.track_dir
        dirs = [d, rotate_2d(d, 60.0), rotate_2d(d, -60.0)]
        self.track_start_pos    = self.x_est[:3].copy()
        self.track_start_signal = signal
        self.track_candidates   = [
            track_next_waypoint(self.x_est, di, terrain, step_m=TRACK_STEP_M, agl=agl)
            for di in dirs
        ]
        self.track_cand_signals = [float('nan')] * 3
        self.track_cand_idx     = 0
        self.waypoints = [self.track_candidates[0]]
        self.wp_idx    = 0

    def init_es(self, terrain: Terrain, agl: float) -> None:
        """
        Inizializza lo stato ES alla transizione SEARCH → TRACK.
        Il riferimento parte dalla posizione stimata corrente;
        α parte da 0 e rampa verso ES_ALPHA_MAX secondo eq. 13.
        """
        self.es_x_ref = float(self.x_est[0])
        self.es_y_ref = float(self.x_est[1])
        self.es_alpha = 0.0
        self.es_time  = 0.0
        z = terrain.agl_z(self.es_x_ref, self.es_y_ref, agl)
        self.waypoints = [np.array([self.es_x_ref, self.es_y_ref, z])]
        self.wp_idx    = 0

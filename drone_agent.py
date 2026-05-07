"""
drone_agent.py
==============
Definizioni dell'agente drone e delle funzioni di navigazione locale.

Contenuto
---------
  DroneState        — enumerazione stati FSM (SEARCH / TRACK)
  DroneAgent        — dataclass con stato reale, filtro IMDCL, hill-climbing
  lawnmower_waypoints  — generatore pattern a greca per la fase SEARCH
  _rotate_2d           — utility: ruota vettore 2D
  track_next_waypoint  — calcola il prossimo waypoint TRACK (hill-climbing)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from imdcl import AgentIMDCL
from mpc_drone import DroneMPC
from terrain import Terrain
from config import (
    AGL_HEIGHT, LANE_SPACING, STOP_THRESH,
    TRACK_STEP_M, TRACK_TURN_DEG,
)


# ============================================================================
# FSM
# ============================================================================

class DroneState(IntEnum):
    SEARCH = 0   # greca lawnmower, ricerca attiva
    TRACK  = 1   # hill-climbing verso la sorgente ARTVA


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

    Returns
    -------
    waypoints : lista di np.array([x, y, z])
    """
    width     = (x_max - x_min) / n_drones
    x0_s      = x_min + drone_id * width
    x1_s      = x0_s + width

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


# ============================================================================
# Hill-climbing utilities
# ============================================================================

def _rotate_2d(v: np.ndarray, deg: float) -> np.ndarray:
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
    """
    Calcola il prossimo waypoint TRACK dato la posizione stimata corrente
    e la direzione di ricerca corrente (vettore 2D unitario).

    Non accede mai al campo ARTVA: la direzione è aggiornata reattivamente
    dal drone in base alle misure reali già raccolte.

    Returns
    -------
    wp : np.array([x, y, z])
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])

    next_xy = current_pos_est[:2] + step_m * direction
    z_next  = terrain.agl_z(next_xy[0], next_xy[1], agl)
    return np.array([next_xy[0], next_xy[1], z_next])


# ============================================================================
# DroneAgent
# ============================================================================

@dataclass
class DroneAgent:
    """
    Un drone con macchina a stati finiti, controllore MPC e filtro IMDCL.

    Attributi principali
    --------------------
    id           : identificatore intero
    state        : DroneState (SEARCH / TRACK)
    x            : stato **reale** (6,) [px, py, pz, vx, vy, vz]
    imdcl        : AgentIMDCL — stima decentralizzata; MPC usa imdcl.x_hat
    waypoints    : lista waypoint correnti (lawnmower oppure TRACK)
    wp_idx       : indice waypoint attuale
    signal_log   : misure ARTVA raccolte [(pos, signal), ...]
    history      : stati reali loggati ad ogni passo
    est_history  : stime IMDCL (x_hat) loggata ad ogni passo

    Campi hill-climbing (stato TRACK)
    ----------------------------------
    track_dir         : direzione di ricerca corrente (2,), unitaria
    track_signal_prev : segnale ARTVA all'ultimo passo TRACK
    track_turn_sign   : +1/-1 — alterna L/R ad ogni fallimento
    track_fail_count  : passi consecutivi con segnale calante
    track_stopped     : True quando segnale ≥ TRACK_STOP_THR → drone in hovering
    track_time        : contatore passi in stato TRACK (usato per diminuire lo step del gradiente nel tempo ed evitare oscillazioni)
    source_est        : stima locale (3,) della posizione sorgente ARTVA (DCGD)
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
    detected:     bool       = False
    estimate_pos: Optional[np.ndarray] = None

    # Hill-climbing state
    track_dir:          Optional[np.ndarray] = None
    track_signal_prev:  float                = 0.0
    track_turn_sign:    int                  = +1
    track_fail_count:   int                  = 0
    track_time:         int                  = 0

    # Stopping e stima distribuita
    track_stopped:      bool                 = False
    source_est:         Optional[np.ndarray] = None  # stima locale [x,y,z] sorgente

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

    # ── Hill-climbing ──────────────────────────────────────────────────────

    def init_track_dir(self, lawnmower_wps: List[np.ndarray]) -> None:
        """
        Inizializza la direzione di ricerca al momento del primo rilevamento.

        Droni di supporto (waypoints = [support_wp]): puntano verso il loro
        unico waypoint, già orientato verso la zona della vittima.
        Droni normali: usano il momentum del lawnmower verso il prossimo wp.
        Fallback: velocità stimata, o asse Est.
        """
        if len(self.waypoints) == 1:
            delta = self.waypoints[0][:2] - self.x_est[:2]
            norm  = np.linalg.norm(delta)
            if norm > 1e-3:
                self.track_dir = delta / norm
                return

        next_idx  = min(self.wp_idx + 1, len(lawnmower_wps) - 1)
        delta     = lawnmower_wps[next_idx][:2] - self.x_est[:2]
        norm      = np.linalg.norm(delta)
        if norm > 1e-3:
            self.track_dir = delta / norm
        else:
            v  = self.x_est[3:5]
            nv = np.linalg.norm(v)
            self.track_dir = v / nv if nv > 1e-3 else np.array([1.0, 0.0])

    def update_track_dir(self, signal_new: float) -> None:
        """
        Aggiorna la direzione di ricerca confrontando il segnale corrente con
        quello del passo precedente (hill-climbing reattivo).

          signal_new >= signal_prev → continua, azzera fail_count
          signal_new <  signal_prev → ruota di TRACK_TURN_DEG (con escalation),
                                      alterna L/R ad ogni fallimento
        """
        if self.track_dir is None:
            return
        if signal_new >= self.track_signal_prev:
            self.track_fail_count = 0
        else:
            self.track_fail_count += 1
            angle = (
                min(TRACK_TURN_DEG * self.track_fail_count, 90.0)
                * self.track_turn_sign
            )
            self.track_dir       = _rotate_2d(self.track_dir, angle)
            self.track_turn_sign = -self.track_turn_sign
        self.track_signal_prev = signal_new

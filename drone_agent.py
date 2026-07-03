"""
drone_agent.py
==============
Drone agent definitions and local navigation functions.

Contents
--------
  DroneState           — 5-state FSM: SEARCH / TRACK / STOP / SUPPORT / FINAL_ORBIT
  DroneAgent           — dataclass with real state, IMDCL filter, navigation
  lawnmower_waypoints  — lawnmower pattern for the SEARCH phase
  rotate_2d            — utility: rotates a 2D vector
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
    SEARCH      = 0   # lawnmower, active search
    TRACK       = 1   # 3-point exploration, convergence towards source
    STOP        = 2   # hovering, TRACK_STOP_THR threshold reached
    SUPPORT     = 3   # positions at 120° around source estimate for triangulation
    FINAL_ORBIT = 4   # final refinement orbit around estimate, before stop


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
    Generates the lawnmower waypoint sequence for drone 'drone_id'.

    The workspace is divided into N_DRONES vertical strips (along x).
    Each drone traverses horizontal lanes (along y) alternating direction.
    Waypoints include altitude z = terrain(x, y) + agl.
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
    """Generates the 2 waypoints of a single vertical lane (y_min↔y_max at fixed x)."""
    y_start, y_end = (y_min, y_max) if go_up else (y_max, y_min)
    return [
        np.array([x, y_start, terrain.agl_z(x, y_start, agl)]),
        np.array([x, y_end,   terrain.agl_z(x, y_end,   agl)]),
    ]


# ============================================================================
# Navigation utilities
# ============================================================================

def rotate_2d(v: np.ndarray, deg: float) -> np.ndarray:
    """Rotates the 2D vector v by 'deg' degrees counter-clockwise."""
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
    Generates n_pts equally spaced waypoints on a circle of given radius
    around center, at constant AGL altitude.
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
    Drone with a 5-state FSM, MPC controller and IMDCL filter.

    Waypoint management
    -------------------
    All states are arrival-gated: the next waypoint is computed
    only when the current one is reached (or on state transition).
    Dispatch happens in simulation.py via _on_wp_reached().

    Source estimation: Particle Filter
    -----------------------------------
    source_est : current estimate of source position (PF weighted mean)
    """

    id:           int
    x:            np.ndarray          # real state (6,)
    waypoints:    List[np.ndarray]    # current waypoints
    ctrl:         DroneMPC
    imdcl:        AgentIMDCL

    state:        DroneState = DroneState.SEARCH
    wp_idx:       int        = 0
    signal_log:   list       = field(default_factory=list)
    history:      list       = field(default_factory=list)
    state_log:    list       = field(default_factory=list)   # DroneState at each step (parallel to history)
    est_history:  list       = field(default_factory=list)
    input_log:    list       = field(default_factory=list)
    solve_t_log:  list       = field(default_factory=list)
    wp_target_log: list      = field(default_factory=list)   # current_target() at each step
    detected:     bool       = False

    # SOURCE ESTIMATION (particle filter)
    pf:                Optional[ParticleFilter] = None
    artva_sigma_noise: float                    = 0.0
    source_est:        Optional[np.ndarray]     = None
    source_est_std:    Optional[np.ndarray]     = None  # weighted std dev on (x,y,z) at end
    source_est_log:    list                     = field(default_factory=list)
    pf_log:            list                     = field(default_factory=list)  # (particles, weights) per step

    # ARTVA distance estimate
    r:     Optional[float] = None
    r_log: list            = field(default_factory=list)

    # SEARCH — global lane lawnmower state
    lane_xs:          np.ndarray = field(default_factory=lambda: np.array([]))
    current_lane_idx: int        = 0
    n_drones_total:   int        = 1
    lane_go_up:       bool       = True

    # TRACK-ES — Extremum Seeking internal state  [Azzollini et al. 2021, eq. 11-13]
    es_x_ref: float = 0.0   # current ES x reference [m]
    es_y_ref: float = 0.0   # current ES y reference [m]
    es_alpha: float = 0.0   # current α value (ramp 0 → ES_ALPHA_MAX)
    es_time:  float = 0.0   # ES internal time [s]

    # ES history — for computing average over a full cycle (used by step 2d)
    es_x_hist: list = field(default_factory=list)
    es_y_hist: list = field(default_factory=list)
    es_active: bool = False   # True after init_es: ES manages drone waypoints

    # SUPPORT — cooperative orbit
    support_center:       Optional[np.ndarray] = None
    support_orbit_radius: float = 0.0
    support_cw:           bool  = False
    support_pending:      bool  = False   # True while still waiting for partners
    support_deadline:     int   = 0       # step beyond which we give up
    support_n_needed:     int   = 0       # partners still missing
    support_failed:       bool  = False   # True if SUPPORT call expired without partners
    final_orbit_done:     bool  = False   # True when drone has reached last waypoint of final orbit


    # ARTVA signal
    sig_filt: Optional[float] = None
    sig_raw_last: float = 0.0
    sig_batch:  deque = field(default_factory=lambda: deque(maxlen=N_SIGNAL_SAMPLES))

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def x_est(self) -> np.ndarray:
        """Current state estimate from IMDCL (used by MPC controller)."""
        return self.imdcl.x_hat

    # ── Waypoint management ────────────────────────────────────────────────

    def current_target(self) -> np.ndarray:
        idx = min(self.wp_idx, len(self.waypoints) - 1)
        return self.waypoints[idx]

    def advance_waypoint(self) -> bool:
        """Advances to next waypoint. Returns True if there are more."""
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
        Updates sig_batch with the new raw sample and returns
        the moving average over the last N_SIGNAL_SAMPLES values.
        """
        self.sig_batch.append(sig)
        return float(np.mean(self.sig_batch))

    def init_es(self, terrain: Terrain, agl: float) -> None:
        """
        Initialises ES state (TRACK or SUPPORT).
        Reference starts from current estimated position;
        α starts at 0 and ramps towards ES_ALPHA_MAX per eq. 13.
        """
        self.es_x_ref  = float(self.x_est[0])
        self.es_y_ref  = float(self.x_est[1])
        self.es_alpha  = 0.0
        self.es_time   = 0.0
        self.es_active = True
        z = terrain.agl_z(self.es_x_ref, self.es_y_ref, agl)
        self.waypoints = [np.array([self.es_x_ref, self.es_y_ref, z])]
        self.wp_idx    = 0

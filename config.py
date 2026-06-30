"""
config.py
=========
Global parameters for the multi-drone avalanche search simulation.

All project modules import constants from here.
Modify this file to change execution parameters.
"""

import numpy as np

# ============================================================================
# Workspace
# ============================================================================
AREA_SIZE_M   = 200.0  # [m]  side of the square area extracted from the DEM

# ============================================================================
# Flight
# ============================================================================
AGL_HEIGHT    = 1.5     # [m]  height above ground (Above Ground Level)
LIDAR_SIGMA   = 0.1    # [m]  standard deviation of Gaussian LiDAR noise

# ============================================================================
# Drones
# ============================================================================
N_DRONES      = 3       # number of drones (configurable via --n)
DEPLOY_OFFSET = 2.0     # [m]  lateral distance between drones at deployment

# ============================================================================
# Lawnmower (SEARCH phase)
# ============================================================================
LANE_SPACING  = 15.0    # [m]  spacing between passes in the sweep lane

# ============================================================================
# ARTVA source
# ============================================================================
ARTVA_MOMENT      = 1.0    # normalised magnetic moment [A·m²]
ARTVA_NOISE_STD   = 1e-6   # additive noise (~1% signal at 40 m)
VICTIM_XY         = None   # [m, m] xy position of the victim in the local workspace;
                            #        None = random (seed from CLI)
VICTIM_DEPTH      = 3      # [m]  burial depth below ground

# ============================================================================
# Noise calibration (pre-flight phase)
# ============================================================================
N_NOISE_CALIB_SAMPLES = 20     # measurements per drone to estimate σ_noise
NOISE_CONSENSUS_ITERS = 10     # average-consensus iterations between drones

# Maximum reliable range for ES — from Azzollini et al. (arXiv:2106.14514),
# Sec. IV-A SITL: initial drone-source distance ≈ 50 m, from which ES
# demonstrates convergence with α=20, κ=0.07, ω=0.65.
# Used as absolute floor for DETECT_THR: prevents very low noise from
# lowering the threshold to the point of triggering TRACK from the entire workspace.
ES_DETECT_MAX_R = 50.0   # [m]

# Radius within which the source estimate is considered "found".
# Azzollini et al. uses 5×5 m as practical bounding box for the rescuer;
# we use 10 m to account for simulation approximations.
FOUND_RADIUS = 10.0      # [m]

# Success criterion based on Particle Filter: a run succeeds if at least
# one drone with active PF has a 95% confidence ellipse (corrected for IMDCL
# drift) that CONTAINS the victim in the xy plane, and is sufficiently concentrated.
# Area threshold: equivalent circle of radius ~5 m → the rescuer searches
# a small area. Consistent with the practical bounding box of Azzollini et al.
FOUND_ELLIPSE_CONF     = 0.95               # ellipse confidence level
FOUND_ELLIPSE_AREA_MAX = np.pi * 5.0**2     # [m²] ≈ 78.5 (equivalent circle r≈5 m)

# Nominal values for visualisation — these are used only for plots.
# The simulation uses dynamically measured thresholds.
ARTVA_DETECT_THR = max(5 * ARTVA_NOISE_STD, ARTVA_MOMENT / ES_DETECT_MAX_R**3)  # ≈ 8e-6
TRACK_STOP_THR   = ARTVA_MOMENT / FOUND_RADIUS**3                                # = 1e-3

# exponential filter for ARTVA signal (to reduce noise effect in decisions)
TAU_FILTER_ARTVA = 0.5   # [s] time constant


# ============================================================================
# Extremum Seeking (ES) — TRACK mode  [Azzollini et al., 2021 — eq. 11-13]
#
#   ẋ_ref = √(α·ω) · cos(ω·t + κ·yt)
#   ẏ_ref = √(α·ω) · sin(ω·t + κ·yt)
#   ẏt    = (−1/λ)·α + (1/λ)·α_max     (α ramp from 0 → α_max)
#
#   yt = 1/∛S  — conditioned signal (min at source, convex)
#
# Velocity constraint: √(α_max · ω) ≤ V_MAX = 3.0 m/s  →  ω = V_MAX²/α_max
# ============================================================================
ES_ALPHA_MAX = 20.0    # [-]   maximum amplitude α (circle radius ≈ √(α/ω) m)
ES_OMEGA     = 0.45    # [rad/s] frequency: √(20·0.45) = 3.0 m/s = V_MAX  ✓
ES_KAPPA     = 0.05    # [-]   conditioned signal feedback gain
ES_LAMBDA    = 15.0    # [s]   time constant for α ramp (α → α_max in ~3λ s)
ES_EPS       = 1e-12   # [-]   floor to avoid 1/cbrt(0)


# ============================================================================
# MPC
# ============================================================================
DT_MPC  = 0.1    # [s]
N_MPC   = 20
A_MAX   = 6.0    # [m/s²]
V_MAX   = 3.0    # [m/s]

# ============================================================================
# IMDCL
# ============================================================================
IMDCL_SIGMA_ACC   = 0.15   # [m/s²] filter process noise (3× SIGMA_ACC_SIM — conservative filter)
IMDCL_P0_POS      = 0.5    # [m]    initial position std. dev.
IMDCL_P0_VEL      = 0.1    # [m/s]  initial velocity std. dev.
IMDCL_COMM_RADIUS = 80  # [m]    inter-drone communication radius
IMDCL_R_MEAS_STD  = 0.3    # [m]    relative position measurement std. dev.
IMDCL_PI_MAX_NORM = 1e4    # [a.u.] maximum Frobenius norm for Pi_jl; beyond this → reset to zero
IMDCL_R_LIDAR_STD = 0.05   # [m]    LiDAR altitude measurement std. dev.
IMDCL_H_LIDAR     = np.array([[0., 0., 1., 0., 0., 0.]])  # H for pz (1×6)

# ============================================================================
# Simulation
# ============================================================================
N_SIM         = 600     # maximum steps
N_STOP        = 3       # number of drones in STOP that triggers simulation termination
DT_SIM        = DT_MPC
N_SIGNAL_SAMPLES  = 5      # [-]  ARTVA measurements stored in drone for moving average
SIGMA_ACC_SIM = 0.05   # [m/s²] simulation acceleration noise (< IMDCL_SIGMA_ACC)
STOP_THRESH   = 0.3     # [m]  waypoint arrival threshold

# ============================================================================
# Particle Filter (source position estimation)
# ============================================================================
PF_N_PARTICLES = 300   # number of particles per drone

# ============================================================================
# SUPPORT — partner selection and cooperative orbit
# ============================================================================
SUPPORT_CIRCLE_N       = 9     # waypoints on the circle traveled by SUPPORT drones
TRIANGULATE_N_PARTNERS = 2     # drones called to support the first STOP
SUPPORT_SEARCH_TIMEOUT = 300  # [steps] max wait to find missing SUPPORT partners
CONSENSUS_K_MAX        = 10    # max min-consensus iterations (≥ estimated network diameter)

# ============================================================================
# Final refinement orbit
# ============================================================================
# Regardless of how the simulation is stopped (N_STOP, SUPPORT expired, or timeout),
# before stopping, all drones with active PF perform a quick orbit
# around the estimated position (static centre captured at the stop instant)
# to collect diverse views and refine the PF before delivering the estimate.
FINAL_ORBIT_RADIUS  = 10.0   # [m]  final orbit radius
FINAL_ORBIT_N_WAYPOINTS = 6  # waypoints on the circle: the orbit ends when
                              # every drone has reached the last waypoint

# ============================================================================
# Visualisation
# ============================================================================
ANIM_SPEED = 5.0   # animation speed multiplier (1.0 = real time)
COLORS = {i: c for i, c in enumerate([
    "#e63946", "#2a9d8f", "#e9c46a", "#a8dadc",
    "#f4a261", "#6a4c93", "#1982c4", "#8ac926",
])}
BG_DARK = "#0d1117"

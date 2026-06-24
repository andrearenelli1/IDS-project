# ARTVA Multi-Drone Avalanche Search Simulation

A Python simulation of cooperative avalanche victim search using multiple autonomous drones equipped with ARTVA (457 kHz magnetic beacon) sensors, flying over a real alpine DEM (TINItaly GeoTIFF).

The project combines:
- terrain-aware flight over a real DEM (TINItaly GeoTIFF, 10 m resolution),
- multi-agent mission logic (SEARCH → TRACK → STOP/SUPPORT),
- Model Predictive Control (MPC) for trajectory tracking,
- distributed cooperative localization (IMDCL, GPS-denied),
- Extremum Seeking (ES) source-seeking control (Azzollini et al., arXiv:2106.14514),
- 3-D cooperative Particle Filter for victim position estimation,
- parametric sweep with full result analysis.

## Mission flow

Each drone starts in **SEARCH** mode flying an adaptive lawnmower pattern (lane spacing
derived from the dynamic detection threshold to guarantee area coverage).

When the ARTVA signal exceeds the dynamically calibrated detection threshold
`τ_detect`, the drone switches to **TRACK** and runs the Extremum Seeking algorithm
to converge toward the signal maximum (victim location).

When signal strength exceeds the stop threshold `τ_stop`, the drone enters **STOP**,
hovers, and recruits two partner drones via min-consensus. The partners transition to
**SUPPORT** and orbit the Stop drone on a circle of radius `(moment/S)^(1/3)`,
one CW and one CCW, providing diverse angular measurements.

A cooperative 3-D Particle Filter runs on all drones that have detected the signal,
fusing measurements from neighbors within communication range. The mission succeeds
when ≥ 3 drones are in STOP **and** the mean PF estimate is within `FOUND_RADIUS`
(10 m) of the true victim.

## Main modules

| File | Role |
|---|---|
| `config.py` | Single source of truth for all parameters |
| `terrain.py` | DEM loading, extraction, interpolation, `Terrain` class |
| `artva.py` | ARTVA magnetic dipole signal model |
| `pf.py` | 3-D Particle Filter for source estimation |
| `drone_agent.py` | Drone FSM (SEARCH/TRACK/STOP/SUPPORT), lawnmower, ES navigation |
| `simulation.py` | Main multi-agent simulation loop (MPC + IMDCL + PF + mission logic) |
| `mpc_drone.py` | MPC controller and 3-D point-mass model |
| `imdcl.py` | Distributed cooperative Kalman filter |
| `visualization.py` | Static plots and 3D animations |
| `main.py` | CLI entry point for single-run simulation |
| `run_experiments.py` | Parametric sweep runner (multi-process, resume-capable) |
| `plot_results.py` | Result analysis and visualization from sweep CSV |

## Requirements

- Python 3.10+
- A local DEM file `w51065_s10.tif` in project root (TINItaly, not versioned)

```bash
pip install numpy scipy matplotlib casadi tifffile tqdm pillow
```

Download the DEM tile from https://tinitaly.pi.ingv.it/Download_Area1_1.html and
place it as `w51065_s10.tif` in the project root.

## Quick start

```bash
# Default simulation (config.py values)
python main.py

# Custom run with 3D animation
python main.py --n 3 --steps 600 --animate

# Replay a specific experiment from a sweep CSV row
python main.py --n 4 --noise 1e-6 --rc 80 --area 100 \
               --victim-x 45.0 --victim-y 80.0 --victim-depth 2.5 \
               --ws 0.60,0.45 --animate

# Full parametric sweep (parallel, with resume)
python run_experiments.py --workers 4 --out results.csv

# Plot sweep results
python plot_results.py results.csv
```

## CLI options — main.py

| Option | Default | Description |
|---|---|---|
| `--n` | config | Number of drones |
| `--agl` | config | Flight height above ground [m] |
| `--steps` | config | Simulation steps |
| `--animate` | off | Show 3D animation |
| `--save` | off | Save animation to disk |
| `--seed` | 42 | Random seed |
| `--noise` | config | ARTVA noise std (override, e.g. `1e-6`) |
| `--rc` | config | Communication radius [m] |
| `--area` | config | Workspace side [m] |
| `--victim-x` | random | Victim x coordinate in workspace [m] |
| `--victim-y` | random | Victim y coordinate [m] |
| `--victim-depth` | config | Burial depth [m] |
| `--ws` | `(0.60,0.58)` | DEM workspace center: `r,c` or `center` |
| `--save-figs` | off | Save PNG figures to `./figures/` |

## CLI options — run_experiments.py

```bash
python run_experiments.py --workers 4     # parallel workers
python run_experiments.py --workers 1 --verbose  # sequential, verbose
python run_experiments.py --dry-run       # show grid without running
python run_experiments.py --out my.csv    # custom output file
```

Sweep parameters are defined at the top of the file:
- `AREA_SIZES`, `N_DRONES_LIST`, `ARTVA_NOISE_STDS`, `ACC_SIM_LIST`, `COMM_RADII`, `WORKSPACE_CENTERS`

The sweep resumes automatically from existing CSV rows.

## Dynamic detection thresholds

Thresholds are not fixed constants. At startup each drone calibrates the local noise floor
from `N_NOISE_CALIB_SAMPLES = 20` measurements at its initial position, then the team
agrees on a shared estimate via average-consensus:

```
σ̂  = average-consensus(local noise samples)
τ_detect = max(mu_noise + 5 × σ̂,  ARTVA_MOMENT / ES_DETECT_MAX_R³)
τ_stop   = ARTVA_MOMENT / FOUND_RADIUS³
```

The physics floor (`ES_DETECT_MAX_R = 50 m`, from Azzollini et al. SITL) prevents very
low noise from triggering TRACK before the drone is within the ES convergence basin.
The lawnmower lane spacing is then recomputed from `τ_detect` to guarantee complete
area coverage.

## Particle Filter source estimation

Each drone initializes a 3-D Particle Filter (`pf.py`) when it first detects the signal.
Particles are drawn in polar coordinates from the ARTVA dipole model (distance from
the signal inversion formula, azimuth and elevation uniform), then converted to
Cartesian workspace coordinates.

At each step:
1. Weights updated with Gaussian likelihood using adaptive sigma `sqrt(σ_n² + (0.20·S)²)`,
   which prevents weight collapse when fusing measurements from drones at very different ranges.
2. Cooperative fusion: each drone also updates its PF with measurements from all neighbors
   within communication range that have an active PF.
3. Systematic resampling only when effective particle count `N_eff < N_p/2`.

Final source estimate: `source_est = Σ w_k · ξ_k` (weighted mean of particles).

## Found criterion

A run is `found = True` only when:
1. ≥ 3 drones reached STOP state, **and**
2. `est_error_2d < FOUND_RADIUS` (10 m) — the mean PF estimate is within 10 m of the true victim.

## Configuration

All parameters centralized in `config.py`. Key groups:

- **Workspace / flight**: `AREA_SIZE_M`, `AGL_HEIGHT`, `LANE_SPACING` (initial only)
- **ARTVA model**: `ARTVA_MOMENT`, `ARTVA_NOISE_STD`, `VICTIM_DEPTH`
- **Threshold calibration**: `N_NOISE_CALIB_SAMPLES`, `ES_DETECT_MAX_R`, `FOUND_RADIUS`
- **Extremum Seeking**: `ES_ALPHA_MAX`, `ES_OMEGA`, `ES_KAPPA`, `ES_LAMBDA`
- **MPC**: `DT_MPC`, `N_MPC`, `A_MAX`, `V_MAX`
- **IMDCL**: `IMDCL_COMM_RADIUS`, `IMDCL_SIGMA_ACC`, etc.
- **Particle Filter**: `PF_N_PARTICLES`

## Simulation flow (high level)

1. Load DEM patch and build local workspace terrain model.
2. Place victim ARTVA source at random (or specified) position.
3. Build drone agents (MPC + IMDCL + initial lawnmower waypoints).
4. Calibrate noise floor via consensus; compute dynamic thresholds; recompute lawnmower spacing.
5. At each step:
   - measure ARTVA signal (noisy), apply sliding-window filter,
   - handle FSM transitions (SEARCH→TRACK→STOP/SUPPORT),
   - update ES reference trajectory (TRACK drones),
   - run MPC using IMDCL estimated state,
   - propagate dynamics with process noise,
   - update IMDCL (LiDAR + cooperative inter-drone UWB),
   - update Particle Filter (local + cooperative measurements).
6. When ≥ 3 drones in STOP: check found criterion, report PF estimate.
7. Generate mission plots and optional 3D animation.

## Reference

Azzollini, I.A., Mimmo, N., Gentilini, L., Marconi, L. (2021).
*UAV-Based Search and Rescue in Avalanches using ARVA: An Extremum Seeking Approach.*
arXiv:2106.14514.

## Troubleshooting

- **DEM not found**: verify `w51065_s10.tif` is in project root.
- **CasADi error**: `pip install casadi`.
- **Animation save fails**: install ffmpeg; code falls back to GIF.

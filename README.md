# ARTVA Multi-Drone Avalanche Search Simulation

A Python simulation of cooperative avalanche victim search using multiple autonomous drones equipped with ARTVA (457 kHz magnetic beacon) sensors.

The project combines:
- terrain-aware flight over a real DEM (TINItaly GeoTIFF),
- multi-agent mission logic (SEARCH → TRACK → STOP/SUPPORT),
- Model Predictive Control (MPC) for trajectory tracking,
- distributed cooperative localization (IMDCL),
- Extremum Seeking (ES) source-seeking control (Azzollini et al., arXiv:2106.14514),
- distributed source estimation (DCGD) to estimate victim position,
- parametric sweep with full result analysis.

## What this project does

Each drone starts in SEARCH mode flying a lawnmower pattern.
When the ARTVA signal exceeds a dynamically calibrated detection threshold, the drone switches to TRACK mode and runs an Extremum Seeking algorithm to converge toward the signal maximum (victim location).
When signal strength exceeds the stop threshold, the drone hovers (STOP) and selects two partner drones to orbit around it (SUPPORT) for triangulation.
When ≥ 3 drones are stopped, a final DCGD refinement estimates the victim's position.

Success requires both ≥ 3 drones in STOP **and** the final position estimate within `FOUND_RADIUS` (10 m) of the true victim.

## Main modules

| File | Role |
|---|---|
| `config.py` | Single source of truth for all parameters |
| `terrain.py` | DEM loading, extraction, interpolation, `Terrain` class |
| `artva.py` | ARTVA magnetic dipole signal model |
| `drone_agent.py` | Drone FSM (SEARCH/TRACK/STOP/SUPPORT), lawnmower, ES navigation |
| `simulation.py` | Main multi-agent simulation loop (MPC + IMDCL + DCGD + mission logic) |
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
pip install numpy scipy matplotlib casadi tifffile
```

Download the DEM tile from https://tinitaly.pi.ingv.it/Download_Area1_1.html and place it as `w51065_s10.tif` in the project root.

## Quick start

```bash
# Default simulation (config.py values)
python main.py

# Custom run with 3D animation
python main.py --n 3 --steps 600 --animate

# Replay a specific experiment from a sweep CSV row
python main.py --n 4 --noise 1e-6 --rc 80 --area 200 \
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

## CLI options — run_experiments.py

```bash
python run_experiments.py --workers 4     # parallel workers
python run_experiments.py --workers 1 --verbose  # sequential, verbose
python run_experiments.py --dry-run       # show grid without running
python run_experiments.py --out my.csv    # custom output file
```

Sweep parameters are defined at the top of the file:
- `AREA_SIZES`, `N_DRONES_LIST`, `ARTVA_NOISE_STDS`, `COMM_RADII`, `WORKSPACE_CENTERS`

The sweep resumes automatically from existing CSV rows.

## Dynamic detection thresholds

Thresholds are not fixed constants. At startup each drone calibrates the local noise floor:

```
σ̂  = average-consensus(local noise samples)
DETECT_THR = max(NOISE_DETECT_FACTOR × σ̂,  ARTVA_MOMENT / ES_DETECT_MAX_R³)
STOP_THR   = max(NOISE_STOP_FACTOR   × σ̂,  10 × ARTVA_MOMENT / ES_DETECT_MAX_R³)
```

The physical floor (`ES_DETECT_MAX_R = 50 m`, from Azzollini et al. SITL) prevents very low noise from triggering TRACK before the drone is within the ES convergence basin.

## Found criterion

A run is `found = True` only when:
1. ≥ 3 drones reached STOP state, **and**
2. `est_error_2d < FOUND_RADIUS` (10 m) — the mean DCGD estimate is within 10 m of the true victim.

This prevents false positives where drones stop at the wrong position (e.g. ES converging to a local attractor far from the source).

## Supported noise levels

Active sweep: `[1e-7, 1e-6, 1e-5]` T·m³ (normalized).
- `1e-7`: low noise, good shielding — realistic best case for drone-mounted sensor
- `1e-6`: nominal — realistic quadrotor with moderate EMI isolation
- `1e-5`: high noise, worst case — heavy drone EMI or adverse environment

The level `1e-8` was removed: with the dynamic threshold, it causes detection from >100 m (entire workspace), outside the ES convergence basin for all practical starting positions.

## Configuration

All parameters centralized in `config.py`. Key groups:

- **Workspace / flight**: `AREA_SIZE_M`, `AGL_HEIGHT`, `LANE_SPACING`
- **ARTVA model**: `ARTVA_MOMENT`, `ARTVA_NOISE_STD`, `VICTIM_DEPTH`
- **Threshold calibration**: `NOISE_DETECT_FACTOR`, `NOISE_STOP_FACTOR`, `ES_DETECT_MAX_R`
- **Found criterion**: `FOUND_RADIUS`
- **Extremum Seeking**: `ES_ALPHA_MAX`, `ES_OMEGA`, `ES_KAPPA`, `ES_LAMBDA`
- **MPC**: `DT_MPC`, `N_MPC`, `A_MAX`, `V_MAX`
- **IMDCL**: `IMDCL_COMM_RADIUS`, `IMDCL_SIGMA_ACC`, etc.
- **DCGD**: `DIST_EST_ALPHA`, `DIST_EST_BETA`, `DIST_EST_REFINE`

## Simulation flow (high level)

1. Load DEM patch and build local workspace terrain model.
2. Place victim ARTVA source at random (or specified) position.
3. Build drone agents (MPC + IMDCL + lawnmower waypoints).
4. Calibrate noise floor via consensus; compute dynamic thresholds.
5. At each step:
   - measure ARTVA signal (noisy),
   - handle FSM transitions (SEARCH→TRACK→STOP/SUPPORT),
   - update ES reference trajectory (TRACK drones),
   - run MPC using IMDCL estimated state,
   - propagate dynamics with process noise,
   - update IMDCL (LiDAR + cooperative inter-drone),
   - update DCGD source estimate (Adapt + Combine).
6. When ≥ 3 drones in STOP: run DCGD refinement, report estimate.
7. Generate mission plots and optional 3D animation.

## Reference

Azzollini, I.A., Mimmo, N., Gentilini, L., Marconi, L. (2021).
*UAV-Based Search and Rescue in Avalanches using ARVA: An Extremum Seeking Approach.*
arXiv:2106.14514.

## Troubleshooting

- **DEM not found**: verify `w51065_s10.tif` is in project root.
- **CasADi error**: `pip install casadi`.
- **Animation save fails**: install ffmpeg; code falls back to GIF.
- **KeyError on noise value in plot_results.py**: the CSV contains a noise level not in `NOISE_STDS`. The loader in `main()` filters automatically; if running custom scripts, filter rows with `r["noise"] in set(NOISE_STDS)`.

# ARTVA Multi-Drone Avalanche Search Simulation

A Python simulation of cooperative avalanche victim search using multiple autonomous drones.

The project combines:
- terrain-aware flight over a real DEM (TINItaly GeoTIFF),
- multi-agent mission logic (SEARCH and TRACK phases),
- Model Predictive Control (MPC) for trajectory tracking,
- distributed cooperative localization (IMDCL),
- distributed source estimation (DCGD) to estimate victim position.

## What this project does

Each drone starts in SEARCH mode with a lawnmower pattern over a terrain patch.
When a drone detects ARTVA signal above threshold, it switches to TRACK mode and performs reactive hill-climbing.
Nearby drones can be called in support and move toward the detector drone.
When TRACK drones reach strong signal, they stop in hover, then a DCGD refinement estimates the victim location.

## Main modules

- `main.py`: CLI entry point for full mission simulation.
- `config.py`: single source of truth for parameters.
- `terrain.py`: DEM loading, extraction, interpolation, terrain queries.
- `artva.py`: ARTVA dipole signal model.
- `drone_agent.py`: drone FSM, SEARCH/TRACK behaviors, waypoint logic.
- `simulation.py`: main simulation loop (MPC + IMDCL + DCGD + mission logic).
- `mpc_drone.py`: MPC controller and point-mass model.
- `imdcl.py`: decentralized cooperative localization filter.
- `visualization.py`: static plots and 3D animations.

## Requirements

- Python 3.10+
- A local DEM file named `w51065_s10.tif` in project root

Python packages:
- numpy
- scipy
- matplotlib
- casadi
- tifffile

Install packages:

```bash
pip install numpy scipy matplotlib casadi tifffile
```

## DEM dataset setup

The DEM tile is not versioned in Git. Download:

- https://tinitaly.pi.ingv.it/Download_Area1_1.html

Then place the file in project root as:

- `w51065_s10.tif`

Without this file, terrain loading will fail.

## Quick start

Run the default simulation:

```bash
python main.py
```

Run with custom settings:

```bash
python main.py --n 3 --steps 600 --agl 1.5 --seed 42
```

Run with 3D animation:

```bash
python main.py --animate
```

Run and save animation:

```bash
python main.py --animate --save
```

## CLI options (main.py)

- `--n`: number of drones.
- `--agl`: flight altitude above ground level in meters.
- `--steps`: number of simulation steps.
- `--animate`: enable 3D mission animation.
- `--save`: save animation to disk.
- `--seed`: random seed for reproducibility.

## Standalone visualization utility

You can run the standalone MPC animation utility from `visualization.py`:

```bash
python visualization.py --speed 2.0
python visualization.py --save --fps 30 --out drone_animation
```

Options:
- `--save`: save animation output.
- `--fps`: output frame rate.
- `--speed`: playback speed multiplier.
- `--out`: output filename prefix.

## Configuration

Tune mission, control, estimation, and thresholds from `config.py`.

Important groups include:
- workspace geometry,
- flight altitude and sensor noise,
- ARTVA model thresholds,
- MPC horizon and constraints,
- IMDCL filter parameters,
- TRACK and DCGD behavior.

## Simulation flow (high level)

1. Load DEM and build local workspace terrain model.
2. Spawn victim ARTVA source in the mission area.
3. Build drone agents (MPC + IMDCL + SEARCH waypoints).
4. At each step:
   - measure ARTVA,
   - handle FSM transitions SEARCH <-> TRACK,
   - update TRACK waypoints via hill-climbing,
   - run MPC using estimated state,
   - propagate dynamics with noise,
   - run IMDCL updates,
   - run DCGD source-estimation step.
5. Stop when TRACK drones are all in hover, refine DCGD, report estimate.
6. Generate mission plots and optional animation.

## Project notes

- Parameters are intended to be centralized in `config.py`.
- In mission control logic, MPC should use estimated state (`ag.x_est`) rather than true state (`ag.x`).
- If you modify a drone waypoint list (`ag.waypoints`), reset `ag.wp_idx = 0`.
- Use `np.random.default_rng(seed)` for deterministic random generation.

## Typical outputs

The simulation prints:
- detection and stop events,
- per-drone mission status snapshots,
- final distributed victim estimate,
- final estimation errors.

Plots include:
- top-view paths over terrain and ARTVA map,
- 3D trajectories,
- ARTVA signal vs time,
- AGL tracking,
- IMDCL estimation errors.

## Troubleshooting

- Error loading DEM:
  - verify `w51065_s10.tif` exists in project root.
- CasADi import error:
  - install with `pip install casadi` in your active environment.
- Animation save fails as MP4:
  - install ffmpeg or let the code fall back to GIF.

## License

No license file is currently included in this repository.
If needed, add a LICENSE file and update this section.

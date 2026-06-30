"""
run_experiments.py
==================
Batch test script for the multi-drone ARTVA search simulation.

For each parameter combination runs the simulation and records in
results.csv:
  - simulated time to find the victim [s]
  - planimetric variance of the position estimate across drones [m²]
  - planimetric estimation error w.r.t. true position [m]
  - timeout note if the time exceeds 15 simulated minutes

Usage
-----
    python run_experiments.py                      # full sweep (4 workers)
    python run_experiments.py --workers 8          # more parallel workers
    python run_experiments.py --workers 1          # sequential (debug)
    python run_experiments.py --workers 1 --verbose
    python run_experiments.py --dry-run            # show combinations, do not run
    python run_experiments.py --out my.csv         # custom output file

Automatic resume
----------------
If results.csv already exists, run_ids already present are skipped and the
sweep resumes from missing jobs (useful after interruption).

Configuration
-------------
Modify the lists in the "PARAMETER GRID" section.
The total number of experiments is printed before execution.

Workspace
---------
WORKSPACE_CENTERS defines 5 distinct terrain patches extracted from the
real TINItaly DEM. Each entry is (row_frac, col_frac) ∈ [0,1]² or None
(= DEM centre, original behaviour). This allows evaluating system
robustness on different alpine terrains.

Random victims
--------------
N_RANDOM_VICTIMS victim positions and burial depths are sampled uniformly
in run_one(), increasing diversity without compromising reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

import numpy as np

from pf import (weighted_mean_cov_xy, run_ellipse_metrics,
                ellipse_contains, ellipse_area)

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ============================================================================
# PARAMETER GRID — modify to restrict/expand the sweep
# ============================================================================

AREA_SIZES = [100, 200]        # [m]  workspace side

N_DRONES_LIST = [3, 4, 5]          # number of agents

# Number of random victim positions: each sampled independently in run_one().
N_RANDOM_VICTIMS = 10

ARTVA_NOISE_STDS = [1e-7, 1e-6, 1e-5]  # ARTVA signal noise

# Simulation acceleration noise:
ACC_SIM_LIST = [0.05, 0.1]             # [m/s²]

# UWB communication radius:
#   120 m → optimal conditions (direct line of sight)
#    60 m → open environment with light obstacles
#    25 m → difficult conditions
COMM_RADII = [25, 60, 120]     # [m]

# Distinct terrain patches extracted from the TINItaly DEM.
# Each entry is (row_frac, col_frac) as fractions of the DEM dimensions.
# None = DEM centre (original and default behaviour).
WORKSPACE_CENTERS = [
    None,            # central patch (default)
    (0.35, 0.4),    # SW patch
    (0.45, 0.55),    # E patch
    (0.60, 0.45),    # N-W patch
    (0.60, 0.58),    # N-E patch
]

# ============================================================================
# Simulation constants
# ============================================================================

MAX_SIM_SECONDS = 900.0             # 15 minuti → soglia timeout
DT_SIM          = 0.1               # [s] — deve coincidere con DT_MPC in config
MAX_STEPS       = int(MAX_SIM_SECONDS / DT_SIM)
SEED            = 42

# ============================================================================
# CSV header
# ============================================================================

CSV_FIELDS = [
    "run_id",
    "area_size_m",
    "n_drones",
    "victim_x_m",
    "victim_y_m",
    "victim_dist_m",
    "victim_depth_m",
    "artva_noise_std",
    "acc_sim_ms2",
    "comm_radius_m",
    "workspace_frac_r",
    "workspace_frac_c",
    "found",
    "time_found_s",
    "n_drones_stopped",
    "n_drones_tracked",
    "pos_variance_m2",
    "pos_std_m",
    "est_error_2d_m",
    "est_error_3d_m",
    "est_depth_m",
    "est_error_depth_m",
    "dcgd_err_final_mean_m",
    "landing_err_mean_m",
    "pf_std_xy_mean_m",
    "pf_ellipse_area_mean_m2",
    "pf_iou_mean",
    "note",
]

# ============================================================================
# Terrain cache — reused within each process
# ============================================================================

_terrain_cache: dict = {}


def _load_terrain(area_size: float, center_frac=None, verbose: bool = False):
    """Returns (terrain_obj, …) for the requested area, with per-process cache."""
    key = (area_size, center_frac)
    if key in _terrain_cache:
        return _terrain_cache[key]

    import terrain as terrain_mod
    from terrain import build_terrain

    old_area = terrain_mod.AREA_SIZE_M
    terrain_mod.AREA_SIZE_M = area_size
    try:
        buf = io.StringIO()
        with redirect_stdout(sys.stdout if verbose else buf):
            result = build_terrain(center_frac=center_frac)
    finally:
        terrain_mod.AREA_SIZE_M = old_area

    _terrain_cache[key] = result
    return result


# ============================================================================
# Single-experiment runner
# ============================================================================

def run_one(
    area_size:   float,
    n_drones:    int,
    noise_std:   float,
    comm_radius: float,
    acc_sim:     float = 0.05,
    center_frac        = None,
    seed:        int   = SEED,
    verbose:     bool  = False,
) -> dict:
    """
    Runs one experiment and returns the metrics.

    Parameters are patched on module globals (without touching config.py on disk)
    and restored in the finally block.
    Each child process has its own module copies → no race conditions.

    center_frac : (row_frac, col_frac) for workspace selection in the DEM,
                  or None for the default centre.
    """
    import config
    import terrain    as terrain_mod
    import artva      as artva_mod
    import simulation as sim_mod
    from artva       import ARTVASource
    from simulation  import build_agents, simulate
    from drone_agent import DroneState

    saved = {
        "terrain_AREA_SIZE_M":   terrain_mod.AREA_SIZE_M,
        "artva_ARTVA_NOISE_STD": artva_mod.ARTVA_NOISE_STD,
        "sim_IMDCL_COMM_RADIUS": sim_mod.IMDCL_COMM_RADIUS,
        "cfg_AREA_SIZE_M":       config.AREA_SIZE_M,
        "cfg_ARTVA_NOISE_STD":   config.ARTVA_NOISE_STD,
        "cfg_IMDCL_COMM_RADIUS": config.IMDCL_COMM_RADIUS,
        "cfg_SIGMA_ACC_SIM":     config.SIGMA_ACC_SIM,
    }
    terrain_mod.AREA_SIZE_M   = area_size
    artva_mod.ARTVA_NOISE_STD = noise_std
    sim_mod.IMDCL_COMM_RADIUS = comm_radius
    config.AREA_SIZE_M        = area_size
    config.ARTVA_NOISE_STD    = noise_std
    config.IMDCL_COMM_RADIUS  = comm_radius
    config.SIGMA_ACC_SIM      = acc_sim

    _pos_rng     = np.random.default_rng()
    vrel_x       = float(_pos_rng.uniform(0.00, 1.00))
    vrel_y       = float(_pos_rng.uniform(0.00, 1.00))
    victim_depth = float(_pos_rng.uniform(1.0, 5.0))

    sink = sys.stdout if verbose else io.StringIO()

    try:
        terrain_obj, *_ = _load_terrain(area_size, center_frac=center_frac,
                                         verbose=verbose)

        span_x   = terrain_obj.x_max - terrain_obj.x_min
        span_y   = terrain_obj.y_max - terrain_obj.y_min
        victim_x = terrain_obj.x_min + vrel_x * span_x
        victim_y = terrain_obj.y_min + vrel_y * span_y
        victim_z = terrain_obj.z(victim_x, victim_y) - victim_depth

        artva = ARTVASource(
            theta=np.array([victim_x, victim_y, victim_z]),
            moment=config.ARTVA_MOMENT,
            seed=seed + 1,
        )
        deploy_xy    = np.array([terrain_obj.x_min + 5.0, terrain_obj.y_min + 5.0])
        victim_dist  = float(np.linalg.norm(np.array([victim_x, victim_y]) - deploy_xy))

        with redirect_stdout(sink):
            agents = build_agents(
                deploy_xy=deploy_xy,
                terrain=terrain_obj,
                n_drones=n_drones,
                agl=config.AGL_HEIGHT,
            )
            agents, *_ = simulate(
                terrain=terrain_obj,
                artva=artva,
                agents=agents,
                n_steps=MAX_STEPS,
                dt=DT_SIM,
                sigma=acc_sim,
                agl=config.AGL_HEIGHT,
                rng_seed=seed,
            )

        steps_run = len(next(iter(agents.values())).history) - 1
        time_s    = steps_run * DT_SIM

        # STOP or FINAL_ORBIT: drones that have reached the source (at the end of the
        # run, those with an active PF are in FINAL_ORBIT for the refinement orbit).
        n_stopped  = sum(1 for ag in agents.values()
                         if ag.state in (DroneState.STOP, DroneState.FINAL_ORBIT))

        n_tracked  = sum(1 for ag in agents.values() if ag.detected)

        # The PF operates in the drone's estimated frame (x_est): to compare with the
        # true victim position each estimate must be corrected for IMDCL drift (x_est − x).
        # All error metrics and inter-drone spread use the corrected estimates.
        corrected_ests = [
            ag.source_est - (ag.x_est[:3] - ag.x[:3])
            for ag in agents.values() if ag.source_est is not None
        ]
        if len(corrected_ests) >= 2:
            ests_arr = np.array(corrected_ests)      # (n, 3)
            var_xy   = np.var(ests_arr[:, :2], axis=0)
            pos_var  = float(var_xy.sum())
            pos_std  = float(np.sqrt(pos_var))
        else:
            pos_var = pos_std = float("nan")

        if corrected_ests:
            ests_arr    = np.array(corrected_ests)   # (n, 3)
            est_mean    = np.mean(ests_arr, axis=0)  # 3D centroid (frame reale)
            est_error   = float(np.linalg.norm(est_mean[:2] - artva._theta[:2]))
            est_error_3d = float(np.linalg.norm(est_mean    - artva._theta))
            est_depth   = float(terrain_obj.z(est_mean[0], est_mean[1]) - est_mean[2])
            est_depth_err = abs(est_depth - victim_depth)
            # mean per-drone XY error (drift-corrected = errore di landing reale)
            dcgd_err_mean = float(np.mean([
                np.linalg.norm(e[:2] - artva._theta[:2]) for e in corrected_ests
            ]))
            landing_err_mean = dcgd_err_mean
            # PF intra-drone std: mean of ||σ_xy|| across drones with active PF
            pf_stds = [
                float(np.linalg.norm(ag.source_est_std[:2]))
                for ag in agents.values()
                if ag.source_est_std is not None
            ]
            pf_std_xy_mean = float(np.mean(pf_stds)) if pf_stds else float("nan")
            # Per-drone confidence ellipses (reused for metrics and found criterion)
            # Centres corrected for IMDCL drift (real frame) + PF covariance.
            # Consistent between found criterion, IoU/area metrics and plots.
            pf_clouds = []
            for ag in agents.values():
                if ag.pf is None or ag.source_est is None:
                    continue
                m_xy, cov_xy = weighted_mean_cov_xy(ag.pf.particles, ag.pf.weights)
                center = m_xy - (ag.x_est[:2] - ag.x[:2])
                pf_clouds.append((center, cov_xy))
            pf_ellipse_area_mean, pf_iou_mean = run_ellipse_metrics(
                [c for c, _ in pf_clouds], [cov for _, cov in pf_clouds])
        else:
            est_error = est_error_3d = est_depth = est_depth_err = float("nan")
            dcgd_err_mean = landing_err_mean = pf_std_xy_mean = float("nan")
            pf_ellipse_area_mean = pf_iou_mean = float("nan")
            pf_clouds = []

        # Found criterion (PF-based): ONE drone with an active PF whose confidence
        # ellipse (corrected for IMDCL drift) contains the victim in the xy plane
        # AND has area ≤ FOUND_ELLIPSE_AREA_MAX is sufficient. The area constraint
        # discards estimates that are too broad.
        from config import FOUND_ELLIPSE_CONF, FOUND_ELLIPSE_AREA_MAX
        found = False
        for center, cov_xy in pf_clouds:
            if (ellipse_area(cov_xy, FOUND_ELLIPSE_CONF) <= FOUND_ELLIPSE_AREA_MAX
                    and ellipse_contains(artva._theta[:2], center, cov_xy,
                                         FOUND_ELLIPSE_CONF)):
                found = True
                break

        if found:
            note = ""
        elif not pf_clouds:
            note = (
                f"victim not found within {MAX_SIM_SECONDS/60:.0f} minutes "
                "(no PF activated)"
            )
        else:
            note = (
                f"victim not found within {MAX_SIM_SECONDS/60:.0f} minutes "
                "(PF ellipse does not contain victim or too large)"
            )

        return {
            "victim_x_m":         round(victim_x, 1),
            "victim_y_m":         round(victim_y, 1),
            "victim_dist_m":      round(victim_dist, 1),
            "victim_depth_m":     round(victim_depth, 2),
            "found":              found,
            "time_found_s":       time_s,
            "n_drones_stopped":   n_stopped,
            "n_drones_tracked":   n_tracked,
            "pos_variance_m2":    pos_var,
            "pos_std_m":          pos_std,
            "est_error_2d_m":     est_error,
            "est_error_3d_m":     est_error_3d,
            "est_depth_m":        round(est_depth, 3) if not math.isnan(est_depth) else float("nan"),
            "est_error_depth_m":     round(est_depth_err, 3) if not math.isnan(est_depth_err) else float("nan"),
            "dcgd_err_final_mean_m": round(dcgd_err_mean, 3) if not math.isnan(dcgd_err_mean) else float("nan"),
            "landing_err_mean_m":    round(landing_err_mean, 3) if not math.isnan(landing_err_mean) else float("nan"),
            "pf_std_xy_mean_m":      round(pf_std_xy_mean, 3) if not math.isnan(pf_std_xy_mean) else float("nan"),
            "pf_ellipse_area_mean_m2": round(pf_ellipse_area_mean, 3) if not math.isnan(pf_ellipse_area_mean) else float("nan"),
            "pf_iou_mean":             round(pf_iou_mean, 4) if not math.isnan(pf_iou_mean) else float("nan"),
            "note":                  note,
        }

    finally:
        terrain_mod.AREA_SIZE_M   = saved["terrain_AREA_SIZE_M"]
        artva_mod.ARTVA_NOISE_STD = saved["artva_ARTVA_NOISE_STD"]
        sim_mod.IMDCL_COMM_RADIUS = saved["sim_IMDCL_COMM_RADIUS"]
        config.AREA_SIZE_M        = saved["cfg_AREA_SIZE_M"]
        config.ARTVA_NOISE_STD    = saved["cfg_ARTVA_NOISE_STD"]
        config.IMDCL_COMM_RADIUS  = saved["cfg_IMDCL_COMM_RADIUS"]
        config.SIGMA_ACC_SIM      = saved["cfg_SIGMA_ACC_SIM"]


# ============================================================================
# Worker — top-level to be picklable with multiprocessing
# ============================================================================

def _worker(job: tuple) -> tuple:
    """Executed in the child process. Returns (run_id, metrics_or_None, tb_or_None)."""
    import matplotlib.pyplot as _plt
    _plt.switch_backend("agg")
    run_id, (area, n_drones, _, noise, acc_sim, rc, ws), seed, verbose = job
    try:
        metrics = run_one(
            area_size=area, n_drones=n_drones,
            noise_std=noise, comm_radius=rc,
            acc_sim=acc_sim,
            center_frac=ws, seed=seed, verbose=verbose,
        )
        return run_id, metrics, None
    except Exception:
        return run_id, None, traceback.format_exc()


# ============================================================================
# Progress bar — tqdm se disponibile, fallback ASCII altrimenti
# ============================================================================

class _FallbackBar:
    """Progress bar without external dependencies."""

    def __init__(self, total: int, initial: int = 0, desc: str = "") -> None:
        self.total  = total
        self.n      = initial
        self._t0    = time.perf_counter()
        self._desc  = desc
        self._pf    = ""
        self._render()

    def update(self, n: int = 1) -> None:
        self.n += n
        self._render()

    def set_postfix_str(self, s: str) -> None:
        self._pf = s
        self._render()

    def _render(self) -> None:
        elapsed = time.perf_counter() - self._t0
        rate    = self.n / elapsed if elapsed > 0 else 0
        eta     = (self.total - self.n) / rate if rate > 0 else float("inf")
        bar_w   = 28
        filled  = int(bar_w * self.n / self.total) if self.total else 0
        bar     = "#" * filled + "." * (bar_w - filled)
        pct     = 100 * self.n / self.total if self.total else 0
        eta_s   = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
        line    = (
            f"\r{self._desc} |{bar}| {self.n}/{self.total} "
            f"[{pct:.0f}%  {elapsed:.0f}s/{eta_s}  {rate:.2f}exp/s]"
        )
        if self._pf:
            line += f"  {self._pf}"
        sys.stderr.write(line)
        sys.stderr.flush()

    def close(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _make_pbar(total: int, initial: int, desc: str):
    if _HAS_TQDM:
        return _tqdm(
            total=total, initial=initial, desc=desc,
            unit="exp", dynamic_ncols=True, file=sys.stderr,
        )
    return _FallbackBar(total=total, initial=initial, desc=desc)


# ============================================================================
# Helper CSV
# ============================================================================

def _read_done_ids(path: str) -> set:
    """Reads already-completed run_ids from the CSV (for resume after interruption).
    Rows with notes starting with 'ERROR:' are not counted as complete and will be
    re-run automatically on the next invocation."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return {int(row["run_id"]) for row in reader
                    if row.get("run_id") and row["run_id"] != "run_id"
                    and not row.get("note", "").startswith("ERROR:")}
    except FileNotFoundError:
        return set()


def _ws_label(ws) -> tuple:
    """Converts workspace centre to (frac_r, frac_c) strings for the CSV."""
    if ws is None:
        return ("center", "center")
    return (f"{ws[0]:.2f}", f"{ws[1]:.2f}")


# ============================================================================
# Preview workspace
# ============================================================================

def preview_workspaces() -> None:
    """
    Single figure: full DEM with workspace rectangles + numbered markers.
    Each marker is connected by a dotted line to a small 3-D inset showing
    that workspace surface (no labels or axes).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LightSource
    from matplotlib.patches import ConnectionPatch
    from terrain import (
        read_geotiff, extract_area, coords_extent,
        interpolate_surface, TIF_PATH,
    )

    print("\nLoading full DEM for workspace preview...")
    dem, transform = read_geotiff(TIF_PATH)
    rows, cols = dem.shape
    xmin_f, xmax_f, ymin_f, ymax_f = coords_extent(dem, transform)

    ls       = LightSource(azdeg=315, altdeg=45)
    hs       = ls.hillshade(np.nan_to_num(dem, nan=0.0), vert_exag=2)
    vmin_dem = np.nanpercentile(dem, 2)
    vmax_dem = np.nanpercentile(dem, 98)

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")

    # ── Main DEM map — centred, flanked by insets ─────────────────────────
    ax_map = fig.add_axes([0.22, 0.07, 0.55, 0.87])
    ax_map.set_facecolor("#dddddd")
    im = ax_map.imshow(
        dem, cmap="terrain", vmin=vmin_dem, vmax=vmax_dem,
        extent=[xmin_f, xmax_f, ymin_f, ymax_f],
        origin="upper", interpolation="bilinear", zorder=2,
    )
    ax_map.imshow(
        hs, cmap="gray", alpha=0.35,
        extent=[xmin_f, xmax_f, ymin_f, ymax_f],
        origin="upper", interpolation="bilinear", zorder=3,
    )

    # Small colorbar inset inside the map (bottom-right corner)
    cbar_ax = ax_map.inset_axes([0.968, 0.03, 0.017, 0.33])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("m a.s.l.", fontsize=7, color="#222222", labelpad=3)
    cbar_ax.tick_params(colors="#222222", labelsize=6)
    cbar_ax.yaxis.set_ticks_position("right")
    cbar_ax.yaxis.set_label_position("right")

    ax_map.set_xlabel("E  [m UTM]", fontsize=13, color="#222222")
    ax_map.set_ylabel("N  [m UTM]", fontsize=13, color="#222222")
    ax_map.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_map.tick_params(labelsize=12, colors="#222222")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax_map.xaxis.offsetText.set_color("#222222")
    ax_map.yaxis.offsetText.set_color("#222222")

    _COLORS = ["#ff4444", "#44ff88", "#4488ff", "#ffdd00", "#ff66ff"]
    area_prev = max(AREA_SIZES)

    # 3-D inset bounding boxes in figure fraction [left, bottom, width, height]
    # Left column  (3 insets): WS 0 top, WS 3 mid, WS 4 bot
    # Right column (2 insets): WS 1 top, WS 2 bot
    inset_pos = [
        [0.01, 0.67, 0.19, 0.28],   # WS 0: left-top
        [0.79, 0.56, 0.19, 0.28],   # WS 1: right-top
        [0.79, 0.21, 0.19, 0.28],   # WS 2: right-bottom
        [0.01, 0.37, 0.19, 0.28],   # WS 3: left-middle
        [0.01, 0.07, 0.19, 0.28],   # WS 4: left-bottom
    ]
    # Anchor = right-centre for left insets, left-centre for right insets
    inset_anchors = [
        (0.20, 0.81),   # WS 0: right-centre  (x=0.01+0.19, y=0.67+0.14)
        (0.79, 0.70),   # WS 1: left-centre   (x=0.79,      y=0.56+0.14)
        (0.79, 0.35),   # WS 2: left-centre   (x=0.79,      y=0.21+0.14)
        (0.20, 0.51),   # WS 3: right-centre  (x=0.01+0.19, y=0.37+0.14)
        (0.20, 0.21),   # WS 4: right-centre  (x=0.01+0.19, y=0.07+0.14)
    ]

    print(f"  RBF interpolation for {len(WORKSPACE_CENTERS)} workspaces "
          f"({area_prev}×{area_prev} m)...")

    for i, ws in enumerate(WORKSPACE_CENTERS):
        color = _COLORS[i % len(_COLORS)]
        if ws is None:
            cr, cc = rows // 2, cols // 2
        else:
            cr = int(np.clip(ws[0] * rows, 0, rows - 1))
            cc = int(np.clip(ws[1] * cols, 0, cols - 1))

        sub_i, x_c, y_c, _ = extract_area(
            dem, transform, center_row=cr, center_col=cc, size_m=area_prev)

        rx, ry = x_c.min(), y_c.min()
        rw, rh = x_c.max() - x_c.min(), y_c.max() - y_c.min()
        cx_map = rx + rw / 2
        cy_map = ry + rh / 2

        # Workspace rectangle on the map
        ax_map.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=0, facecolor=color, alpha=0.18, zorder=5,
        ))
        ax_map.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=2.0, edgecolor=color, facecolor="none",
            linestyle="--", zorder=6,
        ))
        # Numbered circle at workspace center
        ax_map.text(
            cx_map, cy_map, str(i),
            color="white", fontsize=12, fontweight="bold",
            ha="center", va="center", zorder=7,
            bbox=dict(
                boxstyle="circle,pad=0.3",
                facecolor=color, alpha=0.85,
                edgecolor="#333333", linewidth=1.5,
            ),
        )

        # Interpolate 3-D surface for inset
        print(f"    WS {i}...", end=" ", flush=True)
        xi, yi, zi = interpolate_surface(sub_i, x_c, y_c, grid_n=50)
        print("ok")

        valid  = sub_i[~np.isnan(sub_i)]
        vmin_i = float(np.percentile(valid, 1))  if valid.size else 0.0
        vmax_i = float(np.percentile(valid, 99)) if valid.size else 1.0
        norm_z = (zi - vmin_i) / max(vmax_i - vmin_i, 1e-6)

        # Dotted connecting line from workspace number to inset anchor.
        # Added to ax_map so it renders before the 3-D insets (appears behind them).
        anchor = inset_anchors[i]
        con = ConnectionPatch(
            xyA=(cx_map, cy_map),
            xyB=anchor,
            coordsA="data",
            coordsB="figure fraction",
            axesA=ax_map,
            color=color, lw=1.5, linestyle=":", alpha=0.9,
            clip_on=False, zorder=8,
        )
        ax_map.add_artist(con)

        # 3-D inset axes overlaid on the map
        ip = inset_pos[i]
        ax3d = fig.add_axes(ip, projection="3d")
        ax3d.patch.set_facecolor("#f0f0f0")
        ax3d.patch.set_alpha(0.95)
        ax3d.plot_surface(
            xi, yi, zi,
            facecolors=plt.get_cmap("plasma")(norm_z),
            rcount=50, ccount=50, linewidth=0, antialiased=True, alpha=0.95,
        )
        ax3d.set_axis_off()

    print("\nClose the window to start the simulation...")
    plt.show()


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    default_workers = min(os.cpu_count() or 1, 4)

    parser = argparse.ArgumentParser(
        description="Parameter sweep for multi-drone ARTVA search simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out",     default="results.csv", help="Output CSV file")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Parallel processes (1 = sequential)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print simulation output (only with --workers 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show combinations without running")
    parser.add_argument("--seed",    type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    if args.verbose and args.workers > 1:
        print("WARNING: --verbose is ignored with workers > 1 (interleaved output).")
        args.verbose = False

    # ── Grid ──────────────────────────────────────────────────────────────
    grid = list(itertools.product(
        AREA_SIZES, N_DRONES_LIST, range(N_RANDOM_VICTIMS),
        ARTVA_NOISE_STDS, ACC_SIM_LIST, COMM_RADII, WORKSPACE_CENTERS,
    ))
    total = len(grid)

    print(f"Parameter sweep — {total} total experiments")
    print(f"  area_sizes:        {AREA_SIZES}")
    print(f"  n_drones:          {N_DRONES_LIST}")
    print(f"  victim positions:  {N_RANDOM_VICTIMS} (uniform random position and depth in run_one)")
    print(f"  noise_stds:        {ARTVA_NOISE_STDS}")
    print(f"  acc_sim [m/s²]:    {ACC_SIM_LIST}")
    print(f"  comm_radii[m]:     {COMM_RADII}")
    print(f"  workspace_centers: {len(WORKSPACE_CENTERS)} DEM patches")
    print(f"  timeout:          {MAX_SIM_SECONDS:.0f}s ({MAX_SIM_SECONDS/60:.0f}min) / {MAX_STEPS} steps")
    print(f"  workers:          {args.workers}")
    print(f"  output:           {args.out}")

    if args.dry_run:
        print("\n[dry-run] First 10 combinations:")
        for i, (area, nd, vidx, noise, acc_sim, rc, ws) in enumerate(grid[:10], 1):
            ws_str = "center" if ws is None else f"({ws[0]:.2f},{ws[1]:.2f})"
            print(f"  {i:3d}: area={area}m  n={nd}  victim_rep={vidx}  "
                  f"noise={noise:.0e}  acc_sim={acc_sim:.2f}  rc={rc}m  ws={ws_str}")
        if total > 10:
            print(f"  … ({total - 10} more)")
        return

    # ── Preview workspace su DEM + 3-D ────────────────────────────────────
    preview_workspaces()

    # ── Resume ────────────────────────────────────────────────────────────
    done_ids  = _read_done_ids(args.out)
    jobs      = [
        (run_id, combo, args.seed, args.verbose)
        for run_id, combo in enumerate(grid, start=1)
        if run_id not in done_ids
    ]
    remaining = len(jobs)
    if done_ids:
        print(f"\nResume: {len(done_ids)} already completed, {remaining} remaining.")
    if remaining == 0:
        print("No jobs to run.")
        return

    # ── Pre-load terrains in the main process ─────────────────────────────
    # On Linux (fork) workers inherit the cache → no DEM re-read.
    # On macOS/Windows (spawn) each worker rebuilds its own cache.
    unique_terrain_keys = {(a, ws) for a, _, _, _, _, _, ws in grid}
    print(f"\nPre-loading {len(unique_terrain_keys)} terrain configurations...")
    for area, ws in sorted(unique_terrain_keys, key=lambda x: (x[0], str(x[1]))):
        t0     = time.perf_counter()
        ws_str = "center" if ws is None else f"({ws[0]:.2f},{ws[1]:.2f})"
        _load_terrain(area, center_frac=ws, verbose=False)
        print(f"  {area}m × {area}m  ws={ws_str}  ({time.perf_counter() - t0:.1f}s)")

    # ── Esecuzione ────────────────────────────────────────────────────────
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    found_n = timeout_n = error_n = 0

    print()  # riga vuota prima della barra

    with open(args.out, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        def _record(run_id: int, metrics, tb) -> None:
            nonlocal found_n, timeout_n, error_n
            area, n_drones, _, noise, acc_sim, rc, ws = grid[run_id - 1]
            ws_r, ws_c = _ws_label(ws)

            if tb is not None:
                error_n += 1
                metrics = {
                    "victim_x_m":         float("nan"),
                    "victim_y_m":         float("nan"),
                    "victim_dist_m":      float("nan"),
                    "victim_depth_m":     float("nan"),
                    "found":              False,
                    "time_found_s":       float("nan"),
                    "n_drones_stopped":   0,
                    "n_drones_tracked":   0,
                    "pos_variance_m2":    float("nan"),
                    "pos_std_m":          float("nan"),
                    "est_error_2d_m":     float("nan"),
                    "est_error_3d_m":     float("nan"),
                    "est_depth_m":        float("nan"),
                    "est_error_depth_m":     float("nan"),
                    "dcgd_err_final_mean_m": float("nan"),
                    "landing_err_mean_m":    float("nan"),
                    "pf_std_xy_mean_m":      float("nan"),
                    "note":                  f"ERROR: {tb.splitlines()[-1]}",
                }
            elif metrics["found"]:
                found_n += 1
            else:
                timeout_n += 1

            writer.writerow({
                "run_id":           run_id,
                "area_size_m":      area,
                "n_drones":         n_drones,
                "artva_noise_std":  noise,
                "acc_sim_ms2":      acc_sim,
                "comm_radius_m":    rc,
                "workspace_frac_r": ws_r,
                "workspace_frac_c": ws_c,
                **metrics,
            })
            csvfile.flush()
            pbar.update(1)
            pbar.set_postfix_str(f"ok={found_n} TO={timeout_n} E={error_n}")

        with _make_pbar(total, initial=len(done_ids), desc="sweep") as pbar:
            if args.workers == 1:
                for job in jobs:
                    _record(*_worker(job))
            else:
                with mp.Pool(processes=args.workers) as pool:
                    for result in pool.imap_unordered(_worker, jobs):
                        _record(*result)

    print(f"\nDone: {found_n} found, {timeout_n} timeout, {error_n} errors.")
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()

"""
run_experiments.py
==================
Script di test batch per la simulazione di ricerca multi-drone ARTVA.

Per ogni combinazione di parametri esegue la simulazione e registra in
results.csv:
  - tempo simulato per trovare la vittima [s]
  - varianza planimetrica della stima di posizione tra i droni [m²]
  - errore planimetrico della stima rispetto alla posizione reale [m]
  - nota di timeout se il tempo supera 15 minuti simulati

Uso
---
    python run_experiments.py                      # sweep completo (4 worker)
    python run_experiments.py --workers 8          # più paralleli
    python run_experiments.py --workers 1          # sequenziale (debug)
    python run_experiments.py --workers 1 --verbose
    python run_experiments.py --dry-run            # mostra combinazioni, non esegue
    python run_experiments.py --out mio.csv        # file output personalizzato

Resume automatico
-----------------
Se results.csv esiste già, i run_id già presenti vengono saltati e lo sweep
riprende dai job mancanti (utile dopo interruzione).

Configurazione
--------------
Modificare le liste nella sezione "GRID DEI PARAMETRI".
Il numero totale di esperimenti è stampato prima dell'esecuzione.

Workspace
---------
WORKSPACE_CENTERS definisce 5 patch di terreno distinte estratte dal DEM
reale TINItaly. Ogni entry è (row_frac, col_frac) ∈ [0,1]² oppure None
(= centro del DEM, comportamento originale). Questo permette di valutare
la robustezza del sistema su terreni alpini diversi.

Vittime casuali
---------------
N_RANDOM_VICTIMS posizioni vittima e N_RANDOM_DEPTHS profondità di
sepoltura vengono campionate con seme fisso _GRID_RNG_SEED e aggiunte
alle liste fisse, aumentando la diversità senza compromettere la
riproducibilità.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

# Forza backend non-interattivo prima di qualunque import matplotlib
# (necessario nei processi figlio che non hanno display).
import matplotlib
matplotlib.use("Agg")

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ============================================================================
# GRID DEI PARAMETRI — modificare per restringere/ampliare lo sweep
# ============================================================================

AREA_SIZES = [100, 200]        # [m]  lato workspace

N_DRONES_LIST = [3, 4, 5]          # numero di agenti

# Numero di posizioni vittima casuali per ogni combinazione di parametri.
# Ogni simulazione campiona una posizione indipendente dentro run_one().
N_RANDOM_VICTIMS = 3

BURIAL_DEPTHS_FIXED = [1.0, 3.0, 5.0]   # [m]  profondità fisse

N_RANDOM_DEPTHS = 0                 # profondità di sepoltura casuali aggiuntive
_depths_random = [
    float(_rng_grid.uniform(0.5, 4.5)) for _ in range(N_RANDOM_DEPTHS)
]
BURIAL_DEPTHS = BURIAL_DEPTHS_FIXED + _depths_random

ARTVA_NOISE_STDS = [1e-8, 1e-7, 1e-6]  # rumore segnale ARTVA

NOISE_DETECT_FACTORS = [50.0, 100.0]   # fattore soglia rilevamento (DETECT_THR = FACTOR × σ̂)

# Raggio comunicazione UWB:
#   120 m → condizioni ottimali (visibilità diretta)
#    80 m → ambiente aperto con ostacoli leggeri
#    50 m → condizioni miste
#    25 m → condizioni difficili
COMM_RADII = [25, 50, 80, 120]     # [m]

# Patch di terreno distinte estratte dal DEM TINItaly.
# Ogni entry è (row_frac, col_frac) come frazioni delle dimensioni del DEM.
# None = centro del DEM (comportamento originale e di default).
WORKSPACE_CENTERS = [
    None,            # patch centrale (default)
    (0.35, 0.45),    # patch SW
    (0.45, 0.55),    # patch E
    (0.60, 0.45),    # patch N-W
    (0.60, 0.58),    # patch N-E
]

# ============================================================================
# Costanti di simulazione
# ============================================================================

MAX_SIM_SECONDS = 900.0             # 15 minuti → soglia timeout
DT_SIM          = 0.1               # [s] — deve coincidere con DT_MPC in config
MAX_STEPS       = int(MAX_SIM_SECONDS / DT_SIM)
SEED            = 42

# La simulazione termina quando n_stopped >= STOP_THRESHOLD.
# Con n_drones < 3, questa condizione non è mai raggiungibile.
STOP_THRESHOLD  = 3

# ============================================================================
# Intestazione CSV
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
    "noise_detect_factor",
    "comm_radius_m",
    "workspace_frac_r",
    "workspace_frac_c",
    "found",
    "time_found_s",
    "n_drones_stopped",
    "pos_variance_m2",
    "pos_std_m",
    "est_error_2d_m",
    "note",
]

# ============================================================================
# Cache terreno — riutilizzata all'interno di ogni processo
# ============================================================================

_terrain_cache: dict = {}


def _load_terrain(area_size: float, center_frac=None, verbose: bool = False):
    """Restituisce (terrain_obj, …) per l'area richiesta, con cache per processo."""
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
# Runner di un singolo esperimento
# ============================================================================

def run_one(
    area_size:           float,
    n_drones:            int,
    victim_depth:        float,
    noise_std:           float,
    comm_radius:         float,
    noise_detect_factor: float = 100.0,
    center_frac                = None,
    seed:                int   = SEED,
    verbose:             bool  = False,
) -> dict:
    """
    Esegue un esperimento e restituisce le metriche.

    I parametri vengono patchati sulle variabili globali dei moduli (senza
    toccare config.py su disco) e ripristinati nel finally.
    Ogni processo figlio ha la propria copia dei moduli → nessuna race condition.

    center_frac : (row_frac, col_frac) per la selezione del workspace nel DEM,
                  oppure None per il centro di default.
    """
    import config
    import terrain    as terrain_mod
    import artva      as artva_mod
    import simulation as sim_mod
    from artva       import ARTVASource
    from simulation  import build_agents, simulate
    from drone_agent import DroneState

    saved = {
        "terrain_AREA_SIZE_M":        terrain_mod.AREA_SIZE_M,
        "artva_ARTVA_NOISE_STD":      artva_mod.ARTVA_NOISE_STD,
        "sim_IMDCL_COMM_RADIUS":      sim_mod.IMDCL_COMM_RADIUS,
        "sim_NOISE_DETECT_FACTOR":    sim_mod.NOISE_DETECT_FACTOR,
        "cfg_AREA_SIZE_M":            config.AREA_SIZE_M,
        "cfg_ARTVA_NOISE_STD":        config.ARTVA_NOISE_STD,
        "cfg_IMDCL_COMM_RADIUS":      config.IMDCL_COMM_RADIUS,
        "cfg_NOISE_DETECT_FACTOR":    config.NOISE_DETECT_FACTOR,
    }
    terrain_mod.AREA_SIZE_M        = area_size
    artva_mod.ARTVA_NOISE_STD      = noise_std
    sim_mod.IMDCL_COMM_RADIUS      = comm_radius
    sim_mod.NOISE_DETECT_FACTOR    = noise_detect_factor
    config.AREA_SIZE_M             = area_size
    config.ARTVA_NOISE_STD         = noise_std
    config.IMDCL_COMM_RADIUS       = comm_radius
    config.NOISE_DETECT_FACTOR     = noise_detect_factor

    _pos_rng = np.random.default_rng()
    vrel_x   = float(_pos_rng.uniform(0.10, 0.90))
    vrel_y   = float(_pos_rng.uniform(0.10, 0.90))

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
            position=np.array([victim_x, victim_y, victim_z]),
            moment=config.ARTVA_MOMENT,
            rng_seed=seed + 1,
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
            agents, _ = simulate(
                terrain=terrain_obj,
                artva=artva,
                agents=agents,
                n_steps=MAX_STEPS,
                dt=DT_SIM,
                sigma=config.SIGMA_ACC_SIM,
                agl=config.AGL_HEIGHT,
                rng_seed=seed,
            )

        steps_run = len(next(iter(agents.values())).history) - 1
        time_s    = steps_run * DT_SIM

        n_stopped = sum(1 for ag in agents.values() if ag.state == DroneState.STOP)
        found     = n_stopped >= STOP_THRESHOLD

        valid_ests = [ag.source_est for ag in agents.values() if ag.source_est is not None]
        if len(valid_ests) >= 2:
            ests_xy = np.array([e[:2] for e in valid_ests])
            var_xy  = np.var(ests_xy, axis=0)
            pos_var = float(var_xy.sum())
            pos_std = float(np.sqrt(pos_var))
        else:
            pos_var = pos_std = float("nan")

        if valid_ests:
            est_mean  = np.mean([e[:2] for e in valid_ests], axis=0)
            est_error = float(np.linalg.norm(est_mean - artva.position[:2]))
        else:
            est_error = float("nan")

        if found:
            note = ""
        elif n_drones < STOP_THRESHOLD:
            note = (
                f"vittima non trovata entro {MAX_SIM_SECONDS/60:.0f} minuti "
                f"(n_drones={n_drones} < soglia={STOP_THRESHOLD})"
            )
        else:
            note = f"vittima non trovata entro {MAX_SIM_SECONDS/60:.0f} minuti"

        return {
            "victim_x_m":       round(victim_x, 1),
            "victim_y_m":       round(victim_y, 1),
            "victim_dist_m":    round(victim_dist, 1),
            "found":            found,
            "time_found_s":     time_s,
            "n_drones_stopped": n_stopped,
            "pos_variance_m2":  pos_var,
            "pos_std_m":        pos_std,
            "est_error_2d_m":   est_error,
            "note":             note,
        }

    finally:
        terrain_mod.AREA_SIZE_M        = saved["terrain_AREA_SIZE_M"]
        artva_mod.ARTVA_NOISE_STD      = saved["artva_ARTVA_NOISE_STD"]
        sim_mod.IMDCL_COMM_RADIUS      = saved["sim_IMDCL_COMM_RADIUS"]
        sim_mod.NOISE_DETECT_FACTOR    = saved["sim_NOISE_DETECT_FACTOR"]
        config.AREA_SIZE_M             = saved["cfg_AREA_SIZE_M"]
        config.ARTVA_NOISE_STD         = saved["cfg_ARTVA_NOISE_STD"]
        config.IMDCL_COMM_RADIUS       = saved["cfg_IMDCL_COMM_RADIUS"]
        config.NOISE_DETECT_FACTOR     = saved["cfg_NOISE_DETECT_FACTOR"]


# ============================================================================
# Worker — top-level per essere picklable con multiprocessing
# ============================================================================

def _worker(job: tuple) -> tuple:
    """Eseguito nel processo figlio. Restituisce (run_id, metrics_or_None, tb_or_None)."""
    run_id, (area, n_drones, victim_idx, depth, noise, detect_factor, rc, ws), seed, verbose = job
    try:
        metrics = run_one(
            area_size=area, n_drones=n_drones,
            victim_depth=depth, noise_std=noise, comm_radius=rc,
            noise_detect_factor=detect_factor,
            center_frac=ws, seed=seed, verbose=verbose,
        )
        return run_id, metrics, None
    except Exception:
        return run_id, None, traceback.format_exc()


# ============================================================================
# Progress bar — tqdm se disponibile, fallback ASCII altrimenti
# ============================================================================

class _FallbackBar:
    """Progress bar senza dipendenze esterne."""

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
    """Legge i run_id già completati nel CSV (per resume dopo interruzione).
    Le righe con note che iniziano per 'ERRORE:' non vengono conteggiate come
    completate, così vengono rieseguite automaticamente al prossimo run."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return {int(row["run_id"]) for row in reader
                    if row.get("run_id") and row["run_id"] != "run_id"
                    and not row.get("note", "").startswith("ERRORE:")}
    except FileNotFoundError:
        return set()


def _ws_label(ws) -> tuple:
    """Converte workspace center in (frac_r, frac_c) stringhe per il CSV."""
    if ws is None:
        return ("center", "center")
    return (f"{ws[0]:.2f}", f"{ws[1]:.2f}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    default_workers = min(os.cpu_count() or 1, 4)

    parser = argparse.ArgumentParser(
        description="Sweep parametrico simulazione multi-drone ARTVA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out",     default="results.csv", help="File CSV di output")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Processi paralleli (1 = sequenziale)")
    parser.add_argument("--verbose", action="store_true",
                        help="Stampa output simulazione (solo con --workers 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra combinazioni senza eseguire")
    parser.add_argument("--seed",    type=int, default=SEED, help="Seme random")
    args = parser.parse_args()

    if args.verbose and args.workers > 1:
        print("ATTENZIONE: --verbose è ignorato con workers > 1 (output interleaved).")
        args.verbose = False

    # ── Griglia ───────────────────────────────────────────────────────────
    grid = list(itertools.product(
        AREA_SIZES, N_DRONES_LIST, range(N_RANDOM_VICTIMS),
        BURIAL_DEPTHS, ARTVA_NOISE_STDS, NOISE_DETECT_FACTORS, COMM_RADII, WORKSPACE_CENTERS,
    ))
    total = len(grid)

    print(f"Sweep parametrico — {total} esperimenti totali")
    print(f"  area_sizes:          {AREA_SIZES}")
    print(f"  n_drones:            {N_DRONES_LIST}")
    print(f"  victim positions:    {N_RANDOM_VICTIMS} (campionate uniformemente in run_one)")
    print(f"  depths [m]:          {len(BURIAL_DEPTHS)} valori ({len(BURIAL_DEPTHS_FIXED)} fissi + {N_RANDOM_DEPTHS} casuali)")
    print(f"  noise_stds:          {ARTVA_NOISE_STDS}")
    print(f"  noise_detect_factors: {NOISE_DETECT_FACTORS}")
    print(f"  comm_radii[m]:       {COMM_RADII}")
    print(f"  workspace_centers:   {len(WORKSPACE_CENTERS)} patch DEM")
    print(f"  timeout:          {MAX_SIM_SECONDS:.0f}s ({MAX_SIM_SECONDS/60:.0f}min) / {MAX_STEPS} passi")
    print(f"  workers:          {args.workers}")
    print(f"  output:           {args.out}")

    if args.dry_run:
        print("\n[dry-run] Prime 10 combinazioni:")
        for i, (area, nd, vidx, depth, noise, detect_factor, rc, ws) in enumerate(grid[:10], 1):
            ws_str = "center" if ws is None else f"({ws[0]:.2f},{ws[1]:.2f})"
            print(f"  {i:3d}: area={area}m  n={nd}  victim_idx={vidx}  "
                  f"depth={depth:.2f}m  noise={noise:.0e}  detect_factor={detect_factor}  rc={rc}m  ws={ws_str}")
        if total > 10:
            print(f"  … ({total - 10} altre)")
        return

    # ── Resume ────────────────────────────────────────────────────────────
    done_ids  = _read_done_ids(args.out)
    jobs      = [
        (run_id, combo, args.seed, args.verbose)
        for run_id, combo in enumerate(grid, start=1)
        if run_id not in done_ids
    ]
    remaining = len(jobs)
    if done_ids:
        print(f"\nResume: {len(done_ids)} già completati, {remaining} rimanenti.")
    if remaining == 0:
        print("Nessun job da eseguire.")
        return

    # ── Pre-carica terreni nel processo principale ────────────────────────
    # Su Linux (fork) i worker ereditano la cache → non rileggono il DEM.
    # Su macOS/Windows (spawn) ogni worker ricostruisce la propria cache.
    unique_terrain_keys = {(a, ws) for a, _, _, _, _, _, _, ws in grid}
    print(f"\nPre-caricamento {len(unique_terrain_keys)} configurazioni terreno...")
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
            area, n_drones, _vidx, depth, noise, detect_factor, rc, ws = grid[run_id - 1]
            ws_r, ws_c = _ws_label(ws)

            if tb is not None:
                error_n += 1
                metrics = {
                    "victim_x_m":       float("nan"),
                    "victim_y_m":       float("nan"),
                    "victim_dist_m":    float("nan"),
                    "found":            False,
                    "time_found_s":     float("nan"),
                    "n_drones_stopped": 0,
                    "pos_variance_m2":  float("nan"),
                    "pos_std_m":        float("nan"),
                    "est_error_2d_m":   float("nan"),
                    "note":             f"ERRORE: {tb.splitlines()[-1]}",
                }
            elif metrics["found"]:
                found_n += 1
            else:
                timeout_n += 1

            writer.writerow({
                "run_id":              run_id,
                "area_size_m":         area,
                "n_drones":            n_drones,
                "victim_depth_m":      round(depth, 2),
                "artva_noise_std":     noise,
                "noise_detect_factor": detect_factor,
                "comm_radius_m":       rc,
                "workspace_frac_r":    ws_r,
                "workspace_frac_c":    ws_c,
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

    print(f"\nCompletato: {found_n} trovati, {timeout_n} timeout, {error_n} errori.")
    print(f"Risultati: {args.out}")


if __name__ == "__main__":
    main()

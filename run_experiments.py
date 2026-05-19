"""
run_experiments.py
==================
Script di test batch per la simulazione di ricerca multi-drone ARTVA.

Per ogni combinazione di parametri esegue la simulazione e registra in
results.csv:
  - tempo simulato per trovare la vittima [s]
  - varianza planimetrica della stima di posizione tra i droni [m²]
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

AREA_SIZES = [100, 150, 200]        # [m]  lato workspace

N_DRONES_LIST = [3, 4, 5]          # numero di agenti

# Posizione vittima come frazione relativa dell'area [0..1].
# Convertite in coordinate assolute a runtime → sempre valide per qualunque area.
VICTIM_REL_XY = [
    (0.25, 0.25),                   # quadrante SW
    (0.50, 0.50),                   # centro
    (0.75, 0.50),                   # metà E
    (0.35, 0.70),                   # zona N-W
]

BURIAL_DEPTHS = [1.0, 3.0, 5.0]    # [m]  profondità sepoltura

ARTVA_NOISE_STDS = [1e-8, 1e-7, 5e-7]  # rumore segnale ARTVA

# Raggio comunicazione UWB:
#   120 m → condizioni ottimali (visibilità diretta)
#    80 m → ambiente aperto con ostacoli leggeri
#    50 m → condizioni miste
#    25 m → condizioni difficili
COMM_RADII = [25, 50, 80, 120]     # [m]

# ============================================================================
# Costanti di simulazione
# ============================================================================

MAX_SIM_SECONDS = 900.0             # 15 minuti → soglia timeout
DT_SIM          = 0.5               # [s] — deve coincidere con DT_MPC in config
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
    "victim_depth_m",
    "artva_noise_std",
    "comm_radius_m",
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


def _load_terrain(area_size: float, verbose: bool = False):
    """Restituisce (terrain_obj, …) per l'area richiesta, con cache per processo."""
    if area_size in _terrain_cache:
        return _terrain_cache[area_size]

    import terrain as terrain_mod
    from terrain import build_terrain

    old_area = terrain_mod.AREA_SIZE_M
    terrain_mod.AREA_SIZE_M = area_size
    try:
        buf = io.StringIO()
        with redirect_stdout(sys.stdout if verbose else buf):
            result = build_terrain()
    finally:
        terrain_mod.AREA_SIZE_M = old_area

    _terrain_cache[area_size] = result
    return result


# ============================================================================
# Runner di un singolo esperimento
# ============================================================================

def run_one(
    area_size:    float,
    n_drones:     int,
    victim_rel:   tuple,
    victim_depth: float,
    noise_std:    float,
    comm_radius:  float,
    seed:         int  = SEED,
    verbose:      bool = False,
) -> dict:
    """
    Esegue un esperimento e restituisce le metriche.

    I parametri vengono patchati sulle variabili globali dei moduli (senza
    toccare config.py su disco) e ripristinati nel finally.
    Ogni processo figlio ha la propria copia dei moduli → nessuna race condition.
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
    }
    terrain_mod.AREA_SIZE_M   = area_size
    artva_mod.ARTVA_NOISE_STD = noise_std
    sim_mod.IMDCL_COMM_RADIUS = comm_radius
    config.AREA_SIZE_M        = area_size
    config.ARTVA_NOISE_STD    = noise_std
    config.IMDCL_COMM_RADIUS  = comm_radius

    sink = sys.stdout if verbose else io.StringIO()

    try:
        terrain_obj, *_ = _load_terrain(area_size, verbose=verbose)

        span_x   = terrain_obj.x_max - terrain_obj.x_min
        span_y   = terrain_obj.y_max - terrain_obj.y_min
        victim_x = terrain_obj.x_min + victim_rel[0] * span_x
        victim_y = terrain_obj.y_min + victim_rel[1] * span_y
        victim_z = terrain_obj.z(victim_x, victim_y) - victim_depth

        artva = ARTVASource(
            position=np.array([victim_x, victim_y, victim_z]),
            moment=config.ARTVA_MOMENT,
            rng_seed=seed + 1,
        )
        deploy_xy = np.array([terrain_obj.x_min + 5.0, terrain_obj.y_min + 5.0])

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
            "found":            found,
            "time_found_s":     time_s,
            "n_drones_stopped": n_stopped,
            "pos_variance_m2":  pos_var,
            "pos_std_m":        pos_std,
            "est_error_2d_m":   est_error,
            "note":             note,
        }

    finally:
        terrain_mod.AREA_SIZE_M   = saved["terrain_AREA_SIZE_M"]
        artva_mod.ARTVA_NOISE_STD = saved["artva_ARTVA_NOISE_STD"]
        sim_mod.IMDCL_COMM_RADIUS = saved["sim_IMDCL_COMM_RADIUS"]
        config.AREA_SIZE_M        = saved["cfg_AREA_SIZE_M"]
        config.ARTVA_NOISE_STD    = saved["cfg_ARTVA_NOISE_STD"]
        config.IMDCL_COMM_RADIUS  = saved["cfg_IMDCL_COMM_RADIUS"]


# ============================================================================
# Worker — top-level per essere picklable con multiprocessing
# ============================================================================

def _worker(job: tuple) -> tuple:
    """Eseguito nel processo figlio. Restituisce (run_id, metrics_or_None, tb_or_None)."""
    run_id, (area, n_drones, vrel, depth, noise, rc), seed, verbose = job
    try:
        metrics = run_one(
            area_size=area, n_drones=n_drones, victim_rel=vrel,
            victim_depth=depth, noise_std=noise, comm_radius=rc,
            seed=seed, verbose=verbose,
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
        bar     = "█" * filled + "░" * (bar_w - filled)
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
    """Legge i run_id già completati nel CSV (per resume dopo interruzione)."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return {int(row["run_id"]) for row in reader if row.get("run_id")}
    except FileNotFoundError:
        return set()


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
        AREA_SIZES, N_DRONES_LIST, VICTIM_REL_XY,
        BURIAL_DEPTHS, ARTVA_NOISE_STDS, COMM_RADII,
    ))
    total = len(grid)

    print(f"Sweep parametrico — {total} esperimenti totali")
    print(f"  area_sizes:    {AREA_SIZES}")
    print(f"  n_drones:      {N_DRONES_LIST}")
    print(f"  victim_rel_xy: {VICTIM_REL_XY}")
    print(f"  depths [m]:    {BURIAL_DEPTHS}")
    print(f"  noise_stds:    {ARTVA_NOISE_STDS}")
    print(f"  comm_radii[m]: {COMM_RADII}")
    print(f"  timeout:       {MAX_SIM_SECONDS:.0f}s ({MAX_SIM_SECONDS/60:.0f}min) / {MAX_STEPS} passi")
    print(f"  workers:       {args.workers}")
    print(f"  output:        {args.out}")

    if args.dry_run:
        print("\n[dry-run] Prime 10 combinazioni:")
        for i, (area, nd, vrel, depth, noise, rc) in enumerate(grid[:10], 1):
            print(f"  {i:3d}: area={area}m  n={nd}  vrel={vrel}  "
                  f"depth={depth}m  noise={noise:.0e}  rc={rc}m")
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
    print("\nPre-caricamento terreni...")
    for area in AREA_SIZES:
        t0 = time.perf_counter()
        _load_terrain(area, verbose=False)
        print(f"  {area}m × {area}m  ({time.perf_counter() - t0:.1f}s)")

    # ── Esecuzione ────────────────────────────────────────────────────────
    write_header = len(done_ids) == 0
    found_n = timeout_n = error_n = 0

    print()  # riga vuota prima della barra

    with open(args.out, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        def _record(run_id: int, metrics, tb) -> None:
            nonlocal found_n, timeout_n, error_n
            area, n_drones, vrel, depth, noise, rc = grid[run_id - 1]

            if tb is not None:
                error_n += 1
                metrics = {
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
                "run_id":          run_id,
                "area_size_m":     area,
                "n_drones":        n_drones,
                "victim_x_m":      round(vrel[0] * area, 1),
                "victim_y_m":      round(vrel[1] * area, 1),
                "victim_depth_m":  depth,
                "artva_noise_std": noise,
                "comm_radius_m":   rc,
                **metrics,
            })
            csvfile.flush()
            pbar.update(1)
            pbar.set_postfix_str(f"✓{found_n} T{timeout_n} E{error_n}")

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

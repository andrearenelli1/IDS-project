"""
level_dataset.py
================
Livella results.csv: porta TUTTE le celle (area, n_droni) a TARGET vittime
random per combinazione di parametri, aggiungendo SOLO le run mancanti.

Contesto del problema
---------------------
In results.csv ogni cella (area, n_droni) ha le stesse 90 combinazioni di
parametri (3 noise × 2 acc × 3 rc × 5 workspace), ma un numero DIVERSO di
vittime random per combinazione, perché il CSV è stato costruito in più
passate aumentando N_RANDOM_VICTIMS:
    area=100 n=3 → 3 vittime/combo   (sotto-campionata)
    area=100 n=4 → 5 vittime/combo   (sotto-campionata)
    tutte le altre → 13 vittime/combo
Le vittime sono campionate a caso a ogni run (`_pos_rng = default_rng()` SENZA
seed in run_one), quindi ogni chiamata a run_one() = vittima nuova: basta
chiamarla N volte in più per la cella sotto-campionata.

Perché NON usare il resume nativo dello sweep
---------------------------------------------
`run_id` è POSIZIONALE nella griglia `itertools.product(...)` che include
`range(N_RANDOM_VICTIMS)`. Cambiare N_RANDOM_VICTIMS riordina la griglia, e
il resume salta per run_id → mapping run_id↔parametri incoerente col CSV
esistente (che ha run_id fino a 5400 da griglie precedenti). Quindi si fa un
top-up dedicato che APPENDE righe con run_id nuovi, senza toccare il resume.

Uso
---
    python level_dataset.py              # esegue il top-up su results.csv
    python level_dataset.py --dry-run    # mostra solo quante run aggiungerebbe
    python level_dataset.py --workers 8  # più paralleli
Fai un backup prima (lo script lo fa in automatico in results.csv.bak).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import shutil
import traceback
from collections import Counter

from tqdm import tqdm

import run_experiments as R

OUT    = "results.csv"
TARGET = 13


def _key(area, n, noise, acc, rc, ws_r, ws_c):
    return (int(area), int(n), float(noise), float(acc), int(rc), ws_r, ws_c)


def _worker(job: tuple) -> tuple:
    import matplotlib.pyplot as _plt
    _plt.switch_backend("agg")
    area, n, noise, acc, rc, ws = job
    try:
        metrics = R.run_one(area_size=area, n_drones=n, noise_std=noise,
                            comm_radius=rc, acc_sim=acc, center_frac=ws)
        return area, n, noise, acc, rc, ws, metrics, None
    except Exception:
        return area, n, noise, acc, rc, ws, None, traceback.format_exc()


def main() -> None:
    default_workers = min(mp.cpu_count() or 1, 4)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=default_workers,
                    help="Processi paralleli (1 = sequenziale)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.out)))
    have = Counter(
        _key(float(r["area_size_m"]), r["n_drones"], r["artva_noise_std"],
             r["acc_sim_ms2"], float(r["comm_radius_m"]),
             r["workspace_frac_r"], r["workspace_frac_c"])
        for r in rows
    )
    max_id = max(int(r["run_id"]) for r in rows)

    cells  = sorted({(int(float(r["area_size_m"])), int(r["n_drones"])) for r in rows})
    combos = list(itertools.product(
        R.ARTVA_NOISE_STDS, R.ACC_SIM_LIST, R.COMM_RADII, R.WORKSPACE_CENTERS))

    # Costruisce lista piatta di job (uno per run da aggiungere)
    jobs = []
    by_cell: Counter = Counter()
    for area, n in cells:
        for noise, acc, rc, ws in combos:
            ws_r, ws_c = R._ws_label(ws)
            k = _key(area, n, noise, acc, rc, ws_r, ws_c)
            deficit = max(0, TARGET - have.get(k, 0))
            for _ in range(deficit):
                jobs.append((area, n, noise, acc, rc, ws))
            if deficit:
                by_cell[(area, n)] += deficit

    total_new = len(jobs)
    print(f"Run da aggiungere per cella (target {TARGET}/combo):")
    for (area, n), c in sorted(by_cell.items()):
        print(f"  area={area} n={n}: +{c}")
    print(f"Totale nuove run: {total_new}  |  workers: {args.workers}")

    if args.dry_run or total_new == 0:
        return

    shutil.copy(args.out, args.out + ".bak")
    print(f"Backup: {args.out}.bak")

    # Pre-carica terreni nel processo principale (su Linux i fork ereditano la cache)
    unique_terrain_keys = {(j[0], j[5]) for j in jobs}
    print(f"Pre-caricamento {len(unique_terrain_keys)} configurazioni terreno...")
    for area, ws in sorted(unique_terrain_keys, key=lambda x: (x[0], str(x[1]))):
        R._load_terrain(area, center_frac=ws)

    done = 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=R.CSV_FIELDS)
        with tqdm(total=total_new, unit="run", dynamic_ncols=True) as bar:

            def _record(area, n, noise, acc, rc, ws, metrics, tb) -> None:
                nonlocal max_id, done
                ws_r, ws_c = R._ws_label(ws)
                if tb is not None:
                    tqdm.write(f"ERRORE area={area} n={n}: {tb.splitlines()[-1]}")
                    return
                max_id += 1
                w.writerow({
                    "run_id":           max_id,
                    "area_size_m":      area,
                    "n_drones":         n,
                    "artva_noise_std":  noise,
                    "acc_sim_ms2":      acc,
                    "comm_radius_m":    rc,
                    "workspace_frac_r": ws_r,
                    "workspace_frac_c": ws_c,
                    **metrics,
                })
                f.flush()
                done += 1
                bar.update(1)

            if args.workers == 1:
                for job in jobs:
                    _record(*_worker(job))
            else:
                with mp.Pool(processes=args.workers) as pool:
                    for result in pool.imap_unordered(_worker, jobs):
                        _record(*result)

    print(f"Fatto: aggiunte {done} run a {args.out}")


if __name__ == "__main__":
    main()

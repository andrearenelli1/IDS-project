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
Fai un backup prima (lo script lo fa in automatico in results.csv.bak).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
from collections import Counter

import run_experiments as R

OUT    = "results.csv"
TARGET = 13                       # vittime/combinazione obiettivo per ogni cella


def _key(area, n, noise, acc, rc, ws_r, ws_c):
    return (int(area), int(n), float(noise), float(acc), int(rc), ws_r, ws_c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.out)))
    have = Counter(
        _key(float(r["area_size_m"]), r["n_drones"], r["artva_noise_std"],
             r["acc_sim_ms2"], float(r["comm_radius_m"]),
             r["workspace_frac_r"], r["workspace_frac_c"])
        for r in rows
    )
    max_id = max(int(r["run_id"]) for r in rows)

    # Celle e combinazioni (90 = noise × acc × rc × ws)
    cells  = sorted({(int(float(r["area_size_m"])), int(r["n_drones"])) for r in rows})
    combos = list(itertools.product(
        R.ARTVA_NOISE_STDS, R.ACC_SIM_LIST, R.COMM_RADII, R.WORKSPACE_CENTERS))

    # Calcola il piano (deficit per combinazione)
    plan = []  # (area, n, noise, acc, rc, ws, ws_r, ws_c, deficit)
    for area, n in cells:
        for noise, acc, rc, ws in combos:
            ws_r, ws_c = R._ws_label(ws)
            k = _key(area, n, noise, acc, rc, ws_r, ws_c)
            deficit = max(0, TARGET - have.get(k, 0))
            if deficit:
                plan.append((area, n, noise, acc, rc, ws, ws_r, ws_c, deficit))

    total_new = sum(p[-1] for p in plan)
    by_cell = Counter()
    for area, n, *_, deficit in plan:
        by_cell[(area, n)] += deficit
    print(f"Run da aggiungere per cella (target {TARGET}/combo):")
    for (area, n), c in sorted(by_cell.items()):
        print(f"  area={area} n={n}: +{c}")
    print(f"Totale nuove run: {total_new}")

    if args.dry_run or total_new == 0:
        return

    shutil.copy(args.out, args.out + ".bak")
    print(f"Backup: {args.out}.bak")

    done = 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=R.CSV_FIELDS)
        for area, n, noise, acc, rc, ws, ws_r, ws_c, deficit in plan:
            for _ in range(deficit):
                metrics = R.run_one(area_size=area, n_drones=n, noise_std=noise,
                                    comm_radius=rc, acc_sim=acc, center_frac=ws)
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
                if done % 25 == 0:
                    print(f"  {done}/{total_new} ...", flush=True)

    print(f"Fatto: aggiunte {done} run a {args.out}")


if __name__ == "__main__":
    main()

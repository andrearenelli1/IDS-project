"""
plot_results.py
===============
Analisi visiva dello sweep parametrico (results.csv).

Uso:
    python plot_results.py                  # legge results.csv, salva PNG + mostra
    python plot_results.py --no-show        # solo salvataggio PNG
    python plot_results.py --out mio.csv    # file CSV diverso
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import product as iproduct
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────── Caricamento CSV ────────────────────────────────

FIELDS = [
    "run_id", "area_size_m", "n_drones",
    "victim_x_m", "victim_y_m", "victim_depth_m",
    "artva_noise_std", "noise_detect_factor", "comm_radius_m",
    "workspace_frac_r", "workspace_frac_c",
    "found", "time_found_s", "n_drones_stopped",
    "pos_variance_m2", "pos_std_m", "est_error_2d_m", "note",
]

def _flt(v: str) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def load(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "area":    int(float(r["area_size_m"])),
                "n":       int(r["n_drones"]),
                "depth":   _flt(r["victim_depth_m"]),
                "noise":   _flt(r["artva_noise_std"]),
                "detect":  _flt(r.get("noise_detect_factor", "100.0")),
                "rc":      int(float(r["comm_radius_m"])),
                "found":   r["found"].strip() == "True",
                "time":    _flt(r["time_found_s"]),
                "err2d":   _flt(r["est_error_2d_m"]),
                "pos_std": _flt(r["pos_std_m"]),
                "note":    r["note"].strip(),
            })
    return rows


# ────────────────────────── Helper aggregazioni ─────────────────────────────

def groupby(rows: list[dict], *keys: str) -> dict:
    """Raggruppa righe per combinazione di chiavi."""
    out: dict = defaultdict(list)
    for r in rows:
        k = tuple(r[k] for k in keys)
        out[k].append(r)
    return dict(out)


def success_rate(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(r["found"] for r in rows) / len(rows)


def median(vals: list[float]) -> float:
    v = sorted(x for x in vals if not math.isnan(x))
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


# ───────────────────────────── Stile comune ─────────────────────────────────

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f"]   # blu, arancio, verde
AREA_COLORS = {100: "#5778a4", 200: "#e49444"}
DEPTH_COLORS = {1.0: "#4e79a7", 3.0: "#f28e2b", 5.0: "#e15759"}
NOISE_LABELS = {1e-8: "10⁻⁸", 1e-7: "10⁻⁷", 1e-6: "10⁻⁶"}
N_DRONES_LIST  = [3, 4, 5]
AREAS          = [100, 200]
COMM_RADII     = [25, 50, 80, 120]
NOISE_STDS     = [1e-8, 1e-7, 1e-6]
DEPTHS         = [1.0, 3.0, 5.0]
DETECT_FACTORS = [50.0, 100.0]
DETECT_COLORS  = {50.0: "#e15759", 100.0: "#4e79a7"}  # rosso, blu
DETECT_LABELS  = {50.0: "factor = 50", 100.0: "factor = 100"}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    ":",
})


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Tasso di successo: heatmap rumore × raggio comm.
#          una colonna per n_droni, una riga per dimensione area
# ════════════════════════════════════════════════════════════════════════════

def fig_success_heatmap(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                             constrained_layout=True)
    fig.suptitle("Tasso di successo [%]  —  Rumore ARTVA × Raggio comunicazione",
                 fontsize=13, fontweight="bold")

    noise_idx = {n: i for i, n in enumerate(sorted(NOISE_STDS))}
    rc_idx    = {r: i for i, r in enumerate(sorted(COMM_RADII))}

    for col, n in enumerate(N_DRONES_LIST):
        for row_i, area in enumerate(AREAS):
            ax = axes[row_i, col]
            mat = np.full((len(COMM_RADII), len(NOISE_STDS)), float("nan"))

            grp = groupby([r for r in rows if r["n"] == n and r["area"] == area],
                          "rc", "noise")
            for (rc, noise), sub in grp.items():
                mat[rc_idx[rc], noise_idx[noise]] = success_rate(sub)

            im = ax.imshow(mat, vmin=0, vmax=100, cmap="RdYlGn",
                           aspect="auto", origin="lower")
            ax.set_xticks(range(len(NOISE_STDS)))
            ax.set_xticklabels([NOISE_LABELS[n_] for n_ in sorted(NOISE_STDS)],
                               fontsize=9)
            ax.set_yticks(range(len(COMM_RADII)))
            ax.set_yticklabels([f"{r} m" for r in sorted(COMM_RADII)],
                               fontsize=9)
            ax.set_xlabel("Rumore σ", fontsize=9)
            ax.set_ylabel("Raggio comm.", fontsize=9)
            ax.set_title(f"{n} droni — area {area} m", fontsize=10,
                         fontweight="bold")
            ax.grid(False)

            # valori nelle celle
            for i in range(len(COMM_RADII)):
                for j in range(len(NOISE_STDS)):
                    v = mat[i, j]
                    if not math.isnan(v):
                        ax.text(j, i, f"{v:.0f}%",
                                ha="center", va="center",
                                fontsize=8.5,
                                color="black" if v > 45 else "white",
                                fontweight="bold")

        plt.colorbar(im, ax=axes[:, col], shrink=0.6,
                     label="Successo [%]")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Tempo di ricerca: boxplot per n_droni, separati per area
# ════════════════════════════════════════════════════════════════════════════

def fig_time_boxplot(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True,
                             constrained_layout=True)
    fig.suptitle("Tempo di ricerca (run riusciti)  —  Effetto del numero di droni",
                 fontsize=13, fontweight="bold")

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=2.0),
                 flierprops=dict(marker=".", markersize=3, alpha=0.4))

    for ax, area in zip(axes, AREAS):
        data   = [[r["time"] for r in found if r["n"] == n and r["area"] == area]
                  for n in N_DRONES_LIST]
        bp = ax.boxplot(data, positions=range(len(N_DRONES_LIST)), **bp_kw)
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(N_DRONES_LIST)))
        ax.set_xticklabels([f"{n} droni" for n in N_DRONES_LIST], fontsize=10)
        ax.set_ylabel("Tempo [s]", fontsize=10)
        ax.set_title(f"Area {area} × {area} m", fontsize=11, fontweight="bold")

        # Annotazioni: mediana
        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, f" {med:.0f}s",
                        va="center", fontsize=8, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — Errore di stima 2D: dipendenza da rumore e profondità vittima
# ════════════════════════════════════════════════════════════════════════════

def fig_estimation_error(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"] and not math.isnan(r["err2d"])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Errore di stima 2D della sorgente  —  Profondità × Rumore ARTVA",
                 fontsize=13, fontweight="bold")

    noise_sorted = sorted(NOISE_STDS)
    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=2.0),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3))

    for ax, noise in zip(axes, noise_sorted):
        data = [[r["err2d"] for r in found if r["depth"] == d and r["noise"] == noise]
                for d in DEPTHS]
        bp = ax.boxplot(data, positions=range(len(DEPTHS)), **bp_kw)
        for patch, (d, color) in zip(bp["boxes"], DEPTH_COLORS.items()):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels([f"{d:.0f} m" for d in DEPTHS], fontsize=10)
        ax.set_xlabel("Profondità vittima", fontsize=10)
        ax.set_title(f"Rumore σ = {NOISE_LABELS[noise]}", fontsize=11,
                     fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Errore stima 2D [m]", fontsize=10)

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, f" {med:.1f}m",
                        va="center", fontsize=8, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — Raggio di comunicazione: tasso di successo vs comm_radius
#          linee per n_droni, pannelli per area
# ════════════════════════════════════════════════════════════════════════════

def fig_comm_radius(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Effetto del raggio di comunicazione UWB sul tasso di successo",
                 fontsize=13, fontweight="bold")

    rc_sorted = sorted(COMM_RADII)

    for ax, area in zip(axes, AREAS):
        for n, color in zip(N_DRONES_LIST, PALETTE):
            sr = []
            for rc in rc_sorted:
                sub = [r for r in rows if r["n"] == n and r["area"] == area
                       and r["rc"] == rc]
                sr.append(success_rate(sub))
            ax.plot(rc_sorted, sr, marker="o", lw=2.2, ms=8,
                    color=color, label=f"{n} droni")
            for x, y in zip(rc_sorted, sr):
                ax.annotate(f"{y:.0f}%", (x, y),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color=color)

        ax.set_xticks(rc_sorted)
        ax.set_xticklabels([f"{r} m" for r in rc_sorted])
        ax.set_xlabel("Raggio comunicazione [m]", fontsize=10)
        ax.set_ylabel("Tasso di successo [%]", fontsize=10)
        ax.set_title(f"Area {area} × {area} m", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.legend(fontsize=9)

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — CDF del tempo di ricerca per configurazione chiave
# ════════════════════════════════════════════════════════════════════════════

def fig_cdf(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("CDF del tempo di ricerca — distribuzione cumulata",
                 fontsize=13, fontweight="bold")

    for ax, area in zip(axes, AREAS):
        for n, color in zip(N_DRONES_LIST, PALETTE):
            times = sorted(r["time"] for r in found
                           if r["n"] == n and r["area"] == area)
            if not times:
                continue
            cdf = np.arange(1, len(times) + 1) / len(times)
            ax.step(times, cdf * 100, where="post", lw=2.0,
                    color=color, label=f"{n} droni  (n={len(times)})")

        ax.axhline(90, color="grey", lw=1.0, ls="--", alpha=0.6)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] < 900 else 860,
                91, "90%", fontsize=8, color="grey")

        ax.set_xlabel("Tempo di ricerca [s]", fontsize=10)
        ax.set_ylabel("CDF [%]", fontsize=10)
        ax.set_title(f"Area {area} × {area} m", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — Consenso: dispersione delle stime tra i droni (pos_std_m)
# ════════════════════════════════════════════════════════════════════════════

def fig_consensus_spread(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"] and not math.isnan(r["pos_std"])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Dispersione delle stime tra droni (σ posizione)  —  Effetto del rumore",
                 fontsize=13, fontweight="bold")

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=2.0),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3))

    noise_sorted = sorted(NOISE_STDS)
    noise_colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for ax, area in zip(axes, AREAS):
        data = [[r["pos_std"] for r in found if r["noise"] == noise and r["area"] == area]
                for noise in noise_sorted]
        bp = ax.boxplot(data, positions=range(len(noise_sorted)), **bp_kw)
        for patch, color in zip(bp["boxes"], noise_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(noise_sorted)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_sorted], fontsize=10)
        ax.set_xlabel("Rumore ARTVA σ", fontsize=10)
        ax.set_title(f"Area {area} × {area} m", fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("σ posizione [m]", fontsize=10)

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, f" {med:.3f}m",
                        va="center", fontsize=8, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — Effetto della soglia di detect (50 vs 100)
#          Riga superiore: tasso di successo per livello di rumore
#          Riga inferiore: tempo mediano di ricerca (run riusciti)
#          Colonne: dimensione area — linee tratteggiate/continue per i due fattori
# ════════════════════════════════════════════════════════════════════════════

def fig_detect_factor(rows: list[dict]) -> plt.Figure:
    noise_sorted = sorted(NOISE_STDS)
    noise_ticks  = [NOISE_LABELS[n] for n in noise_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True,
                             sharey="row")
    fig.suptitle("Effetto della soglia di rilevamento ARTVA  (NOISE_DETECT_FACTOR)",
                 fontsize=13, fontweight="bold")

    linestyles = {50.0: "--", 100.0: "-"}

    for col, area in enumerate(AREAS):
        ax_sr  = axes[0, col]   # success rate
        ax_t   = axes[1, col]   # median time

        for df in DETECT_FACTORS:
            sr_vals, t_vals = [], []
            for noise in noise_sorted:
                sub = [r for r in rows
                       if r["area"] == area and r["detect"] == df
                       and r["noise"] == noise]
                sr_vals.append(success_rate(sub))

                found_times = [r["time"] for r in sub if r["found"]]
                t_vals.append(median(found_times))

            color = DETECT_COLORS[df]
            ls    = linestyles[df]
            label = DETECT_LABELS[df]

            ax_sr.plot(range(len(noise_sorted)), sr_vals,
                       marker="o", lw=2.2, ms=8, color=color,
                       ls=ls, label=label)
            for x, y in zip(range(len(noise_sorted)), sr_vals):
                if not math.isnan(y):
                    ax_sr.annotate(f"{y:.0f}%", (x, y),
                                   textcoords="offset points", xytext=(0, 7),
                                   ha="center", fontsize=8, color=color)

            valid_t = [(x, y) for x, y in enumerate(t_vals)
                       if not math.isnan(y)]
            if valid_t:
                xs, ys = zip(*valid_t)
                ax_t.plot(xs, ys, marker="s", lw=2.2, ms=8, color=color,
                          ls=ls, label=label)
                for x, y in zip(xs, ys):
                    ax_t.annotate(f"{y:.0f}s", (x, y),
                                  textcoords="offset points", xytext=(0, 7),
                                  ha="center", fontsize=8, color=color)

        for ax in (ax_sr, ax_t):
            ax.set_xticks(range(len(noise_sorted)))
            ax.set_xticklabels(noise_ticks, fontsize=10)
            ax.set_xlabel("Rumore ARTVA σ", fontsize=10)
            ax.set_title(f"Area {area} × {area} m", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)

        ax_sr.set_ylabel("Tasso di successo [%]", fontsize=10)
        ax_sr.set_ylim(0, 108)
        axes[1, 0].set_ylabel("Tempo mediano [s]", fontsize=10)

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizzazione sweep ARTVA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", nargs="?", default="results.csv",
                        help="File CSV prodotto da run_experiments.py")
    args = parser.parse_args()

    print(f"Caricamento {args.csv} …")
    rows = load(args.csv)
    total   = len(rows)
    n_found = sum(r["found"] for r in rows)
    n_fail  = total - n_found
    print(f"  {total} run totali — {n_found} successi ({100*n_found/total:.1f}%) "
          f"— {n_fail} timeout")

    fig_success_heatmap(rows)
    fig_time_boxplot(rows)
    fig_estimation_error(rows)
    fig_comm_radius(rows)
    fig_cdf(rows)
    fig_consensus_spread(rows)
    fig_detect_factor(rows)

    plt.show()


if __name__ == "__main__":
    main()

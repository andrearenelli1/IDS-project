"""
plot_results.py
===============
Analisi visiva dello sweep parametrico (results.csv).

Uso:
    python plot_results.py                  # legge results.csv, mostra
    python plot_results.py results7200.csv  # file CSV alternativo
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

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


def _depth_bin(d: float) -> str:
    if math.isnan(d):
        return "N/A"
    for i, (lo, hi) in enumerate(zip(DEPTH_BIN_EDGES, DEPTH_BIN_EDGES[1:])):
        if lo <= d < hi:
            return DEPTH_BIN_LABELS[i]
    return DEPTH_BIN_LABELS[-1]


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
                "area":      int(float(r["area_size_m"])),
                "n":         int(r["n_drones"]),
                "depth":     _flt(r["victim_depth_m"]),
                "depth_bin": _depth_bin(_flt(r["victim_depth_m"])),
                "noise":     _flt(r["artva_noise_std"]),
                "detect":    _flt(r.get("noise_detect_factor", "100.0")),
                "rc":        int(float(r["comm_radius_m"])),
                "found":     r["found"].strip() == "True",
                "time":      _flt(r["time_found_s"]),
                "dist":      _flt(r.get("victim_dist_m", "nan")),
                "err2d":     _flt(r["est_error_2d_m"]),
                "pos_std":   _flt(r["pos_std_m"]),
                "note":      r["note"].strip(),
            })
    return rows


# ────────────────────────── Helper aggregazioni ─────────────────────────────

def groupby(rows: list[dict], *keys: str) -> dict:
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


# ───────────────────────────── Stile IEEE ───────────────────────────────────

PALETTE          = ["#4e79a7", "#f28e2b", "#59a14f"]
AREA_COLORS      = {100: "#5778a4", 200: "#e49444"}
NOISE_LABELS     = {1e-7: r"$10^{-7}$", 1e-6: r"$10^{-6}$", 1e-5: r"$10^{-5}$"}
N_DRONES_LIST    = [3, 4, 5]
AREAS            = [100, 200]
COMM_RADII       = [25, 50, 80, 120]
NOISE_STDS       = [1e-7, 1e-6, 1e-5]
DEPTHS           = [1.0, 3.0, 5.0]
DEPTH_BIN_EDGES  = [1.0, 2.0, 3.0, 4.0, 5.01]
DEPTH_BIN_LABELS = [r"1--2\,m", r"2--3\,m", r"3--4\,m", r"4--5\,m"]
DEPTH_BIN_COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]
DETECT_FACTORS   = [50.0, 100.0]
DETECT_COLORS    = {50.0: "#e15759", 100.0: "#4e79a7"}
DETECT_LABELS    = {50.0: "factor = 50", 100.0: "factor = 100"}


def _setup_ieee_style() -> None:
    plt.rcParams.update({
        "text.usetex":              True,
        "text.latex.preamble":      r"\usepackage{amsmath}",
        "font.family":              "serif",
        "font.serif":               ["Computer Modern Roman"],
        "axes.labelsize":           9,
        "font.size":                9,
        "legend.fontsize":          7,
        "xtick.labelsize":          8,
        "ytick.labelsize":          8,
        "axes.linewidth":           0.6,
        "grid.linewidth":           0.4,
        "lines.linewidth":          1.5,
        "figure.dpi":               200,
        "axes.grid":                True,
        "grid.alpha":               0.35,
        "grid.linestyle":           ":",
    })


_setup_ieee_style()


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Tasso di successo: heatmap rumore × raggio comm.
# ════════════════════════════════════════════════════════════════════════════

def fig_success_heatmap(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8),
                             constrained_layout=True)
    fig.suptitle(
        r"Success rate [\%] --- ARTVA noise $\sigma$ $\times$ comm.\ radius",
        fontsize=10, fontweight="bold")

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
            ax.set_xticklabels([NOISE_LABELS[n_] for n_ in sorted(NOISE_STDS)])
            ax.set_yticks(range(len(COMM_RADII)))
            ax.set_yticklabels([rf"{r}\,m" for r in sorted(COMM_RADII)])
            ax.set_xlabel(r"Noise $\sigma$")
            ax.set_ylabel(r"Comm.\ radius")
            ax.set_title(rf"{n} drones --- {area}\,m", fontsize=9, fontweight="bold")
            ax.grid(False)

            for i in range(len(COMM_RADII)):
                for j in range(len(NOISE_STDS)):
                    v = mat[i, j]
                    if not math.isnan(v):
                        ax.text(j, i, rf"{v:.0f}\%",
                                ha="center", va="center", fontsize=7,
                                color="black" if v > 45 else "white",
                                fontweight="bold")

        plt.colorbar(im, ax=axes[:, col], shrink=0.6, label=r"Success [\%]")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Tempo di ricerca: boxplot per n_droni, separati per area
# ════════════════════════════════════════════════════════════════════════════

def fig_time_boxplot(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True,
                             constrained_layout=True)
    fig.suptitle(r"Search time (successful runs) --- effect of drone count",
                 fontsize=10, fontweight="bold")

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=1.5),
                 flierprops=dict(marker=".", markersize=3, alpha=0.4))

    for ax, area in zip(axes, AREAS):
        data = [[r["time"] for r in found if r["n"] == n and r["area"] == area]
                for n in N_DRONES_LIST]
        bp = ax.boxplot(data, positions=range(len(N_DRONES_LIST)), **bp_kw)
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(N_DRONES_LIST)))
        ax.set_xticklabels([rf"{n} drones" for n in N_DRONES_LIST])
        ax.set_ylabel(r"Time [s]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, rf" {med:.0f}\,s",
                        va="center", fontsize=7, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — Raggio di comunicazione: tasso di successo vs comm_radius
# ════════════════════════════════════════════════════════════════════════════

def fig_comm_radius(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             constrained_layout=True, sharey=True)
    fig.suptitle(r"Effect of UWB communication radius on success rate",
                 fontsize=10, fontweight="bold")

    rc_sorted = sorted(COMM_RADII)

    for ax, area in zip(axes, AREAS):
        for n, color in zip(N_DRONES_LIST, PALETTE):
            sr = []
            for rc in rc_sorted:
                sub = [r for r in rows if r["n"] == n and r["area"] == area
                       and r["rc"] == rc]
                sr.append(success_rate(sub))
            ax.plot(rc_sorted, sr, marker="o", ms=5,
                    color=color, label=rf"{n} drones")
            for x, y in zip(rc_sorted, sr):
                ax.annotate(rf"{y:.0f}\%", (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=color)

        ax.set_xticks(rc_sorted)
        ax.set_xticklabels([rf"{r}\,m" for r in rc_sorted])
        ax.set_xlabel(r"Comm.\ radius [m]")
        ax.set_ylabel(r"Success rate [\%]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.legend()

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — CDF del tempo di ricerca
# ════════════════════════════════════════════════════════════════════════════

def fig_cdf(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             constrained_layout=True, sharey=True)
    fig.suptitle(r"CDF of search time --- cumulative distribution",
                 fontsize=10, fontweight="bold")

    for ax, area in zip(axes, AREAS):
        for n, color in zip(N_DRONES_LIST, PALETTE):
            times = sorted(r["time"] for r in found
                           if r["n"] == n and r["area"] == area)
            if not times:
                continue
            cdf = np.arange(1, len(times) + 1) / len(times)
            ax.step(times, cdf * 100, where="post",
                    color=color, label=rf"{n} drones  ($n$={len(times)})")

        ax.axhline(90, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] < 900 else 860,
                91, r"90\%", fontsize=7, color="grey")

        ax.set_xlabel(r"Search time [s]")
        ax.set_ylabel(r"CDF [\%]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 105)

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — Consenso: dispersione stime tra droni (pos_std_m)
# ════════════════════════════════════════════════════════════════════════════

def fig_consensus_spread(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"] and not math.isnan(r["pos_std"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             constrained_layout=True, sharey=True)
    fig.suptitle(
        r"Inter-drone estimate spread ($\sigma$ position) --- effect of noise",
        fontsize=10, fontweight="bold")

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=1.5),
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
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_sorted])
        ax.set_xlabel(r"ARTVA noise $\sigma$")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel(r"$\sigma$ position [m]")

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, rf" {med:.3f}\,m",
                        va="center", fontsize=7, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — Tempo vs distanza vittima–start
# ════════════════════════════════════════════════════════════════════════════

def fig_time_vs_distance(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows
             if r["found"] and not math.isnan(r["time"]) and not math.isnan(r["dist"])]
    if not found:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5), constrained_layout=True)
    fig.suptitle(r"Search time vs.\ victim--start distance",
                 fontsize=10, fontweight="bold")

    markers = {3: "o", 4: "s", 5: "^"}

    for col, area in enumerate(AREAS):
        ax_t = axes[0, col]
        ax_n = axes[1, col]

        sub = [r for r in found if r["area"] == area]
        if not sub:
            continue

        for n, color in zip(N_DRONES_LIST, PALETTE):
            pts = [r for r in sub if r["n"] == n]
            if not pts:
                continue
            xs = [r["dist"]             for r in pts]
            ys = [r["time"]             for r in pts]
            yn = [r["time"] / r["dist"] for r in pts]

            kw = dict(color=color, alpha=0.45, s=20, marker=markers[n],
                      label=rf"{n} drones")
            ax_t.scatter(xs, ys, **kw)
            ax_n.scatter(xs, yn, **kw)

            if len(xs) >= 3:
                for ax, ydata in [(ax_t, ys), (ax_n, yn)]:
                    coef  = np.polyfit(xs, ydata, 1)
                    x_fit = np.linspace(min(xs), max(xs), 100)
                    ax.plot(x_fit, np.polyval(coef, x_fit),
                            color=color, ls="--", alpha=0.8)

        for ax in (ax_t, ax_n):
            ax.set_title(rf"Area ${area}\times{area}$\,m",
                         fontsize=9, fontweight="bold")
            ax.legend(markerscale=1.4)

        ax_n.set_xlabel(r"Victim--start distance [m]")
        if col == 0:
            ax_t.set_ylabel(r"Search time [s]")
            ax_n.set_ylabel(r"Time / distance [s/m]")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 8 — Istogrammi del tempo di ricerca per parametro
# ════════════════════════════════════════════════════════════════════════════

def fig_time_histograms(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"] and not math.isnan(r["time"])]
    if not found:
        return None

    params = [
        (r"Number of drones",          "n",         N_DRONES_LIST,
         lambda v: rf"{v} drones",     PALETTE),
        (r"ARTVA noise $\sigma$",       "noise",     sorted(NOISE_STDS),
         lambda v: NOISE_LABELS[v],    ["#4e79a7", "#f28e2b", "#e15759"]),
        (r"Victim depth [m]",           "depth_bin", DEPTH_BIN_LABELS,
         lambda v: v,                   DEPTH_BIN_COLORS),
        (r"Comm.\ radius [m]",          "rc",        sorted(COMM_RADII),
         lambda v: rf"{v}\,m",         ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]),
    ]

    n_rows = len(params)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.16, 2.4 * n_rows),
                             constrained_layout=True, sharey=False)
    fig.suptitle(r"Search time distribution by parameter (successful runs)",
                 fontsize=10, fontweight="bold")

    bins = np.linspace(0, 600, 31)

    for row_i, (xlabel, key, values, label_fn, colors) in enumerate(params):
        for col, area in enumerate(AREAS):
            ax  = axes[row_i, col]
            sub = [r for r in found if r["area"] == area]

            for val, color in zip(values, colors):
                data = [r["time"] for r in sub if r[key] == val]
                if not data:
                    continue
                ax.hist(data, bins=bins, density=False, alpha=0.55,
                        color=color, label=label_fn(val), edgecolor="none")
                med = median(data)
                if not math.isnan(med):
                    ax.axvline(med, color=color, lw=1.2, ls="--", alpha=0.85)

            if row_i == n_rows - 1:
                ax.set_xlabel(r"Search time [s]")
            else:
                ax.set_xlabel("")
            ax.set_ylabel(r"Runs per bin")
            ax.set_title(rf"{xlabel} --- ${area}\times{area}$\,m",
                         fontsize=9, fontweight="bold")
            ax.set_xlim(0, 600)
            ax.legend()
            ax.xaxis.set_major_locator(mticker.MultipleLocator(150))

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

    print(f"Caricamento {args.csv} ...")
    rows = load(args.csv)
    _noise_set = set(NOISE_STDS)
    rows    = [r for r in rows if r["noise"] in _noise_set]
    total   = len(rows)
    n_found = sum(r["found"] for r in rows)
    n_fail  = total - n_found
    print(f"  {total} run totali -- {n_found} successi ({100*n_found/total:.1f}pct) "
          f"-- {n_fail} timeout")

    fig_success_heatmap(rows)
    fig_time_boxplot(rows)
    fig_comm_radius(rows)
    fig_cdf(rows)
    fig_consensus_spread(rows)
    fig_time_vs_distance(rows)
    fig_time_histograms(rows)
    plt.show()


if __name__ == "__main__":
    main()

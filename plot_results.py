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
    "artva_noise_std", "comm_radius_m",
    "workspace_frac_r", "workspace_frac_c",
    "found", "time_found_s", "n_drones_stopped",
    "pos_variance_m2", "pos_std_m",
    "landing_err_mean_m", "est_error_depth_m", "pf_std_xy_mean_m",
    "pf_ellipse_area_mean_m2", "pf_iou_mean", "note",
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
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "area":      int(float(r["area_size_m"])),
                "n":         int(r["n_drones"]),
                "vx":        _flt(r.get("victim_x_m", "nan")),
                "vy":        _flt(r.get("victim_y_m", "nan")),
                "depth":     _flt(r["victim_depth_m"]),
                "depth_bin": _depth_bin(_flt(r["victim_depth_m"])),
                "noise":     _flt(r["artva_noise_std"]),
                "rc":        int(float(r["comm_radius_m"])),
                "acc_sim":   _flt(r.get("acc_sim_ms2", "0.05")),
                "found":     r["found"].strip() == "True",
                "time":      _flt(r["time_found_s"]),
                "dist":      _flt(r.get("victim_dist_m", "nan")),
                "pos_std":     _flt(r["pos_std_m"]),
                "landing_err":   _flt(r.get("landing_err_mean_m",  "nan")),
                "depth_err":     _flt(r.get("est_error_depth_m",   "nan")),
                "pf_std_xy":     _flt(r.get("pf_std_xy_mean_m",    "nan")),
                "pf_area":       _flt(r.get("pf_ellipse_area_mean_m2", "nan")),
                "pf_iou":        _flt(r.get("pf_iou_mean",         "nan")),
                "note":          r["note"].strip(),
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
COMM_RADII       = [25, 60, 120]
NOISE_STDS       = [1e-7, 1e-6, 1e-5]
DEPTHS           = [1.0, 3.0, 5.0]
DEPTH_BIN_EDGES  = [1.0, 2.0, 3.0, 4.0, 5.01]
DEPTH_BIN_LABELS = [r"1--2\,m", r"2--3\,m", r"3--4\,m", r"4--5\,m"]
DEPTH_BIN_COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]
DETECT_FACTORS   = [50.0, 100.0]
DETECT_COLORS    = {50.0: "#e15759", 100.0: "#4e79a7"}
DETECT_LABELS    = {50.0: "factor = 50", 100.0: "factor = 100"}
ACC_SIM_LIST     = [0.05, 0.10]
ACC_SIM_LABELS   = {0.05: r"$0.05\,\mathrm{m/s^2}$ (calm)", 0.10: r"$0.10\,\mathrm{m/s^2}$ (windy)"}
ACC_SIM_COLORS   = {0.05: "#4e79a7", 0.10: "#e15759"}


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
# Fig 4b — Rumore ARTVA: tasso di successo vs noise_std
# ════════════════════════════════════════════════════════════════════════════

def fig_noise_success(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             constrained_layout=True, sharey=True)
    fig.suptitle(r"Effect of ARTVA noise $\sigma$ on success rate",
                 fontsize=10, fontweight="bold")

    noise_sorted = sorted(NOISE_STDS)
    noise_ticks  = [NOISE_LABELS[n] for n in noise_sorted]

    for ax, area in zip(axes, AREAS):
        for n, color in zip(N_DRONES_LIST, PALETTE):
            sr = []
            for noise in noise_sorted:
                sub = [r for r in rows if r["n"] == n and r["area"] == area
                       and r["noise"] == noise]
                sr.append(success_rate(sub))
            xs = range(len(noise_sorted))
            ax.plot(list(xs), sr, marker="o", ms=5,
                    color=color, label=rf"{n} drones")
            for x, y in zip(xs, sr):
                ax.annotate(rf"{y:.0f}\%", (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=color)

        ax.set_xticks(range(len(noise_sorted)))
        ax.set_xticklabels(noise_ticks)
        ax.set_xlabel(r"ARTVA noise $\sigma$")
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

    noise_sorted = sorted(NOISE_STDS)
    noise_colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, axes = plt.subplots(
        2, 2, figsize=(7.16, 5.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle(
        r"Inter-drone estimate spread ($\sigma$ position) --- effect of noise",
        fontsize=10, fontweight="bold")

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=1.5),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3))

    all_vals = [r["pos_std"] for r in found]
    y_min = max(0, min(all_vals) * 0.9) if all_vals else 0
    y_max = max(all_vals) * 1.05 if all_vals else 1
    n_bins = 30
    bins = np.linspace(y_min, y_max, n_bins + 1)

    for row_i, area in enumerate(AREAS):
        ax_bp   = axes[row_i, 0]
        ax_hist = axes[row_i, 1]
        ax_hist.sharey(ax_bp)

        data = [[r["pos_std"] for r in found if r["noise"] == noise and r["area"] == area]
                for noise in noise_sorted]

        # ── boxplot ──────────────────────────────────────────────────────
        bp = ax_bp.boxplot(data, positions=range(len(noise_sorted)), **bp_kw)
        for patch, color in zip(bp["boxes"], noise_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax_bp.set_xticks(range(len(noise_sorted)))
        ax_bp.set_xticklabels([NOISE_LABELS[n] for n in noise_sorted])
        ax_bp.set_xlabel(r"ARTVA noise $\sigma$")
        ax_bp.set_ylabel(r"$\sigma$ position [m]")
        ax_bp.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax_bp.text(pos + 0.32, med, rf" {med:.3f}\,m",
                           va="center", fontsize=7, color="#333333")

        # ── istogramma orizzontale (condivide asse y col boxplot) ─────────
        for d, color, noise in zip(data, noise_colors, noise_sorted):
            if not d:
                continue
            ax_hist.hist(d, bins=bins, orientation="horizontal",
                         alpha=0.55, color=color,
                         label=NOISE_LABELS[noise], edgecolor="none")
            med = median(d)
            if not math.isnan(med):
                ax_hist.axhline(med, color=color, lw=1.1, ls="--", alpha=0.85)

        ax_hist.set_xlabel(r"Count")
        ax_hist.tick_params(labelleft=False)
        ax_hist.legend(fontsize=7)

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
        (r"Number of drones",                    "n",        N_DRONES_LIST,
         lambda v: rf"{v} drones",               PALETTE),
        (r"ARTVA noise $\sigma$",                "noise",    sorted(NOISE_STDS),
         lambda v: NOISE_LABELS[v],              ["#4e79a7", "#f28e2b", "#e15759"]),
        (r"Motion noise $\sigma_{\mathrm{acc}}$","acc_sim",  ACC_SIM_LIST,
         lambda v: ACC_SIM_LABELS.get(v, rf"{v}"),
         [ACC_SIM_COLORS[a] for a in ACC_SIM_LIST]),
        (r"Victim depth [m]",                    "depth_bin",DEPTH_BIN_LABELS,
         lambda v: v,                             DEPTH_BIN_COLORS),
        (r"Comm.\ radius [m]",                   "rc",       sorted(COMM_RADII),
         lambda v: rf"{v}\,m",                   ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]),
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
# Fig 9 — Rumore accelerazione: tasso di successo vs σ_acc
# ════════════════════════════════════════════════════════════════════════════

def fig_acc_sim(rows: list[dict]) -> plt.Figure:
    acc_vals = sorted(set(r["acc_sim"] for r in rows if not math.isnan(r["acc_sim"])))
    if len(acc_vals) < 2:
        return None   # CSV vecchio senza la colonna: nulla da mostrare

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             constrained_layout=True, sharey=True)
    fig.suptitle(
        r"Effect of motion noise $\sigma_{\mathrm{acc}}$ on success rate",
        fontsize=10, fontweight="bold",
    )

    for ax, area in zip(axes, AREAS):
        for n, color in zip(N_DRONES_LIST, PALETTE):
            sr_vals = []
            for acc in acc_vals:
                sub = [r for r in rows if r["n"] == n and r["area"] == area
                       and r["acc_sim"] == acc]
                sr_vals.append(success_rate(sub))
            xs = range(len(acc_vals))
            ax.plot(list(xs), sr_vals, marker="o", ms=5,
                    color=color, label=rf"{n} drones")
            for x, y in zip(xs, sr_vals):
                ax.annotate(rf"{y:.0f}\%", (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=color)

        ax.set_xticks(range(len(acc_vals)))
        ax.set_xticklabels([ACC_SIM_LABELS.get(a, rf"{a:.2f}") for a in acc_vals])
        ax.set_xlabel(r"Motion noise $\sigma_{\mathrm{acc}}$")
        ax.set_ylabel(r"Success rate [\%]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.legend()

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 10 — Mappa posizioni vittima
# ════════════════════════════════════════════════════════════════════════════

def fig_victim_map(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8), constrained_layout=True)
    fig.suptitle(r"Victim positions --- green: found, red: not found",
                 fontsize=10, fontweight="bold")

    for ax, area in zip(axes, AREAS):
        ax.set_xlim(0, area)
        ax.set_ylim(0, area)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")

        sub     = [r for r in rows if r["area"] == area
                   and not math.isnan(r["vx"]) and not math.isnan(r["vy"])]
        found_r = [r for r in sub if r["found"]]
        fail_r  = [r for r in sub if not r["found"]]

        for r_list, color, label in [
            (fail_r,  "#d62728", rf"Not found ({len(fail_r)})"),
            (found_r, "#2ca02c", rf"Found ({len(found_r)})"),
        ]:
            if not r_list:
                continue
            ax.scatter([r["vx"] for r in r_list],
                       [r["vy"] for r in r_list],
                       c=color, s=1, alpha=0.5, zorder=3, label=label)

        ax.legend(fontsize=7, loc="upper right")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 11 — Errore XY di landing e errore stima profondità
# ════════════════════════════════════════════════════════════════════════════

def fig_localization_errors(rows: list[dict]) -> plt.Figure:
    found = [r for r in rows if r["found"]]

    noise_sorted = sorted(NOISE_STDS)
    noise_colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5), constrained_layout=True)
    fig.suptitle(
        r"Localization accuracy: XY landing error and depth estimation error",
        fontsize=10, fontweight="bold",
    )

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=1.5),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3))

    # ── Riga 0: errore XY di landing per livello di rumore ────────────────
    for col, area in enumerate(AREAS):
        ax    = axes[0, col]
        valid = [r for r in found
                 if not math.isnan(r["landing_err"]) and r["area"] == area]
        data  = [[r["landing_err"] for r in valid if r["noise"] == noise]
                 for noise in noise_sorted]

        bp = ax.boxplot(data, positions=range(len(noise_sorted)), **bp_kw)
        for patch, color in zip(bp["boxes"], noise_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(noise_sorted)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_sorted])
        ax.set_xlabel(r"ARTVA noise $\sigma$")
        ax.set_ylabel(r"XY landing error [m]")
        ax.set_title(rf"Landing error --- ${area}\times{area}$\,m",
                     fontsize=9, fontweight="bold")

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, rf" {med:.2f}\,m",
                        va="center", fontsize=7, color="#333333")

    # ── Riga 1: errore stima profondità per bin di profondità ─────────────
    for col, area in enumerate(AREAS):
        ax    = axes[1, col]
        valid = [r for r in found
                 if not math.isnan(r["depth_err"]) and r["area"] == area]
        data  = [[r["depth_err"] for r in valid if r["depth_bin"] == bin_]
                 for bin_ in DEPTH_BIN_LABELS]

        bp = ax.boxplot(data, positions=range(len(DEPTH_BIN_LABELS)), **bp_kw)
        for patch, color in zip(bp["boxes"], DEPTH_BIN_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(DEPTH_BIN_LABELS)))
        ax.set_xticklabels(DEPTH_BIN_LABELS)
        ax.set_xlabel(r"Victim depth [m]")
        ax.set_ylabel(r"Depth estimation error [m]")
        ax.set_title(rf"Depth error --- ${area}\times{area}$\,m",
                     fontsize=9, fontweight="bold")

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, rf" {med:.2f}\,m",
                        va="center", fontsize=7, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 12 — PF intra-drone uncertainty (pf_std_xy_mean_m)
# ════════════════════════════════════════════════════════════════════════════

def fig_pf_uncertainty(rows: list[dict]) -> plt.Figure:
    """
    Incertezza intra-drone del Particle Filter (||σ_xy|| medio).
    Diversa da pos_std_m (dispersione *inter*-drone): questa misura quanto
    ogni singolo drone è incerto sulla posizione della sorgente.
    Atteso: diminuisce con il rumore ARTVA — la soglia di rilevamento dinamica
    (mu+5σ) aumenta con il rumore, quindi il drone rileva la sorgente solo da
    vicino, il PF si inizializza con r piccolo e le particelle restano concentrate.
    """
    found = [r for r in rows if r["found"] and not math.isnan(r["pf_std_xy"])]
    if not found:
        return None

    noise_sorted = sorted(NOISE_STDS)
    noise_colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             constrained_layout=True, sharey=True)
    fig.suptitle(
        r"Particle Filter intra-drone XY uncertainty $\|\sigma_{xy}\|$ vs.\ ARTVA noise",
        fontsize=10, fontweight="bold",
    )

    bp_kw = dict(patch_artist=True, notch=False, widths=0.55,
                 medianprops=dict(color="black", lw=1.5),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3))

    for ax, area in zip(axes, AREAS):
        data = [
            [r["pf_std_xy"] for r in found if r["noise"] == noise and r["area"] == area]
            for noise in noise_sorted
        ]
        bp = ax.boxplot(data, positions=range(len(noise_sorted)), **bp_kw)
        for patch, color in zip(bp["boxes"], noise_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(len(noise_sorted)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_sorted])
        ax.set_xlabel(r"ARTVA noise $\sigma$")
        ax.set_ylabel(r"$\|\sigma_{xy}\|$ PF [m]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")

        for pos, d in enumerate(data):
            if d:
                med = median(d)
                ax.text(pos + 0.32, med, rf" {med:.2f}\,m",
                        va="center", fontsize=7, color="#333333")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Fig 13 — Consenso PF: IoU media inter-drone vs area media ellisse 95%
# ════════════════════════════════════════════════════════════════════════════

def fig_pf_ellipse_consensus(rows: list[dict]) -> plt.Figure:
    """
    Una metrica di consenso inter-drone più informativa di pos_std_m.

    Per ogni run (con ≥2 droni a PF attivo):
      - asse x: IoU media a coppie tra le ellissi di confidenza 95% dei droni
                (consenso: 100% = stime perfettamente sovrapposte);
      - asse y: area media dell'ellisse di confidenza 95% [m²]
                (incertezza intra-drone: più piccola = più confidente).

    Il regime ideale è in basso-a-destra (confidenti E d'accordo). La IoU evita
    il confondente geometrico dell'intersezione grezza (box grandi → più overlap).
    Marker pieno = run riuscito, vuoto = fallito.
    """
    valid = [r for r in rows
             if not math.isnan(r["pf_iou"]) and not math.isnan(r["pf_area"])
             and r["pf_area"] > 0]
    if not valid:
        return None

    color_found  = "#4e79a7"
    color_failed = "#e15759"

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6),
                             constrained_layout=True, sharey=True)
    fig.suptitle(
        r"Inter-drone PF consensus --- mean pairwise IoU vs.\ mean 95\% ellipse area",
        fontsize=10, fontweight="bold")

    for ax, area in zip(axes, AREAS):
        sub   = [r for r in valid if r["area"] == area]
        pts_f = [r for r in sub if r["found"]]
        pts_n = [r for r in sub if not r["found"]]
        if pts_f:
            ax.scatter([100 * r["pf_iou"] for r in pts_f],
                       [r["pf_area"] for r in pts_f],
                       c=color_found, s=16, alpha=0.55, edgecolors="none",
                       label="found")
        if pts_n:
            ax.scatter([100 * r["pf_iou"] for r in pts_n],
                       [r["pf_area"] for r in pts_n],
                       facecolors="none", edgecolors=color_failed, s=16,
                       linewidths=0.8, alpha=0.7, label="failed")

        ax.set_yscale("log")
        ax.set_xlim(-3, 103)
        ax.set_xlabel(r"Mean pairwise IoU [\%]")
        ax.set_ylabel(r"Mean 95\% ellipse area [m$^2$]")
        ax.set_title(rf"Area ${area}\times{area}$\,m", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", ncol=1)

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
    fig_noise_success(rows)
    fig_cdf(rows)
    fig_consensus_spread(rows)
    fig_time_vs_distance(rows)
    fig_time_histograms(rows)
    fig_acc_sim(rows)
    fig_victim_map(rows)
    fig_localization_errors(rows)
    fig_pf_uncertainty(rows)
    fig_pf_ellipse_consensus(rows)
    plt.show()


if __name__ == "__main__":
    main()

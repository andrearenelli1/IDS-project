"""
dem_tinitaly.py
===============
Lettura, visualizzazione e interpolazione di un tile DEM TINItaly
(file: w51065_s10.tif — risoluzione 10 m, proiezione UTM/WGS84).

Dipendenze
----------
    pip install tifffile numpy scipy matplotlib

Utilizzo
--------
    python dem_tinitaly.py

Il file w51065_s10.tif deve trovarsi nella stessa directory dello script
(oppure modificare TIF_PATH qui sotto).

Output
------
  Figura 1 : DEM completo (hillshade + quota)
  Figura 2 : Area 200×200 m + superficie interpolata (RBF e griglia fine)
"""

import os
import sys
import struct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# Percorso al file
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIF_PATH   = os.path.join(SCRIPT_DIR, "w51065_s10.tif")

# ---------------------------------------------------------------------------
# Parametri area di interesse (modifica se vuoi un'area diversa)
# ---------------------------------------------------------------------------
# Offset in pixel dall'angolo top-left del tile per il centro dell'area 200x200 m.
# Con risoluzione 10 m/pixel, 200 m = 20 pixel.
# Puoi modificare CENTER_ROW / CENTER_COL per puntare a una zona diversa.
CENTER_ROW = None   # None → usa il centro del tile
CENTER_COL = None

AREA_SIZE_M  = 200    # lato dell'area quadrata [m]
INTERP_GRID  = 100    # punti per lato nella griglia fine di interpolazione

NODATA_VALUE = -9999  # valore NoData TINItaly

CMAP_DEM     = "terrain"
CMAP_INTERP  = "plasma"


# ===========================================================================
# Lettura GeoTIFF (senza rasterio/GDAL — solo tifffile)
# ===========================================================================

def read_geotiff(path: str):
    """
    Legge un GeoTIFF e restituisce:
      dem   : array 2-D float (righe × colonne) — quota [m]
      transform : (x_origin, pixel_w, y_origin, pixel_h)
                  pixel_h è negativo (y cresce verso il basso nell'immagine)

    Usa tifffile per leggere i dati e i tag TIFF.
    Il tag GeoTIFF ModelPixelScaleTag (33550) e ModelTiepointTag (33922)
    sono estratti manualmente per ricavare la trasformazione affine.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("Installa tifffile:  pip install tifffile")

    with tifffile.TiffFile(path) as tif:
        page   = tif.pages[0]
        dem    = page.asarray().astype(float)

        # — Tag GeoTIFF per la georiferimento —
        pixel_scale = None
        tie_point   = None

        for tag in page.tags.values():
            if tag.code == 33550:   # ModelPixelScaleTag
                pixel_scale = tag.value  # (scale_x, scale_y, scale_z)
            elif tag.code == 33922: # ModelTiepointTag
                # (i, j, k, X, Y, Z) — solitamente un singolo tie-point
                tp = tag.value
                # Ogni tie-point è 6 valori
                tie_point = tp[3], tp[4]   # (X_origin, Y_origin)

    # NoData → NaN
    dem[dem == NODATA_VALUE] = np.nan

    if pixel_scale is not None and tie_point is not None:
        x_origin = tie_point[0]
        y_origin = tie_point[1]
        pixel_w  =  pixel_scale[0]
        pixel_h  = -pixel_scale[1]   # negativo: y decresce con le righe
        transform = (x_origin, pixel_w, y_origin, pixel_h)
    else:
        # Fallback: coordinate pixel (caso in cui i tag GeoTIFF siano assenti)
        print("[WARN] Tag GeoTIFF non trovati — uso coordinate pixel.")
        transform = (0.0, 1.0, dem.shape[0], -1.0)

    return dem, transform


def pixel_to_coords(row, col, transform):
    """Converte indici pixel in coordinate geografiche/proiettate."""
    x_origin, pixel_w, y_origin, pixel_h = transform
    x = x_origin + col * pixel_w
    y = y_origin + row * pixel_h
    return x, y


def coords_extent(dem, transform):
    """Restituisce [xmin, xmax, ymin, ymax] in unità mappa."""
    rows, cols = dem.shape
    x_origin, pixel_w, y_origin, pixel_h = transform
    xmin = x_origin
    xmax = x_origin + cols * pixel_w
    ymin = y_origin + rows * pixel_h
    ymax = y_origin
    return xmin, xmax, min(ymin, ymax), max(ymin, ymax)


# ===========================================================================
# Plot 1 — DEM completo con hillshade
# ===========================================================================

def plot_full_dem(dem, transform, area_x_coords=None, area_y_coords=None):
    """
    Visualizza il DEM completo con colormap terrain + hillshade.
    Se area_x_coords e area_y_coords sono forniti, disegna il riquadro
    dell'area 200×200 m selezionata.
    """
    rows, cols = dem.shape
    xmin, xmax, ymin, ymax = coords_extent(dem, transform)

    # Hillshade (illuminazione artificiale)
    ls  = LightSource(azdeg=315, altdeg=45)
    hs  = ls.hillshade(np.nan_to_num(dem, nan=0.0), vert_exag=2)

    vmin = np.nanpercentile(dem, 2)
    vmax = np.nanpercentile(dem, 98)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("#333333")
    im = ax.imshow(dem, cmap=CMAP_DEM, vmin=vmin, vmax=vmax,
                   extent=[xmin, xmax, ymin, ymax],
                   origin="upper", interpolation="bilinear", zorder=2)
    ax.imshow(hs, cmap="gray", alpha=0.35,
              extent=[xmin, xmax, ymin, ymax],
              origin="upper", interpolation="bilinear", zorder=3)

    # — Riquadro area 200×200 m —
    if area_x_coords is not None and area_y_coords is not None:
        import matplotlib.patches as mpatches
        rx     = area_x_coords.min()
        ry     = area_y_coords.min()
        rw     = area_x_coords.max() - area_x_coords.min()
        rh     = area_y_coords.max() - area_y_coords.min()
        rect   = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=2.0, edgecolor="#ff4444", facecolor="none",
            linestyle="--", zorder=5,
        )
        ax.add_patch(rect)
        # Etichetta con freccia
        cx_area = rx + rw / 2
        cy_area = ry + rh / 2
        ax.annotate(
            f"Area {AREA_SIZE_M}×{AREA_SIZE_M} m",
            xy=(cx_area, ry),
            xytext=(cx_area, ry - (ymax - ymin) * 0.07),
            color="#ff4444", fontsize=9, fontweight="bold",
            ha="center", va="top", zorder=6,
            arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.5),
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Quota [m s.l.m.]", fontsize=10)
    ax.set_xlabel("E  [m UTM]", fontsize=10)
    ax.set_ylabel("N  [m UTM]", fontsize=10)
    ax.set_title(
        f"DEM TINItaly — {os.path.basename(TIF_PATH)}\n"
        f"{cols}×{rows} px  ·  risoluzione {abs(transform[1]):.0f} m/px",
        fontsize=11, fontweight="bold"
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ===========================================================================
# Estrai area 200×200 m
# ===========================================================================

def extract_area(dem, transform, center_row=None, center_col=None,
                 size_m=200):
    """
    Estrae un'area quadrata di 'size_m' metri attorno al pixel centrale.

    Returns
    -------
    sub_dem  : array 2-D  (size_px × size_px)
    x_coords : array 1-D coordinate E dei pixel [m]
    y_coords : array 1-D coordinate N dei pixel [m]
    (cr, cc) : centro in pixel (row, col)
    """
    _, pixel_w, _, pixel_h = transform
    pixel_size = abs(pixel_w)       # m/pixel (quadrato)
    half_px    = size_m / (2 * pixel_size)  # pixel dal centro al bordo
    half_px    = int(np.round(half_px))

    rows, cols = dem.shape
    if center_row is None:
        center_row = rows // 2
    if center_col is None:
        center_col = cols // 2

    r0 = max(0, center_row - half_px)
    r1 = min(rows, center_row + half_px)
    c0 = max(0, center_col - half_px)
    c1 = min(cols, center_col + half_px)

    sub_dem = dem[r0:r1, c0:c1]

    # Coordinate in unità mappa (m UTM)
    col_idx = np.arange(c0, c1)
    row_idx = np.arange(r0, r1)
    x_origin, pw, y_origin, ph = transform
    x_coords = x_origin + col_idx * pw
    y_coords = y_origin + row_idx * ph   # ph < 0

    return sub_dem, x_coords, y_coords, (center_row, center_col)


# ===========================================================================
# Interpolazione RBF sulla superficie 200×200 m
# ===========================================================================

def interpolate_surface(sub_dem, x_coords, y_coords, grid_n=INTERP_GRID):
    """
    Interpola il DEM su una griglia più fine con RBFInterpolator (thin-plate).

    Returns
    -------
    xi, yi : griglie fine meshgrid (grid_n × grid_n)
    zi     : quota interpolata sulla griglia fine
    """
    # — Punti di controllo (solo pixel validi, senza NaN) —
    XX, YY = np.meshgrid(x_coords, y_coords)   # (r, c) → (y, x)
    mask   = ~np.isnan(sub_dem)

    x_ctrl = XX[mask].ravel()
    y_ctrl = YY[mask].ravel()
    z_ctrl = sub_dem[mask].ravel()

    # Normalizza per migliore condizionamento numerico
    x_mean, x_std = x_ctrl.mean(), x_ctrl.std()
    y_mean, y_std = y_ctrl.mean(), y_ctrl.std()

    pts_norm = np.column_stack([
        (x_ctrl - x_mean) / x_std,
        (y_ctrl - y_mean) / y_std,
    ])

    # RBF thin-plate spline
    rbf = RBFInterpolator(pts_norm, z_ctrl, kernel="thin_plate_spline",
                          smoothing=0.1)

    # — Griglia fine —
    xi_1d = np.linspace(x_coords.min(), x_coords.max(), grid_n)
    yi_1d = np.linspace(y_coords.min(), y_coords.max(), grid_n)
    xi, yi = np.meshgrid(xi_1d, yi_1d)

    query_norm = np.column_stack([
        (xi.ravel() - x_mean) / x_std,
        (yi.ravel() - y_mean) / y_std,
    ])
    zi = rbf(query_norm).reshape(grid_n, grid_n)

    return xi, yi, zi


# ===========================================================================
# Plot 2 — Area 200×200 m + superficie interpolata
# ===========================================================================

def plot_area_and_interpolation(sub_dem, x_coords, y_coords,
                                xi, yi, zi, center_xy):
    """
    Layout a 4 pannelli:
      (A) DEM originale dell'area 200×200 m
      (B) Superficie interpolata (griglia fine)
      (C) Vista 3-D superficie interpolata
      (D) Differenza interpolazione − DEM originale
    """
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()   # y_coords scende

    # Rimappa y per extent imshow
    ey = [ymax, ymin] if y_coords[0] > y_coords[-1] else [ymin, ymax]
    extent_orig = [xmin, xmax, ey[0], ey[1]]
    extent_interp = [xi.min(), xi.max(), yi.min(), yi.max()]

    # Vmin/vmax comuni
    vmin = np.nanpercentile(sub_dem, 1)
    vmax = np.nanpercentile(sub_dem, 99)

    ls = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("#ffffff")

    # ── A: DEM originale ─────────────────────────────────────────────────
    ax_a = fig.add_subplot(2, 2, 1)
    im_a = ax_a.imshow(sub_dem, cmap=CMAP_DEM, vmin=vmin, vmax=vmax,
                       extent=extent_orig, origin="upper",
                       interpolation="nearest")
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb_a.set_label("Quota [m]", fontsize=9)
    ax_a.set_title("A — DEM originale 200×200 m", fontweight="bold",
                   fontsize=10)
    ax_a.set_xlabel("E [m UTM]", fontsize=9)
    ax_a.set_ylabel("N [m UTM]", fontsize=9)
    ax_a.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_a.tick_params(labelsize=8)
    ax_a.set_facecolor("#cccccc")

    # ── B: Superficie interpolata ─────────────────────────────────────────
    ax_b = fig.add_subplot(2, 2, 2)
    im_b = ax_b.imshow(zi, cmap=CMAP_INTERP, vmin=vmin, vmax=vmax,
                       extent=extent_interp, origin="lower",
                       interpolation="bilinear")
    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label("Quota [m]", fontsize=9)
    ax_b.set_title(f"B — Interpolata RBF (thin-plate)  {INTERP_GRID}²",
                   fontweight="bold", fontsize=10)
    ax_b.set_xlabel("E [m UTM]", fontsize=9)
    ax_b.set_ylabel("N [m UTM]", fontsize=9)
    ax_b.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_b.tick_params(labelsize=8)

    # ── C: Vista 3-D superficie interpolata ───────────────────────────────
    ax_c = fig.add_subplot(2, 2, 3, projection="3d")
    hs_3d = ls.hillshade(zi, vert_exag=2)
    surf  = ax_c.plot_surface(
        xi, yi, zi,
        facecolors=plt.get_cmap(CMAP_INTERP)(
            (zi - vmin) / max(vmax - vmin, 1e-6)
        ),
        rcount=60, ccount=60,
        linewidth=0, antialiased=True, alpha=0.95,
    )
    ax_c.set_xlabel("E [m]", fontsize=8, labelpad=3)
    ax_c.set_ylabel("N [m]", fontsize=8, labelpad=3)
    ax_c.set_zlabel("z [m]", fontsize=8, labelpad=3)
    ax_c.set_title("C — Vista 3-D (RBF)", fontweight="bold", fontsize=10)
    ax_c.tick_params(labelsize=7)
    ax_c.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax_c.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # ── D: Differenza ─────────────────────────────────────────────────────
    # Ricampiona il DEM originale sulla stessa griglia dell'interpolazione
    from scipy.interpolate import RegularGridInterpolator
    XX_orig, YY_orig = np.meshgrid(x_coords, y_coords)
    # y_coords è decrescente: inverti per RegularGridInterpolator (vuole valori crescenti)
    y_asc = np.sort(y_coords)
    sub_asc = sub_dem[::-1, :] if y_coords[0] > y_coords[-1] else sub_dem

    # Sostituisci NaN con media per evitare problemi nell'interpolatore
    sub_filled = sub_asc.copy()
    mean_z     = np.nanmean(sub_filled)
    sub_filled[np.isnan(sub_filled)] = mean_z

    rgi   = RegularGridInterpolator(
        (y_asc, x_coords), sub_filled, method="linear",
        bounds_error=False, fill_value=np.nan,
    )
    pts   = np.column_stack([yi.ravel(), xi.ravel()])
    z_ref = rgi(pts).reshape(INTERP_GRID, INTERP_GRID)
    diff  = zi - z_ref

    ax_d = fig.add_subplot(2, 2, 4)
    vd   = np.nanpercentile(np.abs(diff), 98)
    im_d = ax_d.imshow(diff, cmap="RdBu_r", vmin=-vd, vmax=vd,
                       extent=extent_interp, origin="lower",
                       interpolation="bilinear")
    cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb_d.set_label("Δ quota [m]", fontsize=9)
    ax_d.set_title("D — Residuo: interpolata − originale",
                   fontweight="bold", fontsize=10)
    ax_d.set_xlabel("E [m UTM]", fontsize=9)
    ax_d.set_ylabel("N [m UTM]", fontsize=9)
    ax_d.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_d.tick_params(labelsize=8)

    fig.suptitle(
        f"TINItaly DEM — Area 200×200 m  (centro: "
        f"E={center_xy[0]:.0f}, N={center_xy[1]:.0f})\n"
        f"Interpolazione RBF thin-plate su griglia {INTERP_GRID}×{INTERP_GRID}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    if not os.path.isfile(TIF_PATH):
        print(
            f"[ERRORE] File non trovato: {TIF_PATH}\n"
            "Scarica 'w51065_s10.tif' da:\n"
            "  https://tinitaly.pi.ingv.it/Download_Area1_1.html\n"
            "e posizionalo nella stessa directory di questo script."
        )
        sys.exit(1)

    print(f"Lettura: {TIF_PATH}")
    dem, transform = read_geotiff(TIF_PATH)
    rows, cols = dem.shape
    pixel_size = abs(transform[1])
    print(f"  Dimensioni : {cols} × {rows} pixel")
    print(f"  Risoluzione: {pixel_size:.1f} m/pixel")
    print(f"  Quota min/max (no-data escluso): "
          f"{np.nanmin(dem):.1f} / {np.nanmax(dem):.1f} m")

    # ── Estrai area 200×200 m ────────────────────────────────────────────
    sub_dem, x_coords, y_coords, (cr, cc) = extract_area(
        dem, transform,
        center_row=CENTER_ROW,
        center_col=CENTER_COL,
        size_m=AREA_SIZE_M,
    )
    cx, cy = pixel_to_coords(cr, cc, transform)
    print(f"\nArea selezionata: {AREA_SIZE_M}×{AREA_SIZE_M} m  "
          f"(centro pixel {cr},{cc} → E={cx:.0f} N={cy:.0f})")
    print(f"  Dimensioni sub-DEM: {sub_dem.shape[1]}×{sub_dem.shape[0]} px")
    print(f"  Quota min/max area: "
          f"{np.nanmin(sub_dem):.1f} / {np.nanmax(sub_dem):.1f} m")

    # ── Plot 1: DEM completo con riquadro area ────────────────────────────
    fig1 = plot_full_dem(dem, transform,
                         area_x_coords=x_coords, area_y_coords=y_coords)

    # ── Interpolazione RBF ───────────────────────────────────────────────
    print(f"\nInterpolazione RBF thin-plate su griglia "
          f"{INTERP_GRID}×{INTERP_GRID}...")
    xi, yi, zi = interpolate_surface(sub_dem, x_coords, y_coords)
    print("  Completata.")

    # ── Plot 2: area + interpolazione ────────────────────────────────────
    fig2 = plot_area_and_interpolation(
        sub_dem, x_coords, y_coords, xi, yi, zi, (cx, cy)
    )

    plt.show()
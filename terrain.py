"""
terrain.py
==========
Lettura DEM TINItaly, estrazione area di lavoro e wrapper interpolato.

Consolida dem_tinitaly.py + terrain.py (eliminando ridondanze).

Funzioni pubbliche
------------------
  read_geotiff          — legge un GeoTIFF senza GDAL (solo tifffile)
  pixel_to_coords       — converti indici pixel → coordinate mappa
  coords_extent         — bounding box dell'intero tile
  extract_area          — ritaglia area size_m × size_m dal DEM
  interpolate_surface   — griglia fine RBF thin-plate (solo per plot DEM)
  build_terrain         — pipeline completa → oggetto Terrain

Classe
------
  Terrain               — DEM interpolato interrogabile come f(x, y)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator

from config import AREA_SIZE_M, AGL_HEIGHT, LIDAR_SIGMA

# ---------------------------------------------------------------------------
# Percorso al file GeoTIFF
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIF_PATH   = os.path.join(SCRIPT_DIR, "w51065_s10.tif")

# ---------------------------------------------------------------------------
# Costanti interne (usate solo dalle funzioni di visualizzazione DEM raw)
# ---------------------------------------------------------------------------
_NODATA_VALUE = -9999
_INTERP_GRID  = 100
_CMAP_DEM     = "terrain"
_CMAP_INTERP  = "plasma"


# ============================================================================
# I/O GeoTIFF
# ============================================================================

def read_geotiff(path: str):
    """
    Legge un GeoTIFF e restituisce:
      dem       : array 2-D float (righe × colonne) — quota [m]
      transform : (x_origin, pixel_w, y_origin, pixel_h)

    Usa tifffile; i tag GeoTIFF ModelPixelScaleTag (33550) e
    ModelTiepointTag (33922) sono estratti manualmente.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("Installa tifffile:  pip install tifffile")

    with tifffile.TiffFile(path) as tif:
        page        = tif.pages[0]
        dem         = page.asarray().astype(float)
        pixel_scale = None
        tie_point   = None
        for tag in page.tags.values():
            if tag.code == 33550:
                pixel_scale = tag.value
            elif tag.code == 33922:
                tp = tag.value
                tie_point = tp[3], tp[4]

    dem[dem == _NODATA_VALUE] = np.nan

    if pixel_scale is not None and tie_point is not None:
        x_origin = tie_point[0]
        y_origin = tie_point[1]
        pixel_w  =  pixel_scale[0]
        pixel_h  = -pixel_scale[1]
        transform = (x_origin, pixel_w, y_origin, pixel_h)
    else:
        print("[WARN] Tag GeoTIFF non trovati — uso coordinate pixel.")
        transform = (0.0, 1.0, dem.shape[0], -1.0)

    return dem, transform


def pixel_to_coords(row, col, transform):
    """Converte indici pixel in coordinate geografiche/proiettate."""
    x_origin, pixel_w, y_origin, pixel_h = transform
    return x_origin + col * pixel_w, y_origin + row * pixel_h


def coords_extent(dem, transform):
    """Restituisce [xmin, xmax, ymin, ymax] in unità mappa."""
    rows, cols = dem.shape
    x_origin, pixel_w, y_origin, pixel_h = transform
    xmin = x_origin
    xmax = x_origin + cols * pixel_w
    ymin = y_origin + rows * pixel_h
    ymax = y_origin
    return xmin, xmax, min(ymin, ymax), max(ymin, ymax)


# ============================================================================
# Estrazione area
# ============================================================================

def extract_area(dem, transform, center_row=None, center_col=None, size_m=200):
    """
    Estrae un'area quadrata di 'size_m' metri attorno al pixel centrale.
    TODOS: punto generico passato in coordinate mappa invece che pixel; gestione bordi più elegante.

    Returns
    -------
    sub_dem  : array 2-D  (size_px × size_px)
    x_coords : array 1-D coordinate E dei pixel [m]
    y_coords : array 1-D coordinate N dei pixel [m]
    (cr, cc) : centro in pixel (row, col)
    """
    _, pixel_w, _, pixel_h = transform
    pixel_size = abs(pixel_w)
    half_px    = int(np.round(size_m / (2 * pixel_size)))

    rows, cols = dem.shape
    if center_row is None:
        center_row = rows // 2
    if center_col is None:
        center_col = cols // 2

    # Clamp so the full patch always fits within the DEM — no border truncation.
    center_row = int(np.clip(center_row, half_px, rows - half_px))
    center_col = int(np.clip(center_col, half_px, cols - half_px))

    r0 = center_row - half_px
    r1 = center_row + half_px
    c0 = center_col - half_px
    c1 = center_col + half_px

    sub_dem  = dem[r0:r1, c0:c1]
    x_origin, pw, y_origin, ph = transform
    x_coords = x_origin + np.arange(c0, c1) * pw
    y_coords = y_origin + np.arange(r0, r1) * ph   # ph < 0

    return sub_dem, x_coords, y_coords, (center_row, center_col)


# ============================================================================
# Interpolazione RBF fine (usata dai plot diagnostici del DEM)
# ============================================================================

def interpolate_surface(sub_dem, x_coords, y_coords, grid_n=_INTERP_GRID):
    """
    Interpola il DEM su griglia fine con RBFInterpolator (thin-plate spline).

    Returns
    -------
    xi, yi : meshgrid (grid_n × grid_n)
    zi     : quota interpolata
    """
    XX, YY = np.meshgrid(x_coords, y_coords)
    mask   = ~np.isnan(sub_dem)
    x_ctrl = XX[mask].ravel()
    y_ctrl = YY[mask].ravel()
    z_ctrl = sub_dem[mask].ravel()

    x_mean, x_std = x_ctrl.mean(), x_ctrl.std()
    y_mean, y_std = y_ctrl.mean(), y_ctrl.std()
    pts_norm = np.column_stack([
        (x_ctrl - x_mean) / x_std,
        (y_ctrl - y_mean) / y_std,
    ])
    rbf = RBFInterpolator(pts_norm, z_ctrl, kernel="thin_plate_spline",
                          smoothing=0.1)

    xi_1d = np.linspace(x_coords.min(), x_coords.max(), grid_n)
    yi_1d = np.linspace(y_coords.min(), y_coords.max(), grid_n)
    xi, yi = np.meshgrid(xi_1d, yi_1d)
    query_norm = np.column_stack([
        (xi.ravel() - x_mean) / x_std,
        (yi.ravel() - y_mean) / y_std,
    ])
    zi = rbf(query_norm).reshape(grid_n, grid_n)
    return xi, yi, zi


# ============================================================================
# Classe Terrain
# ============================================================================

class Terrain:
    """
    Incapsula il DEM interpolato e fornisce:
      z(x, y)          — quota terreno [m]
      z_lidar(x, y)    — quota con rumore LiDAR gaussiano
      agl_z(x, y, agl) — quota assoluta per volare a 'agl' m sopra suolo

    Le coordinate x, y sono in metri locali del workspace (origine = angolo SW
    dell'area estratta). Usare utm_origin per ricavare le coordinate UTM assolute.
    """

    def __init__(
        self,
        rbf_interp: RegularGridInterpolator,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        utm_origin: tuple = (0.0, 0.0),
    ) -> None:
        self._interp   = rbf_interp
        self.x_min = x_min;  self.x_max = x_max
        self.y_min = y_min;  self.y_max = y_max
        self.utm_origin = utm_origin   # (E_utm, N_utm) dell'angolo SW [m]
        self._rng  = np.random.default_rng(0)

    def z(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
    ) -> float | np.ndarray:
        """Quota terreno interpolata [m]. Clamp ai bordi dell'area."""
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)
        pts = np.column_stack([np.atleast_1d(y).ravel(),
                               np.atleast_1d(x).ravel()])
        z = self._interp(pts)
        return float(z[0]) if np.ndim(x) == 0 else z

    def z_lidar(self, x: float, y: float) -> float:
        """Quota terreno con rumore LiDAR gaussiano."""
        return self.z(x, y) + self._rng.normal(0, LIDAR_SIGMA)

    def agl_z(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        agl: float = AGL_HEIGHT,
    ) -> float | np.ndarray:
        """Quota assoluta per volare a 'agl' m sopra il terreno."""
        return self.z(x, y) + agl


# ============================================================================
# Pipeline principale
# ============================================================================

def build_terrain(center_frac=None, tif_path: str = TIF_PATH):
    """
    Legge il GeoTIFF, estrae area AREA_SIZE_M × AREA_SIZE_M m,
    costruisce un oggetto Terrain interrogabile.

    Strategia: si estrae un patch leggermente più grande (AREA_SIZE_M + 2 pixel
    di margine per lato) per costruire il RegularGridInterpolator, poi il dominio
    del Terrain rimane esattamente [0, AREA_SIZE_M]². In questo modo non si
    arriva mai al fill_value dell'interpolatore ai bordi.

    Parameters
    ----------
    tif_path    : percorso al file GeoTIFF
    center_frac : (row_frac, col_frac) ∈ [0,1]² — centro dell'area come
                  frazione delle dimensioni del DEM. None = centro del DEM.

    Returns
    -------
    terrain  : Terrain
    x_coords : np.ndarray  (coordinate E dei pixel dell'area nominale)
    y_coords : np.ndarray  (coordinate N dei pixel dell'area nominale)
    sub_dem  : np.ndarray  (quota grezza dell'area nominale, per i plot)
    transform: tuple       (x_origin, pixel_w, y_origin, pixel_h)
    """
    dem, transform = read_geotiff(tif_path)
    _, pixel_w, _, pixel_h = transform
    pixel_size = abs(pixel_w)

    center_row, center_col = None, None
    if center_frac is not None:
        rows, cols = dem.shape
        center_row = int(np.clip(center_frac[0] * rows, 0, rows - 1))
        center_col = int(np.clip(center_frac[1] * cols, 0, cols - 1))

    # Patch nominale (usata per i plot diagnostici e come riferimento UTM)
    sub_dem, x_coords, y_coords, (cr, cc) = extract_area(
        dem, transform,
        center_row=center_row, center_col=center_col,
        size_m=AREA_SIZE_M,
    )

    # Patch allargata: +2 pixel per lato → l'interpolatore non ha bordi "vuoti"
    # all'interno del dominio [0, AREA_SIZE_M]²
    margin_px   = 2
    size_large  = AREA_SIZE_M + 2 * margin_px * pixel_size
    sub_big, x_big, y_big, _ = extract_area(
        dem, transform,
        center_row=cr, center_col=cc,
        size_m=size_large,
    )

    if y_big[0] > y_big[-1]:
        y_big_asc   = y_big[::-1]
        sub_big_asc = sub_big[::-1, :]
    else:
        y_big_asc   = y_big
        sub_big_asc = sub_big

    # Origine UTM = angolo SW del patch nominale (invariato rispetto a prima)
    x_min_utm = float(x_coords.min())
    y_min_utm = float((y_coords[::-1] if y_coords[0] > y_coords[-1] else y_coords).min())

    x_big_local   = x_big     - x_min_utm
    y_big_asc_loc = y_big_asc - y_min_utm

    mean_z      = np.nanmean(sub_big_asc)
    sub_filled  = np.where(np.isnan(sub_big_asc), mean_z, sub_big_asc)

    rgi = RegularGridInterpolator(
        (y_big_asc_loc, x_big_local), sub_filled,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    terrain = Terrain(
        rbf_interp=rgi,
        x_min=0.0,  x_max=float(AREA_SIZE_M),
        y_min=0.0,  y_max=float(AREA_SIZE_M),
        utm_origin=(x_min_utm, y_min_utm),
    )

    # y_loc_orig per i plot: ordine originale dei pixel nominali
    if y_coords[0] > y_coords[-1]:
        y_loc_orig = y_coords - y_min_utm
    else:
        y_loc_orig = y_coords - y_min_utm
    x_local = x_coords - x_min_utm

    return terrain, x_local, y_loc_orig, sub_dem, transform


# ============================================================================
# Plot diagnostici DEM (standalone — non usati dalla simulazione)
# ============================================================================

def plot_full_dem(dem, transform, area_x_coords=None, area_y_coords=None):
    """DEM completo con hillshade; opzionalmente disegna il riquadro area."""
    rows, cols = dem.shape
    xmin, xmax, ymin, ymax = coords_extent(dem, transform)
    ls  = LightSource(azdeg=315, altdeg=45)
    hs  = ls.hillshade(np.nan_to_num(dem, nan=0.0), vert_exag=2)
    vmin = np.nanpercentile(dem, 2)
    vmax = np.nanpercentile(dem, 98)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("#333333")
    im = ax.imshow(dem, cmap=_CMAP_DEM, vmin=vmin, vmax=vmax,
                   extent=[xmin, xmax, ymin, ymax],
                   origin="upper", interpolation="bilinear", zorder=2)
    ax.imshow(hs, cmap="gray", alpha=0.35,
              extent=[xmin, xmax, ymin, ymax],
              origin="upper", interpolation="bilinear", zorder=3)

    if area_x_coords is not None and area_y_coords is not None:
        rx   = area_x_coords.min()
        ry   = area_y_coords.min()
        rw   = area_x_coords.max() - area_x_coords.min()
        rh   = area_y_coords.max() - area_y_coords.min()
        rect = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=2.0, edgecolor="#ff4444", facecolor="none",
            linestyle="--", zorder=5,
        )
        ax.add_patch(rect)
        cx_area = rx + rw / 2
        ax.annotate(
            f"Area {AREA_SIZE_M}×{AREA_SIZE_M} m",
            xy=(cx_area, ry),
            xytext=(cx_area, ry - (ymax - ymin) * 0.07),
            color="#ff4444", fontsize=9, fontweight="bold",
            ha="center", va="top", zorder=6,
            arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.5),
        )

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Quota [m s.l.m.]", fontsize=10)
    ax.set_xlabel("E  [m UTM]", fontsize=10)
    ax.set_ylabel("N  [m UTM]", fontsize=10)
    ax.set_title(
        f"DEM TINItaly — {os.path.basename(TIF_PATH)}\n"
        f"{cols}×{rows} px  ·  risoluzione {abs(transform[1]):.0f} m/px",
        fontsize=11, fontweight="bold",
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_area_and_interpolation(sub_dem, x_coords, y_coords,
                                xi, yi, zi, center_xy):
    """Layout 4 pannelli: DEM originale / RBF / 3-D / residuo."""
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()
    ey = [ymax, ymin] if y_coords[0] > y_coords[-1] else [ymin, ymax]
    extent_orig   = [xmin, xmax, ey[0], ey[1]]
    extent_interp = [xi.min(), xi.max(), yi.min(), yi.max()]
    vmin = np.nanpercentile(sub_dem, 1)
    vmax = np.nanpercentile(sub_dem, 99)
    ls   = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("#ffffff")

    ax_a = fig.add_subplot(2, 2, 1)
    im_a = ax_a.imshow(sub_dem, cmap=_CMAP_DEM, vmin=vmin, vmax=vmax,
                       extent=extent_orig, origin="upper", interpolation="nearest")
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04).set_label("Quota [m]", fontsize=9)
    ax_a.set_title("A — DEM originale 200×200 m", fontweight="bold", fontsize=10)
    ax_a.set_xlabel("E [m UTM]", fontsize=9); ax_a.set_ylabel("N [m UTM]", fontsize=9)
    ax_a.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_a.tick_params(labelsize=8); ax_a.set_facecolor("#cccccc")

    ax_b = fig.add_subplot(2, 2, 2)
    im_b = ax_b.imshow(zi, cmap=_CMAP_INTERP, vmin=vmin, vmax=vmax,
                       extent=extent_interp, origin="lower", interpolation="bilinear")
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04).set_label("Quota [m]", fontsize=9)
    ax_b.set_title(f"B — Interpolata RBF (thin-plate)  {_INTERP_GRID}²",
                   fontweight="bold", fontsize=10)
    ax_b.set_xlabel("E [m UTM]", fontsize=9); ax_b.set_ylabel("N [m UTM]", fontsize=9)
    ax_b.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_b.tick_params(labelsize=8)

    ax_c = fig.add_subplot(2, 2, 3, projection="3d")
    ax_c.plot_surface(
        xi, yi, zi,
        facecolors=plt.get_cmap(_CMAP_INTERP)((zi - vmin) / max(vmax - vmin, 1e-6)),
        rcount=60, ccount=60, linewidth=0, antialiased=True, alpha=0.95,
    )
    ax_c.set_xlabel("E [m]", fontsize=8, labelpad=3)
    ax_c.set_ylabel("N [m]", fontsize=8, labelpad=3)
    ax_c.set_zlabel("z [m]", fontsize=8, labelpad=3)
    ax_c.set_title("C — Vista 3-D (RBF)", fontweight="bold", fontsize=10)
    ax_c.tick_params(labelsize=7)
    ax_c.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax_c.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Pannello D — residuo
    y_asc     = np.sort(y_coords)
    sub_asc   = sub_dem[::-1, :] if y_coords[0] > y_coords[-1] else sub_dem
    sub_fill2 = sub_asc.copy(); sub_fill2[np.isnan(sub_fill2)] = np.nanmean(sub_fill2)
    rgi       = RegularGridInterpolator(
        (y_asc, x_coords), sub_fill2, method="linear",
        bounds_error=False, fill_value=np.nan,
    )
    z_ref = rgi(np.column_stack([yi.ravel(), xi.ravel()])).reshape(_INTERP_GRID, _INTERP_GRID)
    diff  = zi - z_ref

    ax_d = fig.add_subplot(2, 2, 4)
    vd   = np.nanpercentile(np.abs(diff), 98)
    im_d = ax_d.imshow(diff, cmap="RdBu_r", vmin=-vd, vmax=vd,
                       extent=extent_interp, origin="lower", interpolation="bilinear")
    fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04).set_label("Δ quota [m]", fontsize=9)
    ax_d.set_title("D — Residuo: interpolata − originale", fontweight="bold", fontsize=10)
    ax_d.set_xlabel("E [m UTM]", fontsize=9); ax_d.set_ylabel("N [m UTM]", fontsize=9)
    ax_d.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_d.tick_params(labelsize=8)

    fig.suptitle(
        f"TINItaly DEM — Area 200×200 m  (centro: "
        f"E={center_xy[0]:.0f}, N={center_xy[1]:.0f})\n"
        f"Interpolazione RBF thin-plate su griglia {_INTERP_GRID}×{_INTERP_GRID}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ============================================================================
# Main standalone (diagnostico DEM)
# ============================================================================

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
    print(f"  Dimensioni : {cols} × {rows} pixel")
    print(f"  Risoluzione: {abs(transform[1]):.1f} m/pixel")
    print(f"  Quota min/max: {np.nanmin(dem):.1f} / {np.nanmax(dem):.1f} m")

    sub_dem, x_coords, y_coords, (cr, cc) = extract_area(dem, transform, size_m=AREA_SIZE_M)
    cx, cy = pixel_to_coords(cr, cc, transform)
    print(f"\nArea selezionata: {AREA_SIZE_M}×{AREA_SIZE_M} m  (centro E={cx:.0f} N={cy:.0f})")

    fig1 = plot_full_dem(dem, transform, area_x_coords=x_coords, area_y_coords=y_coords)

    print(f"\nInterpolazione RBF thin-plate su griglia {_INTERP_GRID}×{_INTERP_GRID}...")
    xi, yi, zi = interpolate_surface(sub_dem, x_coords, y_coords)
    print("  Completata.")

    fig2 = plot_area_and_interpolation(sub_dem, x_coords, y_coords, xi, yi, zi, (cx, cy))
    plt.show()
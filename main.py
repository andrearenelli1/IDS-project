"""
main.py
=======
Simulazione di ricerca in valanga multi-agente con droni MPC.

Architettura
------------
  Terrain    : DEM TINItaly (w51065_s10.tif) interpolato con RBF → f(x,y)→z
  Workspace  : area 200×200 m estratta dal tile
  ARTVA      : sorgente dipolo magnetico 3-D (campo ∝ 1/r³)
  LiDAR sim  : f(x,y) + rumore gaussiano → stima altezza terreno
  Droni      : N agenti MPC (doppio integratore 3-D)
               Stato 0 — SEARCH : greca (lawnmower) assegnata a ciascun drone
               Stato 1 — TRACK  : gradient ascent del segnale ARTVA
                                   + triangolazione con i 2 droni più vicini

Vincoli MPC aggiuntivi rispetto a mpc_drone.py
-----------------------------------------------
  z_drone(k) = terrain(x,y) + AGL_HEIGHT   [quota assoluta]
  implementato come soft constraint sul termine z della posizione
  negli waypoint (gli waypoint stessi sono già calcolati a quota corretta).

TODO (versioni future)
----------------------
  * Stima stato via IMDCL (imdcl.py) al posto della posizione reale
  * Grafo di comunicazione a raggio limitato + consensus

Dipendenze
----------
    pip install numpy scipy matplotlib tifffile casadi
"""

from __future__ import annotations

import os
import sys
import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dem_tinitaly import (
    read_geotiff, extract_area, interpolate_surface,
    pixel_to_coords, TIF_PATH,
)
from mpc_drone import DroneMPC, PointMass3DModel

# ============================================================================
# Parametri globali
# ============================================================================

# — Workspace —
AREA_SIZE_M   = 200.0          # [m] lato area quadrata
INTERP_GRID   = 120           # punti per lato interpolazione RBF

# — Volo —
AGL_HEIGHT    = 1.5            # [m] altezza sopra il terreno (Above Ground Level)
LIDAR_SIGMA   = 0.05           # [m] rumore gaussiano simulazione LiDAR

# — Droni —
N_DRONES      = 2              # numero di droni (modificabile)
DEPLOY_OFFSET = 2.0            # [m] distanza laterale tra droni al deployment

# — Lawnmower —
LANE_OVERLAP  = 5.0            # [m] sovrapposizione laterale tra corsie
LANE_SPACING  = 15.0           # [m] distanza tra passaggi nella striscia

# — ARTVA (dipolo magnetico) —
ARTVA_MOMENT  = 1.0            # momento magnetico normalizzato [A·m²]
ARTVA_DETECT_THR = 1e-5        # soglia rilevamento segnale — rilevabile a ~40 m
ARTVA_NOISE_STD  = 1e-7        # rumore additivo sulla misura ARTVA (~1% del segnale a 40m)

# — Gradient ascent (stato TRACK) —
GRAD_STEP_M   = 3.0            # [m] passo del gradient ascent nel piano
GRAD_N_AHEAD  = 4              # numero di waypoint look-ahead nel gradient ascent

# — Triangolazione —
TRIANGULATE_N_PARTNERS = 2     # quanti droni chiamare in supporto

# — MPC —
DT_MPC        = 0.1            # [s]
N_MPC         = 20
A_MAX         = 6.0            # [m/s²]
V_MAX         = 3.0            # [m/s]

# — Simulazione —
N_SIM         = 600            # passi massimi
DT_SIM        = DT_MPC
SIGMA_ACC_SIM = 0.05           # rumore accelerazione nella simulazione
STOP_THRESH   = 0.3            # [m] soglia raggiungimento waypoint

# — Plot —
COLORS = {i: c for i, c in enumerate(
    ["#e63946", "#2a9d8f", "#e9c46a", "#a8dadc", "#f4a261",
     "#6a4c93", "#1982c4", "#8ac926"]
)}
BG_DARK = "#0d1117"


# ============================================================================
# Enumerazione stati FSM
# ============================================================================

class DroneState(IntEnum):
    SEARCH = 0   # greca, ricerca attiva
    TRACK  = 1   # gradient ascent verso la vittima


# ============================================================================
# Terrain: wrapper RBF interpolato
# ============================================================================

class Terrain:
    """
    Incapsula il DEM interpolato e fornisce:
      z(x, y)      — quota terreno (scalar o array)
      z_lidar(x,y) — quota con rumore LiDAR simulato
    """

    def __init__(self, rbf_interp: RegularGridInterpolator,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float) -> None:
        self._interp = rbf_interp
        self.x_min = x_min;  self.x_max = x_max
        self.y_min = y_min;  self.y_max = y_max
        self._rng  = np.random.default_rng(0)

    def z(self, x: float | np.ndarray,
              y: float | np.ndarray) -> float | np.ndarray:
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

    def agl_z(self, x: float | np.ndarray,
                   y: float | np.ndarray,
                   agl: float = AGL_HEIGHT) -> float | np.ndarray:
        """Quota assoluta per volare a 'agl' metri sopra il terreno."""
        return self.z(x, y) + agl


def build_terrain(tif_path: str = TIF_PATH) -> Terrain:
    """
    Legge il GeoTIFF, estrae area 200×200 m, interpola RBF e
    costruisce un Terrain interrogabile come f(x, y).
    """
    dem, transform = read_geotiff(tif_path)
    sub_dem, x_coords, y_coords, _ = extract_area(
        dem, transform, size_m=AREA_SIZE_M
    )

    # y_coords è decrescente → inverti per RegularGridInterpolator
    if y_coords[0] > y_coords[-1]:
        y_asc   = y_coords[::-1]
        sub_asc = sub_dem[::-1, :]
    else:
        y_asc   = y_coords
        sub_asc = sub_dem

    # Riempi NaN con media (bordi NoData)
    mean_z = np.nanmean(sub_asc)
    sub_filled = np.where(np.isnan(sub_asc), mean_z, sub_asc)

    rgi = RegularGridInterpolator(
        (y_asc, x_coords), sub_filled,
        method="linear", bounds_error=False, fill_value=mean_z,
    )

    return Terrain(
        rbf_interp=rgi,
        x_min=float(x_coords.min()), x_max=float(x_coords.max()),
        y_min=float(y_asc.min()),    y_max=float(y_asc.max()),
    ), x_coords, y_coords, sub_dem, transform


# ============================================================================
# Sorgente ARTVA — dipolo magnetico 3-D
# ============================================================================

class ARTVASource:
    """
    Sorgente ARTVA modellata come dipolo magnetico verticale.

    Campo in un punto r = (x, y, z) rispetto alla sorgente in r0:

        B(r) = (μ₀/4π) · m · [3(m̂·r̂)r̂ − m̂] / |r|³

    Per un dipolo verticale (m̂ = ẑ):

        |B|² ∝ m² · (1 + 3cos²θ) / r⁶

    Qui restituiamo |B| normalizzato:

        S(r) = moment · sqrt(1 + 3·cos²θ) / r³

    dove θ è l'angolo tra r e l'asse z del dipolo.
    """

    def __init__(self, position: np.ndarray,
                 moment: float = ARTVA_MOMENT,
                 rng_seed: int = 1) -> None:
        self.position = np.asarray(position, dtype=float)  # (3,) [x, y, z]
        self.moment   = moment
        self._rng     = np.random.default_rng(rng_seed)

    def signal(self, pos: np.ndarray, noisy: bool = True) -> float:
        """
        Intensità segnale ARTVA in 'pos' (3,).

        Parameters
        ----------
        pos   : posizione del sensore [x, y, z]
        noisy : se True aggiunge rumore gaussiano

        Returns
        -------
        S ≥ 0
        """
        r_vec = np.asarray(pos) - self.position
        r     = np.linalg.norm(r_vec)
        if r < 1e-3:
            r = 1e-3   # evita singolarità
        cos_theta = r_vec[2] / r          # componente z / modulo
        S = self.moment * np.sqrt(1.0 + 3.0 * cos_theta**2) / r**3
        if noisy:
            S += self._rng.normal(0, ARTVA_NOISE_STD)
        return max(0.0, S)

    def gradient_xy(self, pos: np.ndarray, eps: float = 0.5) -> np.ndarray:
        """
        Gradiente numerico del segnale ARTVA nel piano (x, y) in 'pos'.
        Usato per il gradient ascent nel tracker.
        """
        sx_p = self.signal(pos + [eps, 0,   0], noisy=False)
        sx_m = self.signal(pos - [eps, 0,   0], noisy=False)
        sy_p = self.signal(pos + [0,   eps, 0], noisy=False)
        sy_m = self.signal(pos - [0,   eps, 0], noisy=False)
        grad = np.array([(sx_p - sx_m) / (2*eps),
                         (sy_p - sy_m) / (2*eps)])
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 1e-9 else grad


# ============================================================================
# Lawnmower pattern generator
# ============================================================================

def lawnmower_waypoints(
    drone_id: int,
    n_drones: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    terrain: Terrain,
    lane_spacing: float = LANE_SPACING,
    agl: float = AGL_HEIGHT,
) -> List[np.ndarray]:
    """
    Genera la sequenza di waypoint della greca per il drone 'drone_id'.

    Strategia: il workspace è diviso in N_DRONES strisce verticali
    (lungo x). Ogni drone riceve una striscia e percorre le corsie
    orizzontali (lungo y) alternando il verso.

    I waypoint includono la quota z = terrain(x, y) + agl.

    Returns
    -------
    waypoints : lista di np.array([x, y, z])
    """
    # — Divisione in strisce verticali —
    width  = (x_max - x_min) / n_drones
    x0_s   = x_min + drone_id * width
    x1_s   = x0_s + width

    # Corsie lungo y (passo = lane_spacing)
    x_positions = np.arange(x0_s + lane_spacing / 2,
                             x1_s, lane_spacing)
    if len(x_positions) == 0:
        x_positions = np.array([(x0_s + x1_s) / 2])

    waypoints: List[np.ndarray] = []
    go_up = True
    for x in x_positions:
        y_start, y_end = (y_min, y_max) if go_up else (y_max, y_min)
        # 2 waypoint per corsia (inizio e fine)
        for y in (y_start, y_end):
            z = terrain.agl_z(x, y, agl)
            waypoints.append(np.array([x, y, z]))
        go_up = not go_up

    return waypoints


# ============================================================================
# Gradient-ascent waypoints (stato TRACK)
# ============================================================================

def track_waypoints(
    current_pos: np.ndarray,
    artva: ARTVASource,
    terrain: Terrain,
    n_ahead: int = GRAD_N_AHEAD,
    step_m: float = GRAD_STEP_M,
    agl: float = AGL_HEIGHT,
) -> List[np.ndarray]:
    """
    Genera 'n_ahead' waypoint lungo la direzione di gradient ascent del
    segnale ARTVA nel piano xy, proiettati sulla superficie del terreno + AGL.

    Returns
    -------
    waypoints : lista di n_ahead np.array([x, y, z])
    """
    pos    = current_pos[:3].copy()
    wps: List[np.ndarray] = []
    for _ in range(n_ahead):
        grad = artva.gradient_xy(pos)
        pos_next_xy = pos[:2] + step_m * grad
        z_next      = terrain.agl_z(pos_next_xy[0], pos_next_xy[1], agl)
        wp = np.array([pos_next_xy[0], pos_next_xy[1], z_next])
        wps.append(wp)
        pos = np.array([wp[0], wp[1], z_next])
    return wps


# ============================================================================
# Agente drone — FSM + MPC
# ============================================================================

@dataclass
class DroneAgent:
    """
    Un drone con macchina a stati finiti e controllore MPC.

    Attributi pubblici principali
    -----------------------------
    id          : identificatore intero
    state       : DroneState (SEARCH / TRACK)
    x           : stato corrente (6,) [px, py, pz, vx, vy, vz]
    waypoints   : lista di waypoint correnti (può essere aggiornata dalla FSM)
    wp_idx      : indice waypoint attuale
    signal_log  : misure ARTVA raccolte [(pos, signal), ...]
    """
    id:          int
    x:           np.ndarray          # stato iniziale (6,)
    waypoints:   List[np.ndarray]    # waypoint iniziali (lawnmower)
    ctrl:        DroneMPC
    state:       DroneState = DroneState.SEARCH
    wp_idx:      int        = 0
    signal_log:  list       = field(default_factory=list)
    history:     list       = field(default_factory=list)
    input_log:   list       = field(default_factory=list)
    solve_t_log: list       = field(default_factory=list)
    detected:    bool       = False  # ha rilevato il segnale almeno una volta
    estimate_pos: Optional[np.ndarray] = None  # stima posizione vittima

    def current_target(self) -> np.ndarray:
        idx = min(self.wp_idx, len(self.waypoints) - 1)
        return self.waypoints[idx]

    def advance_waypoint(self) -> bool:
        """Avanza al prossimo waypoint. Restituisce True se ce ne sono altri."""
        if self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            return True
        return False

    def all_waypoints_done(self) -> bool:
        return (self.wp_idx >= len(self.waypoints) - 1 and
                np.linalg.norm(self.x[:3] - self.current_target()) < STOP_THRESH)


# ============================================================================
# Costruzione degli agenti
# ============================================================================

def build_agents(
    deploy_xy: np.ndarray,
    terrain: Terrain,
    n_drones: int = N_DRONES,
    agl: float = AGL_HEIGHT,
    rng_seed: int = 42,
) -> Dict[int, DroneAgent]:
    """
    Crea N droni dal punto di deployment condiviso, ognuno con:
      - posizione iniziale leggermente distanziata lateralmente
      - waypoint lawnmower precompilati
      - controllore MPC con warm-start
    """
    rng = np.random.default_rng(rng_seed)
    agents: Dict[int, DroneAgent] = {}

    x_min = terrain.x_min
    x_max = terrain.x_max
    y_min = terrain.y_min
    y_max = terrain.y_max

    for i in range(n_drones):
        # Offset laterale dal punto di deployment (lungo y)
        offset = (i - (n_drones - 1) / 2) * DEPLOY_OFFSET
        px0 = deploy_xy[0]
        py0 = deploy_xy[1] + offset
        pz0 = terrain.agl_z(px0, py0, agl)

        x0 = np.array([px0, py0, pz0, 0.0, 0.0, 0.0])

        wps = lawnmower_waypoints(
            i, n_drones, x_min, x_max, y_min, y_max,
            terrain, agl=agl,
        )

        ctrl = DroneMPC(dt=DT_MPC, N=N_MPC, a_max=A_MAX, v_max=V_MAX)

        agent = DroneAgent(
            id=i, x=x0.copy(), waypoints=wps,
            ctrl=ctrl, state=DroneState.SEARCH,
        )
        agent.history.append(x0.copy())
        agents[i] = agent

    # Warm-start MPC per tutti
    print("Warm-start MPC droni...")
    for i, ag in agents.items():
        ag.ctrl.warm_start(ag.x, ag.current_target())
        print(f"  Drone {i}: wp[0]={ag.current_target().round(1)}")

    return agents


# ============================================================================
# Stima posizione vittima per triangolazione
# ============================================================================

def triangulate_victim(
    agents: Dict[int, DroneAgent],
    artva: ARTVASource,
) -> np.ndarray:
    """
    Stima semplice della posizione vittima: media pesata delle posizioni
    dei droni in stato TRACK, ponderata per l'intensità del segnale.
    (Placeholder — in versioni future: least-squares su misure di campo)
    """
    positions = []
    weights   = []
    for ag in agents.values():
        if ag.state == DroneState.TRACK and len(ag.signal_log) > 0:
            last_sig = ag.signal_log[-1][1]
            if last_sig > 0:
                positions.append(ag.x[:3])
                weights.append(last_sig)

    if not positions:
        return artva.position.copy()   # fallback

    positions = np.array(positions)
    weights   = np.array(weights)
    weights  /= weights.sum()
    return (positions * weights[:, None]).sum(axis=0)


# ============================================================================
# Loop principale di simulazione
# ============================================================================

def simulate(
    terrain:  Terrain,
    artva:    ARTVASource,
    agents:   Dict[int, DroneAgent],
    n_steps:  int   = N_SIM,
    dt:       float = DT_SIM,
    sigma:    float = SIGMA_ACC_SIM,
    agl:      float = AGL_HEIGHT,
    rng_seed: int   = 42,
) -> Dict[int, DroneAgent]:
    """
    Esegue la simulazione multi-agente.

    Per ogni passo:
      1. Misura ARTVA in posizione corrente
      2. FSM: transizione di stato se soglia superata / waypoint esauriti
      3. MPC step → u_opt
      4. Aggiornamento dinamica (con rumore)
      5. Controllo altezza AGL via quota wp
    """
    rng        = np.random.default_rng(rng_seed)
    model      = PointMass3DModel(sigma_acc=sigma)
    drone_ids  = list(agents.keys())

    # Header log
    print(f"\n{'Step':>5}  {'t[s]':>6}  " +
          "  ".join(f"D{i}:state/wp/dist" for i in drone_ids))

    for step in range(n_steps):
        t = step * dt

        # ── 1. Misura ARTVA e transizioni FSM ────────────────────────────
        for i in drone_ids:
            ag  = agents[i]
            sig = artva.signal(ag.x[:3], noisy=True)
            ag.signal_log.append((ag.x[:3].copy(), sig))

            # Transizione SEARCH → TRACK
            if ag.state == DroneState.SEARCH and sig >= ARTVA_DETECT_THR:
                ag.state    = DroneState.TRACK
                ag.detected = True
                print(f"\n  ★ Drone {i} RILEVATO segnale ARTVA "
                      f"(S={sig:.4f}) al passo {step} (t={t:.2f}s)")
                print(f"    Posizione: {ag.x[:3].round(2)}")

                # Chiama i 2 droni più vicini in supporto
                dists = {
                    j: np.linalg.norm(agents[j].x[:3] - ag.x[:3])
                    for j in drone_ids if j != i
                }
                partners = sorted(dists, key=dists.get)[:TRIANGULATE_N_PARTNERS]
                for j in partners:
                    if agents[j].state == DroneState.SEARCH:
                        agents[j].state = DroneState.TRACK
                        print(f"    → Drone {j} chiamato in supporto "
                              f"(dist={dists[j]:.1f} m)")

            # Aggiorna waypoint per stato TRACK
            if ag.state == DroneState.TRACK:
                ag.waypoints = track_waypoints(
                    ag.x, artva, terrain, n_ahead=GRAD_N_AHEAD,
                    step_m=GRAD_STEP_M, agl=agl,
                )
                ag.wp_idx = 0

        # ── 2. MPC step per ogni drone ────────────────────────────────────
        for i in drone_ids:
            ag  = agents[i]
            tgt = ag.current_target()

            from time import perf_counter
            t0    = perf_counter()
            u_opt = ag.ctrl.step(ag.x, tgt)
            dt_s  = perf_counter() - t0

            # Rumore processo
            noise = rng.multivariate_normal(np.zeros(3),
                                            np.diag([sigma**2]*3))
            ag.x = model.f(ag.x, u_opt + noise, dt)

            # Clamp quota minima: non scendere sotto terreno + AGL/2
            z_floor = terrain.agl_z(ag.x[0], ag.x[1], agl * 0.5)
            if ag.x[2] < z_floor:
                ag.x[2]  = z_floor
                ag.x[5]  = max(0.0, ag.x[5])   # azzera velocità z negativa

            ag.history.append(ag.x.copy())
            ag.input_log.append(u_opt.copy())
            ag.solve_t_log.append(dt_s)

        # ── 3. Avanza waypoint se raggiunto ───────────────────────────────
        for i in drone_ids:
            ag   = agents[i]
            tgt  = ag.current_target()
            dist = np.linalg.norm(ag.x[:3] - tgt)
            if dist < STOP_THRESH:
                ag.advance_waypoint()

        # ── 4. Log periodico ─────────────────────────────────────────────
        if (step + 1) % 20 == 0:
            row = f"{step+1:>5}  {(step+1)*dt:>5.1f}s  "
            for i in drone_ids:
                ag   = agents[i]
                dist = np.linalg.norm(ag.x[:3] - ag.current_target())
                st   = "SRCH" if ag.state == DroneState.SEARCH else "TRCK"
                row += f"  {st}/{ag.wp_idx:02d}/{dist:5.2f}m"
            print(row)

        # ── 5. Stop se tutti in TRACK e vicini alla vittima ──────────────
        all_track  = all(ag.state == DroneState.TRACK for ag in agents.values())
        all_close  = all(
            np.linalg.norm(ag.x[:3] - artva.position) < 5.0
            for ag in agents.values()
        )
        if all_track and all_close:
            print(f"\n  ✔ Tutti i droni vicini alla vittima "
                  f"al passo {step+1} (t={(step+1)*dt:.2f}s)")
            break

    # Stima finale posizione vittima
    est = triangulate_victim(agents, artva)
    err = np.linalg.norm(est[:2] - artva.position[:2])
    print(f"\n  Posizione vittima reale  : {artva.position.round(2)}")
    print(f"  Stima triangolazione     : {est.round(2)}")
    print(f"  Errore planimetrico      : {err:.2f} m")

    return agents


# ============================================================================
# Visualizzazione risultati
# ============================================================================

def plot_mission(
    terrain:  Terrain,
    artva:    ARTVASource,
    agents:   Dict[int, DroneAgent],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    sub_dem:  np.ndarray,
) -> plt.Figure:
    """
    Figura missione:
      (A) Vista 2-D dall'alto — traiettorie + segnale ARTVA
      (B) Vista 3-D — traiettorie sul terreno
      (C) Segnale ARTVA nel tempo per ogni drone
      (D) Altezza AGL nel tempo (verifica vincolo)
    """
    drone_ids = list(agents.keys())

    # Mappa segnale ARTVA sull'area
    nx, ny = 80, 80
    xs = np.linspace(terrain.x_min, terrain.x_max, nx)
    ys = np.linspace(terrain.y_min, terrain.y_max, ny)
    XS, YS = np.meshgrid(xs, ys)
    ZS = terrain.z(XS.ravel(), YS.ravel()).reshape(ny, nx)
    ARTVA_MAP = np.array([
        [artva.signal([xs[j], ys[i], ZS[i, j] + AGL_HEIGHT], noisy=False)
         for j in range(nx)]
        for i in range(ny)
    ])

    # Extent
    ext = [terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#ffffff")

    # ── A: Vista dall'alto ───────────────────────────────────────────────
    ax_a = fig.add_subplot(2, 2, 1)
    ax_a.set_facecolor("#1a1a2e")
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(sub_dem, nan=0.0), vert_exag=3)
    # y_coords potrebbe essere decrescente
    yext = [float(np.min(y_coords)), float(np.max(y_coords))]
    ax_a.imshow(hs, cmap="gray", alpha=0.5,
                extent=[float(x_coords.min()), float(x_coords.max()),
                        yext[0], yext[1]],
                origin="upper" if y_coords[0] > y_coords[-1] else "lower",
                interpolation="bilinear", zorder=1)
    im_artva = ax_a.imshow(
        np.log1p(ARTVA_MAP), cmap="inferno", alpha=0.55,
        extent=ext, origin="lower", zorder=2,
    )
    fig.colorbar(im_artva, ax=ax_a, fraction=0.046, pad=0.04,
                 label="log(1+ARTVA) [a.u.]")

    for i in drone_ids:
        ag   = agents[i]
        traj = np.array(ag.history)
        c    = COLORS.get(i, "#aaaaaa")
        # Traiettoria (search in tenue, track in pieno)
        search_mask = []
        state_seq   = _reconstruct_state_sequence(ag)
        n = min(len(traj), len(state_seq))
        s_idx = [k for k in range(n) if state_seq[k] == DroneState.SEARCH]
        t_idx = [k for k in range(n) if state_seq[k] == DroneState.TRACK]
        if s_idx:
            ax_a.plot(traj[s_idx, 0], traj[s_idx, 1],
                      color=c, lw=1.0, alpha=0.5, ls="--")
        if t_idx:
            ax_a.plot(traj[t_idx, 0], traj[t_idx, 1],
                      color=c, lw=2.0, alpha=0.9)
        ax_a.plot(*traj[0, :2], "o", color=c, ms=7,
                  mec="white", mew=1.0, zorder=6)
        ax_a.plot(*traj[-1, :2], "^", color=c, ms=9,
                  mec="white", mew=0.8, zorder=6)

    # Vittima
    ax_a.plot(*artva.position[:2], "*", color="white", ms=18, zorder=10,
              mec="yellow", mew=1.5, label="Vittima ARTVA")
    ax_a.set_xlabel("E [m UTM]", fontsize=9)
    ax_a.set_ylabel("N [m UTM]", fontsize=9)
    ax_a.set_title("A — Traiettorie + mappa segnale ARTVA",
                   fontweight="bold", fontsize=10)
    ax_a.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax_a.tick_params(labelsize=8)

    handles = [mpatches.Patch(color=COLORS.get(i, "#aaa"),
                               label=f"Drone {i}") for i in drone_ids]
    handles += [
        plt.Line2D([0],[0], color="w", lw=2.0, label="Fase TRACK"),
        plt.Line2D([0],[0], color="w", lw=1.0, ls="--", label="Fase SEARCH"),
        plt.Line2D([0],[0], marker="*", color="w", mfc="yellow",
                   ms=12, label="Vittima", linestyle="None"),
    ]
    ax_a.legend(handles=handles, fontsize=7.5, loc="upper left",
                framealpha=0.75)

    # ── B: Vista 3-D ─────────────────────────────────────────────────────
    ax_b = fig.add_subplot(2, 2, 2, projection="3d")
    ax_b.set_facecolor("#0d1117")

    # Superficie terreno (campionata a bassa risoluzione per velocità)
    xs_3d = np.linspace(terrain.x_min, terrain.x_max, 40)
    ys_3d = np.linspace(terrain.y_min, terrain.y_max, 40)
    X3, Y3 = np.meshgrid(xs_3d, ys_3d)
    Z3 = terrain.z(X3.ravel(), Y3.ravel()).reshape(X3.shape)
    ax_b.plot_surface(X3, Y3, Z3, cmap="terrain", alpha=0.45,
                      rcount=40, ccount=40, linewidth=0)

    for i in drone_ids:
        ag   = agents[i]
        traj = np.array(ag.history)
        c    = COLORS.get(i, "#aaaaaa")
        ax_b.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color=c, lw=1.5, alpha=0.85, label=f"Drone {i}")
        ax_b.scatter(*traj[0, :3], color=c, s=40, zorder=6)
        ax_b.scatter(*traj[-1, :3], marker="^", color=c, s=60, zorder=6)

    ax_b.scatter(*artva.position, marker="*", color="yellow",
                 s=250, zorder=10, edgecolors="red", linewidths=1)
    ax_b.set_xlabel("E [m]", fontsize=8, labelpad=3)
    ax_b.set_ylabel("N [m]", fontsize=8, labelpad=3)
    ax_b.set_zlabel("z [m]", fontsize=8, labelpad=3)
    ax_b.set_title("B — Vista 3-D", fontweight="bold", fontsize=10)
    ax_b.tick_params(labelsize=7)
    ax_b.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax_b.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_b.legend(fontsize=7.5, loc="upper left")

    # ── C: Segnale ARTVA nel tempo ───────────────────────────────────────
    ax_c = fig.add_subplot(2, 2, 3)
    ax_c.set_facecolor("#f8f8f8")
    for i in drone_ids:
        ag   = agents[i]
        c    = COLORS.get(i, "#aaaaaa")
        sigs = [s for _, s in ag.signal_log]
        time = np.arange(len(sigs)) * DT_SIM
        ax_c.semilogy(time, np.maximum(sigs, 1e-8),
                      color=c, lw=1.3, alpha=0.85, label=f"Drone {i}")
    ax_c.axhline(ARTVA_DETECT_THR, color="red", lw=1.2, ls="--",
                 label=f"Soglia ({ARTVA_DETECT_THR:.0e})")
    ax_c.set_xlabel("Tempo [s]", fontsize=9)
    ax_c.set_ylabel("Segnale ARTVA [a.u.]", fontsize=9)
    ax_c.set_title("C — Segnale ARTVA nel tempo", fontweight="bold",
                   fontsize=10)
    ax_c.legend(fontsize=8)
    ax_c.grid(True, ls=":", alpha=0.5)
    ax_c.tick_params(labelsize=8)

    # ── D: Altezza AGL nel tempo ─────────────────────────────────────────
    ax_d = fig.add_subplot(2, 2, 4)
    ax_d.set_facecolor("#f8f8f8")
    for i in drone_ids:
        ag   = agents[i]
        c    = COLORS.get(i, "#aaaaaa")
        traj = np.array(ag.history)
        time = np.arange(len(traj)) * DT_SIM
        # Calcola AGL reale ad ogni step
        z_terrain = np.array([
            terrain.z(traj[k, 0], traj[k, 1])
            for k in range(len(traj))
        ])
        agl_real = traj[:, 2] - z_terrain
        ax_d.plot(time, agl_real, color=c, lw=1.2, alpha=0.85,
                  label=f"Drone {i}")
    ax_d.axhline(AGL_HEIGHT, color="green", lw=1.2, ls="--",
                 label=f"AGL target ({AGL_HEIGHT} m)")
    ax_d.axhline(0, color="red", lw=1.0, ls=":", alpha=0.7,
                 label="Terreno")
    ax_d.set_xlabel("Tempo [s]", fontsize=9)
    ax_d.set_ylabel("Altezza sopra terreno [m]", fontsize=9)
    ax_d.set_title("D — Altezza AGL (vincolo quota)", fontweight="bold",
                   fontsize=10)
    ax_d.legend(fontsize=8)
    ax_d.grid(True, ls=":", alpha=0.5)
    ax_d.tick_params(labelsize=8)
    ax_d.set_ylim(bottom=-0.5)

    fig.suptitle(
        f"Ricerca valanga multi-agente  ·  {len(agents)} droni  ·  "
        f"AGL={AGL_HEIGHT} m  ·  N_MPC={N_MPC}  ·  dt={DT_SIM} s",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def _reconstruct_state_sequence(ag: DroneAgent) -> List[DroneState]:
    """
    Ricostruisce la sequenza di stati dal log dei segnali:
    SEARCH fino al primo rilevamento, poi TRACK.
    """
    n = len(ag.history)
    if not ag.detected:
        return [DroneState.SEARCH] * n
    # Trova il passo del primo rilevamento
    for k, (_, s) in enumerate(ag.signal_log):
        if s >= ARTVA_DETECT_THR:
            return ([DroneState.SEARCH] * (k + 1) +
                    [DroneState.TRACK] * max(0, n - k - 1))
    return [DroneState.SEARCH] * n


# ============================================================================
# Animazione (opzionale)
# ============================================================================

def animate_mission(
    terrain:   Terrain,
    artva:     ARTVASource,
    agents:    Dict[int, DroneAgent],
    dt:        float = DT_SIM,
    fps:       int   = 30,
    speed:     float = 2.0,
    save:      bool  = False,
    save_path: str   = "mission_animation",
) -> FuncAnimation:
    """
    Animazione 3-D della missione: droni sul terreno + segnale ARTVA.
    """
    drone_ids = list(agents.keys())
    T = max(len(ag.history) for ag in agents.values())

    # Superficie terreno (bassa risoluzione)
    xs_3d = np.linspace(terrain.x_min, terrain.x_max, 35)
    ys_3d = np.linspace(terrain.y_min, terrain.y_max, 35)
    X3, Y3 = np.meshgrid(xs_3d, ys_3d)
    Z3 = terrain.z(X3.ravel(), Y3.ravel()).reshape(X3.shape)

    fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
    ax  = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(BG_DARK)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.plot_surface(X3, Y3, Z3, cmap="terrain", alpha=0.35,
                    rcount=35, ccount=35, linewidth=0, zorder=1)
    ax.scatter(*artva.position, marker="*", color="yellow",
               s=300, zorder=10, edgecolors="red", linewidths=1.5)

    TRAIL_LEN = 50
    trails, dots = {}, {}
    for i in drone_ids:
        c = COLORS.get(i, "#aaaaaa")
        tr, = ax.plot([], [], [], color=c, lw=1.5, alpha=0.8)
        dt_, = ax.plot([], [], [], "o", color=c, ms=7,
                       mec="white", mew=0.8, zorder=8)
        trails[i] = tr
        dots[i]   = dt_

    info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                     color="#c9d1d9", fontsize=8, va="top",
                     fontfamily="monospace")

    ax.set_xlabel("E [m]", fontsize=8, labelpad=4)
    ax.set_ylabel("N [m]", fontsize=8, labelpad=4)
    ax.set_zlabel("z [m]", fontsize=8, labelpad=4)
    ax.tick_params(colors="#c9d1d9", labelsize=7)
    fig.suptitle("Ricerca valanga multi-agente", color="#c9d1d9",
                 fontsize=11, fontweight="bold")

    step_skip    = max(1, int(round(1.0 / (dt * fps) * speed)))
    frame_idx    = list(range(0, T, step_skip))

    def init():
        for i in drone_ids:
            trails[i].set_data([], []); trails[i].set_3d_properties([])
            dots[i].set_data([], []);   dots[i].set_3d_properties([])
        info.set_text("")
        return list(trails.values()) + list(dots.values()) + [info]

    def update(f):
        t_step = frame_idx[f]
        lines  = [f"t = {t_step * dt:.2f} s  step {t_step}/{T-1}"]
        for i in drone_ids:
            ag   = agents[i]
            traj = np.array(ag.history)
            ti   = min(t_step, len(traj) - 1)
            ts   = max(0, ti - TRAIL_LEN)
            trails[i].set_data(traj[ts:ti+1, 0], traj[ts:ti+1, 1])
            trails[i].set_3d_properties(traj[ts:ti+1, 2])
            dots[i].set_data([traj[ti, 0]], [traj[ti, 1]])
            dots[i].set_3d_properties([traj[ti, 2]])
            st = "SRCH" if ti < len(ag.signal_log) and \
                ag.signal_log[ti][1] < ARTVA_DETECT_THR else "TRCK"
            lines.append(f"D{i}: {st}  z={traj[ti,2]:.1f}m")
        info.set_text("\n".join(lines))
        return list(trails.values()) + list(dots.values()) + [info]

    anim = FuncAnimation(fig, update, frames=len(frame_idx),
                         init_func=init, blit=True,
                         interval=1000 // fps)

    if save:
        try:
            out = save_path + ".mp4"
            anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=1800),
                      dpi=130, savefig_kwargs={"facecolor": BG_DARK})
            print(f"Animazione → {out}")
        except Exception as e:
            try:
                out = save_path + ".gif"
                anim.save(out, writer=PillowWriter(fps=fps), dpi=90,
                          savefig_kwargs={"facecolor": BG_DARK})
                print(f"Animazione → {out}")
            except Exception as e2:
                print(f"Salvataggio animazione fallito: {e2}")

    return anim


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Simulazione ricerca valanga multi-drone MPC"
    )
    parser.add_argument("--n",      type=int,   default=N_DRONES,
                        help="Numero di droni")
    parser.add_argument("--agl",    type=float, default=AGL_HEIGHT,
                        help="Altezza sopra terreno [m]")
    parser.add_argument("--steps",  type=int,   default=N_SIM,
                        help="Passi simulazione")
    parser.add_argument("--animate",action="store_true",
                        help="Mostra animazione 3-D")
    parser.add_argument("--save",   action="store_true",
                        help="Salva animazione su disco")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    print("=" * 62)
    print("  ARTVA Search & Rescue — simulazione multi-drone MPC")
    print(f"  Droni: {args.n}   AGL: {args.agl} m   Passi: {args.steps}")
    print("=" * 62)

    # — Costruzione ambiente —
    print("\nLettura e interpolazione DEM...")
    terrain_obj, x_coords, y_coords, sub_dem, transform = build_terrain(TIF_PATH)
    print(f"  Workspace: E=[{terrain_obj.x_min:.0f}, {terrain_obj.x_max:.0f}]  "
          f"N=[{terrain_obj.y_min:.0f}, {terrain_obj.y_max:.0f}]")

    # — Posizione deployment (angolo SW del workspace) —
    deploy_xy = np.array([
        terrain_obj.x_min + 5.0,
        terrain_obj.y_min + 5.0,
    ])

    # — Posizione vittima (casuale nell'area centrale) —
    rng_main = np.random.default_rng(args.seed)
    victim_x = rng_main.uniform(terrain_obj.x_min + 30,
                                terrain_obj.x_max - 30)
    victim_y = rng_main.uniform(terrain_obj.y_min + 30,
                                terrain_obj.y_max - 30)
    victim_z = terrain_obj.z(victim_x, victim_y) - 0.5   # sepolta 0.5 m
    artva = ARTVASource(
        position=np.array([victim_x, victim_y, victim_z]),
        moment=ARTVA_MOMENT,
        rng_seed=args.seed + 1,
    )
    print(f"\n  Vittima posizionata in: "
          f"E={victim_x:.1f} N={victim_y:.1f} z={victim_z:.1f} m")

    # — Costruzione agenti —
    print(f"\nCostruzione {args.n} droni dal punto di deployment "
          f"({deploy_xy.round(1)})...")
    agents = build_agents(
        deploy_xy=deploy_xy,
        terrain=terrain_obj,
        n_drones=args.n,
        agl=args.agl,
        rng_seed=args.seed,
    )

    # — Simulazione —
    print("\nAvvio simulazione...\n")
    agents = simulate(
        terrain=terrain_obj,
        artva=artva,
        agents=agents,
        n_steps=args.steps,
        dt=DT_SIM,
        sigma=SIGMA_ACC_SIM,
        agl=args.agl,
        rng_seed=args.seed,
    )

    # — Plot risultati —
    fig_mission = plot_mission(
        terrain_obj, artva, agents, x_coords, y_coords, sub_dem
    )

    # — Animazione (opzionale) —
    anim = None
    if args.animate:
        anim = animate_mission(
            terrain_obj, artva, agents,
            dt=DT_SIM, speed=2.0,
            save=args.save,
        )

    plt.show()
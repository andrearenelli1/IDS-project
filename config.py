"""
config.py
=========
Parametri globali della simulazione di ricerca valanga multi-drone.

Tutti i moduli del progetto importano le costanti da qui.
Modificare questo file per cambiare i parametri di esecuzione.
"""

import numpy as np

# ============================================================================
# Workspace
# ============================================================================
AREA_SIZE_M   = 200.0  # [m]  lato area quadrata estratta dal DEM

# ============================================================================
# Volo
# ============================================================================
AGL_HEIGHT    = 1.5     # [m]  altezza sopra il terreno (Above Ground Level)
LIDAR_SIGMA   = 0.1    # [m]  deviazione standard rumore gaussiano LiDAR

# ============================================================================
# Droni
# ============================================================================
N_DRONES      = 3       # numero di droni (modificabile via --n)
DEPLOY_OFFSET = 2.0     # [m]  distanza laterale tra droni al deployment

# ============================================================================
# Lawnmower (fase SEARCH)
# ============================================================================
LANE_SPACING  = 15.0    # [m]  distanza tra passaggi nella striscia

# ============================================================================
# Sorgente ARTVA
# ============================================================================
ARTVA_MOMENT      = 1.0    # momento magnetico normalizzato [A·m²]
ARTVA_NOISE_STD   = 1e-6   # rumore additivo (~1% segnale a 40 m)
VICTIM_XY         = None   # [m, m] posizione xy vittima nel workspace locale;
                            #        None = casuale (seed da CLI)
VICTIM_DEPTH      = 3      # [m]  profondità di sepoltura sotto il terreno

# ============================================================================
# Calibrazione rumore (fase pre-volo)
# ============================================================================
N_NOISE_CALIB_SAMPLES = 20     # misure per drone per stimare σ_noise
NOISE_CONSENSUS_ITERS = 10     # iterazioni average-consensus tra droni

# Portata massima affidabile per l'ES — da Azzollini et al. (arXiv:2106.14514),
# Sec. IV-A SITL: distanza iniziale drone-sorgente ≈ 50 m, da cui l'ES
# dimostra convergenza con α=20, κ=0.07, ω=0.65.
# Usata come floor assoluto per DETECT_THR: impedisce che rumore bassissimo
# abbassi la soglia fino a triggerare TRACK dall'intero workspace.
ES_DETECT_MAX_R = 50.0   # [m]

# Raggio entro cui la stima della sorgente è considerata "trovata".
# Azzollini et al. usa 5×5 m come bounding box pratica per il soccorritore;
# usiamo 10 m per tenere conto delle approssimazioni della simulazione.
FOUND_RADIUS = 10.0      # [m]

# Valori nominali per la visualizzazione — questi servono solo ai plot.
# La simulazione usa soglie dinamiche misurate.
ARTVA_DETECT_THR = max(5 * ARTVA_NOISE_STD, ARTVA_MOMENT / ES_DETECT_MAX_R**3)  # ≈ 8e-6
TRACK_STOP_THR   = ARTVA_MOMENT / FOUND_RADIUS**3                                # = 1e-3

# filtro esponenziale per segnale ARTVA (per ridurre l'effetto del rumore nelle decisioni)
TAU_FILTER_ARTVA = 0.5   # [s] costante di tempo


# ============================================================================
# Extremum Seeking (ES) — TRACK mode  [Azzollini et al., 2021 — eq. 11-13]
#
#   ẋ_ref = √(α·ω) · cos(ω·t + κ·yt)
#   ẏ_ref = √(α·ω) · sin(ω·t + κ·yt)
#   ẏt    = (−1/λ)·α + (1/λ)·α_max     (rampa α da 0 → α_max)
#
#   yt = 1/∛S  — segnale condizionato (min in sorgente, convesso)
#
# Vincolo velocità: √(α_max · ω) ≤ V_MAX = 3.0 m/s  →  ω = V_MAX²/α_max
# ============================================================================
ES_ALPHA_MAX = 20.0    # [-]   ampiezza massima α (raggio cerchio ≈ √(α/ω) m)
ES_OMEGA     = 0.45    # [rad/s] frequenza: √(20·0.45) = 3.0 m/s = V_MAX  ✓
ES_KAPPA     = 0.05    # [-]   guadagno feedback segnale condizionato
ES_LAMBDA    = 15.0    # [s]   costante di tempo rampa α (α → α_max in ~3λ s)
ES_EPS       = 1e-12   # [-]   floor per evitare 1/cbrt(0)

# ============================================================================
# DICT — Distributed Iterative Consensus Triangulation
# ============================================================================
DICT_BETA        = 0.4   # [-]  peso consensus inter-drone (Combine step)
DICT_XY_ITERS    = 100    # [-]  iterazioni XY consensus a 3 droni (dopo disambiguazione)
DICT_DEPTH_ITERS = 100    # [-]  iterazioni depth consensus (fase 3)

# ============================================================================
# DCGD — Distributed Consensus Gradient Descent (fase SUPPORT, XY refinement)
# ============================================================================
# Algoritmo Adapt+Combine: ogni drone fa un passo di GD sulla propria misura
# ARTVA, poi media con i vicini. Si esegue mentre i droni sono fermi sul
# cerchio a 120°; al termine, l'orbita viene ricentrata sulla stima raffinata
# e si avvia la stima di profondità dalla posizione aggiornata.
DCGD_ITERS   = 100   # [-]  iterazioni di Adapt+Combine
DCGD_STEP_XY = 0.1   # [m]  step per iterazione (gradiente normalizzato)

# ============================================================================
# Triangolazione
# ============================================================================
CONVERGE_RADIUS = 5.0   # [m] raggio circonferenza finale attorno a source_est (depth phase)

# ============================================================================
# MPC
# ============================================================================
DT_MPC  = 0.1    # [s]
N_MPC   = 20
A_MAX   = 6.0    # [m/s²]
V_MAX   = 3.0    # [m/s]

# ============================================================================
# IMDCL
# ============================================================================
IMDCL_SIGMA_ACC   = 0.15   # [m/s²] rumore processo del filtro (3× SIGMA_ACC_SIM — filtro conservativo)
IMDCL_P0_POS      = 0.5    # [m]    dev.std iniziale posizione
IMDCL_P0_VEL      = 0.1    # [m/s]  dev.std iniziale velocità
IMDCL_COMM_RADIUS = 80  # [m]    raggio comunicazione inter-drone
IMDCL_R_MEAS_STD  = 0.3    # [m]    dev.std misura relativa posizione
IMDCL_PI_MAX_NORM = 1e4    # [a.u.] norma Frobenius massima per Pi_jl; oltre → reset a zero
IMDCL_R_LIDAR_STD = 0.05   # [m]    dev.std misura LiDAR quota
IMDCL_H_LIDAR     = np.array([[0., 0., 1., 0., 0., 0.]])  # H per pz (1×6)

# ============================================================================
# Simulazione
# ============================================================================
N_SIM         = 600     # passi massimi
DT_SIM        = DT_MPC
N_SIGNAL_SAMPLES  = 5      # [-]  misure ARTVA salvate nel drone per media mobile
SIGMA_ACC_SIM = 0.05   # [m/s²] rumore accelerazione simulazione (< IMDCL_SIGMA_ACC)
STOP_THRESH   = 0.3     # [m]  soglia raggiungimento waypoint

# ============================================================================
# Visualizzazione
# ============================================================================
ANIM_SPEED = 10.0   # fattore di accelerazione animazione (1.0 = tempo reale)
COLORS = {i: c for i, c in enumerate([
    "#e63946", "#2a9d8f", "#e9c46a", "#a8dadc",
    "#f4a261", "#6a4c93", "#1982c4", "#8ac926",
])}
BG_DARK = "#0d1117"

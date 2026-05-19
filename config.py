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
NOISE_DETECT_FACTOR   = 100.0  # DETECT_THR = FACTOR × σ̂_noise  (SEARCH→TRACK)
NOISE_STOP_FACTOR     = 1000.0 # STOP_THR   = FACTOR × σ̂_noise  (TRACK/SUPPORT→STOP)

# Valori nominali per la visualizzazione (derivati da ARTVA_NOISE_STD × fattori).
# La simulazione usa soglie dinamiche misurate — questi servono solo ai plot.
ARTVA_DETECT_THR = NOISE_DETECT_FACTOR * ARTVA_NOISE_STD   # ≈ 1e-5
TRACK_STOP_THR   = NOISE_STOP_FACTOR   * ARTVA_NOISE_STD   # ≈ 1e-4

# ============================================================================
# Hill-climbing online (fase TRACK)
# ============================================================================
TRACK_STEP_M      = 5.0    # [m]  passo nel piano xy
TRACK_TURN_DEG    = 60.0   # [°]  rotazione quando il segnale cala
SUPPORT_CIRCLE_N  = 9      # [-]  punti per la circonferenza percorsa dai droni SUPPORT
N_SIGNAL_SAMPLES  = 5      # [-]  misure ARTVA per step (interpolate lungo il moto)

# ============================================================================
# Stima distribuita posizione sorgente — DCGD (fase TRACK)
# ============================================================================
DIST_EST_ALPHA    = 0.3    # [m]     passo normalizzato discesa del gradiente
DIST_EST_BETA     = 0.4    # [-]     peso consensus inter-drone
DIST_EST_H        = 0.1    # [m]     passo differenze finite per gradiente numerico
DIST_EST_REFINE   = 300    # [-]     iterazioni extra di raffinamento post-blocco
DIST_EST_BATCH    = 5      # [-]     misure recenti usate per ogni aggiornamento online

# ============================================================================
# Triangolazione
# ============================================================================
TRIANGULATE_N_PARTNERS = 2  # droni chiamati in supporto al rilevamento

# ============================================================================
# Consenso distribuito selezione partner (min-consensus su grafo limitato)
# ============================================================================
CONSENSUS_K_MAX = 10   # iterazioni max: deve essere ≥ diametro stimato della rete

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
IMDCL_SIGMA_ACC   = 0.05   # [m/s²] rumore processo del filtro
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
SIGMA_ACC_SIM = IMDCL_SIGMA_ACC   # rumore accelerazione = rumore filtro
STOP_THRESH   = 0.3     # [m]  soglia raggiungimento waypoint

# ============================================================================
# Visualizzazione
# ============================================================================
COLORS = {i: c for i, c in enumerate([
    "#e63946", "#2a9d8f", "#e9c46a", "#a8dadc",
    "#f4a261", "#6a4c93", "#1982c4", "#8ac926",
])}
BG_DARK = "#0d1117"

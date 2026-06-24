# ARTVA Search & Rescue — Substrate

> Entry point per AI tools e nuovi collaboratori.
> Leggere questo file prima di qualsiasi altro.

## Cosa fa questo progetto

Simulazione multi-agente di ricerca in valanga con droni autonomi.
Ogni drone vola a quota costante sopra un DEM reale (TINItaly 10 m).
FSM a 5 stati, tutti con waypoint arrival-gated:

- **SEARCH**: lawnmower con lane spacing adattivo; → TRACK quando segnale ≥ `artva_detect_thr` (dinamica, con floor fisico)
- **TRACK**: Extremum Seeking (Azzollini et al. arXiv:2106.14514) — il drone percorre una traiettoria circolare il cui centro converge verso il massimo del segnale ARTVA (= sorgente); → STOP quando segnale ≥ `track_stop_thr` (dinamica)
- **STOP**: hovering; seleziona 2 droni SUPPORT via min-consensus; il Particle Filter continua ad affinare la stima
- **SUPPORT**: percorre una circonferenza di raggio `(moment/S)^(1/3)` centrata sul drone STOP (uno CW, uno CCW); → STOP a `track_stop_thr` (dinamica)
- **FINAL_ORBIT**: alla terminazione, tutti i droni con PF attivo orbitano un cerchio di `FINAL_ORBIT_N_WAYPOINTS` waypoint attorno alla stima congelata per affinare il PF; fine quando ogni drone raggiunge l'ultimo waypoint

Terminazione della ricerca: (1) `N_STOP`=3 droni in STOP; (2) chiamata SUPPORT scaduta (`SUPPORT_SEARCH_TIMEOUT`, sempre attesa anche se un partner si è già fermato); (3) timeout globale. In ogni caso, prima dell'arresto, segue l'orbita finale FINAL_ORBIT. Il successo (`found`) è valutato a posteriori sul PF: ellisse di confidenza 95% **drift-corretta** che contiene la vittima ED ha area ≤ `FOUND_ELLIPSE_AREA_MAX` (non più "≥3 STOP entro 10 m").
Il PF lavora nel **frame stimato** `x_est` (GPS-denied): `source_est` va confrontato con la vittima dopo aver rimosso il drift `(x_est − x)`.
Il controllo traiettoria usa MPC a orizzonte finito; la localizzazione distribuita usa filtro IMDCL (cooperative Kalman).
Tutte le coordinate sono in **metri locali del workspace** (origine = angolo SW dell'area DEM estratta); `terrain.utm_origin` contiene l'offset UTM.

## Soglie dinamiche e lane spacing adattivo

Le soglie di rilevamento non sono costanti — vengono calcolate a runtime in `simulate()`:
```
σ̂ = consensus(misure locali di ogni drone)
DETECT_THR = max(mu_noise + 5 × σ̂,  ARTVA_MOMENT / ES_DETECT_MAX_R³)
STOP_THR   = ARTVA_MOMENT / FOUND_RADIUS³
```
Il floor fisico (`ES_DETECT_MAX_R = 50 m`, da paper SITL) garantisce che il TRACK parta sempre entro il bacino di convergenza dell'ES.

Subito dopo, il lane spacing del lawnmower viene ricalcolato adattivamente:
```
r_detect     = (ARTVA_MOMENT / DETECT_THR)^(1/3)
n_lanes      = ceil(workspace_x / (2 × r_detect × n_drones))
lane_spacing = workspace_x / (n_lanes × n_drones)
```
Questo garantisce copertura completa senza gap per qualsiasi livello di rumore.
I waypoint vengono rigenerati e `wp_idx` azzerato per ogni drone dopo il calcolo.

## Stima sorgente: Particle Filter (pf.py)

Ogni drone inizializza un `ParticleFilter` (300 particelle) al momento del primo rilevamento.
Le particelle sono campionate in coordinate polari relative alla posizione del drone:
- $r_0$ dalla formula del modello ARTVA inverso
- $\phi \sim \mathcal{U}[0, 2\pi]$, $\sin\psi \sim \mathcal{U}[0,1]$

Ad ogni passo:
1. **update_weights** con misura locale (likelihood Gaussiana con sigma adattivo = sqrt(σ_n² + (0.20·S)²))
2. **update cooperativo**: per ogni vicino entro `IMDCL_COMM_RADIUS` che ha già il PF attivo, update aggiuntivo con la sua misura
3. **resample** sistematico solo se N_eff < N/2

La stima finale è la media pesata delle particelle: `source_est = Σ w_k · ξ_k`.

## Livelli di rumore attivi

`[1e-7, 1e-6, 1e-5]` — il livello `1e-8` è stato rimosso perché la portata di rilevamento supera 100 m, fuori dal bacino ES per qualsiasi posizione di deployment.

## Stack tecnico

- **Python 3.10+**
- `numpy`, `scipy` — algebra, interpolazione DEM, filtro
- `casadi` — ottimizzazione NLP per MPC
- `tifffile` — lettura GeoTIFF senza GDAL
- `matplotlib` — visualizzazione statica e animazioni
- DEM: tile TINItaly `w51065_s10.tif` (non versionato, ~30 MB)

## File principali

| File | Ruolo |
|---|---|
| `config.py` | **Unica sorgente di verità** per tutti i parametri |
| `terrain.py` | I/O DEM + classe `Terrain` + plot diagnostici |
| `artva.py` | Modello dipolo magnetico sorgente ARTVA |
| `mpc_drone.py` | Controllore MPC + modello punto-massa 3-D |
| `imdcl.py` | Filtro Kalman cooperativo distribuito |
| `pf.py` | Particle Filter 3-D per stima sorgente |
| `drone_agent.py` | FSM drone (SEARCH/TRACK ES/STOP/SUPPORT/FINAL_ORBIT), lawnmower, ES nav |
| `simulation.py` | Loop temporale multi-agente + calibrazione soglie |
| `visualization.py` | Plot statici + animazioni |
| `main.py` | Entry point CLI (singola run, replay da parametri CSV) |
| `run_experiments.py` | Sweep parametrico multi-processo con resume automatico |
| `plot_results.py` | Analisi e visualizzazione risultati sweep CSV |

## File rimossi (consolidati)

- `dem_tinitaly.py` → fuso in `terrain.py`
- `animate_drone.py` → fuso in `visualization.py` (`animate_mpc_standalone`)

## Entry point

```bash
# Simulazione base
python main.py --n 3 --steps 600 --animate

# Replay esatto di un run dal CSV (per debug)
python main.py --n 4 --noise 1e-6 --rc 80 --area 200 \
               --victim-x 45.0 --victim-y 80.0 --victim-depth 2.5 \
               --ws 0.60,0.45 --animate

# Sweep parametrico (riparte dai run mancanti se il CSV esiste già)
python run_experiments.py --workers 4 --out results.csv

# Plot risultati
python plot_results.py results.csv
```

## Dove guardare per...

| Obiettivo | File/Sezione |
|---|---|
| Cambiare parametri (AGL, N_MPC, N droni…) | `config.py` |
| Cambiare portata massima ES (floor rilevamento) | `config.py → ES_DETECT_MAX_R` |
| Cambiare criterio "found" (ellisse) | `config.py → FOUND_ELLIPSE_CONF, FOUND_ELLIPSE_AREA_MAX` |
| Cambiare orbita finale | `config.py → FINAL_ORBIT_RADIUS, FINAL_ORBIT_N_WAYPOINTS` |
| Calibrazione rumore (campioni, consensus) | `simulation.py → _calibrate_noise` |
| Parametri ES (alpha, omega, kappa) | `config.py → ES_*` |
| Cambiare il pattern SEARCH | `drone_agent.py → lawnmower_waypoints` |
| Lane spacing adattivo | `simulation.py` (formula dopo `_calibrate_noise`) |
| Cambiare la navigazione TRACK | `simulation.py → _es_update` |
| Cambiare la circonferenza SUPPORT | `drone_agent.py → circle_waypoints`, `simulation.py → _assign_support_partners` |
| Migliorare la stima posizione | `pf.py`, `simulation.py` (sezione PF update) |
| Nuovi plot o metriche | `plot_results.py`, `visualization.py` |
| Aggiungere livelli di rumore allo sweep | `run_experiments.py → ARTVA_NOISE_STDS` + `plot_results.py → NOISE_STDS/NOISE_LABELS` |
| Usare un DEM diverso | `terrain.py → TIF_PATH` |

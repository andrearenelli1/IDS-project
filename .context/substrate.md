# ARTVA Search & Rescue — Substrate

> Entry point per AI tools e nuovi collaboratori.
> Leggere questo file prima di qualsiasi altro.

## Cosa fa questo progetto

Simulazione multi-agente di ricerca in valanga con droni autonomi.
Ogni drone vola a quota costante sopra un DEM reale (TINItaly 10 m).
FSM a 4 stati, tutti con waypoint arrival-gated:

- **SEARCH**: lawnmower; → TRACK quando segnale ≥ `artva_detect_thr` (dinamica, con floor fisico)
- **TRACK**: Extremum Seeking (Azzollini et al. arXiv:2106.14514) — il drone percorre una traiettoria circolare il cui centro converge verso il massimo del segnale ARTVA (= sorgente); → STOP quando segnale ≥ `track_stop_thr` (dinamica)
- **STOP**: hovering; seleziona 2 droni SUPPORT via min-consensus; raffinamento DCGD finale
- **SUPPORT**: percorre una circonferenza di raggio `min(r_segnale, r_dcgd)` centrata sul drone STOP (uno CW, uno CCW); → STOP a `track_stop_thr` (dinamica)

Terminazione: ≥ 3 droni in STOP **E** stima entro `FOUND_RADIUS` (10 m) dalla vittima → raffinamento DCGD → stima posizione vittima.
Il controllo traiettoria usa MPC a orizzonte finito; la localizzazione distribuita usa filtro IMDCL (cooperative Kalman).
Tutte le coordinate sono in **metri locali del workspace** (origine = angolo SW dell'area DEM estratta); `terrain.utm_origin` contiene l'offset UTM.

## Soglie dinamiche

Le soglie di rilevamento non sono costanti — vengono calcolate a runtime:
```
σ̂ = consensus(misure locali di ogni drone)
DETECT_THR = max(NOISE_DETECT_FACTOR × σ̂,  ARTVA_MOMENT / ES_DETECT_MAX_R³)
STOP_THR   = max(NOISE_STOP_FACTOR   × σ̂,  10 × ARTVA_MOMENT / ES_DETECT_MAX_R³)
```
Il floor fisico (`ES_DETECT_MAX_R = 50 m`, da paper SITL) garantisce che il TRACK parta sempre entro il bacino di convergenza dell'ES.

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
| `drone_agent.py` | FSM drone (SEARCH/TRACK ES/STOP/SUPPORT), lawnmower, ES nav |
| `simulation.py` | Loop temporale multi-agente + calibrazione soglie |
| `visualization.py` | Plot statici + animazioni (missione e MPC standalone) |
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

# Plot risultati (filtra automaticamente rumore non in NOISE_STDS)
python plot_results.py results.csv
```

## Dove guardare per...

| Obiettivo | File/Sezione |
|---|---|
| Cambiare parametri (AGL, N_MPC, N droni…) | `config.py` |
| Cambiare fattori soglia SNR | `config.py → NOISE_DETECT_FACTOR / NOISE_STOP_FACTOR` |
| Cambiare portata massima ES | `config.py → ES_DETECT_MAX_R` |
| Cambiare raggio "found" | `config.py → FOUND_RADIUS` |
| Calibrazione rumore (campioni, consensus) | `simulation.py → _calibrate_noise` |
| Parametri ES (alpha, omega, kappa) | `config.py → ES_*` |
| Cambiare il pattern SEARCH | `drone_agent.py → lawnmower_waypoints` |
| Cambiare la navigazione TRACK | `drone_agent.py` (ES logic) |
| Cambiare la circonferenza SUPPORT | `drone_agent.py → circle_waypoints`, `simulation.py → _transition_to_stop` |
| Migliorare la stima posizione | `imdcl.py`, `simulation.py → _dcgd_step` |
| Nuovi plot o metriche | `plot_results.py`, `visualization.py` |
| Aggiungere livelli di rumore allo sweep | `run_experiments.py → ARTVA_NOISE_STDS` + `plot_results.py → NOISE_STDS/NOISE_LABELS` |
| Usare un DEM diverso | `terrain.py → TIF_PATH` |

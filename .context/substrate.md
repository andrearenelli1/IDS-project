# ARTVA Search & Rescue — Substrate

> Entry point per AI tools e nuovi collaboratori.
> Leggere questo file prima di qualsiasi altro.

## Cosa fa questo progetto

Simulazione multi-agente di ricerca in valanga con droni autonomi.
Ogni drone vola a quota costante sopra un DEM reale (TINItaly 10 m).
FSM a 4 stati, tutti con waypoint arrival-gated:

- **SEARCH**: lawnmower; → TRACK quando segnale ≥ ARTVA_DETECT_THR
- **TRACK**: esplorazione 3 candidati (avanti, ±60°, stessa distanza TRACK_STEP_M);
  si visita ognuno, si sceglie il più alto, si aggiorna la direzione; → STOP a TRACK_STOP_THR
- **STOP**: hovering; seleziona 2 droni SUPPORT via min-consensus; raffinamento DCGD finale
- **SUPPORT**: percorre una circonferenza di raggio = distanza stimata dalla sorgente,
  centrata sul drone STOP (un CW, uno CCW); → STOP a TRACK_STOP_THR

Terminazione: ≥ 3 droni in STOP → raffinamento DCGD → stima posizione vittima.
Il controllo traiettoria usa MPC a orizzonte finito; la localizzazione del
drone è distribuita tramite filtro IMDCL (cooperative Kalman).
Tutte le coordinate sono in **metri locali del workspace** (origine = angolo SW
dell'area DEM estratta); `terrain.utm_origin` contiene l'offset UTM.

## Stack tecnico

- **Python 3.10+**
- `numpy`, `scipy` — algebra, interpolazione DEM, filtro
- `casadi` — ottimizzazione NLP per MPC
- `tifffile` — lettura GeoTIFF senza GDAL
- `matplotlib` — visualizzazione statica e animazioni
- DEM: tile TINItaly `w51065_s10.tif` (non versionato, ~30 MB)

## File principali (dopo refactoring)

| File | Ruolo |
|---|---|
| `config.py` | **Unica sorgente di verità** per tutti i parametri |
| `terrain.py` | I/O DEM + classe `Terrain` + plot diagnostici |
| `artva.py` | Modello dipolo magnetico sorgente ARTVA |
| `mpc_drone.py` | Controllore MPC + modello punto-massa 3-D |
| `imdcl.py` | Filtro Kalman cooperativo distribuito |
| `drone_agent.py` | FSM drone (SEARCH/TRACK/STOP/SUPPORT), lawnmower, esplorazione 3 punti |
| `simulation.py` | Loop temporale multi-agente |
| `visualization.py` | Plot statici + animazioni (missione e MPC standalone) |
| `main.py` | Entry point CLI |

## File rimossi (consolidati)

- `dem_tinitaly.py` → fuso in `terrain.py`
- `animate_drone.py` → fuso in `visualization.py` (`animate_mpc_standalone`)

## Entry point

```bash
# Simulazione base (2 droni, 600 passi)
python main.py

# Con animazione 3-D salvata su disco
python main.py --n 3 --steps 600 --animate --save

# Animazione MPC standalone
python visualization.py --save --speed 2.0
```

## Dove guardare per...

| Obiettivo | File/Sezione |
|---|---|
| Cambiare parametri (AGL, N_MPC, N droni…) | `config.py` |
| Aggiungere un nuovo tipo di sensore | `drone_agent.py` + `simulation.py` |
| Cambiare il pattern SEARCH | `drone_agent.py → lawnmower_waypoints` |
| Cambiare il pattern TRACK (3 punti) | `drone_agent.py → init_track_round / _track_on_wp_reached` |
| Cambiare la circonferenza SUPPORT | `drone_agent.py → circle_waypoints`, `config.py → SUPPORT_CIRCLE_N` |
| Migliorare la stima posizione | `imdcl.py` |
| Nuovi plot o metriche | `visualization.py` |
| Usare un DEM diverso | `terrain.py → TIF_PATH` |
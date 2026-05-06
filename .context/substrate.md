# ARTVA Search & Rescue — Substrate

> Entry point per AI tools e nuovi collaboratori.
> Leggere questo file prima di qualsiasi altro.

## Cosa fa questo progetto

Simulazione multi-agente di ricerca in valanga con droni autonomi.
Ogni drone vola a quota costante sopra un DEM reale (TINItaly 10 m),
cerca la sorgente ARTVA con pattern lawnmower (fase SEARCH), poi converge
su di essa con hill-climbing reattivo (fase TRACK).
Quando il segnale ARTVA supera `TRACK_STOP_THR` (~15 m dalla sorgente), il
drone si ferma in hovering; quando tutti i droni TRACK sono fermi si esegue
il raffinamento DCGD per la stima distribuita della posizione della vittima.
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
| `drone_agent.py` | FSM drone (SEARCH/TRACK), lawnmower, hill-climbing |
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
| Cambiare il pattern di ricerca | `drone_agent.py → lawnmower_waypoints` |
| Migliorare la stima posizione | `imdcl.py` |
| Nuovi plot o metriche | `visualization.py` |
| Usare un DEM diverso | `terrain.py → TIF_PATH` |
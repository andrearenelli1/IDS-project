# CLAUDE.md

> Istruzioni per Claude Code e altri AI tool.
> Leggere `.context/substrate.md` per il quadro completo.

## Regole rapide

1. **Parametri → sempre da `config.py`**, mai inline.
2. **MPC usa `ag.x_est`**, mai `ag.x`.
3. **`plt.show()` solo in `main.py`** e negli `__main__` dei moduli.
4. **Modificare `ag.waypoints` → resettare sempre `ag.wp_idx = 0`**.
5. **`np.random.default_rng(seed)`** ovunque, mai `np.random.*` globale.
6. **FSM 4 stati**: SEARCH (lawnmower) → TRACK (3 candidati ±60°, arrival-gated) → STOP (hovering + sceglie 2 SUPPORT) / SUPPORT (circonferenza CW/CCW attorno al drone STOP, raggio = distanza stimata sorgente) → STOP. Tutte le assegnazioni waypoint sono arrival-gated; eccezione solo su transizione di stato.
   - **Consenso SUPPORT**: i droni in stato SUPPORT partecipano al consenso sulla source estimate insieme al drone STOP.
   - **Timeout chiamata SUPPORT**: se entro `SUPPORT_CALL_TIMEOUT` step (default 1000) non si trovano 2 droni nel raggio di comunicazione, l'affinazione della stima avviene comunque con i droni disponibili (anche solo 1 SUPPORT + STOP).
7. **Soglie dinamiche**: `ARTVA_DETECT_THR` e `TRACK_STOP_THR` **non esistono più** come costanti. Le soglie vengono calcolate a runtime in `simulate()` tramite `_calibrate_noise()`: ogni drone misura la σ locale (std di `N_NOISE_CALIB_SAMPLES` misure) → average-consensus → soglie = `NOISE_DETECT_FACTOR × σ̂` e `NOISE_STOP_FACTOR × σ̂`. Modificare i fattori in `config.py`, mai le soglie direttamente.

## Struttura .context/

```
.context/
  substrate.md          ← START HERE
  ai-rules.md           ← vincoli obbligatori
  anti-patterns.md      ← cosa NON fare
  glossary.md           ← terminologia di dominio
  decisions.md          ← perché le scelte architetturali sono quelle
  architecture/
    overview.md         ← diagrammi e flusso dati
```

## File rimossi (consolidati)

- `dem_tinitaly.py` è stato fuso in `terrain.py`
- `animate_drone.py` è stato fuso in `visualization.py` → `animate_mpc_standalone()`

Se vedi import a questi file, aggiornali.

## Dipendenze

```
pip install numpy scipy matplotlib casadi tifffile
```

Il file DEM `w51065_s10.tif` non è versionato. Scaricarlo da:
https://tinitaly.pi.ingv.it/Download_Area1_1.html
e posizionarlo nella root del progetto.
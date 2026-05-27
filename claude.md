# CLAUDE.md

> Istruzioni per Claude Code e altri AI tool.
> Leggere `.context/substrate.md` per il quadro completo.

## Regole rapide

1. **Parametri → sempre da `config.py`**, mai inline.
2. **MPC usa `ag.x_est`**, mai `ag.x`.
3. **`plt.show()` solo in `main.py`** e negli `__main__` dei moduli.
4. **Modificare `ag.waypoints` → resettare sempre `ag.wp_idx = 0`**.
5. **`np.random.default_rng(seed)`** ovunque, mai `np.random.*` globale.
6. **FSM 4 stati**: SEARCH (lawnmower) → TRACK (Extremum Seeking ES, Azzollini et al. arXiv:2106.14514) → STOP (hovering + sceglie 2 SUPPORT) / SUPPORT (circonferenza CW/CCW attorno al drone STOP, raggio = `min(r_segnale, r_dcgd)` con `r_segnale = (moment/S)^(1/3)`) → STOP. Tutte le assegnazioni waypoint sono arrival-gated; eccezione solo su transizione di stato.
   - **Consenso SUPPORT**: i droni in stato SUPPORT partecipano al consenso sulla source estimate insieme al drone STOP.
   - **Timeout chiamata SUPPORT**: se entro `SUPPORT_CALL_TIMEOUT` step (default 1000) non si trovano 2 droni nel raggio di comunicazione, l'affinazione della stima avviene comunque con i droni disponibili.
7. **Soglie dinamiche con floor fisico**: le soglie si calcolano a runtime in `simulate()` tramite `_calibrate_noise()`:
   ```
   σ̂ = average-consensus(misure locali)
   DETECT_THR = max(NOISE_DETECT_FACTOR × σ̂,  ARTVA_MOMENT / ES_DETECT_MAX_R³)
   STOP_THR   = max(NOISE_STOP_FACTOR   × σ̂,  10 × ARTVA_MOMENT / ES_DETECT_MAX_R³)
   ```
   Il floor fisico (`ES_DETECT_MAX_R = 50 m`, da paper SITL) impedisce che rumore bassissimo abbassi la soglia fuori dal bacino di convergenza dell'ES. Modificare i fattori e il raggio in `config.py`, mai le soglie direttamente.
8. **Criterio `found`**: un run è `found = True` solo se `n_stopped >= 3` **AND** `est_error < FOUND_RADIUS` (10 m). Verificare sempre entrambe le condizioni — solo `n_stopped` è un falso positivo se l'ES converge lontano dalla sorgente.
9. **Livelli di rumore attivi**: `[1e-7, 1e-6, 1e-5]`. Il livello `1e-8` è stato rimosso — con threshold dinamica, la portata di rilevamento supera 100 m (intero workspace), fuori dal bacino ES. Non reintrodurlo senza modificare `ES_DETECT_MAX_R` di conseguenza.

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

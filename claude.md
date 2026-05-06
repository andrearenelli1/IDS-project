# CLAUDE.md

> Istruzioni per Claude Code e altri AI tool.
> Leggere `.context/substrate.md` per il quadro completo.

## Regole rapide

1. **Parametri → sempre da `config.py`**, mai inline.
2. **MPC usa `ag.x_est`**, mai `ag.x`.
3. **`plt.show()` solo in `main.py`** e negli `__main__` dei moduli.
4. **Modificare `ag.waypoints` → resettare sempre `ag.wp_idx = 0`**.
5. **`np.random.default_rng(seed)`** ovunque, mai `np.random.*` globale.
6. **Partner droni**: quando chiamati rimangono **SEARCH** fino a raggiungere il drone detettore. Waypoint = posizione detettore. Auto-transizionano a **TRACK** al rilevamento ARTVA (≥ TRACK_STOP_THR).

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
# CLAUDE.md

> Istruzioni per Claude Code e altri AI tool.
> Leggere `.context/substrate.md` per il quadro completo.

## Regole rapide

1. **Parametri → sempre da `config.py`**, mai inline.
2. **MPC e Particle Filter usano `ag.x_est`**, mai `ag.x` (GPS-denied: il drone conosce solo la propria stima IMDCL). Quindi `source_est` è nel **frame stimato**: per confrontarlo con la vittima vera depurare il drift → `source_est − (x_est − x)`.
3. **`plt.show()` solo in `main.py`** e negli `__main__` dei moduli.
4. **Modificare `ag.waypoints` → resettare sempre `ag.wp_idx = 0`**.
5. **`np.random.default_rng(seed)`** ovunque, mai `np.random.*` globale. ⚠️ `pf.py` viola ancora questa regola (usa `np.random` globale → run non riproducibili); da sistemare passando un `Generator` seedato al PF.
6. **FSM 5 stati**: SEARCH (lawnmower) → TRACK (Extremum Seeking ES, Azzollini et al. arXiv:2106.14514) → STOP (hovering + sceglie 2 SUPPORT) / SUPPORT (circonferenza CW/CCW attorno al drone STOP, raggio = `(moment/S)^(1/3)`) → STOP; più **FINAL_ORBIT** (orbita finale di raffinamento, vedi regola 10). Tutte le assegnazioni waypoint sono arrival-gated; eccezione solo su transizione di stato.
   - **Lane spacing adattivo**: il lane spacing del lawnmower non è fisso ma viene ricalcolato in `simulate()` dopo la calibrazione del rumore: `n_lanes = ceil(strip_width / (2 × r_detect))`, `lane_spacing = strip_width / n_lanes`. Garantisce copertura completa senza gap. I waypoint vengono rigenerati e `wp_idx = 0` dopo il calcolo.
   - **Stima sorgente**: `ParticleFilter` (pf.py), inizializzato in coordinate polari al primo rilevamento, aggiornato cooperativamente con le misure dei vicini entro `IMDCL_COMM_RADIUS`. `ag.source_est` è la media pesata delle particelle. Init/update/resample usano `ag.x_est` (vicini: `agents[j].x_est`), **non** la posizione vera (vedi regola 2).
     - **Metriche di incertezza (in pf.py)**: covarianza pesata 2×2 (`weighted_mean_cov_xy`), area dell'ellisse di confidenza 95% (`ellipse_area`, k²=χ²₂(0.95)≈5.991), contenenza/Mahalanobis (`ellipse_contains`), IoU tra ellissi via clipping di poligoni convessi (`ellipse_iou`), aggregati per run (`run_ellipse_metrics`). Plot consenso IoU-vs-area in `plot_results.py`.
     - **Vincolo terreno (LiDAR)**: la vittima è sepolta, quindi ogni particella deve stare sotto la superficie, `ξz ≤ z_ground = p_z − AGL_HEIGHT` (terreno sotto il drone dal LiDAR, ipotesi piatto locale). Si passa `z_ground` sia a `initialize_particles` che a `resample_particles`. All'init il vincolo diventa un tetto su `sinpsi`: profondità `d = r·cosψ ≥ AGL_HEIGHT`, monotòna decrescente in `sinpsi` → `sinpsi_max` (scan su griglia in `_sinpsi_max`), si campiona `sinpsi ~ U[0, sinpsi_max]` (+ clamp di sicurezza nel caso degenere). Nel resampling, dopo il jitter le particelle con `ξz > z_ground` vengono **riflesse** sotto: `ξz ← 2·z_ground − ξz`.
   - **Timeout SUPPORT**: la chiamata di reclutamento dura `SUPPORT_SEARCH_TIMEOUT` step (config). Il timeout va **sempre atteso**, anche se nel frattempo un partner si è già fermato (team < 3): si dà al team la sua possibilità di reclutare fino a 3 droni prima di rinunciare quando ne sono raggiungibili meno. Alla scadenza si imposta `support_failed` → stop (→ orbita finale).
7. **Soglie dinamiche con floor fisico**: le soglie si calcolano a runtime in `simulate()` tramite `_calibrate_noise()`:
   ```
   (mu_noise, sigma_noise) = average-consensus(misure locali)
   DETECT_THR = max(mu_noise + 5 × sigma_noise,  ARTVA_MOMENT / ES_DETECT_MAX_R³)
   STOP_THR   = ARTVA_MOMENT / FOUND_RADIUS³
   ```
   Il floor fisico (`ES_DETECT_MAX_R = 50 m`, da paper SITL) impedisce che rumore bassissimo abbassi la soglia fuori dal bacino di convergenza dell'ES. Modificare `ES_DETECT_MAX_R` e `FOUND_RADIUS` in `config.py`.
8. **Criterio `found` (PF-based)**: NON più `n_stopped >= 3`. Un run è `found = True` se **almeno un drone con PF attivo** ha l'**ellisse di confidenza 95% drift-corretta** che (i) **contiene** la vittima nel piano (Mahalanobis ≤ k²) **AND** (ii) ha **area ≤ `FOUND_ELLIPSE_AREA_MAX`** (≈78.5 m², cerchio r≈5 m). Un solo drone basta a localizzare. Costanti: `FOUND_ELLIPSE_CONF`, `FOUND_ELLIPSE_AREA_MAX` in `config.py`. (Il vecchio `FOUND_RADIUS` resta solo per `TRACK_STOP_THR`.)
9. **Livelli di rumore attivi**: `[1e-7, 1e-6, 1e-5]`. Il livello `1e-8` è stato rimosso — con threshold dinamica, la portata di rilevamento supera 100 m (intero workspace), fuori dal bacino ES. Non reintrodurlo senza modificare `ES_DETECT_MAX_R` di conseguenza.
10. **Orbita finale (FINAL_ORBIT)**: comunque la ricerca si fermi (team in STOP=`N_STOP`, timeout SUPPORT scaduto, o timeout globale), prima dell'arresto **tutti i droni con PF attivo** orbitano attorno alla stima (centro **statico** = media dei `source_est` catturata all'istante di stop) per raccogliere viste diverse e affinare il PF. La durata **non è a tempo**: cerchio di `FINAL_ORBIT_N_WAYPOINTS` waypoint, finisce quando ogni drone ha raggiunto l'ultimo waypoint (flag `final_orbit_done`); solo un tetto di sicurezza anti-stallo. Il loop di `simulate()` è un `while` che può proseguire oltre `n_steps` per completare l'orbita. Config: `FINAL_ORBIT_RADIUS`, `FINAL_ORBIT_N_WAYPOINTS`. Nelle visualizzazioni (animazione, `plot_pf_evolution`, `plot_final_positions`) le particelle/stime vanno mostrate **drift-corrette**.

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

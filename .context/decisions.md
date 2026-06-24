# Decisioni architetturali

## TRACK: Extremum Seeking invece di hill-climbing

**Decisione**: la fase TRACK usa l'algoritmo ES (Azzollini et al. arXiv:2106.14514) invece del hill-climbing a 3 candidati.

**Perché**: il hill-climbing con 3 candidati ±60° era fragile in presenza di rumore (scelta errata della direzione, oscillazioni). L'ES è model-free, garantisce convergenza semiglobale al massimo del segnale e ha stabilità teorica dimostrata (Proposizione 1 del paper). Il raggio dell'orbita stazionaria (`√(α·ω) ≈ 3 m`) e la velocità massima (`√(α_max·ω) = V_MAX`) sono parametri direttamente raccordabili ai vincoli fisici del drone.

---

## Stima sorgente: Particle Filter invece di DCGD

**Decisione**: la stima della posizione della sorgente usa un Particle Filter 3-D (pf.py) invece del DCGD (Distributed Consensus Gradient Descent) usato in precedenza.

**Perché**: il DCGD richiedeva una stima iniziale ragionevole di `source_est` e convergeva lentamente in 3-D (profondità difficile da stimare con gradient descent su dati angolari). Il PF:
- Si inizializza in coordinate polari direttamente dalla misura, senza stima iniziale esterna
- Stima naturalmente la profondità 3-D (particelle nella shell 3-D)
- Ha una sigma adattiva (`sqrt(σ_n² + (0.20·S)²)`) che permette update cooperativi cross-drone anche quando i segnali differiscono di ordini di grandezza
- Il resampling lazy (solo quando N_eff < N/2) preserva la diversità vicino alla sorgente

---

## Soglie dinamiche con floor fisico

**Decisione**: `DETECT_THR = max(mu_noise + 5·σ̂, moment / r_max³)` con `r_max = 50 m`.

**Perché**: con soglie puramente proporzionali al rumore, rumore bassissimo (es. 1e-8) abbassa la soglia fino a triggerare TRACK dal punto di deployment, fuori dal bacino di convergenza dell'ES. Il floor fissa la portata massima di rilevamento a 50 m — il valore usato nelle simulazioni SITL del paper, dove la convergenza è dimostrata. Questo rende il comportamento indipendente dal livello di rumore per ciò che riguarda la distanza di innesco.

---

## Criterio `found` con soglia di posizione

**Decisione**: `found = (n_stopped >= 3) AND (est_error < FOUND_RADIUS)` con `FOUND_RADIUS = 10 m`.

**Perché**: senza la condizione sulla posizione, i run in cui l'ES converge su un ottimo locale lontano dalla sorgente venivano contati come successi (i droni si fermano perché il segnale è alto, non perché sono vicini alla vittima). Il valore 10 m è più permissivo della soglia operativa del paper (5×5 m box) per tenere conto delle approssimazioni della simulazione.

---

## Raggio orbita SUPPORT: (moment/S)^(1/3)

**Decisione**: `radius = (moment/S_stop)^(1/3)` — raggio fisicamente derivato dal segnale al punto di STOP.

**Perché**: in precedenza si usava la distanza dalla stima della sorgente. Quando la stima è ancora lontana dalla realtà (inizio convergenza PF), il raggio diventava dell'ordine di 80-100 m — i droni SUPPORT uscivano dall'area o volavano inutilmente lontano. La stima dal segnale locale `r_signal = (moment/S)^(1/3)` è sempre fisicamente affidabile al punto di STOP e fornisce la distanza reale stimata dalla sorgente.

---

## Rimozione livello di rumore 1e-8

**Decisione**: `ARTVA_NOISE_STDS = [1e-7, 1e-6, 1e-5]` — rimosso 1e-8.

**Perché**: con `noise=1e-8` la soglia dinamica scende a ~9e-7, corrispondente a una portata di rilevamento di ~100 m. Su aree 100×100 m il segnale è sopra soglia in tutto lo spazio fin dal deployment — il lawnmower non viene mai eseguito e l'ES parte sempre da distanze incompatibili con la convergenza garantita. Il livello 1e-8 non corrisponde a nessuno scenario operativo realistico per un drone (il rumore elettronico è dominato dall'EMI dei motori).

---

## Lane spacing adattivo al raggio di rilevamento

**Decisione**: il lane spacing del lawnmower non è fisso (`LANE_SPACING = 15 m`), ma viene calcolato in `simulate()` dopo la calibrazione del rumore: `n_lanes = ceil(strip_width / (2 × r_detect))`, `lane_spacing = strip_width / n_lanes`.

**Perché**: con lane spacing fisso, il lawnmower era sovra-denso per rumore basso (es. 1e-7, r_detect ≈ 47 m → 15 m spacing = 3× ridondanza inutile, ricerca lenta) e potenzialmente sotto-denso per rumore alto (1e-5, r_detect ≈ 13 m → spacing fisso 15 m > 2×13 = 26 m → piccoli gap di copertura). Il lane spacing adattivo garantisce copertura esatta con il minimo numero di corsie, rendendo il tempo di ricerca proporzionale alla difficoltà del problema.

---

## Replay da CLI (main.py --noise --rc --area --victim-x/y/depth --ws)

**Decisione**: aggiunta di parametri CLI a `main.py` per replicare esattamente i run dal CSV dello sweep.

**Perché**: durante l'analisi dei risultati è necessario ispezionare visualmente i run anomali con `--animate`. Senza questi parametri, occorreva modificare `config.py` a mano — error-prone e non riproducibile.

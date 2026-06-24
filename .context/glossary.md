# Glossario

## Termini di dominio

| Termine | Significato |
|---|---|
| **ARTVA** | *Apparecchio di Ricerca dei Travolti in Valanga* — trasmettitore/ricevitore magnetico a 457 kHz portato da sciatori |
| **AGL** | *Above Ground Level* — altezza del drone sopra il terreno (non quota assoluta) |
| **DEM** | *Digital Elevation Model* — modello digitale del terreno (file GeoTIFF) |
| **TINItaly** | Progetto INGV: DEM ad alta risoluzione dell'Italia (10 m/pixel) |
| **UTM** | Sistema di coordinate proiettate in metri (Est/Nord) usato nel DEM |
| **tile** | Un singolo file GeoTIFF del DEM (es. `w51065_s10.tif`) |

## Componenti software

| Termine | Significato |
|---|---|
| **MPC** | *Model Predictive Control* — controllore che ottimizza la traiettoria su un orizzonte futuro di N passi |
| **IMDCL** | *Information-based Multi-agent Decentralized Cooperative Localization* — filtro Kalman distribuito che combina misure locali e relative inter-drone |
| **FSM** | *Finite State Machine* — macchina a stati del drone: `SEARCH` → `TRACK` → `STOP/SUPPORT` → `FINAL_ORBIT` |
| **FINAL_ORBIT** | Stato finale: alla terminazione tutti i droni con PF attivo orbitano un cerchio di `FINAL_ORBIT_N_WAYPOINTS` waypoint attorno alla stima congelata, per affinare il PF; fine al raggiungimento dell'ultimo waypoint |
| **ellisse di confidenza** | Regione 95% della stima PF (covarianza 2×2 pesata, k²=χ²₂(0.95)≈5.991); usata per il criterio `found` (drift-corretta, deve contenere la vittima con area ≤ `FOUND_ELLIPSE_AREA_MAX`) e per le metriche di consenso (area media, IoU a coppie) |
| **lawnmower** | Pattern di volo a greca (corsie parallele) per coprire sistematicamente l'area; lane spacing calcolato adattivamente da `r_detect` |
| **ES** | *Extremum Seeking* — algoritmo model-free che fa percorrere al drone una traiettoria circolare il cui centro converge verso il massimo del segnale ARTVA; usato in fase TRACK (Azzollini et al. arXiv:2106.14514) |
| **PF** | *Particle Filter* — filtro a particelle 3-D usato da ogni drone per stimare la posizione della sorgente ARTVA; inizializzato in coordinate polari, aggiornato cooperativamente dai vicini |
| **warm-start** | Inizializzazione del solver MPC con una soluzione ammissibile per ridurre il tempo di calcolo al primo passo |
| **landmark message** | Messaggio IMDCL che codifica la stima di posizione di un drone per essere usata come riferimento da un vicino |
| **min-consensus** | Algoritmo distribuito per propagare la distanza geodesica lungo la rete di comunicazione; usato per selezionare i 2 partner SUPPORT più vicini al drone STOP |

## Variabili chiave

| Variabile | Shape | Unità | Descrizione |
|---|---|---|---|
| `x` | (6,) | m, m/s | Stato reale del drone `[px,py,pz,vx,vy,vz]` in coordinate locali |
| `x_hat` | (6,) | m, m/s | Stima IMDCL dello stato |
| `u` | (3,) | m/s² | Accelerazione comandata `[ax,ay,az]` |
| `wp` | (3,) | m | Waypoint `[x,y,z]` in coordinate locali workspace |
| `S` | scalare | a.u. | Intensità segnale ARTVA (adimensionale normalizzato) |
| `P` | (6,6) | misto | Matrice covarianza stima IMDCL |
| `source_est` | (3,) | m | Stima locale del drone della posizione sorgente (media pesata PF) |

## Parametri soglie (calcolati a runtime)

| Termine | Significato |
|---|---|
| `artva_detect_thr` | Soglia dinamica rilevamento: `max(mu + 5·σ̂, moment/ES_DETECT_MAX_R³)` — il drone passa da SEARCH a TRACK |
| `track_stop_thr` | Soglia di stop: `moment / FOUND_RADIUS³` — il drone si ferma quando supera questa |
| `ES_DETECT_MAX_R` | Portata massima affidabile per l'ES (50 m, da paper SITL); usata come floor per `DETECT_THR` |
| `FOUND_RADIUS` | Raggio usato solo per `track_stop_thr` (10 m); **non** è più il criterio `found` |
| `FOUND_ELLIPSE_AREA_MAX` | Area massima dell'ellisse 95% perché un run sia `found=True` (≈78.5 m², cerchio r≈5 m); la vittima deve cadere nell'ellisse drift-corretta |

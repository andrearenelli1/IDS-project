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
| **FSM** | *Finite State Machine* — macchina a stati del drone: `SEARCH` → `TRACK` |
| **lawnmower** | Pattern di volo a greca (corsie parallele) per coprire sistematicamente l'area |
| **ES** | *Extremum Seeking* — algoritmo model-free che fa percorrere al drone una traiettoria circolare il cui centro converge verso il massimo del segnale ARTVA; usato in fase TRACK (Azzollini et al. arXiv:2106.14514) |
| **hill-climbing** | ~~Algoritmo greedy reattivo~~ — **sostituito da ES** in fase TRACK; il termine compare ancora nel codice legacy ma non è il comportamento attivo |
| **warm-start** | Inizializzazione del solver MPC con una soluzione ammissibile per ridurre il tempo di calcolo al primo passo |
| **landmark message** | Messaggio IMDCL che codifica la stima di posizione di un drone per essere usata come riferimento da un vicino |

## Variabili chiave

| Variabile | Shape | Unità | Descrizione |
|---|---|---|---|
| `x` | (6,) | m, m/s | Stato reale del drone `[px,py,pz,vx,vy,vz]` in coordinate locali |
| `x_hat` | (6,) | m, m/s | Stima IMDCL dello stato |
| `u` | (3,) | m/s² | Accelerazione comandata `[ax,ay,az]` |
| `wp` | (3,) | m | Waypoint `[x,y,z]` in coordinate locali workspace |
| `S` | scalare | a.u. | Intensità segnale ARTVA (adimensionale normalizzato) |
| `P` | (6,6) | misto | Matrice covarianza stima IMDCL |
| `source_est` | (3,) | m | Stima locale del drone della posizione sorgente (DCGD) |

## Algoritmo stima sorgente — DCGD

| Termine | Significato |
|---|---|
| **DCGD** | *Distributed Consensus Gradient Descent* — stima online distribuita della posizione sorgente ARTVA |
| **Adapt** | Passo locale: `theta_i -= alpha * grad_J_i / \|\|grad_J_i\|\|` su batch di misure recenti |
| **Combine** | Passo consensus: media pesata con vicini TRACK entro `IMDCL_COMM_RADIUS` |
| `TRACK_STOP_THR` | Soglia dinamica: `max(NOISE_STOP_FACTOR × σ̂, floor)` — il drone si ferma quando il segnale la supera; con i parametri nominali corrisponde a ~20 m dalla sorgente |
| `ES_DETECT_MAX_R` | Portata massima affidabile per l'ES (50 m, da paper SITL); usata come floor per `DETECT_THR` e `STOP_THR` |
| `FOUND_RADIUS` | Raggio entro cui la stima DCGD deve cadere perché un run sia `found=True` (10 m) |
| **raffinamento** | `DIST_EST_REFINE` iterazioni DCGD extra eseguite quando tutti i droni TRACK si fermano |
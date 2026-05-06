# Regole per AI — Vincoli obbligatori

> Queste regole NON sono negoziabili.
> Violazioni causano bug silenti o inconsistenze architetturali.

## 1. config.py è l'unica sorgente di verità

**Non definire mai** costanti numeriche o soglie direttamente nei moduli.
Importa **sempre** da `config.py`.

```python
# ✅ CORRETTO
from config import AGL_HEIGHT, ARTVA_DETECT_THR

# ❌ SBAGLIATO
AGL = 1.5   # duplicazione → deriva silenziosa
```

## 2. Il controllore MPC usa sempre la stima IMDCL

Il MPC riceve `ag.x_est` (= `ag.imdcl.x_hat`), **mai** `ag.x` (posizione reale).
Questo replica il comportamento reale: il drone non conosce la propria posizione esatta.

```python
# ✅ CORRETTO
u_opt = ag.ctrl.step(ag.x_est, ag.current_target())

# ❌ SBAGLIATO
u_opt = ag.ctrl.step(ag.x, ag.current_target())  # cheating!
```

## 3. Stato del drone: convenzioni array

- Posizione: indici `[0:3]` → `[px, py, pz]` in UTM [m]
- Velocità:  indici `[3:6]` → `[vx, vy, vz]` in [m/s]
- Waypoint:  `np.ndarray` shape `(3,)` → `[x, y, z]`

## 4. Coordinate: sistema locale del workspace

Tutte le coordinate spaziali usate nella simulazione sono in **metri locali**:
- Origine = angolo SW dell'area estratta dal DEM
- x ∈ [0, AREA_SIZE_M], y ∈ [0, AREA_SIZE_M]
- Recupera coordinate UTM assolute con `terrain.utm_origin = (E_utm, N_utm)`

Non mescolare coordinate pixel, coordinate UTM assolute o coordinate locali.
La conversione locale → UTM è `x_utm = x_local + terrain.utm_origin[0]`.

## 5. Rumore: usa `numpy.random.default_rng`, non `numpy.random`

```python
# ✅ CORRETTO
rng = np.random.default_rng(seed)
noise = rng.normal(0, sigma)

# ❌ SBAGLIATO
noise = np.random.normal(0, sigma)  # stato globale → non riproducibile
```

## 6. Terrain.z() è thread-safe ma non vettorizzata per y variabile

Per query massive usa `ravel()` + `reshape()` come fa internamente.

## 7. IMDCL: apply_update è broadcast a tutto il team

Quando un drone riceve un update relativo cooperativo, questo viene
applicato a **tutti** i droni del team (`for k in drone_ids`), non solo ai due coinvolti.
Questa è una proprietà fondamentale del filtro IMDCL distribuito.

## 8. Nomi funzione: convenzioni esistenti

| Concetto | Nome atteso |
|---|---|
| Loop principale | `simulate()` in `simulation.py` |
| Costruzione agenti | `build_agents()` in `simulation.py` |
| Terreno interrogabile | `terrain.z(x, y)` / `terrain.agl_z(x, y, agl)` |
| Step MPC | `ag.ctrl.step(x_est, target)` |
| Segnale ARTVA | `artva.signal(pos, noisy=True)` |
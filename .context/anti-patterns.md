# Anti-pattern da evitare

## Simulazione e controllo

### ❌ Accedere alla posizione reale dal controllore
```python
# SBAGLIATO — il drone reale non conosce la propria posizione esatta
u = ctrl.step(ag.x[:3], target)

# CORRETTO
u = ctrl.step(ag.x_est[:3], target)
```

### ❌ Confrontare waypoint TRACK con `all_waypoints_done()`
In fase TRACK la lista waypoint viene sovrascritta ad ogni passo
(`ag.waypoints = [wp_nuovo]`). Non usare `all_waypoints_done()` per
decidere la fine della missione in TRACK: usare la distanza alla sorgente.

### ❌ Modificare `ag.waypoints` senza resettare `ag.wp_idx`
```python
# SBAGLIATO
ag.waypoints = nuovi_waypoint  # wp_idx potrebbe puntare fuori

# CORRETTO
ag.waypoints = nuovi_waypoint
ag.wp_idx    = 0
```

## Terreno e DEM

### ❌ Chiamare `terrain.z()` con coordinate pixel
`Terrain.z(x, y)` si aspetta coordinate UTM in metri, non indici riga/colonna.

### ❌ Usare `interpolate_surface()` in tempo reale
`interpolate_surface()` è costosa (RBF su ~400 punti). È solo per i plot diagnostici del DEM, non per interrogazioni durante la simulazione.

### ❌ Ignorare `NaN` nel DEM
Bordi del tile TINItaly possono contenere `NaN` (NoData). `build_terrain()` li
riempie con la media prima di costruire l'interpolatore, ma `sub_dem` grezzo
li mantiene. Usare `np.nanmean` / `np.nan_to_num` nei plot.

## IMDCL

### ❌ Chiamare `propagate()` dopo `apply_update()` nello stesso step
L'ordine corretto in ogni passo è: propagate → update → step_no_measurement.
Invertire causa double-counting del rumore di processo.

### ❌ Condividere lo stesso oggetto `AgentIMDCL` tra droni diversi
Ogni drone deve avere la propria istanza con `x_hat` e `P` indipendenti.

## Visualizzazione

### ❌ Chiamare `plt.show()` dentro `plot_mission()` o `animate_mission()`
Le funzioni restituiscono figure/anim; `plt.show()` è responsabilità di `main.py`.

### ❌ Salvare animazioni con `PillowWriter` a alta risoluzione
Il fallback GIF (Pillow) è accettabile solo per debug. Per produzione usare ffmpeg.
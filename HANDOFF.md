# Handoff — Livellare `results.csv` (progetto IDS, ricerca valanga multi-drone)

## Obiettivo
Bilanciare il dataset `results.csv`: portare **tutte** le celle `(area, n_droni)` a
**13 vittime random per combinazione di parametri**. Oggi due celle sono sotto-campionate:
- `area=100, n=3` → 3 vittime/combo (servono +900 run)
- `area=100, n=4` → 5 vittime/combo (servono +720 run)
- tutte le altre celle → già 13/combo (OK)

Totale run da aggiungere: **1620**.

## Contesto chiave (perché è così)
- Ogni cella ha le stesse **90 combinazioni** di parametri: `noise(3) × acc(2) × rc(3) × workspace(5)`.
  Nessuna combinazione manca: cambia solo il numero di **vittime random** per combinazione.
- Il CSV è stato costruito in più passate aumentando `N_RANDOM_VICTIMS` (vedi git log
  "new partial/complete data"); le celle completate prima sono rimaste a 3/5 vittime.
- Le vittime sono campionate a caso a ogni run: in `run_one()` c'è
  `_pos_rng = np.random.default_rng()` **senza seed** → ogni chiamata = vittima/profondità
  nuove. Quindi per livellare basta **chiamare `run_one` N volte in più** per le combo
  sotto-campionate.

## ⚠️ Gotcha importante — NON usare il resume nativo dello sweep
`run_id` è **posizionale** nella griglia `itertools.product(..., range(N_RANDOM_VICTIMS), ...)`
e il resume salta per `run_id`. Cambiare `N_RANDOM_VICTIMS` riordina la griglia → il mapping
`run_id ↔ parametri` non corrisponde più al CSV esistente (che ha `run_id` fino a 5400 da
griglie precedenti). Quindi **non** rilanciare `run_experiments.py` sullo stesso file:
corromperebbe la corrispondenza. Si fa un **top-up dedicato** che appende righe con `run_id` nuovi.

## Cosa fare
È già pronto lo script **`level_dataset.py`** nella root del repo. Fa esattamente il top-up
sicuro: legge `results.csv`, calcola il deficit per ogni combo, chiama `run_one` il numero
giusto di volte e **appende** righe con `run_id = max+1, max+2, …` e lo schema `CSV_FIELDS`
identico (verificato: match esatto delle colonne, comprese `pf_ellipse_area_mean_m2` e
`pf_iou_mean`). Fa anche un backup automatico in `results.csv.bak`.

```bash
cd /home/andrea/IDS-project
python level_dataset.py --dry-run     # verifica il piano (deve dire +900 n=3, +720 n=4)
python level_dataset.py               # esegue (≈1620 simulazioni; ci vuole tempo)
```

Se preferisci scrivere su file separato e fare il merge dopo: `level_dataset.py` scrive
in-place sul file passato con `--out`; per separarlo, copia prima `results.csv` su un nuovo
file e passa `--out quel_file`.

## Verifica finale (dopo l'esecuzione)
```bash
python3 -c "
import csv
from collections import Counter
rows=list(csv.DictReader(open('results.csv')))
def combo(r): return (r['artva_noise_std'],r['acc_sim_ms2'],r['comm_radius_m'],r['workspace_frac_r'],r['workspace_frac_c'])
for area,n in sorted({(int(float(r['area_size_m'])),int(r['n_drones'])) for r in rows}):
    sub=[r for r in rows if int(float(r['area_size_m']))==area and int(r['n_drones'])==n]
    reps=Counter(Counter(combo(r) for r in sub).values())
    print(f'area={area} n={n}: righe={len(sub)} vittime/combo={dict(reps)}')
"
```
Atteso: tutte le celle a `{13: 90}` e 1170 righe ciascuna.

## Note / caveat
- **Non riproducibile:** il PF usa `np.random` globale e `_pos_rng` è senza seed → le run
  aggiunte non sono ripetibili bit-a-bit (coerente col resto del dataset, generato così).
- Il dataset esistente è **già coerente col codice attuale** (colonne ellisse popolate;
  criterio `found` PF-based, con `found=True` anche per `n_drones_stopped` 1–2). Le righe
  aggiunte useranno il codice corrente: assicurati che il working tree non abbia modifiche
  non committate alla simulazione prima di girare, così tutte le righe sono dello stesso "vintage".
- Dopo il livellamento: **rigenera i plot** (`python plot_results.py results.csv`) e **riempi
  le tabelle/numeri** nel `report.tex` (ora placeholder `---`).
- Memoria di progetto rilevante: `pf-localization-design`, `final-orbit-termination`,
  `report-results-pending`.

## File coinvolti
- `level_dataset.py` — script di top-up (pronto, root del repo)
- `run_experiments.py` — `run_one()`, `CSV_FIELDS`, `_ws_label()`, griglia parametri
- `results.csv` — dataset da livellare (backup automatico in `results.csv.bak`)

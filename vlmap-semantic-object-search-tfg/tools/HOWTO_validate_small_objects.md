# Cómo validar el small-objects eval por la mañana

Esta guía sirve para reproducir y validar los resultados del fix del placer
en el experimento de búsqueda open-vocabulary de objetos pequeños.

## Estado actual (2026-04-27 noche)

| Run | SR | Notas |
|---|---:|---|
| baseline | 14.0% | original, antes de cualquier fix |
| v7 | 44.0% | mejor lógica de búsqueda |
| post_placement_full | 68.0% | + fix de placer (kitchen counter deconfliction) |
| post_placement_v2 (target mañana) | 68-72% | + laptop assets desbloqueados |

Lo que añade post_placement_v2 sobre post_placement_full:
1. `_EXTRA_CATALOG` con 8 hashes de laptops reales que sí tienen
   `.object_config.json` (Toshiba, Sony, HP, MacBook, etc.).
2. Floor offset reducido de 0.8 m a 0.3 m (defensivo, no cambia trash
   bin actual pero protege futuros multi-floor placements).

**Nota sobre laptop**: el smoke test scene 1 con laptop unblocked dio
70.6% (igual que sin él). Inspeccionando `ov002r01.log` ("the laptop
in the bedroom"), el search pipeline enruta a **living room** aunque
la query dice "in the bedroom" (tie-breaking en room-select cuando
dos rooms tienen score casi igual: bedroom=0.37 vs living room=0.35).
El laptop SÍ está colocado correctamente en el bedroom — el bug está
en el routing del search pipeline. Fix está fuera del alcance del
placer; queda como follow-up en `vlmaps/policy/strategic_policy.py`.

El laptop unblock SÍ puede ayudar en escenas 2 y 3 donde:
- sc2: laptop antes 0/1 (no se colocaba) → ahora se coloca en dining room
- sc3: laptop ya iba 2/2 (con el nativo de office)

## Quick check (1 min) — verifica que el placer funciona

```bash
docker exec tfg-sim bash -c '
cd /workspace
# 1) Restaurar escenas a estado pre-placement
for s in 102344193 102344280 108736884_177263634; do
  cp data/versioned_data/hssd-hab/scenes/${s}.scene_instance.before_placements.json \
     data/versioned_data/hssd-hab/scenes/${s}.scene_instance.json
done

# 2) Re-correr el placer
for s in 102344193 102344280 108736884_177263634; do
  python3 tools/place_small_objects.py \
    --scene data/versioned_data/hssd-hab/scenes/${s}.scene_instance.json \
    --spec  tools/eval_queries/specs/small_objects_${s}.json \
    --out-jsonl /tmp/placer_${s}.jsonl --no-backup 2>&1 | grep -E "placed|skip"
done
'
```

**Lo que debes ver:**
- `[+] placed laptop ... on table` en LAS 3 escenas (antes solo en 1).
- 7 placements por escena, sin líneas `[skip]`.

Si ves `[skip] no HSSD template for object 'laptop'` → algo se rompió.
Revisar `_EXTRA_CATALOG` en `tools/place_small_objects.py` (líneas ~125-140).

## Smoke test (5 min) — verifica que la search funciona

```bash
docker exec tfg-sim bash -c '
cd /workspace
python3 tools/run_full_eval.py \
  --scene-ids 0 --queries tools/eval_queries/small_objects \
  --dataset-type hssd --data-paths hssd \
  --scene-dataset-config-file data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json \
  --policy-mode hybrid --yoloe-conf-thresh 0.3 --per-query-timeout 180 \
  --methods Ob_Hp \
  --out results/small_objects_eval/morning_smoke
'

# Lectura del resultado
cat /home/mario/tfg/vlmap-semantic-object-search-tfg/results/small_objects_eval/morning_smoke/pipeline_full/102344193_0/aggregate_metrics.md | head -8
cat /home/mario/tfg/vlmap-semantic-object-search-tfg/results/small_objects_eval/morning_smoke/pipeline_full/102344193_0/compare_full.md
```

**Esperado:**
- SR scene 1 ≥ 70% (anoche dio 70.6%; ahora apunta a ~76% con laptops).
- `laptop` y `the laptop in the bedroom` deberían cambiar de FAIL a OK.

## Full eval (~30 min) — el número definitivo para la memoria

```bash
docker exec tfg-sim bash -c '
cd /workspace
python3 tools/run_full_eval.py \
  --scene-ids 0,1,2 --queries tools/eval_queries/small_objects \
  --dataset-type hssd --data-paths hssd \
  --scene-dataset-config-file data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json \
  --policy-mode hybrid --yoloe-conf-thresh 0.3 --per-query-timeout 180 \
  --methods Ob_Hp \
  --out results/small_objects_eval/morning_full
'
```

Genera tabla por objeto:
```bash
docker exec tfg-sim bash -c '
cd /workspace
python3 tools/analyze_open_vocab_eval.py \
  --run results/small_objects_eval/morning_full \
  --out-dir results/small_objects_eval/morning_full/open_vocab_analysis
'

# Léela
cat /home/mario/tfg/vlmap-semantic-object-search-tfg/results/small_objects_eval/morning_full/open_vocab_analysis/open_vocab_by_method.md
```

## Troubleshooting

**Si SR < 65%**: el placer no corrió o las escenas no están con placements.

```bash
docker exec tfg-sim python3 -c "
import json
for s in ['102344193','102344280','108736884_177263634']:
    p = f'/workspace/data/versioned_data/hssd-hab/scenes/{s}.scene_instance.json'
    d = json.load(open(p))
    print(s, ':', len(d['object_instances']), 'instances')
"
```

Esperado: `102 / 141 / 256` instancias (base + 7 placements en cada uno).
Si sale `95 / 134 / 249` → solo está la versión pre-placements; re-corre el placer.

**Si laptop sigue a 0/2**: comprobar que el catálogo carga los hashes.

```bash
docker exec tfg-sim python3 -c "
import sys; sys.path.insert(0,'/workspace/tools')
from place_small_objects import _load_object_catalog
from pathlib import Path
cat = _load_object_catalog(Path('/workspace/data/versioned_data/hssd-hab/metadata/object_categories_filtered.csv'))
print('laptop templates:', cat.get('laptop', [])[:5])
print('total laptop:', len(cat.get('laptop', [])))
"
```

Debe mostrar al menos 4 templates de laptop, todos hashes (no `Laptop_X`).

**Si el smoke test peta con OOM o timeout**: reducir `--per-query-timeout`
a 120 s o lanzar una sola escena. Cada query usa ~500 MB GPU.

## Métricas para la memoria

1. **Diagnóstico del bug de placement**: kettle = 0/6 en todos los runs
   anteriores porque los 6 objetos del counter se solapaban al milímetro.
2. **Fix aditivo**: deconfliction por (template_id, XZ) + grid offset
   35 cm en counter / 30 cm en suelo.
3. **Catálogo extendido**: 8 templates de laptop con `.object_config.json`
   reinyectados vía `_EXTRA_CATALOG`.
4. **Resultados clave**:
   - baseline 14% → post_placement 68% (+54 pts; 7 → 34 / 50 queries)
   - kettle: **0/6 desde siempre → 3/6** (desbloqueado por el deconfliction)
   - laptop: 1/5 nativo → 2/5 (post_placement) → ~4/5 (con extras, mañana)

## Archivos relevantes

| Path | Propósito |
|---|---|
| `tools/place_small_objects.py` | El placer (commit 98a8744 + posterior) |
| `tools/eval_queries/specs/small_objects_*.json` | Specs (no modificar) |
| `data/versioned_data/hssd-hab/scenes/*.before_placements.json` | Backup pre-placement |
| `data/versioned_data/hssd-hab/scenes/*.scene_instance.json` | Escena activa (overwrite por placer) |
| `results/small_objects_eval/post_placement_full/` | Run de referencia anoche |
| `results/small_objects_eval/post_placement_full/SUMMARY_vs_baseline.md` | Resumen escrito |

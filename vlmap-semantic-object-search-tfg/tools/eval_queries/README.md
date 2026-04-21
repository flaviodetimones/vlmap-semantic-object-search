# Evaluation Query JSONL

This directory stores normalized `JSONL` batches for evaluation, one file per
scene:

- `tools/eval_queries/{scene_name}.jsonl`

Each line is a single query record with this schema:

```json
{
  "id": "q001",
  "scene_id": 0,
  "scene_name": "102344193_0",
  "query": "bed",
  "query_type": "object",
  "target_label": "bed",
  "expected_rooms": ["bedroom"],
  "expected_room_polygons": [{"label": "bedroom", "instance_idx": 0}],
  "tags": ["single_object"]
}
```

## How it was generated

Generator script:

```bash
python /workspace/tools/build_eval_queries.py \
  --scene-ids 0,1 \
  --queries-per-scene 50 \
  --seed 21042026
```

Default assumptions:

- dataset: `HSSD`
- scenes: `0,1`
- minimum navigable room ratio: `0.25`
- total: `50` queries per scene
- deterministic seed: `21042026`

## Sampling criteria

The generator reuses the same scene setup used by the online batch tooling:

- rooms come from `SearchState` + `SemanticSceneRoomProvider`
- the object pool comes from present VLMap categories in the scene
- `expected_rooms` for `object` queries are derived deterministically from
  `vlmaps/utils/room_priors.py` with manual priors only (`llm_output={}`)
- `expected_room_polygons` are resolved against the actual HSSD room polygons
  exposed by the room provider

## Query-type distribution

Per scene, the default distribution is:

- `60%` `object`
- `20%` `room`
- `15%` `room_object`
- `5%` `compound`

For `50` queries per scene, that is:

- `30` object
- `10` room
- `8` room_object
- `2` compound

## Notes

- `expected_rooms` uses canonical room families for `object` queries.
- For `room`, `room_object`, and `compound`, `expected_rooms` follows the room
  family named in the query.
- `expected_room_polygons` stores the actual scene room family label and a
  deterministic `instance_idx` inside that family.
- If a query cannot disambiguate a duplicated room family, all matching
  polygons are included.
- `tags` are intended for later slicing, for example:
  `single_object`, `explicit_instance`, `multi_instance`, `ambiguous_room`,
  `canonical_alias`, `low_prior`, `multi_step`.

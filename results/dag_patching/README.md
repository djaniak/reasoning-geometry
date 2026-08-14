# DAG patching evidence package

The eight `*.json` runs here are the complete record of the 2026-08-13 arithmetic
DAG patching pilot (`EXPERIMENT_LOG.md`, entry of that date). They are **immutable**:
committed byte-for-byte as produced, content-addressed in `MANIFEST.json`, and
never rewritten. Anything recovered after the fact is stored *beside* them, never
merged into them.

They are deliberately in git rather than DVC. They are not a DVC stage and never
were — they appear in neither `dvc.yaml` nor `dvc.lock` — and there is no DVC
remote on this host, which would make `.dvc/cache` the only copy.

## Files

| File | What it is |
|:---|:---|
| `feasibility.json` | first run, consistent (`both`) donor edit |
| `result_only.json`, `operand_only.json` | the donor mechanism split |
| `depth{1,2,3}_gap0.json` | the path-depth ladder |
| `depth1_gap{1,2}.json` | token-distance controls for depth 1 |
| `MANIFEST.json` | sha256, run commit, model and tokenizer revision, replay command, schema version, derived fields, and inferred fields |
| `tokenizer_alignment.json` | the three checkpoints tokenize the same trace identically — a precondition for any Base/Instruct/Distill comparison — checked per item family, for every arm that has been or will be run |

## Item family

All eight runs used the `v1_unpaired` generator, which is **not paired across
depth**: its chain steps draw from the main random stream, so an item re-rolls
entirely when depth changes and `depth1_gap0` / `depth2_gap0` / `depth3_gap0` are
three different families. Their depth comparison is between-family and cannot be
read item by item.

`v2_paired` is the generator for new runs and is the default. `v1_unpaired` stays
reachable only so these artifacts remain re-derivable — `dag_evidence` pins it
for any report that does not name a generator, which is all eight of them.

## Schema versions

`feasibility`, `result_only`, and `operand_only` are **v0**: they predate the
`depth`, `gap`, and `ancestor_distance` fields (and `feasibility` predates
`condition`, which was added with the donor split).

Those values are recovered by regenerating the items — generation is
deterministic and tokenizer-only — and written into `MANIFEST.json` under
`derived_fields`. A derivation is accepted only if the regenerated items
reproduce the archived measurements: item count, token count, target value, and
the kind, node, and `distance_to_read` of every edit in recorded order.

`derived` and `inferred` are kept apart on purpose:

- **derived** — recovered and checked against the measurement (`depth`, `gap`,
  `ancestor_distance`).
- **inferred** — taken from the run order and the log, *not* checkable. Listed
  per artifact from what that report omits, so it is honest in both directions.
  Only `feasibility` lacks `condition`; the other two v0 runs state theirs.
  `n_decoys` is unrecorded in all eight, v1 included, so it is flagged
  everywhere — though a wrong value would change the token count and be rejected.

## Provenance field names

- `run_commit` — the commit that produced the run. No report recorded it. It is
  recovered from the artifact mtime bracketed against the commit timeline, and
  corroborated independently by which schema fields the report carries: a report
  with `condition` cannot predate the commit that added it, one without cannot
  postdate it. Both signals agree for all eight. The values are frozen in
  `dag_evidence.RUN_COMMITS` because git does not preserve mtimes, so a clone
  loses the first signal; `mtime_still_confirms` records whether this checkout
  can still see it.
- `replay_command` — reconstructed from the recorded settings. It reproduces the
  run; it is not a transcript of what was typed.
- `manifest_generation_commit` — when the manifest was built. Not when any run
  was produced.

## Scoring

Verdicts are a policy over the rows, not part of the measurement, so any gate
revision is re-runnable here without a GPU:

```
uv run python dag_patching.py --rescore results/dag_patching/result_only.json
uv run python dag_evidence.py --table
```

Two surface-control policies are reported for every run. `v1_two_sided` is the
rule as originally registered; `v2_one_sided` is the active post-hoc amendment.
See `dag_patching._surface_gate` for why, and `EXPERIMENT_LOG.md` for what
passing v2 does and does not establish.

Each run also reports `prospective_joint_layer`: the layers at which every gate
clears together. It is frozen for the next paired run and does not decide any
archived verdict — see `dag_patching._joint_layer_gate`.

## Regenerating this package

```
uv run python dag_evidence.py --manifest --tokenizer_report --table
```

This reads the artifacts and writes only `MANIFEST.json` and
`tokenizer_alignment.json`. It never loads a model and never touches the eight
runs.

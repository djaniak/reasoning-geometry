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
| `MANIFEST.json` | sha256, source commit, model and tokenizer revision, exact command, schema version, and derived fields |
| `tokenizer_alignment.json` | the three checkpoints tokenize the same trace identically — a precondition for any Base/Instruct/Distill comparison |

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
- **inferred** — taken from the run order and the log, *not* checkable. The donor
  `condition` changes only the donor text; positions, token count, target value,
  and every distance are identical across conditions, so no archived field can
  confirm it. `n_decoys` was never recorded either, though a wrong value would
  change the token count and be rejected.

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

## Regenerating this package

```
uv run python dag_evidence.py --manifest --tokenizer_report --table
```

This reads the artifacts and writes only `MANIFEST.json` and
`tokenizer_alignment.json`. It never loads a model and never touches the eight
runs.

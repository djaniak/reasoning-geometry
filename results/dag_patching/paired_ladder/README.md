# Paired depth ladder

Rerun of the depth ladder from `results/dag_patching/` (which is immutable and
was **not** touched) using the `v2_paired` generator, so the five arms below
are the same item family at every depth/gap instead of three re-rolled ones.
See `EXPERIMENT_LOG.md`, 2026-08-14, "The paired depth ladder is rerun."

Run commit for all five: `20ea135e36fceca36909c8122b84b6d02a68a5d2`. Model
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, seed 0, `n_items 5`,
`n_decoys 6`, `condition both` — identical to the archived ladder's settings,
generator changed only.

| File | Replay command |
|:---|:---|
| `depth1_gap0.json` | `uv run python dag_patching.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --condition both --seed 0 --n_items 5 --depth 1 --gap 0 --generator v2_paired --output results/dag_patching/paired_ladder/depth1_gap0.json` |
| `depth2_gap0.json` | same, `--depth 2 --gap 0` |
| `depth3_gap0.json` | same, `--depth 3 --gap 0` |
| `depth1_gap1.json` | same, `--depth 1 --gap 1` |
| `depth1_gap2.json` | same, `--depth 1 --gap 2` |

These are not in `MANIFEST.json` or `tokenizer_alignment.json` — those cover
only the eight archived, provenance-recovered runs. This directory needs no
recovery: every field the archived package had to infer or derive is recorded
directly in these reports (`generator`, `n_decoys`, `depth`, `gap`).

Not scored under `v1_two_sided` retroactively for interpretation, though
`--rescore` reports it: the active policy is `v2_one_sided`, and the point of
this run was to score under it, and under `joint_layer`, from the start.

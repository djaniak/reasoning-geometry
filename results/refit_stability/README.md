# Outer-refit sweep

The evidence for the 2026-08-21 review's outer-refit blocker: every interval
elsewhere in this project bootstraps *prompts* with the fitted pipeline frozen
at `seed=42`, which answers "different prompts to score on" and not "different
partition to fit on". Produced by `controls/refit_stability.py`; registered
before it ran, in the `EXPERIMENT_LOG.md` entry of 2026-08-22, and read out in
the entry of 2026-08-25.

Three seeds (42, 101, 202) x two long-trace models, run `--skip_peer`. Seed 42
is the frozen partition and so is the sweep's own control rather than an
independent refit; two of the three are genuinely new partitions.

## Files

| File | What it is |
|:---|:---|
| `refit_stability_partial_report.md` | the per-refit values and the cross-refit stability table |
| `refit_stability_partial_results.json` | the same numbers machine-readable, plus `summary` |
| `work/seed_<n>/<model>/decomposition/` | the refitted OOF report and results JSON |
| `work/seed_<n>/<model>/abstention/` | the refitted prompt-level readouts — where `B1 - B0` comes from |
| `work/seed_<n>/<model>/probe/` | the refitted last-token probe, in-fold layer and penalty selection included |
| `work/seed_<n>/<model>/*.log` | per-step stdout; the `PCA var=` and `memory-mapping` lines live here |
| `work/seed_<n>/<model>/.done_*` | one marker per completed step, recording the exact command it ran |

The `partial` in those two filenames is structural, not a defect:
`protocol_complete` requires all four registered seeds **and** all three models
**and** the peer ladder, so any run missing one of those writes the partial
stem. This run is missing all three of those things by design.

## What is not here

The `math500_prompt_decomposition_oof.csv` files, one per seed and model. They
are 14 MB against 1.6 MB for everything else, and they are intermediates:
`abstention` and `probe` consume them, and both of those outputs *are*
committed. Regenerate a missing one by re-running the sweep — `Step.done`
requires the marker **and** its artifact **and** a matching command, so a
checkout without the CSVs re-runs the decompositions rather than skipping them
on the strength of a marker alone.

Note that `analysis/rmd_window_sensitivity.py` reads the three **seed-42** OOF
CSVs directly. That analysis therefore needs a regenerated seed-42 decomposition
before it can run from a fresh checkout.

## Cost

~10 h wall for the three seeds, dominated by six decomposition steps at roughly
80 minutes each. Two things are load-bearing in that number and neither is
obvious:

- `analysis/layer_cache.py` must have a valid cache for the layer being fitted.
  A cache hit is a memmap slice, which moves 82.8 GiB (deepseek) and 93.1 GiB
  (deepseek_llama) of hidden states out of anonymous memory and into droppable
  page cache. Each `decomposition.log` records which path it took; the line to
  look for is `memory-mapping <n> traces (<size>) from the layer cache`. Its
  absence means the cache was refused and the NPZ path ran instead.
- Transparent huge pages must be disabled for the process tree. On a
  long-uptime host with fragmented free memory, every 2 MB anonymous fault
  triggers a synchronous compaction scan; the first attempt at this sweep spent
  15 h on one fold of one step at 100% system time and 0% user before being
  killed. `~/no_thp.py` sets `PR_SET_THP_DISABLE`, which is inherited across
  fork and preserved across execve.

## Completing it

`peer_residual_aurc` is `n: 0` here — the peer ladder is only a control if
target and peers share a partition, so it needs all three models at a shared
seed. Closing that half means qwen at seeds 101 and 202, then the peer step at
each seed. A full `protocol_complete` report additionally needs seed 303 and
qwen throughout.

# Remaining refit sweep: job specification

The one outstanding evidence gate that cannot be closed in a working session. It is
specified here so it can be started in tmux and left alone. **Do not start it from
an agent session** — the standing rule is that multi-hour jobs are launched by hand.

## What is missing

The registered protocol is four seeds (42, 101, 202, 303) x three models x the peer
ladder. Completed: seeds 42/101/202 x {deepseek, deepseek_llama}, `--skip_peer`.

| Missing piece | Steps | Peak RAM | Notes |
|---|---|---|---|
| qwen at seed 101 | decomposition, abstention, probe | **140 GB** (decomposition) | the expensive one |
| qwen at seed 202 | decomposition, abstention, probe | **140 GB** | the expensive one |
| peer ladder at seeds 42, 101, 202 | peer step | 8 GB each | `peer_step` iterates over **all** selected specs, so at seeds 101 and 202 it needs qwen — which does not exist there yet. Step 1 below must finish before step 2 can run at those seeds. Seed 42 alone is runnable today. |
| seed 303, all three models | everything | 140 GB per decomposition | the fourth registered seed; untouched |

`peak_gb` values are read from `controls/refit_stability.py` (`decomposition_step`
140, `probe_step` 8, `peer_step` 8), not estimated.

## Checked resource requirements on this host (2026-09-07)

| Resource | Required | Available | Verdict |
|---|---|---|---|
| RAM | 140 GB peak, one decomposition at a time | 503 GB total, 244 GB available | **OK**, but do not run two decompositions concurrently |
| Disk | measured: a completed seed directory is 6.7 MB (seed 42, 3 models) / 4.5 MB (seeds 101, 202, 2 models); each OOF CSV is **2.0 MB** and is *not* committed (`.gitignore` ignores `results/refit_stability/**/*.csv`); the committed remainder is ~0.47 MB per seed | 2.4 TB free on `/home` | **OK** |
| GPU | none — the sweep is CPU-only | — | **OK** |
| Time | ~2.5 h for qwen x 2 seeds + the three peer steps; roughly double if seed 303 is included | — | run in tmux |

Two conditions are load-bearing and were established during the 2026-08-23
layer-cache validation. Both must hold or the decomposition step will thrash:

1. **The `analysis/layer_cache.py` memmap must be in use.** The `memory-mapping`
   line in `work/seed_<n>/<model>/decomposition.log` is the confirmation.
2. **Transparent huge pages must be disabled for the process**
   (`PR_SET_THP_DISABLE`). The helper for this currently lives only at
   `~/no_thp.py`, outside the repo, alongside `~/run_refit_long_models.sh`.
   **Move both into the repo before relying on this document** — a job spec that
   depends on two files in someone's home directory is not reproducible.

## Command

Run from the repo root, in tmux. `Step.done` requires the marker **and** its
artifact **and** a matching command, so completed work is skipped and interrupted
work resumes correctly. Re-running is safe.

```bash
tmux new -s refit
cd /home/djaniak/projects/reasoning-geometry-probe
export PYTHONPATH=.

# 1. qwen at the two new partitions (the 140 GB steps, serial)
python ~/no_thp.py .venv/bin/python controls/refit_stability.py \
  --seed 101 --seed 202 --model qwen --skip_peer \
  --work_dir results/refit_stability/work \
  --output_dir results/refit_stability

# 2. the peer ladder at each seed, now that all three models exist there
python ~/no_thp.py .venv/bin/python controls/refit_stability.py \
  --seed 42 --seed 101 --seed 202 \
  --work_dir results/refit_stability/work \
  --output_dir results/refit_stability
```

Step 2 drops `--skip_peer`, which is what flips `require_peer` and lets
`protocol_complete` become reachable. Adding `--seed 303` to both invocations
completes the registered protocol; without it the report will still, correctly,
say the protocol is not complete.

## What closing it changes

The report stem switches from `refit_stability_partial` to `refit_stability`, and
the generator's "Registered protocol: NOT complete" line becomes a completion line.
Two quantities currently reported with `n: 0` — `peer_residual_aurc` and
`peer_residual_deployable_aurc` — get values for the first time.

**The registered decision rule has already returned its verdict on the two
quantities that carry the headline.** `b1_minus_b0_aurc` is sign-stable on both
refitted models with spread narrower than its bootstrap interval, so under the rule
fixed on 2026-08-22 the claim stands and both are reported. Closing the gate is
about completeness, not about rescuing the headline. What is genuinely unknown is
whether qwen — the model with the largest increment and no refit at a new partition
— behaves like the other two.

## Do not

* Run a bare `dvc repro`. Name the stage. (`EXPERIMENT_LOG.md`, 2026-08-22.)
* Run `dvc checkout` or `dvc gc`. There is no DVC remote on this host, so
  `.dvc/cache` is the only copy of the data and both are irreversible.
* Run two decomposition steps concurrently. 2 x 140 GB exceeds available RAM.

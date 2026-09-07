# Remaining registered refit sweep

Prepared 2026-09-07. The sweep has not been launched by this repair.
All four seeds (42, 101, 202, 303), all three models, and both peer-residual
quantities are required for protocol completion. Seed 303 is not optional.

## Existing work and inputs

Qwen has a frozen-seed reproduction. The two distilled models have seeds
42/101/202 without the peer step. Remaining work includes Qwen 101/202, missing
peer steps at every seed, and all three models at 303. Preserve the existing
`results/refit_stability/work` directory and its command markers.

Required inputs: original model NPZs, valid layer caches, the existing `.venv`,
and the completed work directories. Committed result JSONs permit inspection;
a fresh clone without these inputs cannot reproduce the fits.

The Linux process wrapper is now `controls/no_thp.py`, ported from the existing
home-directory helper. It accepts a **module**, not an executable or file path.
The old `~/run_refit_long_models.sh` is not needed: it selects only the two
long-trace models, skips peers, and defaults to three seeds. The native refit
planner already runs every step serially and skips matching completed commands.
The home-directory originals are preserved as historical launchers.

## Resources and preflight

The code's `peak_gb=140` for decomposition is a rough scheduling estimate,
not a measured peak for every model. Probe/peer estimates are 8 GB. On argon
at preparation, `/proc/meminfo` reported 229,517,828 kB available (about 219 GiB),
`/home` had 2.4 TB free, and SSH-cgroup full-memory-pressure averages were zero.
These are snapshots, not reservations. Recheck immediately before launch.
Use one sweep at a time. The remaining wall time is unverified; budget several
hours, including all of seed 303, rather than relying on the earlier 2.5 h estimate.

From the development repository root:

```bash
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
df -h .
cat /sys/fs/cgroup/system.slice/ssh.service/memory.pressure
PYTHONPATH=. .venv/bin/python -m controls.no_thp controls.refit_stability \
  --seed 42 --seed 101 --seed 202 --seed 303 \
  --work_dir results/refit_stability/work \
  --output_dir results/refit_stability --dry_run
```

All three geometry-layer caches validated on argon during preparation.
The verified dry run skips 21 completed steps and plans 19: Qwen 101/202
(6), all three models at 303 (9), and four peer steps.
Review the skip/plan list before launch. A missing CSV or mismatched marker
command causes its step to run again. Verified executable aliases in the same
virtual environment are treated as equivalent; other environments and changed
analysis arguments remain distinct. Do not rewrite markers merely to force a skip.
The dry run does not load hidden states or validate layer caches. Check the cache
manifests using `analysis.layer_cache.LayerCache.open` (which returns `None` for
missing/stale caches), and confirm `memory-mapping` in each actual decomposition log.
Caches live under repository `.layer_cache/` by default, outside the DVC data tree.
Do not run DVC checkout, cleanup, or broad reproduction commands for this job.

## Launch after preflight

Start tmux, change to the development repository root, and run:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" .venv/bin/python -u -m controls.no_thp \
  controls.refit_stability \
  --seed 42 --seed 101 --seed 202 --seed 303 \
  --work_dir results/refit_stability/work \
  --output_dir results/refit_stability
```

This single invocation includes peers and all three models. It needs no external
launcher, and serial execution is built into `plan_seed`/`run_step`.

## Completion criteria

Check `complete: true`, four registered seeds, all three models, and populated
`peer_residual_aurc` and `peer_residual_deployable_aurc` summaries. Inspect each
per-seed result, not just the report heading. Apply the registered sign/spread
rule from the experiment log's 2026-08-22 full-refit registration. Retain the
frozen seed-42 headline and disclose the new partitions separately. Until all
criteria pass, the manuscript must mark the full-refit protocol pending.

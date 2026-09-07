# Preprint repair verification — 2026-09-07

Base: `8c1b63e`. Applied to the development checkout on argon. The knowledge-base
`raw/` snapshot is unchanged. This is a bounded repair, not another full audit.

## Verified

- Corrected contrast definitions in the claim table and current summaries. The
  DeepConf key without `_baseline` includes DeepConf on both sides; the tail-over-
  full key compares the combined readout against B0 plus whole-trace RMD.
- Frozen headline point estimates checked against all three committed abstention
  JSONs; DeepConf values checked against its primary-population artifact.
- `837 passed` on argon in 15.54 s:
  `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q --ignore=tests/test_posttraining_error_recovery.py`.
- New regression checks cover peer report completeness, same-environment Python
  aliases versus changed environments/arguments, and Linux THP flag inheritance.
- Full four-seed dry-run command executes successfully: 21 skipped, 19 planned,
  zero research stages executed. Qwen seed 42 is reused rather than recomputed
  because of the python/python3 alias difference.
- `LayerCache.open` validates the geometry layer for each of the three models.
  This checks source fingerprints and maps the arrays; it does not recompute scores.
- OPI v3 §8.6 and Appendix L.1 checked: selective-prediction result exists and its
  reported coverage range is 0.50–1.00. Updated the current related-work discussion.
- Current experiment-log pointers converted to section-title references. Existing
  log entries left unchanged; the correction entry is appended at the end.

## Deliverables

- Corrected README, RMD strategy, strategy index, related-work discussion, claim
  table, and primary-population plan references.
- Historical notice on FINDINGS.md; historical content preserved.
- `controls/no_thp.py` and corrected refit planner/report behavior, with checks.
- Repository-contained launch instructions with every registered seed and the
  actual runtime preflight results. No dependency on either home-directory launcher.
- Working manuscript: `rmd-preprint-draft.md`. It is not a submission-ready paper.

## Not checked or not performed

- The documented posttraining test collection failure remains excluded; bare-root
  pytest collection and the unrelated free-energy collection problem were not fixed.
- No full refit, generation, bootstrap, or other new research analysis was run.
- No independent subagent review was commissioned in this repair. The validation
  above is a separate verification pass by the same agent, not independent review.
- No full audit of historical FINDINGS sections, DAG strategy, or executed notebook
  outputs. DAG work is a separate paper. The draft cites current source artifacts.
- No clean-clone full pipeline reproduction. Committed JSONs make results
  inspectable; original NPZs, caches, and fitting inputs are still required to rerun.
- Final figures, complete model/generation metadata, authors, affiliations, and
  publication formatting remain to be supplied/checked. The literature pass was
  limited to the OPI correction and the existing related-work record.
- No data files, numerical result JSONs, DVC state, old home helpers, or the
  pre-existing untracked `posttraining_error_recovery.py` were changed.
- No commit or push. Changes are in the argon working tree for review.

## Next action

Launch the verified four-seed sweep from the job specification after rechecking
current memory availability. Fill in the pending manuscript paragraph only after
examining the completed artifacts and registered decision rule. Do not select a
new seed or cutoff because its point estimate is more favorable.

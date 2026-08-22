# E3 clustered summaries for the DAG workshop draft

Source: `raw/repos/reasoning-geometry` commit `127606253d1265c3a293fc39070ebb306495cea2`; `results/dag_patching/e3_ladder/ANALYSIS.json` SHA-256 `42ef4e45a2f13adf8bb356ff7e01e3fc3ba64e27a87bbd7b2147d1ba22244391`.

`555/603` and `0/432` are descriptive site counts. Sites share generated items, arms, gap placements, and seeds. The table below keeps the original site estimand but shows the result separately for each of the three E3 seeds. It does not turn three seed rates into a confidence interval or a new significance test.

| semantic outcome | seed 0 | seed 1 | seed 2 | pooled descriptive count |
|:--|--:|--:|--:|--:|
| One written operation remains | 190/206 (92.2%) | 174/190 (91.6%) | 191/207 (92.3%) | 555/603 (92.0%) |
| Two or more written operations remain | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/432 (0.0%) |

The one-step denominator combines the three depth-1 gap placements and the one-step chain sites in the depth-2 and depth-3 arms. The depth-1 gap placements reuse the same generated spines within a seed, so they are not independent replications. The seed table makes that dependence visible without changing the registered E3 reading.

## Per-seed site rates

| site | seed 0 | seed 1 | seed 2 |
|:--|--:|--:|--:|
| Depth 1, gap 0 ancestor, one step | 35/39 (89.7%) | 33/37 (89.2%) | 36/41 (87.8%) |
| Depth 1, gap 1 ancestor, one step | 32/39 (82.1%) | 23/29 (79.3%) | 32/36 (88.9%) |
| Depth 1, gap 2 ancestor, one step | 27/32 (84.4%) | 22/28 (78.6%) | 27/34 (79.4%) |
| Depth 2, chain `n`, one step | 48/48 (100%) | 48/48 (100%) | 48/48 (100%) |
| Depth 2, ancestor, two steps | 0/48 (0.0%) | 0/48 (0.0%) | 0/48 (0.0%) |
| Depth 3, chain `n`, one step | 48/48 (100%) | 48/48 (100%) | 48/48 (100%) |
| Depth 3, chain `m`, two steps | 0/48 (0.0%) | 0/48 (0.0%) | 0/48 (0.0%) |
| Depth 3, ancestor, three steps | 0/48 (0.0%) | 0/48 (0.0%) | 0/48 (0.0%) |

Every seed shows the same qualitative split. The least favorable one-step cell is depth 1, gap 2, seed 1 at 22/28. Every multi-step cell is zero in every seed.

## Same-trace item pairs

Each row below uses 48 distinct items per seed. “Chain only” means the chain-site intervention put the implied digit alone on top and the ancestor intervention did not. These are the item-level clusters behind the pooled 144-pair tests.

| comparison | seed 0 | seed 1 | seed 2 | pooled |
|:--|:--|:--|:--|:--|
| Depth 2: chain `n` (one step) vs ancestor (two steps) | 48 chain only; 0 ancestor only | 48; 0 | 48; 0 | 144; 0 |
| Depth 3: chain `n` (one step) vs ancestor (three steps) | 48 chain only; 0 ancestor only | 48; 0 | 48; 0 | 144; 0 |
| Depth 3: chain `m` (two steps) vs ancestor (three steps) | 48 neither | 48 neither | 48 neither | 144 neither |

The published draft can quote the pre-registered D1 matched-pair test as its inferential headline. For E3, report the pooled rates alongside this seed table and the same-trace item-pair counts. Do not describe 1,035 sites as independent trials.

## Reproduction

The committed reader reproduces the source ledger without a model run:

```bash
cd raw/repos/reasoning-geometry
uv run python -m dag.dag_e3_ladder --output -
```

The tables here sum its layer-13 eligible records by `seed`, `steps`, and the same-trace `(seed, depth, item)` pairs. The source reader and its tests fix the inherited layer, eligibility rule, and pairing key.

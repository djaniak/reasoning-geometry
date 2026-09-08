# Incremental abstention analysis — deepseek_llama / math500 (L24)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.634; automatic failures 1; capped prompts 92; unparsed traces 229.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.747 | 0.251 | 0.212 | 0.612 |
| B1 | 0.795 | 0.203 | 0.200 | 0.581 |
| dumb_cap_count | 0.659 | 0.339 | 0.224 | 0.640 |
| dumb_unparsed_count | 0.654 | 0.344 | 0.224 | 0.642 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.048 [0.020, 0.078] p=0.000 | -0.048 [-0.077, -0.021] p=0.000 | -0.012 [-0.023, -0.004] p=0.000 | -0.030 [-0.050, -0.011] p=0.000 |

## valid_plurality

499 prompts; base accuracy 0.635; automatic failures 0; capped prompts 91; unparsed traces 221.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.747 | 0.251 | 0.212 | 0.613 |
| B1 | 0.796 | 0.202 | 0.200 | 0.582 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.049 [0.021, 0.074] p=0.000 | -0.049 [-0.081, -0.021] p=0.000 | -0.012 [-0.023, -0.003] p=0.010 | -0.031 [-0.052, -0.013] p=0.000 |

## cap_free_valid_plurality

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.763 | 0.234 | 0.209 | 0.608 |
| B1 | 0.813 | 0.185 | 0.198 | 0.577 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.050 [0.012, 0.098] p=0.020 | -0.050 [-0.081, -0.009] p=0.020 | -0.011 [-0.023, -0.000] p=0.030 | -0.031 [-0.056, -0.004] p=0.010 |

## cap_free_full_population

408 prompts; base accuracy 0.674; automatic failures 0; capped prompts 0; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.763 | 0.234 | 0.209 | 0.608 |
| B1 | 0.813 | 0.185 | 0.198 | 0.577 |
| dumb_cap_count | 0.656 | 0.342 | 0.220 | 0.633 |
| dumb_unparsed_count | 0.656 | 0.342 | 0.220 | 0.633 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.050 [0.012, 0.098] p=0.020 | -0.050 [-0.081, -0.009] p=0.020 | -0.011 [-0.023, -0.000] p=0.030 | -0.031 [-0.056, -0.004] p=0.010 |

## all_eight_parseable

411 prompts; base accuracy 0.672; automatic failures 0; capped prompts 3; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.763 | 0.234 | 0.209 | 0.606 |
| B1 | 0.813 | 0.184 | 0.197 | 0.575 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.050 [0.017, 0.087] p=0.010 | -0.050 [-0.082, -0.013] p=0.020 | -0.012 [-0.022, -0.002] p=0.020 | -0.032 [-0.056, -0.004] p=0.040 |

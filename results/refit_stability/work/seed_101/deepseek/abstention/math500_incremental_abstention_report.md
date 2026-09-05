# Incremental abstention analysis — deepseek / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.750; automatic failures 7; capped prompts 107; unparsed traces 351.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.846 | 0.152 | 0.170 | 0.518 |
| B1 | 0.870 | 0.128 | 0.162 | 0.493 |
| dumb_cap_count | 0.771 | 0.227 | 0.174 | 0.531 |
| dumb_unparsed_count | 0.769 | 0.229 | 0.173 | 0.529 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.024 [-0.002, 0.050] p=0.100 | -0.024 [-0.052, 0.002] p=0.070 | -0.008 [-0.016, -0.001] p=0.030 | -0.024 [-0.046, -0.004] p=0.030 |

## valid_plurality

493 prompts; base accuracy 0.761; automatic failures 0; capped prompts 100; unparsed traces 295.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.847 | 0.151 | 0.173 | 0.525 |
| B1 | 0.871 | 0.127 | 0.164 | 0.501 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.024 [-0.004, 0.055] p=0.100 | -0.024 [-0.054, 0.005] p=0.130 | -0.008 [-0.017, -0.001] p=0.010 | -0.024 [-0.046, -0.005] p=0.010 |

## cap_free_valid_plurality

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.858 | 0.140 | 0.157 | 0.490 |
| B1 | 0.889 | 0.108 | 0.150 | 0.466 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.032 [0.002, 0.069] p=0.020 | -0.032 [-0.059, 0.001] p=0.060 | -0.007 [-0.017, 0.002] p=0.140 | -0.024 [-0.049, 0.009] p=0.090 |

## cap_free_full_population

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.858 | 0.140 | 0.157 | 0.490 |
| B1 | 0.889 | 0.108 | 0.150 | 0.466 |
| dumb_cap_count | 0.768 | 0.230 | 0.163 | 0.507 |
| dumb_unparsed_count | 0.756 | 0.241 | 0.164 | 0.510 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.032 [0.002, 0.069] p=0.020 | -0.032 [-0.059, 0.001] p=0.060 | -0.007 [-0.017, 0.002] p=0.140 | -0.024 [-0.049, 0.009] p=0.090 |

## all_eight_parseable

384 prompts; base accuracy 0.799; automatic failures 0; capped prompts 4; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.858 | 0.140 | 0.156 | 0.487 |
| B1 | 0.895 | 0.103 | 0.147 | 0.457 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.037 [0.001, 0.072] p=0.050 | -0.037 [-0.071, -0.005] p=0.010 | -0.009 [-0.018, 0.002] p=0.100 | -0.030 [-0.059, 0.002] p=0.070 |

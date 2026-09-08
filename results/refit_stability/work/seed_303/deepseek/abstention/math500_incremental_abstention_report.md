# Incremental abstention analysis — deepseek / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.750; automatic failures 7; capped prompts 107; unparsed traces 351.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.824 | 0.174 | 0.171 | 0.523 |
| B1 | 0.864 | 0.134 | 0.164 | 0.499 |
| dumb_cap_count | 0.767 | 0.231 | 0.173 | 0.529 |
| dumb_unparsed_count | 0.769 | 0.229 | 0.172 | 0.527 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.041 [0.017, 0.067] p=0.000 | -0.041 [-0.077, -0.015] p=0.000 | -0.008 [-0.014, -0.001] p=0.040 | -0.024 [-0.041, -0.006] p=0.040 |

## valid_plurality

493 prompts; base accuracy 0.761; automatic failures 0; capped prompts 100; unparsed traces 295.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.824 | 0.174 | 0.174 | 0.530 |
| B1 | 0.865 | 0.133 | 0.166 | 0.507 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.042 [0.014, 0.072] p=0.000 | -0.042 [-0.072, -0.016] p=0.000 | -0.008 [-0.015, -0.001] p=0.030 | -0.024 [-0.040, -0.003] p=0.020 |

## cap_free_valid_plurality

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.831 | 0.167 | 0.159 | 0.498 |
| B1 | 0.891 | 0.106 | 0.149 | 0.463 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.061 [0.028, 0.092] p=0.000 | -0.061 [-0.091, -0.033] p=0.000 | -0.010 [-0.020, 0.001] p=0.070 | -0.035 [-0.063, -0.006] p=0.010 |

## cap_free_full_population

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.831 | 0.167 | 0.159 | 0.498 |
| B1 | 0.891 | 0.106 | 0.149 | 0.463 |
| dumb_cap_count | 0.762 | 0.235 | 0.163 | 0.508 |
| dumb_unparsed_count | 0.751 | 0.247 | 0.164 | 0.511 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.061 [0.028, 0.092] p=0.000 | -0.061 [-0.091, -0.033] p=0.000 | -0.010 [-0.020, 0.001] p=0.070 | -0.035 [-0.063, -0.006] p=0.010 |

## all_eight_parseable

384 prompts; base accuracy 0.799; automatic failures 0; capped prompts 4; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.828 | 0.169 | 0.158 | 0.495 |
| B1 | 0.899 | 0.099 | 0.147 | 0.455 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.036, 0.100] p=0.000 | -0.070 [-0.109, -0.038] p=0.000 | -0.011 [-0.021, -0.001] p=0.040 | -0.040 [-0.064, -0.013] p=0.000 |

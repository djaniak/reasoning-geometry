# Incremental abstention analysis — deepseek / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.750; automatic failures 7; capped prompts 107; unparsed traces 351.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.843 | 0.155 | 0.170 | 0.518 |
| B1 | 0.878 | 0.120 | 0.161 | 0.491 |
| dumb_cap_count | 0.761 | 0.237 | 0.174 | 0.532 |
| dumb_unparsed_count | 0.768 | 0.230 | 0.172 | 0.528 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.034 [0.005, 0.059] p=0.020 | -0.034 [-0.059, -0.011] p=0.000 | -0.009 [-0.017, -0.002] p=0.040 | -0.027 [-0.047, -0.009] p=0.000 |

## valid_plurality

493 prompts; base accuracy 0.761; automatic failures 0; capped prompts 100; unparsed traces 295.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.844 | 0.154 | 0.173 | 0.525 |
| B1 | 0.879 | 0.119 | 0.163 | 0.497 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.035 [0.010, 0.059] p=0.010 | -0.035 [-0.062, -0.009] p=0.000 | -0.009 [-0.016, -0.002] p=0.020 | -0.027 [-0.047, -0.007] p=0.000 |

## cap_free_valid_plurality

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.863 | 0.134 | 0.158 | 0.491 |
| B1 | 0.902 | 0.095 | 0.149 | 0.461 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.039 [0.007, 0.066] p=0.030 | -0.039 [-0.069, -0.010] p=0.020 | -0.009 [-0.018, 0.003] p=0.130 | -0.030 [-0.059, -0.001] p=0.030 |

## cap_free_full_population

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.863 | 0.134 | 0.158 | 0.491 |
| B1 | 0.902 | 0.095 | 0.149 | 0.461 |
| dumb_cap_count | 0.747 | 0.250 | 0.163 | 0.508 |
| dumb_unparsed_count | 0.740 | 0.258 | 0.164 | 0.511 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.039 [0.007, 0.066] p=0.030 | -0.039 [-0.069, -0.010] p=0.020 | -0.009 [-0.018, 0.003] p=0.130 | -0.030 [-0.059, -0.001] p=0.030 |

## all_eight_parseable

384 prompts; base accuracy 0.799; automatic failures 0; capped prompts 4; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.863 | 0.134 | 0.157 | 0.490 |
| B1 | 0.903 | 0.095 | 0.147 | 0.459 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.040 [0.002, 0.076] p=0.040 | -0.040 [-0.075, -0.011] p=0.010 | -0.010 [-0.020, 0.001] p=0.060 | -0.032 [-0.064, 0.002] p=0.070 |

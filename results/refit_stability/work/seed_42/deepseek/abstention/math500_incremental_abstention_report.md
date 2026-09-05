# Incremental abstention analysis — deepseek / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.750; automatic failures 7; capped prompts 107; unparsed traces 351.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.834 | 0.164 | 0.169 | 0.517 |
| B1 | 0.862 | 0.136 | 0.161 | 0.493 |
| dumb_cap_count | 0.766 | 0.232 | 0.173 | 0.529 |
| dumb_unparsed_count | 0.766 | 0.232 | 0.172 | 0.526 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.028 [0.003, 0.050] p=0.010 | -0.028 [-0.058, -0.007] p=0.010 | -0.008 [-0.015, -0.002] p=0.010 | -0.024 [-0.040, -0.007] p=0.000 |

## valid_plurality

493 prompts; base accuracy 0.761; automatic failures 0; capped prompts 100; unparsed traces 295.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.834 | 0.164 | 0.172 | 0.524 |
| B1 | 0.863 | 0.135 | 0.163 | 0.500 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.029 [0.005, 0.054] p=0.000 | -0.029 [-0.055, -0.007] p=0.020 | -0.009 [-0.015, -0.003] p=0.000 | -0.024 [-0.042, -0.007] p=0.000 |

## cap_free_valid_plurality

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.845 | 0.152 | 0.158 | 0.494 |
| B1 | 0.881 | 0.117 | 0.151 | 0.471 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.036 [0.009, 0.064] p=0.000 | -0.036 [-0.065, -0.008] p=0.000 | -0.007 [-0.016, 0.001] p=0.090 | -0.022 [-0.044, 0.004] p=0.090 |

## cap_free_full_population

393 prompts; base accuracy 0.796; automatic failures 0; capped prompts 0; unparsed traces 15.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.845 | 0.152 | 0.158 | 0.494 |
| B1 | 0.881 | 0.117 | 0.151 | 0.471 |
| dumb_cap_count | 0.761 | 0.237 | 0.163 | 0.508 |
| dumb_unparsed_count | 0.753 | 0.244 | 0.163 | 0.508 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.036 [0.009, 0.064] p=0.000 | -0.036 [-0.065, -0.008] p=0.000 | -0.007 [-0.016, 0.001] p=0.090 | -0.022 [-0.044, 0.004] p=0.090 |

## all_eight_parseable

384 prompts; base accuracy 0.799; automatic failures 0; capped prompts 4; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.841 | 0.156 | 0.157 | 0.491 |
| B1 | 0.880 | 0.117 | 0.149 | 0.467 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.039 [0.010, 0.068] p=0.000 | -0.039 [-0.067, -0.009] p=0.000 | -0.008 [-0.016, -0.001] p=0.010 | -0.024 [-0.043, -0.003] p=0.030 |

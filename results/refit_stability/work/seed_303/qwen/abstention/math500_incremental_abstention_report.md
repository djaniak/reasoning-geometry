# Incremental abstention analysis — qwen / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.620; automatic failures 2; capped prompts 108; unparsed traces 328.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.762 | 0.236 | 0.193 | 0.571 |
| B1 | 0.817 | 0.181 | 0.157 | 0.489 |
| dumb_cap_count | 0.656 | 0.342 | 0.221 | 0.634 |
| dumb_unparsed_count | 0.656 | 0.342 | 0.221 | 0.635 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.055 [0.019, 0.100] p=0.000 | -0.055 [-0.096, -0.014] p=0.000 | -0.035 [-0.050, -0.022] p=0.000 | -0.082 [-0.118, -0.045] p=0.000 |

## valid_plurality

498 prompts; base accuracy 0.622; automatic failures 0; capped prompts 106; unparsed traces 312.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.763 | 0.235 | 0.193 | 0.573 |
| B1 | 0.818 | 0.180 | 0.158 | 0.491 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.055 [0.020, 0.100] p=0.010 | -0.055 [-0.088, -0.021] p=0.010 | -0.035 [-0.049, -0.022] p=0.000 | -0.082 [-0.116, -0.042] p=0.000 |

## cap_free_valid_plurality

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.779 | 0.218 | 0.192 | 0.570 |
| B1 | 0.849 | 0.149 | 0.149 | 0.472 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.025, 0.118] p=0.010 | -0.070 [-0.117, -0.028] p=0.000 | -0.043 [-0.056, -0.026] p=0.000 | -0.098 [-0.146, -0.048] p=0.000 |

## cap_free_full_population

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.779 | 0.218 | 0.192 | 0.570 |
| B1 | 0.849 | 0.149 | 0.149 | 0.472 |
| dumb_cap_count | 0.656 | 0.341 | 0.215 | 0.621 |
| dumb_unparsed_count | 0.656 | 0.341 | 0.215 | 0.621 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.025, 0.118] p=0.010 | -0.070 [-0.117, -0.028] p=0.000 | -0.043 [-0.056, -0.026] p=0.000 | -0.098 [-0.146, -0.048] p=0.000 |

## all_eight_parseable

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 1; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.778 | 0.220 | 0.191 | 0.569 |
| B1 | 0.847 | 0.150 | 0.150 | 0.473 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.019, 0.118] p=0.030 | -0.070 [-0.117, -0.025] p=0.000 | -0.042 [-0.058, -0.025] p=0.000 | -0.096 [-0.143, -0.043] p=0.000 |

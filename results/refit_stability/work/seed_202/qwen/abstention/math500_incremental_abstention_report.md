# Incremental abstention analysis — qwen / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.620; automatic failures 2; capped prompts 108; unparsed traces 328.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.771 | 0.227 | 0.193 | 0.571 |
| B1 | 0.830 | 0.168 | 0.157 | 0.489 |
| dumb_cap_count | 0.664 | 0.334 | 0.220 | 0.632 |
| dumb_unparsed_count | 0.663 | 0.335 | 0.220 | 0.632 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.059 [0.030, 0.091] p=0.000 | -0.059 [-0.090, -0.029] p=0.000 | -0.036 [-0.049, -0.021] p=0.000 | -0.082 [-0.120, -0.043] p=0.000 |

## valid_plurality

498 prompts; base accuracy 0.622; automatic failures 0; capped prompts 106; unparsed traces 312.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.771 | 0.227 | 0.194 | 0.573 |
| B1 | 0.831 | 0.167 | 0.158 | 0.491 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.060 [0.029, 0.092] p=0.000 | -0.060 [-0.094, -0.023] p=0.000 | -0.036 [-0.050, -0.022] p=0.000 | -0.082 [-0.114, -0.038] p=0.000 |

## cap_free_valid_plurality

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.796 | 0.202 | 0.191 | 0.568 |
| B1 | 0.865 | 0.132 | 0.151 | 0.476 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.037, 0.105] p=0.000 | -0.070 [-0.107, -0.031] p=0.000 | -0.041 [-0.061, -0.024] p=0.000 | -0.092 [-0.138, -0.042] p=0.000 |

## cap_free_full_population

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.796 | 0.202 | 0.191 | 0.568 |
| B1 | 0.865 | 0.132 | 0.151 | 0.476 |
| dumb_cap_count | 0.664 | 0.333 | 0.214 | 0.619 |
| dumb_unparsed_count | 0.665 | 0.332 | 0.214 | 0.619 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.037, 0.105] p=0.000 | -0.070 [-0.107, -0.031] p=0.000 | -0.041 [-0.061, -0.024] p=0.000 | -0.092 [-0.138, -0.042] p=0.000 |

## all_eight_parseable

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 1; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.794 | 0.203 | 0.191 | 0.568 |
| B1 | 0.865 | 0.133 | 0.152 | 0.479 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.070 [0.033, 0.112] p=0.000 | -0.070 [-0.104, -0.038] p=0.000 | -0.040 [-0.059, -0.020] p=0.000 | -0.089 [-0.135, -0.045] p=0.000 |

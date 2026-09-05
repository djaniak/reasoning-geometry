# Incremental abstention analysis — qwen / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.620; automatic failures 2; capped prompts 108; unparsed traces 328.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.773 | 0.225 | 0.190 | 0.565 |
| B1 | 0.825 | 0.173 | 0.156 | 0.485 |
| dumb_cap_count | 0.679 | 0.319 | 0.219 | 0.631 |
| dumb_unparsed_count | 0.679 | 0.319 | 0.220 | 0.632 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.052 [0.025, 0.080] p=0.000 | -0.052 [-0.082, -0.019] p=0.000 | -0.034 [-0.047, -0.022] p=0.000 | -0.080 [-0.114, -0.049] p=0.000 |

## valid_plurality

498 prompts; base accuracy 0.622; automatic failures 0; capped prompts 106; unparsed traces 312.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.774 | 0.224 | 0.191 | 0.566 |
| B1 | 0.826 | 0.172 | 0.157 | 0.487 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.052 [0.018, 0.084] p=0.000 | -0.052 [-0.076, -0.020] p=0.000 | -0.034 [-0.049, -0.022] p=0.000 | -0.080 [-0.115, -0.045] p=0.000 |

## cap_free_valid_plurality

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.801 | 0.196 | 0.187 | 0.557 |
| B1 | 0.860 | 0.137 | 0.148 | 0.468 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.059 [0.025, 0.098] p=0.000 | -0.059 [-0.103, -0.020] p=0.000 | -0.039 [-0.055, -0.024] p=0.000 | -0.089 [-0.138, -0.042] p=0.000 |

## cap_free_full_population

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.801 | 0.196 | 0.187 | 0.557 |
| B1 | 0.860 | 0.137 | 0.148 | 0.468 |
| dumb_cap_count | 0.675 | 0.322 | 0.214 | 0.619 |
| dumb_unparsed_count | 0.675 | 0.322 | 0.214 | 0.619 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.059 [0.025, 0.098] p=0.000 | -0.059 [-0.103, -0.020] p=0.000 | -0.039 [-0.055, -0.024] p=0.000 | -0.089 [-0.138, -0.042] p=0.000 |

## all_eight_parseable

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 1; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.800 | 0.197 | 0.187 | 0.557 |
| B1 | 0.859 | 0.138 | 0.149 | 0.470 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.059 [0.021, 0.100] p=0.010 | -0.059 [-0.096, -0.018] p=0.000 | -0.038 [-0.057, -0.018] p=0.000 | -0.087 [-0.133, -0.041] p=0.000 |

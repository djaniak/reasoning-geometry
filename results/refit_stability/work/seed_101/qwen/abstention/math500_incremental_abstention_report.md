# Incremental abstention analysis — qwen / math500 (L21)

Brier and log loss are OOF probabilistic-forecast scores after logistic calibration; they do not isolate calibration from discrimination/resolution.

`B0` = length + global entropy + global log-probability + vote agreement. `B1` adds tail RMD. All deltas below use paired prompt bootstrap.

## full_population

500 prompts; base accuracy 0.620; automatic failures 2; capped prompts 108; unparsed traces 328.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.768 | 0.230 | 0.192 | 0.569 |
| B1 | 0.820 | 0.178 | 0.159 | 0.492 |
| dumb_cap_count | 0.642 | 0.356 | 0.220 | 0.631 |
| dumb_unparsed_count | 0.642 | 0.356 | 0.220 | 0.632 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.052 [0.025, 0.088] p=0.000 | -0.052 [-0.084, -0.016] p=0.010 | -0.033 [-0.047, -0.020] p=0.000 | -0.077 [-0.103, -0.038] p=0.000 |

## valid_plurality

498 prompts; base accuracy 0.622; automatic failures 0; capped prompts 106; unparsed traces 312.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.769 | 0.229 | 0.193 | 0.571 |
| B1 | 0.821 | 0.177 | 0.160 | 0.494 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.052 [0.013, 0.087] p=0.000 | -0.052 [-0.075, -0.018] p=0.010 | -0.033 [-0.049, -0.019] p=0.000 | -0.077 [-0.115, -0.040] p=0.000 |

## cap_free_valid_plurality

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.791 | 0.206 | 0.189 | 0.564 |
| B1 | 0.855 | 0.143 | 0.152 | 0.476 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.064 [0.023, 0.104] p=0.000 | -0.064 [-0.099, -0.019] p=0.000 | -0.037 [-0.054, -0.020] p=0.000 | -0.088 [-0.127, -0.046] p=0.000 |

## cap_free_full_population

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 0; unparsed traces 1.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.791 | 0.206 | 0.189 | 0.564 |
| B1 | 0.855 | 0.143 | 0.152 | 0.476 |
| dumb_cap_count | 0.638 | 0.359 | 0.215 | 0.622 |
| dumb_unparsed_count | 0.638 | 0.359 | 0.215 | 0.622 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.064 [0.023, 0.104] p=0.000 | -0.064 [-0.099, -0.019] p=0.000 | -0.037 [-0.054, -0.020] p=0.000 | -0.088 [-0.127, -0.046] p=0.000 |

## all_eight_parseable

392 prompts; base accuracy 0.691; automatic failures 0; capped prompts 1; unparsed traces 0.

| model | AUACC (higher) | conventional AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.788 | 0.209 | 0.189 | 0.563 |
| B1 | 0.854 | 0.144 | 0.152 | 0.477 |

### Paired increments

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | 0.065 [0.028, 0.105] p=0.000 | -0.065 [-0.105, -0.026] p=0.000 | -0.037 [-0.054, -0.020] p=0.000 | -0.086 [-0.125, -0.040] p=0.000 |

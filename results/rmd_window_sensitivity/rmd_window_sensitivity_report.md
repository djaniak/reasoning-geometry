# RMD tail-window sensitivity

Post-hoc sensitivity analysis. `rmd_tail_q20` remains frozen; this run does not select a detector.

AURC deltas are `B0 + detector` minus `B0`; lower is better.

| model | detector | delta AURC [95% CI] |
|---|---|---:|
| qwen | `rmd_tail_q10` | -0.0559 [-0.0889, -0.0228] |
| qwen | `rmd_tail_q20` | -0.0520 [-0.0832, -0.0218] |
| qwen | `rmd_tail_q50` | -0.0304 [-0.0596, -0.0012] |
| qwen | `rmd_full` | -0.0178 [-0.0435, +0.0105] |
| qwen | `rmd_high_entropy_q20` | -0.0148 [-0.0357, +0.0051] |
| qwen | `rmd_random_q20` | -0.0184 [-0.0440, +0.0096] |
| deepseek | `rmd_tail_q10` | -0.0342 [-0.0599, -0.0088] |
| deepseek | `rmd_tail_q20` | -0.0284 [-0.0531, -0.0060] |
| deepseek | `rmd_tail_q50` | -0.0376 [-0.0612, -0.0159] |
| deepseek | `rmd_full` | -0.0287 [-0.0534, -0.0055] |
| deepseek | `rmd_high_entropy_q20` | -0.0052 [-0.0258, +0.0144] |
| deepseek | `rmd_random_q20` | -0.0295 [-0.0541, -0.0060] |
| deepseek_llama | `rmd_tail_q10` | -0.0572 [-0.0900, -0.0258] |
| deepseek_llama | `rmd_tail_q20` | -0.0469 [-0.0773, -0.0178] |
| deepseek_llama | `rmd_tail_q50` | -0.0477 [-0.0766, -0.0194] |
| deepseek_llama | `rmd_full` | -0.0445 [-0.0739, -0.0151] |
| deepseek_llama | `rmd_high_entropy_q20` | -0.0274 [-0.0517, -0.0027] |
| deepseek_llama | `rmd_random_q20` | -0.0444 [-0.0741, -0.0141] |

**Cutoff-robustness rule:** q10, q20, and q50 each improve AURC over B0 with a 95% interval below zero on every checkpoint

**Verdict:** We fixed 20% as a simple localized window; sensitivity analysis shows that the result is not specific to this exact cutoff.

ATRMD, high-entropy-q20, and random-q20 are contextual controls. They do not justify selecting 20% as optimal.

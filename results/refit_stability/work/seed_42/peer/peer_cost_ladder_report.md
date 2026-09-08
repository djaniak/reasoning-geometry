# Peer baseline and cost ladder

Every rung scored against what it costs at decision time. `B1` adds `rmd_tail_q20` to `B0` and buys **no** extra generations -- it reads hidden states of the eight the target already produced -- so it is free at the margin and no peer rung is cost matched to it.

Two readouts of the same purchased generations:

* **`graded`** -- the fraction of the peer's samples that were *correct*. Needs the gold answer, so it is **not computable at decision time**. It is an upper bound on what any peer method could deliver, not a baseline.
* **`agree`** -- the fraction of the peer's samples that returned the *target's own answer*. No gold needed; this is the peer ensemble a reviewer could actually deploy.

AURC, lower is better. A negative `B1 - rung` favours `B1`.

Headline population: `full_population`. Ladder sizes: 1, 2, 4, 8 siblings per peer.

## 1. Floors and the two free rungs

AURC does not bottom out at zero. Where a rung approaches the oracle floor, a delta can no longer separate "this is better" from "there was nothing left to remove".

| Model | n | Base acc. | Oracle AURC | `B0` | `B1` | B0 headroom |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 500 | 0.620 | 0.0836 | 0.2251 | 0.1730 | 0.1414 |
| deepseek | 500 | 0.750 | 0.0342 | 0.1643 | 0.1359 | 0.1301 |
| deepseek_llama | 500 | 0.634 | 0.0771 | 0.2480 | 0.2010 | 0.1709 |

## 2. The ladder in cost order

`Extra` is the margin over `B0` per prompt. `AURC` is the mean over 25 independent re-draws of which siblings were bought (sub-sampled rungs only; the full-cache rungs are deterministic). `removed` is the fraction of `B0`'s headroom the rung takes out -- at 1.00 the rung is on the oracle floor.

| Model | Rung | Kind | Extra calls | Extra tokens | AURC | across-draw range | removed |
|---|---|---|---:|---:|---:|---|---:|
| qwen | `B0` | none | 0 | 0 | 0.2251 | -- | 0.00 |
| qwen | `B1` | none | 0 | 0 | 0.1730 | -- | 0.37 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1477 | [0.1401, 0.1546] | 0.55 |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.1441 | [0.1256, 0.1579] | 0.57 |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.1488 | [0.1387, 0.1621] | 0.54 |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1213 | [0.1126, 0.1287] | 0.73 |
| qwen | `B0_agree_both_m1` | agree | 2 | 6150 | 0.1384 | [0.1266, 0.1475] | 0.61 |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1383 | [0.1338, 0.1440] | 0.61 |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.1418 | [0.1274, 0.1525] | 0.59 |
| qwen | `B0_graded_both_m1` | graded | 2 | 6150 | 0.1211 | [0.1143, 0.1299] | 0.74 |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.1431 | [0.1373, 0.1487] | 0.58 |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1188 | [0.1132, 0.1262] | 0.75 |
| qwen | `B0_agree_both_m2` | agree | 4 | 12301 | 0.1336 | [0.1265, 0.1497] | 0.65 |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1343 | [0.1305, 0.1404] | 0.64 |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.1392 | [0.1312, 0.1475] | 0.61 |
| qwen | `B0_graded_both_m2` | graded | 4 | 12301 | 0.1200 | [0.1138, 0.1245] | 0.74 |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.1389 | [0.1362, 0.1431] | 0.61 |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1149 | [0.1122, 0.1199] | 0.78 |
| qwen | `B0_agree_both_m4` | agree | 8 | 24601 | 0.1303 | [0.1243, 0.1372] | 0.67 |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1317 | -- | 0.66 |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.1373 | -- | 0.62 |
| qwen | `B0_graded_both_m4` | graded | 8 | 24601 | 0.1175 | [0.1149, 0.1199] | 0.76 |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.1378 | -- | 0.62 |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1132 | -- | 0.79 |
| qwen | `B0_agree_both_m8` | agree | 16 | 49202 | 0.1322 | -- | 0.66 |
| qwen | `B0_graded_both_m8` | graded | 16 | 49202 | 0.1170 | -- | 0.76 |
| deepseek | `B0` | none | 0 | 0 | 0.1643 | -- | 0.00 |
| deepseek | `B1` | none | 0 | 0 | 0.1359 | -- | 0.22 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1688 | [0.1608, 0.1756] | -0.03 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1255 | [0.1136, 0.1433] | 0.30 |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.0739 | [0.0657, 0.0835] | 0.70 |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.0761 | [0.0641, 0.0946] | 0.68 |
| deepseek | `B0_agree_both_m1` | agree | 2 | 3606 | 0.1228 | [0.1069, 0.1408] | 0.32 |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1681 | [0.1627, 0.1742] | -0.03 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1252 | [0.1180, 0.1308] | 0.30 |
| deepseek | `B0_graded_both_m1` | graded | 2 | 3606 | 0.0534 | [0.0479, 0.0606] | 0.85 |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.0639 | [0.0612, 0.0678] | 0.77 |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.0642 | [0.0542, 0.0735] | 0.77 |
| deepseek | `B0_agree_both_m2` | agree | 4 | 7211 | 0.1179 | [0.1095, 0.1287] | 0.36 |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1682 | [0.1649, 0.1717] | -0.03 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1266 | [0.1164, 0.1355] | 0.29 |
| deepseek | `B0_graded_both_m2` | graded | 4 | 7211 | 0.0472 | [0.0427, 0.0525] | 0.90 |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.0600 | [0.0577, 0.0633] | 0.80 |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.0537 | [0.0501, 0.0620] | 0.85 |
| deepseek | `B0_agree_both_m4` | agree | 8 | 14422 | 0.1195 | [0.1102, 0.1329] | 0.34 |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1676 | -- | -0.03 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1259 | -- | 0.30 |
| deepseek | `B0_graded_both_m4` | graded | 8 | 14422 | 0.0438 | [0.0410, 0.0480] | 0.93 |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.0583 | -- | 0.81 |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.0494 | -- | 0.88 |
| deepseek | `B0_agree_both_m8` | agree | 16 | 28844 | 0.1190 | -- | 0.35 |
| deepseek | `B0_graded_both_m8` | graded | 16 | 28844 | 0.0414 | -- | 0.95 |
| deepseek_llama | `B0` | none | 0 | 0 | 0.2480 | -- | 0.00 |
| deepseek_llama | `B1` | none | 0 | 0 | 0.2010 | -- | 0.27 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.2022 | [0.1941, 0.2164] | 0.27 |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1472 | [0.1351, 0.1594] | 0.59 |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1797 | [0.1644, 0.1950] | 0.40 |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.1980 | [0.1819, 0.2179] | 0.29 |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 3657 | 0.1419 | [0.1324, 0.1592] | 0.62 |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.2121 | [0.2032, 0.2351] | 0.21 |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1474 | [0.1382, 0.1562] | 0.59 |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 3657 | 0.1760 | [0.1631, 0.1911] | 0.42 |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1794 | [0.1675, 0.1995] | 0.40 |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.1923 | [0.1789, 0.2022] | 0.33 |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 7314 | 0.1453 | [0.1354, 0.1556] | 0.60 |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.2111 | [0.1978, 0.2247] | 0.22 |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1443 | [0.1374, 0.1534] | 0.61 |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 7314 | 0.1756 | [0.1616, 0.1838] | 0.42 |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1736 | [0.1659, 0.1827] | 0.44 |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.1884 | [0.1800, 0.2020] | 0.35 |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 14629 | 0.1429 | [0.1351, 0.1504] | 0.61 |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.2082 | -- | 0.23 |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1439 | -- | 0.61 |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 14629 | 0.1725 | [0.1634, 0.1794] | 0.44 |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1731 | -- | 0.44 |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.1875 | -- | 0.35 |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 29258 | 0.1441 | -- | 0.61 |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 29258 | 0.1715 | -- | 0.45 |

## 3. Does a bought rung beat the free one?

`B1 - rung` on AURC. Negative favours `B1`, the rung that costs no extra generations. The interval is the frozen prompt bootstrap on the median draw; `sign stable` reports whether every draw agreed on the direction, which is the separate question of whether the verdict depends on which siblings you happened to buy.

| Model | Rung | Kind | Extra calls | B1 - rung | 95% CI | excludes 0 | sign stable | winner |
|---|---|---|---:|---:|---|:--:|:--:|---|
| qwen | `B0` | none | 0 | -0.0520 | [-0.0816, -0.0186] | yes | -- | B1 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 0.0250 | [-0.0030, +0.0545] | no | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 0.0250 | [-0.0059, +0.0603] | no | yes | tie |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0238 | [-0.0034, +0.0510] | no | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 0.0520 | [+0.0272, +0.0799] | yes | yes | peer |
| qwen | `B0_agree_both_m1` | agree | 2 | 0.0340 | [+0.0042, +0.0589] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 0.0346 | [+0.0068, +0.0626] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 0.0312 | [-0.0016, +0.0674] | no | yes | tie |
| qwen | `B0_graded_both_m1` | graded | 2 | 0.0527 | [+0.0297, +0.0799] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0295 | [+0.0042, +0.0588] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 0.0545 | [+0.0289, +0.0836] | yes | yes | peer |
| qwen | `B0_agree_both_m2` | agree | 4 | 0.0407 | [+0.0088, +0.0703] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 0.0393 | [+0.0117, +0.0671] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 0.0337 | [+0.0018, +0.0667] | yes | yes | peer |
| qwen | `B0_graded_both_m2` | graded | 4 | 0.0538 | [+0.0275, +0.0806] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0343 | [+0.0063, +0.0648] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 0.0585 | [+0.0373, +0.0846] | yes | yes | peer |
| qwen | `B0_agree_both_m4` | agree | 8 | 0.0423 | [+0.0111, +0.0698] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 0.0414 | [+0.0145, +0.0682] | yes | -- | peer |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 0.0358 | [+0.0048, +0.0710] | yes | -- | peer |
| qwen | `B0_graded_both_m4` | graded | 8 | 0.0557 | [+0.0302, +0.0868] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0353 | [+0.0063, +0.0674] | yes | -- | peer |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 0.0598 | [+0.0373, +0.0868] | yes | -- | peer |
| qwen | `B0_agree_both_m8` | agree | 16 | 0.0409 | [+0.0098, +0.0729] | yes | -- | peer |
| qwen | `B0_graded_both_m8` | graded | 16 | 0.0561 | [+0.0288, +0.0862] | yes | -- | peer |
| deepseek | `B0` | none | 0 | -0.0284 | [-0.0579, -0.0068] | yes | -- | B1 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | -0.0341 | [-0.0576, -0.0067] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 0.0129 | [-0.0203, +0.0465] | no | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0625 | [+0.0387, +0.0879] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 0.0589 | [+0.0249, +0.0960] | yes | yes | peer |
| deepseek | `B0_agree_both_m1` | agree | 2 | 0.0139 | [-0.0240, +0.0462] | no | no | tie |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | -0.0322 | [-0.0560, -0.0047] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 0.0100 | [-0.0324, +0.0475] | no | yes | tie |
| deepseek | `B0_graded_both_m1` | graded | 2 | 0.0820 | [+0.0517, +0.1133] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0721 | [+0.0476, +0.0967] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 0.0725 | [+0.0440, +0.1042] | yes | yes | peer |
| deepseek | `B0_agree_both_m2` | agree | 4 | 0.0180 | [-0.0164, +0.0499] | no | yes | tie |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | -0.0325 | [-0.0513, -0.0054] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 0.0097 | [-0.0302, +0.0487] | no | yes | tie |
| deepseek | `B0_graded_both_m2` | graded | 4 | 0.0895 | [+0.0616, +0.1210] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0759 | [+0.0526, +0.1005] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 0.0838 | [+0.0568, +0.1129] | yes | yes | peer |
| deepseek | `B0_agree_both_m4` | agree | 8 | 0.0167 | [-0.0243, +0.0550] | no | yes | tie |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | -0.0316 | [-0.0554, -0.0043] | yes | -- | B1 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 0.0101 | [-0.0325, +0.0510] | no | -- | tie |
| deepseek | `B0_graded_both_m4` | graded | 8 | 0.0924 | [+0.0654, +0.1218] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0776 | [+0.0542, +0.1046] | yes | -- | peer |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 0.0865 | [+0.0587, +0.1157] | yes | -- | peer |
| deepseek | `B0_agree_both_m8` | agree | 16 | 0.0169 | [-0.0241, +0.0545] | no | -- | tie |
| deepseek | `B0_graded_both_m8` | graded | 16 | 0.0946 | [+0.0677, +0.1241] | yes | -- | peer |
| deepseek_llama | `B0` | none | 0 | -0.0469 | [-0.0751, -0.0188] | yes | -- | B1 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | 0.0014 | [-0.0310, +0.0356] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 0.0544 | [+0.0249, +0.0841] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 0.0207 | [-0.0180, +0.0589] | no | yes | tie |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 0.0049 | [-0.0315, +0.0459] | no | no | tie |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 0.0603 | [+0.0292, +0.0916] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | -0.0093 | [-0.0451, +0.0241] | no | yes | tie |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 0.0532 | [+0.0151, +0.0876] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 0.0250 | [-0.0098, +0.0610] | no | yes | tie |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 0.0215 | [-0.0166, +0.0610] | no | yes | tie |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 0.0095 | [-0.0282, +0.0468] | no | no | tie |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 0.0549 | [+0.0151, +0.0921] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | -0.0092 | [-0.0431, +0.0245] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 0.0572 | [+0.0233, +0.0910] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 0.0250 | [-0.0105, +0.0594] | no | yes | tie |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 0.0274 | [-0.0112, +0.0573] | no | yes | tie |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 0.0137 | [-0.0205, +0.0527] | no | no | tie |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 0.0591 | [+0.0212, +0.0972] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | -0.0072 | [-0.0402, +0.0293] | no | -- | tie |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 0.0571 | [+0.0201, +0.0891] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 0.0286 | [-0.0040, +0.0618] | no | yes | tie |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 0.0280 | [-0.0066, +0.0571] | no | -- | tie |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 0.0135 | [-0.0240, +0.0513] | no | -- | tie |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 0.0569 | [+0.0181, +0.0968] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 0.0295 | [-0.0091, +0.0638] | no | -- | tie |

## 4. The one-extra-generation question

The cheapest thing on the ladder that is not already paid for is one sample from one peer. If it beats `B1`, a cheap peer wins outright. If it does not, `B1` wins at strictly lower cost. Only the `agree` rows are a fair answer to that question -- the `graded` rows need the gold answer and are reported as the bound they are.

| Model | Rung | Kind | Deployable | B1 - rung | 95% CI | sign stable | winner |
|---|---|---|:--:|---:|---|:--:|---|
| qwen | `B0_agree_deepseek_llama_m1` | agree | yes | 0.0250 | [-0.0030, +0.0545] | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | yes | 0.0250 | [-0.0059, +0.0603] | yes | tie |
| qwen | `B0_graded_deepseek_llama_m1` | graded | no | 0.0238 | [-0.0034, +0.0510] | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | no | 0.0520 | [+0.0272, +0.0799] | yes | peer |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | yes | -0.0341 | [-0.0576, -0.0067] | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | yes | 0.0129 | [-0.0203, +0.0465] | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | no | 0.0625 | [+0.0387, +0.0879] | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | no | 0.0589 | [+0.0249, +0.0960] | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | yes | 0.0014 | [-0.0310, +0.0356] | no | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | yes | 0.0544 | [+0.0249, +0.0841] | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | no | 0.0207 | [-0.0180, +0.0589] | yes | tie |
| deepseek_llama | `B0_graded_qwen_m1` | graded | no | 0.0049 | [-0.0315, +0.0459] | no | tie |

## 5. Saturated comparisons

Rungs that have removed at least 90% of `B0`'s headroom. On these the delta against `B1` is not evidence about which readout is better; there is almost nothing left for either to remove.

* **qwen** -- headroom 0.1414 above a floor of 0.0836; no rung saturated
* **deepseek** -- headroom 0.1301 above a floor of 0.0342; 3 rung(s) saturated: `B0_graded_both_m2`, `B0_graded_both_m4`, `B0_graded_both_m8`
* **deepseek_llama** -- headroom 0.1709 above a floor of 0.0771; no rung saturated

## What the cost model does not charge for

`B1` needs the target's hidden states retained and a Mahalanobis readout fitted over them. That is real work and real memory; it is not a generation, and it does not scale with the number of models you are willing to run. Charging it as tokens would be the mirror image of the error this rung exists to correct, so it is named here and left uncosted rather than silently folded in.

The `graded` rungs are not charged for the gold answer they consume, because it cannot be bought at decision time at any price. They bound the peer family from above and are reported for that reason only.

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
| qwen | 500 | 0.620 | 0.0836 | 0.2271 | 0.1678 | 0.1435 |
| deepseek | 500 | 0.750 | 0.0342 | 0.1545 | 0.1202 | 0.1203 |
| deepseek_llama | 500 | 0.634 | 0.0771 | 0.2411 | 0.2117 | 0.1640 |

## 2. The ladder in cost order

`Extra` is the margin over `B0` per prompt. `AURC` is the mean over 25 independent re-draws of which siblings were bought (sub-sampled rungs only; the full-cache rungs are deterministic). `removed` is the fraction of `B0`'s headroom the rung takes out -- at 1.00 the rung is on the oracle floor.

| Model | Rung | Kind | Extra calls | Extra tokens | AURC | across-draw range | removed |
|---|---|---|---:|---:|---:|---|---:|
| qwen | `B0` | none | 0 | 0 | 0.2271 | -- | 0.00 |
| qwen | `B1` | none | 0 | 0 | 0.1678 | -- | 0.41 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1463 | [0.1399, 0.1544] | 0.56 |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.1446 | [0.1252, 0.1596] | 0.58 |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.1501 | [0.1443, 0.1579] | 0.54 |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1256 | [0.1200, 0.1335] | 0.71 |
| qwen | `B0_agree_both_m1` | agree | 2 | 6150 | 0.1340 | [0.1227, 0.1451] | 0.65 |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1383 | [0.1295, 0.1487] | 0.62 |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.1443 | [0.1284, 0.1618] | 0.58 |
| qwen | `B0_graded_both_m1` | graded | 2 | 6150 | 0.1247 | [0.1196, 0.1335] | 0.71 |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.1440 | [0.1385, 0.1502] | 0.58 |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1196 | [0.1136, 0.1245] | 0.75 |
| qwen | `B0_agree_both_m2` | agree | 4 | 12301 | 0.1290 | [0.1186, 0.1391] | 0.68 |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1318 | [0.1253, 0.1402] | 0.66 |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.1422 | [0.1331, 0.1579] | 0.59 |
| qwen | `B0_graded_both_m2` | graded | 4 | 12301 | 0.1202 | [0.1155, 0.1247] | 0.74 |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.1406 | [0.1376, 0.1435] | 0.60 |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1167 | [0.1141, 0.1228] | 0.77 |
| qwen | `B0_agree_both_m4` | agree | 8 | 24601 | 0.1286 | [0.1231, 0.1352] | 0.69 |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1273 | -- | 0.70 |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.1419 | -- | 0.59 |
| qwen | `B0_graded_both_m4` | graded | 8 | 24601 | 0.1184 | [0.1162, 0.1215] | 0.76 |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.1392 | -- | 0.61 |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1148 | -- | 0.78 |
| qwen | `B0_agree_both_m8` | agree | 16 | 49202 | 0.1286 | -- | 0.69 |
| qwen | `B0_graded_both_m8` | graded | 16 | 49202 | 0.1173 | -- | 0.77 |
| deepseek | `B0` | none | 0 | 0 | 0.1545 | -- | 0.00 |
| deepseek | `B1` | none | 0 | 0 | 0.1202 | -- | 0.28 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | 3049 | 0.1600 | [0.1523, 0.1669] | -0.05 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1258 | [0.1170, 0.1343] | 0.24 |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 3049 | 0.0745 | [0.0679, 0.0831] | 0.67 |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.0778 | [0.0622, 0.0890] | 0.64 |
| deepseek | `B0_agree_both_m1` | agree | 2 | 3606 | 0.1177 | [0.1022, 0.1344] | 0.31 |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | 6099 | 0.1603 | [0.1523, 0.1660] | -0.05 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1235 | [0.1154, 0.1289] | 0.26 |
| deepseek | `B0_graded_both_m1` | graded | 2 | 3606 | 0.0535 | [0.0469, 0.0635] | 0.84 |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 6099 | 0.0652 | [0.0612, 0.0713] | 0.74 |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.0673 | [0.0536, 0.0777] | 0.72 |
| deepseek | `B0_agree_both_m2` | agree | 4 | 7211 | 0.1168 | [0.1071, 0.1275] | 0.31 |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | 12197 | 0.1611 | [0.1563, 0.1664] | -0.05 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1238 | [0.1139, 0.1307] | 0.26 |
| deepseek | `B0_graded_both_m2` | graded | 4 | 7211 | 0.0477 | [0.0435, 0.0523] | 0.89 |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 12197 | 0.0604 | [0.0578, 0.0632] | 0.78 |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.0548 | [0.0512, 0.0676] | 0.83 |
| deepseek | `B0_agree_both_m4` | agree | 8 | 14422 | 0.1171 | [0.1058, 0.1286] | 0.31 |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | 24394 | 0.1593 | -- | -0.04 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1231 | -- | 0.26 |
| deepseek | `B0_graded_both_m4` | graded | 8 | 14422 | 0.0436 | [0.0410, 0.0485] | 0.92 |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 24394 | 0.0582 | -- | 0.80 |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.0496 | -- | 0.87 |
| deepseek | `B0_agree_both_m8` | agree | 16 | 28844 | 0.1131 | -- | 0.34 |
| deepseek | `B0_graded_both_m8` | graded | 16 | 28844 | 0.0411 | -- | 0.94 |
| deepseek_llama | `B0` | none | 0 | 0 | 0.2411 | -- | 0.00 |
| deepseek_llama | `B1` | none | 0 | 0 | 0.2117 | -- | 0.18 |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | 3101 | 0.2053 | [0.1834, 0.2280] | 0.22 |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 556 | 0.1433 | [0.1327, 0.1604] | 0.60 |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 3101 | 0.1725 | [0.1595, 0.1811] | 0.42 |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 556 | 0.1945 | [0.1822, 0.2020] | 0.28 |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 3657 | 0.1359 | [0.1264, 0.1449] | 0.64 |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | 6202 | 0.2161 | [0.1972, 0.2349] | 0.15 |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 1112 | 0.1388 | [0.1324, 0.1475] | 0.62 |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 3657 | 0.1724 | [0.1613, 0.1803] | 0.42 |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 6202 | 0.1668 | [0.1537, 0.1768] | 0.45 |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 1112 | 0.1904 | [0.1828, 0.1992] | 0.31 |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 7314 | 0.1339 | [0.1259, 0.1437] | 0.65 |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | 12404 | 0.2241 | [0.2118, 0.2345] | 0.10 |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 2225 | 0.1380 | [0.1322, 0.1447] | 0.63 |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 7314 | 0.1688 | [0.1597, 0.1763] | 0.44 |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 12404 | 0.1650 | [0.1574, 0.1741] | 0.46 |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 2225 | 0.1878 | [0.1793, 0.1993] | 0.33 |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 14629 | 0.1352 | [0.1282, 0.1411] | 0.65 |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | 24808 | 0.2178 | -- | 0.14 |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 4450 | 0.1342 | -- | 0.65 |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 14629 | 0.1680 | [0.1630, 0.1749] | 0.45 |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 24808 | 0.1655 | -- | 0.46 |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 4450 | 0.1860 | -- | 0.34 |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 29258 | 0.1339 | -- | 0.65 |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 29258 | 0.1692 | -- | 0.44 |

## 3. Does a bought rung beat the free one?

`B1 - rung` on AURC. Negative favours `B1`, the rung that costs no extra generations. The interval is the frozen prompt bootstrap on the median draw; `sign stable` reports whether every draw agreed on the direction, which is the separate question of whether the verdict depends on which siblings you happened to buy.

| Model | Rung | Kind | Extra calls | B1 - rung | 95% CI | excludes 0 | sign stable | winner |
|---|---|---|---:|---:|---|:--:|:--:|---|
| qwen | `B0` | none | 0 | -0.0593 | [-0.0901, -0.0293] | yes | -- | B1 |
| qwen | `B0_agree_deepseek_llama_m1` | agree | 1 | 0.0215 | [-0.0087, +0.0460] | no | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | 1 | 0.0231 | [-0.0032, +0.0466] | no | yes | tie |
| qwen | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0184 | [-0.0073, +0.0460] | no | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | 1 | 0.0421 | [+0.0184, +0.0637] | yes | yes | peer |
| qwen | `B0_agree_both_m1` | agree | 2 | 0.0355 | [+0.0059, +0.0653] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m2` | agree | 2 | 0.0291 | [+0.0037, +0.0546] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m2` | agree | 2 | 0.0228 | [-0.0108, +0.0566] | no | yes | tie |
| qwen | `B0_graded_both_m1` | graded | 2 | 0.0435 | [+0.0193, +0.0718] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0241 | [-0.0017, +0.0476] | no | yes | tie |
| qwen | `B0_graded_deepseek_m2` | graded | 2 | 0.0492 | [+0.0272, +0.0687] | yes | yes | peer |
| qwen | `B0_agree_both_m2` | agree | 4 | 0.0393 | [+0.0112, +0.0630] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m4` | agree | 4 | 0.0361 | [+0.0096, +0.0618] | yes | yes | peer |
| qwen | `B0_agree_deepseek_m4` | agree | 4 | 0.0263 | [-0.0122, +0.0544] | no | yes | tie |
| qwen | `B0_graded_both_m2` | graded | 4 | 0.0479 | [+0.0186, +0.0744] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0272 | [+0.0010, +0.0500] | yes | yes | peer |
| qwen | `B0_graded_deepseek_m4` | graded | 4 | 0.0515 | [+0.0301, +0.0742] | yes | yes | peer |
| qwen | `B0_agree_both_m4` | agree | 8 | 0.0399 | [+0.0127, +0.0616] | yes | yes | peer |
| qwen | `B0_agree_deepseek_llama_m8` | agree | 8 | 0.0405 | [+0.0150, +0.0659] | yes | -- | peer |
| qwen | `B0_agree_deepseek_m8` | agree | 8 | 0.0259 | [-0.0099, +0.0569] | no | -- | tie |
| qwen | `B0_graded_both_m4` | graded | 8 | 0.0497 | [+0.0219, +0.0775] | yes | yes | peer |
| qwen | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0286 | [+0.0019, +0.0520] | yes | -- | peer |
| qwen | `B0_graded_deepseek_m8` | graded | 8 | 0.0530 | [+0.0313, +0.0733] | yes | -- | peer |
| qwen | `B0_agree_both_m8` | agree | 16 | 0.0392 | [+0.0121, +0.0635] | yes | -- | peer |
| qwen | `B0_graded_both_m8` | graded | 16 | 0.0505 | [+0.0251, +0.0769] | yes | -- | peer |
| deepseek | `B0` | none | 0 | -0.0343 | [-0.0592, -0.0115] | yes | -- | B1 |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | 1 | -0.0390 | [-0.0667, -0.0161] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | 1 | -0.0057 | [-0.0354, +0.0254] | no | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | 1 | 0.0474 | [+0.0268, +0.0701] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | 1 | 0.0417 | [+0.0167, +0.0761] | yes | yes | peer |
| deepseek | `B0_agree_both_m1` | agree | 2 | 0.0033 | [-0.0333, +0.0283] | no | no | tie |
| deepseek | `B0_agree_deepseek_llama_m2` | agree | 2 | -0.0408 | [-0.0682, -0.0183] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m2` | agree | 2 | -0.0030 | [-0.0360, +0.0299] | no | no | tie |
| deepseek | `B0_graded_both_m1` | graded | 2 | 0.0668 | [+0.0429, +0.0935] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m2` | graded | 2 | 0.0554 | [+0.0312, +0.0777] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m2` | graded | 2 | 0.0539 | [+0.0294, +0.0802] | yes | yes | peer |
| deepseek | `B0_agree_both_m2` | agree | 4 | 0.0044 | [-0.0277, +0.0340] | no | no | tie |
| deepseek | `B0_agree_deepseek_llama_m4` | agree | 4 | -0.0409 | [-0.0671, -0.0178] | yes | yes | B1 |
| deepseek | `B0_agree_qwen_m4` | agree | 4 | -0.0042 | [-0.0415, +0.0331] | no | no | tie |
| deepseek | `B0_graded_both_m2` | graded | 4 | 0.0731 | [+0.0509, +0.1009] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m4` | graded | 4 | 0.0596 | [+0.0355, +0.0808] | yes | yes | peer |
| deepseek | `B0_graded_qwen_m4` | graded | 4 | 0.0673 | [+0.0477, +0.0929] | yes | yes | peer |
| deepseek | `B0_agree_both_m4` | agree | 8 | 0.0019 | [-0.0323, +0.0354] | no | no | tie |
| deepseek | `B0_agree_deepseek_llama_m8` | agree | 8 | -0.0391 | [-0.0656, -0.0144] | yes | -- | B1 |
| deepseek | `B0_agree_qwen_m8` | agree | 8 | -0.0029 | [-0.0368, +0.0325] | no | -- | tie |
| deepseek | `B0_graded_both_m4` | graded | 8 | 0.0771 | [+0.0563, +0.1046] | yes | yes | peer |
| deepseek | `B0_graded_deepseek_llama_m8` | graded | 8 | 0.0620 | [+0.0374, +0.0850] | yes | -- | peer |
| deepseek | `B0_graded_qwen_m8` | graded | 8 | 0.0706 | [+0.0506, +0.0970] | yes | -- | peer |
| deepseek | `B0_agree_both_m8` | agree | 16 | 0.0071 | [-0.0206, +0.0349] | no | -- | tie |
| deepseek | `B0_graded_both_m8` | graded | 16 | 0.0791 | [+0.0578, +0.1064] | yes | -- | peer |
| deepseek_llama | `B0` | none | 0 | -0.0294 | [-0.0566, +0.0007] | no | -- | tie |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | 1 | 0.0063 | [-0.0335, +0.0450] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | 1 | 0.0694 | [+0.0378, +0.1011] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | 1 | 0.0390 | [+0.0010, +0.0716] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m1` | graded | 1 | 0.0165 | [-0.0183, +0.0512] | no | yes | tie |
| deepseek_llama | `B0_agree_both_m1` | agree | 2 | 0.0769 | [+0.0423, +0.1083] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m2` | agree | 2 | -0.0037 | [-0.0482, +0.0370] | no | no | tie |
| deepseek_llama | `B0_agree_qwen_m2` | agree | 2 | 0.0744 | [+0.0378, +0.1055] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m1` | graded | 2 | 0.0382 | [+0.0005, +0.0778] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m2` | graded | 2 | 0.0452 | [+0.0102, +0.0747] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m2` | graded | 2 | 0.0223 | [-0.0108, +0.0586] | no | yes | tie |
| deepseek_llama | `B0_agree_both_m2` | agree | 4 | 0.0777 | [+0.0410, +0.1097] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m4` | agree | 4 | -0.0147 | [-0.0594, +0.0320] | no | yes | tie |
| deepseek_llama | `B0_agree_qwen_m4` | agree | 4 | 0.0736 | [+0.0329, +0.1067] | yes | yes | peer |
| deepseek_llama | `B0_graded_both_m2` | graded | 4 | 0.0427 | [+0.0075, +0.0782] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m4` | graded | 4 | 0.0466 | [+0.0145, +0.0756] | yes | yes | peer |
| deepseek_llama | `B0_graded_qwen_m4` | graded | 4 | 0.0243 | [-0.0096, +0.0622] | no | yes | tie |
| deepseek_llama | `B0_agree_both_m4` | agree | 8 | 0.0758 | [+0.0392, +0.1096] | yes | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m8` | agree | 8 | -0.0061 | [-0.0473, +0.0415] | no | -- | tie |
| deepseek_llama | `B0_agree_qwen_m8` | agree | 8 | 0.0775 | [+0.0412, +0.1075] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m4` | graded | 8 | 0.0437 | [+0.0125, +0.0830] | yes | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m8` | graded | 8 | 0.0462 | [+0.0139, +0.0771] | yes | -- | peer |
| deepseek_llama | `B0_graded_qwen_m8` | graded | 8 | 0.0257 | [-0.0091, +0.0678] | no | -- | tie |
| deepseek_llama | `B0_agree_both_m8` | agree | 16 | 0.0779 | [+0.0407, +0.1092] | yes | -- | peer |
| deepseek_llama | `B0_graded_both_m8` | graded | 16 | 0.0426 | [+0.0087, +0.0803] | yes | -- | peer |

## 4. The one-extra-generation question

The cheapest thing on the ladder that is not already paid for is one sample from one peer. If it beats `B1`, a cheap peer wins outright. If it does not, `B1` wins at strictly lower cost. Only the `agree` rows are a fair answer to that question -- the `graded` rows need the gold answer and are reported as the bound they are.

| Model | Rung | Kind | Deployable | B1 - rung | 95% CI | sign stable | winner |
|---|---|---|:--:|---:|---|:--:|---|
| qwen | `B0_agree_deepseek_llama_m1` | agree | yes | 0.0215 | [-0.0087, +0.0460] | yes | tie |
| qwen | `B0_agree_deepseek_m1` | agree | yes | 0.0231 | [-0.0032, +0.0466] | yes | tie |
| qwen | `B0_graded_deepseek_llama_m1` | graded | no | 0.0184 | [-0.0073, +0.0460] | yes | tie |
| qwen | `B0_graded_deepseek_m1` | graded | no | 0.0421 | [+0.0184, +0.0637] | yes | peer |
| deepseek | `B0_agree_deepseek_llama_m1` | agree | yes | -0.0390 | [-0.0667, -0.0161] | yes | B1 |
| deepseek | `B0_agree_qwen_m1` | agree | yes | -0.0057 | [-0.0354, +0.0254] | no | tie |
| deepseek | `B0_graded_deepseek_llama_m1` | graded | no | 0.0474 | [+0.0268, +0.0701] | yes | peer |
| deepseek | `B0_graded_qwen_m1` | graded | no | 0.0417 | [+0.0167, +0.0761] | yes | peer |
| deepseek_llama | `B0_agree_deepseek_m1` | agree | yes | 0.0063 | [-0.0335, +0.0450] | no | tie |
| deepseek_llama | `B0_agree_qwen_m1` | agree | yes | 0.0694 | [+0.0378, +0.1011] | yes | peer |
| deepseek_llama | `B0_graded_deepseek_m1` | graded | no | 0.0390 | [+0.0010, +0.0716] | yes | peer |
| deepseek_llama | `B0_graded_qwen_m1` | graded | no | 0.0165 | [-0.0183, +0.0512] | yes | tie |

## 5. Saturated comparisons

Rungs that have removed at least 90% of `B0`'s headroom. On these the delta against `B1` is not evidence about which readout is better; there is almost nothing left for either to remove.

* **qwen** -- headroom 0.1435 above a floor of 0.0836; no rung saturated
* **deepseek** -- headroom 0.1203 above a floor of 0.0342; 2 rung(s) saturated: `B0_graded_both_m4`, `B0_graded_both_m8`
* **deepseek_llama** -- headroom 0.1640 above a floor of 0.0771; no rung saturated

## What the cost model does not charge for

`B1` needs the target's hidden states retained and a Mahalanobis readout fitted over them. That is real work and real memory; it is not a generation, and it does not scale with the number of models you are willing to run. Charging it as tokens would be the mirror image of the error this rung exists to correct, so it is named here and left uncosted rather than silently folded in.

The `graded` rungs are not charged for the gold answer they consume, because it cannot be bought at decision time at any price. They bound the peer family from above and are reported for that reason only.

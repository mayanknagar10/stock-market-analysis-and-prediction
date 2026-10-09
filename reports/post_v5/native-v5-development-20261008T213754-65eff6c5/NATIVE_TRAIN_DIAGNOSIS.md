# Native V5 train/validation diagnostic addendum

Train errors replay base-fitting rows with the released blend/calibration learned later in development; they are optimistic in-sample diagnostics, not OOS evidence.

Exact original snapshots were trimmed before feature/target transformations. Reconstructed fit matrices and labels must match saved analogue arrays. Validation replay must match stored development metrics. No new training/calibration, final data, model selection or architecture change occurs.

| Family | Sessions | Base-fit N | Fit features | Train MAE | Validation MAE | MAE degradation | Train ROC-AUC | Validation ROC-AUC | Train Brier | Validation Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| india_equity | 1 | 2196 | 110 | 0.008745 | 0.007495 | -14.3% | 0.857 | 0.565 | 0.2031 | 0.2527 |
| india_equity | 5 | 2172 | 119 | 0.017879 | 0.019256 | +7.7% | 0.888 | 0.520 | 0.1721 | 0.2619 |
| india_equity | 10 | 2142 | 119 | 0.021748 | 0.029634 | +36.3% | 0.924 | 0.575 | 0.1366 | 0.3073 |
| india_equity | 20 | 2082 | 110 | 0.027743 | 0.053051 | +91.2% | 0.063 | 0.555 | 0.3382 | 0.3219 |
| us_equity | 1 | 2235 | 64 | 0.011154 | 0.009620 | -13.8% | 0.151 | 0.521 | 0.2746 | 0.2455 |
| us_equity | 5 | 2211 | 119 | 0.022290 | 0.024260 | +8.8% | 0.895 | 0.476 | 0.1756 | 0.2342 |
| us_equity | 10 | 2181 | 119 | 0.028998 | 0.029475 | +1.6% | 0.922 | 0.552 | 0.1529 | 0.2288 |
| us_equity | 20 | 2121 | 127 | 0.035632 | 0.039779 | +11.6% | 0.946 | 0.540 | 0.1310 | 0.1932 |

Depth3/120tree heads and many features fit only three stocks per family. The table measures in-sample optimism and later development degradation; it cannot identify a single causal failure mechanism. Correlated/overlapping labels reduce effective support below row counts.

The already-published failed V5 final gate remains historical context (1/0/2/7/3); its rows and report were not opened or rescored. The all100-member registered development study, rather than the consumed final result, governs the provisional next architecture. This addendum leaves original V5 and the first failure-analysis report unchanged.

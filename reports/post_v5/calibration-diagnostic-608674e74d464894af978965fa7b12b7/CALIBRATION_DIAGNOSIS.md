# Released V5 calibration diagnostic addendum

These are in-sample base-row diagnostics under released later-development weights/calibration. Negative Platt slope reverses ranking; it may reflect unstable calibration-period ranking and is not proof of a causal mechanism. No calibrator, probability or threshold is changed.

| Family | Sessions | Fit N | Raw fit ROC-AUC | Issued fit ROC-AUC | Platt coefficient | Ranking reversed |
|---|---:|---:|---:|---:|---:|---|
| india_equity | 1 | 2196 | 0.857 | 0.857 | Unavailable | False |
| india_equity | 5 | 2172 | 0.888 | 0.888 | 0.9310 | False |
| india_equity | 10 | 2142 | 0.924 | 0.924 | Unavailable | False |
| india_equity | 20 | 2082 | 0.937 | 0.063 | -0.2430 | True |
| us_equity | 1 | 2235 | 0.849 | 0.151 | -0.5163 | True |
| us_equity | 5 | 2211 | 0.895 | 0.895 | Unavailable | False |
| us_equity | 10 | 2181 | 0.922 | 0.922 | Unavailable | False |
| us_equity | 20 | 2121 | 0.946 | 0.946 | Unavailable | False |

Raw base-fit ROC-AUC measures fitted discrimination, not generalization. The native train/validation addendum reports issued scores, so accepted negative-slope calibrations can show fit AUC below 0.5 despite strong raw in-sample ranking. The new research protocol requires stable held-out discrimination before calibration. No consumed-period rows or outcomes enter this diagnostic or any model choice.

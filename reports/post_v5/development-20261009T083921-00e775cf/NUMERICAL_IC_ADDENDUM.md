# Additive numerical IC audit

Audit numerical-ic-audit-v1 of development-20261009T083921-00e775cf. No model refits, no original report/CSV/JSON edits, no consumed final inputs. Raw scores remain immutable. Read this addendum with the five original reports.

Reference-only predictors can be identical within a date but differ at machine precision after a matrix product. Ranking these tiny differences is a numerical artifact. Within-origin Spearman IC is therefore undefined when prediction or outcome standard deviation is <=1e-12; zero is not substituted for undefined signal. At least8 finite rows are required. The threshold addresses floating-point validity, not performance selection.

| horizon | model | raw_date_rank_ic | tolerance_corrected_date_rank_ic | meaningful_dates | near_constant_dates | date_blocks_with_defined_ic | block_ic_se |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | commodity_only | -0.007522 | — | 0 | 114 | 0 | — |
| 1 | fx_only | -0.007522 | — | 0 | 114 | 0 | — |
| 1 | global_only | 0.007522 | — | 0 | 114 | 0 | — |
| 1 | market_only | 0.007522 | — | 0 | 114 | 0 | — |
| 1 | shallow_boost | -0.001557 | -0.001557 | 83 | 31 | 83 | 0.016437 |
| 5 | commodity_only | -0.116597 | — | 0 | 110 | 0 | — |
| 5 | global_only | 0.116597 | — | 0 | 110 | 0 | — |
| 5 | reduced_stable | -0.025747 | -0.025747 | 62 | 48 | 13 | 0.055922 |
| 10 | commodity_only | -0.124120 | — | 0 | 105 | 0 | — |
| 10 | fx_only | -0.124120 | — | 0 | 105 | 0 | — |
| 10 | global_only | -0.124120 | — | 0 | 105 | 0 | — |
| 10 | reduced_stable | -0.090470 | -0.090470 | 62 | 43 | 7 | 0.042293 |
| 10 | shallow_boost | -0.086396 | -0.086396 | 18 | 87 | 3 | 0.079836 |
| 20 | commodity_only | -0.018806 | — | 0 | 95 | 0 | — |
| 20 | fx_only | 0.018806 | — | 0 | 95 | 0 | — |
| 20 | shallow_boost | -0.133094 | -0.133094 | 24 | 71 | 3 | 0.071612 |

Use meaningful-date counts and date-block dispersion with corrected IC; raw near-constant rankings must not support architecture or promotion. These descriptive standard errors are not adjusted hypothesis-test confidence intervals. Linear sector-excess and decomposition are exactly equivalent in this registered implementation (same alpha-target/head and supporting cohort); adding component heads did not create extra predictive information. Their support differs from full-return models and is paired explicitly in PAIRED/PAIRED_STRATA.

Raw uncalibrated classifier aggregates (fold and full stock/industry/regime/trend strata retained separately):

| horizon | n | dates | roc_auc | balanced_accuracy | brier |
| --- | --- | --- | --- | --- | --- |
| 1 | 10774 | 248 | 0.513172 | 0.500169 | 0.238101 |
| 5 | 10386 | 244 | 0.557271 | 0.521002 | 0.253390 |
| 10 | 9901 | 239 | 0.506522 | 0.498850 | 0.276786 |
| 20 | 8931 | 229 | 0.565554 | 0.500610 | 0.306602 |

Aggregates can mix date-level class-rate shifts; four held-out folds and stable date-level ranking remain necessary before calibration. CLASSIFIER_STRATA and PAIRED_STRATA provide the supplementary breakdown. Adequate paired strata here means >=200 rows and >=5 horizon-session date blocks, a disclosed descriptive reporting threshold with no fitting or tuned selection. Sector/regime claims require these paired cohorts; unsupported strata remain explicit. Tiny MAE improvements within1% are pre-registered ties, not successor signals. All native/model fitting configurations and all original negative results remain unchanged.

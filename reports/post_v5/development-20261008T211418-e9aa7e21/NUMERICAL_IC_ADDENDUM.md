# Additive numerical IC audit

Audit numerical-ic-audit-v1 of development-20261008T211418-e9aa7e21. No model refits, no original report/CSV/JSON edits, no consumed final inputs. Raw scores remain immutable. Read this addendum with the five original reports.

Reference-only predictors can be identical within a date but differ at machine precision after a matrix product. Ranking these tiny differences is a numerical artifact. Within-origin Spearman IC is therefore undefined when prediction or outcome standard deviation is <=1e-12; zero is not substituted for undefined signal. At least8 finite rows are required. The threshold addresses floating-point validity, not performance selection.

| horizon | model | raw_date_rank_ic | tolerance_corrected_date_rank_ic | meaningful_dates | near_constant_dates | date_blocks_with_defined_ic | block_ic_se |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | commodity_only | -0.030089 | — | 0 | 228 | 0 | — |
| 1 | fx_only | 0.016184 | — | 0 | 228 | 0 | — |
| 1 | global_only | -0.032712 | — | 0 | 228 | 0 | — |
| 1 | market_only | 0.025189 | — | 0 | 228 | 0 | — |
| 1 | shallow_boost | 0.006435 | 0.006435 | 171 | 57 | 171 | 0.011090 |
| 5 | commodity_only | 0.055371 | — | 0 | 224 | 0 | — |
| 5 | global_only | -0.064726 | — | 0 | 224 | 0 | — |
| 5 | market_only | 0.039968 | — | 0 | 224 | 0 | — |
| 5 | reduced_stable | -0.004285 | — | 0 | 224 | 0 | — |
| 10 | commodity_only | 0.004159 | — | 0 | 219 | 0 | — |
| 10 | fx_only | 0.081016 | — | 0 | 219 | 0 | — |
| 10 | global_only | 0.024233 | — | 0 | 219 | 0 | — |
| 10 | market_only | 0.124120 | — | 0 | 219 | 0 | — |
| 10 | reduced_stable | -0.083363 | -0.083698 | 62 | 157 | 7 | 0.041462 |
| 10 | shallow_boost | -0.093990 | -0.093990 | 12 | 207 | 3 | 0.060789 |
| 20 | commodity_only | 0.018806 | — | 0 | 209 | 0 | — |
| 20 | fx_only | 0.013737 | — | 0 | 209 | 0 | — |
| 20 | global_only | 0.041563 | — | 0 | 209 | 0 | — |
| 20 | reduced_stable | -0.088007 | -0.118915 | 115 | 94 | 6 | 0.042873 |
| 20 | shallow_boost | -0.073738 | -0.073738 | 63 | 146 | 6 | 0.047867 |

Use meaningful-date counts and date-block dispersion with corrected IC; raw near-constant rankings must not support architecture or promotion. These descriptive standard errors are not adjusted hypothesis-test confidence intervals. Linear sector-excess and decomposition are exactly equivalent in this registered implementation (same alpha-target/head and supporting cohort); adding component heads did not create extra predictive information. Their support differs from full-return models and is paired explicitly in PAIRED/PAIRED_STRATA.

Raw uncalibrated classifier aggregates (fold and full stock/industry/regime/trend strata retained separately):

| horizon | n | dates | roc_auc | balanced_accuracy | brier |
| --- | --- | --- | --- | --- | --- |
| 1 | 21829 | 248 | 0.521826 | 0.502208 | 0.238509 |
| 5 | 21445 | 244 | 0.496796 | 0.481104 | 0.252838 |
| 10 | 20965 | 239 | 0.466339 | 0.486507 | 0.261624 |
| 20 | 20003 | 229 | 0.451145 | 0.501418 | 0.271662 |

Aggregates can mix date-level class-rate shifts; four held-out folds and stable date-level ranking remain necessary before calibration. CLASSIFIER_STRATA and PAIRED_STRATA provide the supplementary breakdown. Adequate paired strata here means >=200 rows and >=5 horizon-session date blocks, a disclosed descriptive reporting threshold with no fitting or tuned selection. Sector/regime claims require these paired cohorts; unsupported strata remain explicit. Tiny MAE improvements within1% are pre-registered ties, not successor signals. All native/model fitting configurations and all original negative results remain unchanged.

# Corrected research decision and support addendum

Study development-20261009T083921-00e775cf replaces INVALID_FOR_MODEL_SELECTION predecessor development-20261008T211418-e9aa7e21 for development decisions. Only corrected-study evidence below supports architecture; predecessor metrics remain preserved audit history and were not used to choose any model, target, feature or threshold. All models/parameters/folds/protocol are unchanged. No model refits followed scoring, no calibration or successor final fit, no V5 writes.

The source boolean mask now uses positional values instead of naive-to-aware Series alignment. Independent SOURCE_MASK_AUDIT verifies39992 scored stock-origin/horizon rows: no invalid200-session past or intervening forward bars, exact calendar returns and conservative information-time cutoffs; all observations/labels pre2025-10-01. Weekend special sessions retain their true maturity session while frozen cutoffs deliberately defer information time to next UTC day. First audit wrongly equated information date and session date; its registration/review are retained, corrected v2 passes with no output edits.

All100 members were attempted;97 have historical eligible rows and17industries appear overall. 95 of97 captured preboundary stock frames contain a zero/invalid session2025-03-18. The fixed200-session guard removes most later2025 origins. Aggregate97-stock counts do not imply representative four-quarter support. In particular quarter3 has1stock/1industry; quarter4 has at most2stocks. Missing study members/industries remain explicit in ELIGIBILITY and FOLD_SUPPORT, not replaced or filled.

| horizon | fold | train_rows | validation_rows | validation_stocks | validation_industries | validation_dates | dates_with_at_least8_stocks | mean_fixed_universe_eligible_fraction | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 60122 | 5629 | 95 | 17 | 62 | 62 | 0.907903 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 1 | 2 | 65741 | 5000 | 96 | 17 | 62 | 52 | 0.806452 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 1 | 3 | 70835 | 61 | 1 | 1 | 61 | 0 | 0.010000 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 1 | 4 | 70896 | 84 | 2 | 2 | 63 | 0 | 0.013333 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 5 | 1 | 59718 | 5629 | 95 | 17 | 62 | 62 | 0.907903 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 5 | 2 | 65297 | 4620 | 96 | 17 | 62 | 48 | 0.745161 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 5 | 3 | 70387 | 61 | 1 | 1 | 61 | 0 | 0.010000 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 5 | 4 | 70448 | 76 | 2 | 2 | 59 | 0 | 0.012881 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 10 | 1 | 59213 | 5629 | 95 | 17 | 62 | 62 | 0.907903 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 10 | 2 | 64742 | 4145 | 96 | 17 | 62 | 43 | 0.668548 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 10 | 3 | 69827 | 61 | 1 | 1 | 61 | 0 | 0.010000 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 10 | 4 | 69888 | 66 | 2 | 2 | 54 | 0 | 0.012222 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 20 | 1 | 58251 | 5629 | 95 | 17 | 62 | 62 | 0.907903 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 20 | 2 | 63678 | 3195 | 96 | 17 | 62 | 33 | 0.515323 | OBSERVED_SUPPORT_COUNTS_ONLY |
| 20 | 3 | 68752 | 61 | 1 | 1 | 61 | 0 | 0.010000 | SPARSE_CROSS_SECTIONAL_SUPPORT |
| 20 | 4 | 68813 | 46 | 2 | 2 | 44 | 0 | 0.010455 | SPARSE_CROSS_SECTIONAL_SUPPORT |

Corrected return-space results:

| horizon | model | n | mae | rmse | dates | nonoverlap_blocks |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | decomposition | 9630 | 0.014940 | 0.021190 | 248 | 248 |
| 1 | elastic_net | 10774 | 0.014815 | 0.021024 | 248 | 248 |
| 1 | reduced_stable | 10774 | 0.014761 | 0.020983 | 248 | 248 |
| 1 | ridge | 10774 | 0.016288 | 0.022357 | 248 | 248 |
| 1 | shallow_boost | 10774 | 0.014947 | 0.021156 | 248 | 248 |
| 1 | zero | 10774 | 0.014780 | 0.020994 | 248 | 248 |
| 5 | decomposition | 9282 | 0.036184 | 0.049635 | 244 | 49 |
| 5 | elastic_net | 10386 | 0.035493 | 0.048765 | 244 | 49 |
| 5 | reduced_stable | 10386 | 0.035667 | 0.048919 | 244 | 49 |
| 5 | ridge | 10386 | 0.050208 | 0.064619 | 244 | 49 |
| 5 | shallow_boost | 10386 | 0.035474 | 0.048677 | 244 | 49 |
| 5 | zero | 10386 | 0.034626 | 0.047657 | 244 | 49 |
| 10 | decomposition | 8847 | 0.052769 | 0.068930 | 239 | 24 |
| 10 | elastic_net | 9901 | 0.056882 | 0.073043 | 239 | 24 |
| 10 | reduced_stable | 9901 | 0.054345 | 0.070532 | 239 | 24 |
| 10 | ridge | 9901 | 0.079967 | 0.099379 | 239 | 24 |
| 10 | shallow_boost | 9901 | 0.051487 | 0.066744 | 239 | 24 |
| 10 | zero | 9901 | 0.047734 | 0.062655 | 239 | 24 |
| 20 | decomposition | 7977 | 0.077945 | 0.099997 | 229 | 12 |
| 20 | elastic_net | 8931 | 0.084020 | 0.105124 | 229 | 12 |
| 20 | reduced_stable | 8931 | 0.088081 | 0.110552 | 229 | 12 |
| 20 | ridge | 8931 | 0.082113 | 0.104133 | 229 | 12 |
| 20 | shallow_boost | 8931 | 0.076879 | 0.097751 | 229 | 12 |
| 20 | zero | 8931 | 0.065607 | 0.085083 | 229 | 12 |

Zero is strongest simple at every horizon. No complex model passes positive aggregate improvement +3of4folds +60%stocks (screening configurations count=0). H1 reduced-stable advantage is0.12622982470668065%, a pre-registered1% tie;3improving folds but only53.60824742268041%stocks. Sparse last-half support would independently bar broad model claims. Sector-excess/decomposition retain9630/9282/8847/7977 paired-supported rows versus10774/10386/9901/8931 full rows. These independent linear-head versions are equivalent here; decomposition adds no information. All negative results remain visible.

Classifier aggregates are raw uncalibrated cost-positive scores:

| horizon | n | dates | roc_auc | balanced_accuracy | brier |
| --- | --- | --- | --- | --- | --- |
| 1 | 10774 | 248 | 0.513172 | 0.500169 | 0.238101 |
| 5 | 10386 | 244 | 0.557271 | 0.521002 | 0.253390 |
| 10 | 9901 | 239 | 0.506522 | 0.498850 | 0.276786 |
| 20 | 8931 | 229 | 0.565554 | 0.500610 | 0.306602 |

H5 aggregate AUC0.5572714747061387 and balanced accuracy0.5210021940400381 narrowly clear scalar discrimination thresholds, but that alone does not establish stable held-out discrimination. H5 has ZERO meaningful within-origin stock-ranking dates; its probability is effectively shared within each date, aggregate scores mix date-level class-rate changes, and the last two quarters lack broad stock/industry support. Brier0.25339034328970506 is worse than coin-flip.25. H20 AUC clears.55 but balanced accuracy remains.50061 and no meaningful within-date ranking. No downstream calibrator was fit or justified by this evidence.

Tolerance-corrected rank and classifier diagnostics (distinct from return MAE):

| horizon | model | tolerance_corrected_date_rank_ic | meaningful_dates | insufficient_dates | date_blocks_with_defined_ic | block_ic_se |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | positive_direction | 0.036373 | 114 | 134 | 114 | 0.011975 |
| 1 | rank | 0.011692 | 114 | 134 | 114 | 0.014265 |
| 1 | reduced_stable | 0.030377 | 114 | 134 | 114 | 0.011737 |
| 5 | positive_direction | — | 0 | 134 | 0 | — |
| 5 | rank | 0.014440 | 110 | 134 | 22 | 0.029802 |
| 5 | reduced_stable | -0.025747 | 62 | 134 | 13 | 0.055922 |
| 10 | positive_direction | -0.072504 | 10 | 134 | 1 | — |
| 10 | rank | 0.017181 | 105 | 134 | 11 | 0.056978 |
| 10 | reduced_stable | -0.090470 | 62 | 134 | 7 | 0.042293 |
| 20 | positive_direction | — | 0 | 134 | 0 | — |
| 20 | rank | -0.038777 | 95 | 134 | 5 | 0.055805 |
| 20 | reduced_stable | -0.150097 | 95 | 134 | 5 | 0.057532 |

NUMERICAL_IC_AUDIT preserves raw IC and flags beside corrected diagnostics:16model/horizon artifacts, undefined when prediction/outcome std<=1e-12 or fewer than8finite stocks.134dates per horizon lack8stocks. Do not interpret undefined as zero or successful ranking. Stock/industry/regime/trend paired strata preserve exact supported cohorts. Descriptive block SE is not a calibrated confidence interval; H20 has only12raw nonoverlapping date blocks and much less representative cross-sectional support.

Architecture decision: NO SUCCESSOR CANDIDATE FIT. Retain immutable capture, source-health and positional session guards plus strongest simple research comparator. Stop complexity expansion; pooled stable/sector/decomposition/rank branches are diagnostic only. Defer calibration and NLP/news. Restore evidence through new immutable prospective captures and mature labels under the registered calendar, not by weakening guards, filling source gaps, using invalid predecessor metrics, consumed V5 final outcomes or changing thresholds. Keep abstention and research-only status. Source revision/current-membership survivorship bias and historical release uncertainty remain. Root-owned native V5 train/calibration addenda are separate preboundary diagnosis/history, not successor architecture selectors.

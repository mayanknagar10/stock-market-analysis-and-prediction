# Development baseline comparison

Experiment development-20261009T083921-00e775cf; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261009T083921-00e775cf/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

| horizon | model | n | mae | rmse | direction_accuracy | ic | dates | nonoverlap_blocks | date_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | elastic_net | 10774 | 0.014815 | 0.021024 | 0.495452 | 0.042368 | 248 | 248 | -0.015481 |
| 1 | historical_mean | 10774 | 0.014899 | 0.021107 | 0.459161 | -0.001135 | 248 | 248 | — |
| 1 | market_only | 10774 | 0.014946 | 0.021143 | 0.458604 | -0.007723 | 248 | 248 | 0.007522 |
| 1 | market_sector | 10774 | 0.014952 | 0.021152 | 0.465565 | -0.012535 | 248 | 248 | -0.025034 |
| 1 | momentum20 | 10774 | 0.015225 | 0.021540 | 0.490533 | -0.046765 | 248 | 248 | 0.001052 |
| 1 | reversion5 | 10774 | 0.016322 | 0.022794 | 0.507518 | 0.033464 | 248 | 248 | 0.022832 |
| 1 | ridge | 10774 | 0.016288 | 0.022357 | 0.478467 | 0.021744 | 248 | 248 | 0.015178 |
| 1 | shallow_boost | 10774 | 0.014947 | 0.021156 | 0.454706 | -0.030693 | 248 | 248 | -0.001557 |
| 1 | zero | 10774 | 0.014780 | 0.020994 | 0.540839 | — | 248 | 248 | — |
| 5 | elastic_net | 10386 | 0.035493 | 0.048765 | 0.530714 | 0.156886 | 244 | 49 | -0.003531 |
| 5 | historical_mean | 10386 | 0.035619 | 0.048781 | 0.436357 | 0.033568 | 244 | 49 | — |
| 5 | market_only | 10386 | 0.036140 | 0.049320 | 0.436260 | 0.054805 | 244 | 49 | — |
| 5 | market_sector | 10386 | 0.036086 | 0.049291 | 0.438282 | 0.062527 | 244 | 49 | -0.020865 |
| 5 | momentum20 | 10386 | 0.039180 | 0.053549 | 0.491142 | -0.098710 | 244 | 49 | -0.012498 |
| 5 | reversion5 | 10386 | 0.047908 | 0.062975 | 0.519738 | 0.127168 | 244 | 49 | 0.041847 |
| 5 | ridge | 10386 | 0.050208 | 0.064619 | 0.523012 | 0.098071 | 244 | 49 | 0.023626 |
| 5 | shallow_boost | 10386 | 0.035474 | 0.048677 | 0.450607 | 0.032910 | 244 | 49 | — |
| 5 | zero | 10386 | 0.034626 | 0.047657 | 0.563643 | — | 244 | 49 | — |
| 10 | elastic_net | 9901 | 0.056882 | 0.073043 | 0.475306 | 0.143388 | 239 | 24 | 0.006172 |
| 10 | historical_mean | 9901 | 0.050887 | 0.066177 | 0.385012 | 0.007356 | 239 | 24 | — |
| 10 | market_only | 9901 | 0.052222 | 0.067742 | 0.385012 | 0.085241 | 239 | 24 | — |
| 10 | market_sector | 9901 | 0.051947 | 0.067448 | 0.385113 | 0.091780 | 239 | 24 | -0.005416 |
| 10 | momentum20 | 9901 | 0.057653 | 0.076629 | 0.509645 | -0.089797 | 239 | 24 | -0.017107 |
| 10 | reversion5 | 9901 | 0.082957 | 0.109141 | 0.505504 | 0.106805 | 239 | 24 | 0.034373 |
| 10 | ridge | 9901 | 0.079967 | 0.099379 | 0.471367 | 0.085186 | 239 | 24 | 0.026218 |
| 10 | shallow_boost | 9901 | 0.051487 | 0.066744 | 0.388446 | 0.060903 | 239 | 24 | -0.086396 |
| 10 | zero | 9901 | 0.047734 | 0.062655 | 0.614988 | — | 239 | 24 | — |
| 20 | elastic_net | 8931 | 0.084020 | 0.105124 | 0.386519 | 0.137345 | 229 | 12 | 0.006364 |
| 20 | historical_mean | 8931 | 0.074179 | 0.094884 | 0.352256 | 0.030261 | 229 | 12 | — |
| 20 | market_only | 8931 | 0.078534 | 0.099509 | 0.352256 | 0.104378 | 229 | 12 | — |
| 20 | market_sector | 8931 | 0.077562 | 0.098497 | 0.352256 | 0.098656 | 229 | 12 | 0.035707 |
| 20 | momentum20 | 8931 | 0.091744 | 0.120315 | 0.508454 | -0.109497 | 229 | 12 | 0.011470 |
| 20 | reversion5 | 8931 | 0.154206 | 0.205667 | 0.513828 | 0.104487 | 229 | 12 | 0.022521 |
| 20 | ridge | 8931 | 0.082113 | 0.104133 | 0.403762 | 0.063738 | 229 | 12 | 0.026338 |
| 20 | shallow_boost | 8931 | 0.076879 | 0.097751 | 0.353488 | 0.094852 | 229 | 12 | -0.133094 |
| 20 | zero | 8931 | 0.065607 | 0.085083 | 0.647744 | — | 229 | 12 | — |

Errors are cumulative-log-return MAE/RMSE; direction is positive versus nonpositive. Historical mean is fixed once per purged fold. Momentum = ret20 * horizon/20; reversion = -ret5 * horizon/5. Ridge alpha10, elastic-net alpha.001/l1.5, shallow boosting 50iterations/depth2/minleaf100/lr.03/seed42/early_stopping=False. All median/scale fitting is train-only; no search.

| horizon | strongest_simple | simple_mae |
| --- | --- | --- |
| 1 | zero | 0.014780 |
| 5 | zero | 0.034626 |
| 10 | zero | 0.047734 |
| 20 | zero | 0.065607 |

Paired cohort comparisons (positive improvement favors model):

| horizon | model | baseline | n | mae_improvement | stocks_improved_fraction | folds_improved | nonoverlap_blocks | block_delta_mean | block_delta_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | elastic_net | zero | 10774 | -0.002326 | 0.381443 | 1 | 248 | 0.000024 | 0.000039 |
| 1 | elastic_net | historical_mean | 10774 | 0.005662 | 0.814433 | 4 | 248 | -0.000059 | 0.000036 |
| 1 | elastic_net | momentum20 | 10774 | 0.026974 | 0.907216 | 4 | 248 | -0.000338 | 0.000090 |
| 1 | elastic_net | reversion5 | 10774 | 0.092359 | 0.958763 | 3 | 248 | -0.000735 | 0.000185 |
| 1 | market_only | zero | 10774 | -0.011250 | 0.082474 | 1 | 248 | 0.000097 | 0.000049 |
| 1 | market_only | historical_mean | 10774 | -0.003190 | 0.134021 | 1 | 248 | 0.000014 | 0.000027 |
| 1 | market_only | momentum20 | 10774 | 0.018311 | 0.814433 | 4 | 248 | -0.000264 | 0.000097 |
| 1 | market_only | reversion5 | 10774 | 0.084279 | 0.948454 | 3 | 248 | -0.000662 | 0.000185 |
| 1 | market_sector | zero | 10774 | -0.011624 | 0.082474 | 0 | 248 | 0.000107 | 0.000049 |
| 1 | market_sector | historical_mean | 10774 | -0.003561 | 0.216495 | 1 | 248 | 0.000025 | 0.000032 |
| 1 | market_sector | momentum20 | 10774 | 0.017948 | 0.804124 | 4 | 248 | -0.000254 | 0.000097 |
| 1 | market_sector | reversion5 | 10774 | 0.083940 | 0.958763 | 3 | 248 | -0.000652 | 0.000187 |
| 1 | ridge | zero | 10774 | -0.102009 | 0.000000 | 0 | 248 | 0.002778 | 0.000344 |
| 1 | ridge | historical_mean | 10774 | -0.093226 | 0.010309 | 0 | 248 | 0.002695 | 0.000321 |
| 1 | ridge | momentum20 | 10774 | -0.069796 | 0.123711 | 0 | 248 | 0.002417 | 0.000353 |
| 1 | ridge | reversion5 | 10774 | 0.002093 | 0.443299 | 2 | 248 | 0.002019 | 0.000370 |
| 1 | shallow_boost | zero | 10774 | -0.011264 | 0.103093 | 1 | 248 | 0.000071 | 0.000075 |
| 1 | shallow_boost | historical_mean | 10774 | -0.003204 | 0.340206 | 1 | 248 | -0.000012 | 0.000059 |
| 1 | shallow_boost | momentum20 | 10774 | 0.018297 | 0.835052 | 4 | 248 | -0.000290 | 0.000104 |
| 1 | shallow_boost | reversion5 | 10774 | 0.084266 | 0.948454 | 3 | 248 | -0.000688 | 0.000195 |
| 5 | elastic_net | zero | 10386 | -0.025023 | 0.453608 | 1 | 49 | 0.001992 | 0.000760 |
| 5 | elastic_net | historical_mean | 10386 | 0.003550 | 0.597938 | 1 | 49 | 0.001496 | 0.000657 |
| 5 | elastic_net | momentum20 | 10386 | 0.094100 | 0.845361 | 3 | 49 | -0.000994 | 0.001106 |
| 5 | elastic_net | reversion5 | 10386 | 0.259145 | 0.989691 | 4 | 49 | -0.005524 | 0.001615 |
| 5 | market_only | zero | 10386 | -0.043707 | 0.061856 | 1 | 49 | 0.000657 | 0.000412 |
| 5 | market_only | historical_mean | 10386 | -0.014613 | 0.092784 | 1 | 49 | 0.000160 | 0.000171 |
| 5 | market_only | momentum20 | 10386 | 0.077587 | 0.865979 | 4 | 49 | -0.002329 | 0.000739 |
| 5 | market_only | reversion5 | 10386 | 0.245641 | 1.000000 | 4 | 49 | -0.006860 | 0.001393 |
| 5 | market_sector | zero | 10386 | -0.042152 | 0.092784 | 1 | 49 | 0.000799 | 0.000388 |
| 5 | market_sector | historical_mean | 10386 | -0.013102 | 0.175258 | 1 | 49 | 0.000302 | 0.000170 |
| 5 | market_sector | momentum20 | 10386 | 0.078961 | 0.855670 | 4 | 49 | -0.002187 | 0.000746 |
| 5 | market_sector | reversion5 | 10386 | 0.246765 | 1.000000 | 4 | 49 | -0.006718 | 0.001387 |
| 5 | ridge | zero | 10386 | -0.450001 | 0.041237 | 1 | 49 | 0.018340 | 0.002700 |
| 5 | ridge | historical_mean | 10386 | -0.409583 | 0.041237 | 1 | 49 | 0.017844 | 0.002582 |
| 5 | ridge | momentum20 | 10386 | -0.281490 | 0.164948 | 1 | 49 | 0.015354 | 0.002795 |
| 5 | ridge | reversion5 | 10386 | -0.048016 | 0.350515 | 1 | 49 | 0.010823 | 0.003172 |
| 5 | shallow_boost | zero | 10386 | -0.024476 | 0.206186 | 1 | 49 | 0.000357 | 0.000367 |
| 5 | shallow_boost | historical_mean | 10386 | 0.004081 | 0.628866 | 2 | 49 | -0.000140 | 0.000226 |
| 5 | shallow_boost | momentum20 | 10386 | 0.094583 | 0.927835 | 4 | 49 | -0.002630 | 0.000715 |
| 5 | shallow_boost | reversion5 | 10386 | 0.259540 | 1.000000 | 4 | 49 | -0.007160 | 0.001364 |
| 10 | elastic_net | zero | 9901 | -0.191657 | 0.113402 | 1 | 24 | 0.010336 | 0.002590 |
| 10 | elastic_net | historical_mean | 9901 | -0.117828 | 0.185567 | 1 | 24 | 0.008526 | 0.002387 |
| 10 | elastic_net | momentum20 | 9901 | 0.013370 | 0.515464 | 1 | 24 | 0.003270 | 0.003629 |
| 10 | elastic_net | reversion5 | 9901 | 0.314318 | 0.989691 | 3 | 24 | -0.010558 | 0.004492 |
| 10 | market_only | zero | 9901 | -0.094025 | 0.092784 | 1 | 24 | 0.002154 | 0.001212 |
| 10 | market_only | historical_mean | 9901 | -0.026244 | 0.123711 | 2 | 24 | 0.000344 | 0.000477 |
| 10 | market_only | momentum20 | 9901 | 0.094205 | 0.762887 | 4 | 24 | -0.004912 | 0.002217 |
| 10 | market_only | reversion5 | 9901 | 0.370496 | 1.000000 | 4 | 24 | -0.018740 | 0.003554 |
| 10 | market_sector | zero | 9901 | -0.088260 | 0.113402 | 1 | 24 | 0.002051 | 0.001191 |
| 10 | market_sector | historical_mean | 9901 | -0.020837 | 0.226804 | 1 | 24 | 0.000241 | 0.000490 |
| 10 | market_sector | momentum20 | 9901 | 0.098978 | 0.783505 | 4 | 24 | -0.005015 | 0.002207 |
| 10 | market_sector | reversion5 | 9901 | 0.373813 | 1.000000 | 4 | 24 | -0.018842 | 0.003590 |
| 10 | ridge | zero | 9901 | -0.675260 | 0.000000 | 1 | 24 | 0.039581 | 0.007139 |
| 10 | ridge | historical_mean | 9901 | -0.571469 | 0.000000 | 1 | 24 | 0.037770 | 0.006737 |
| 10 | ridge | momentum20 | 9901 | -0.387028 | 0.092784 | 1 | 24 | 0.032514 | 0.007610 |
| 10 | ridge | reversion5 | 9901 | 0.036052 | 0.463918 | 1 | 24 | 0.018687 | 0.008763 |
| 10 | shallow_boost | zero | 9901 | -0.078632 | 0.123711 | 0 | 24 | 0.003021 | 0.001286 |
| 10 | shallow_boost | historical_mean | 9901 | -0.011806 | 0.360825 | 1 | 24 | 0.001211 | 0.000916 |
| 10 | shallow_boost | momentum20 | 9901 | 0.106949 | 0.824742 | 3 | 24 | -0.004045 | 0.002241 |
| 10 | shallow_boost | reversion5 | 9901 | 0.379353 | 1.000000 | 4 | 24 | -0.017873 | 0.003811 |
| 20 | elastic_net | zero | 8931 | -0.280663 | 0.051546 | 0 | 12 | 0.023015 | 0.004736 |
| 20 | elastic_net | historical_mean | 8931 | -0.132662 | 0.051546 | 0 | 12 | 0.018060 | 0.003407 |
| 20 | elastic_net | momentum20 | 8931 | 0.084188 | 0.608247 | 2 | 12 | 0.001933 | 0.007730 |
| 20 | elastic_net | reversion5 | 8931 | 0.455144 | 1.000000 | 4 | 12 | -0.035915 | 0.010846 |
| 20 | market_only | zero | 8931 | -0.197036 | 0.113402 | 0 | 12 | 0.006936 | 0.003316 |
| 20 | market_only | historical_mean | 8931 | -0.058699 | 0.103093 | 1 | 12 | 0.001980 | 0.001219 |
| 20 | market_only | momentum20 | 8931 | 0.143990 | 0.711340 | 3 | 12 | -0.014146 | 0.005847 |
| 20 | market_only | reversion5 | 8931 | 0.490723 | 1.000000 | 4 | 12 | -0.051995 | 0.008314 |
| 20 | market_sector | zero | 8931 | -0.182227 | 0.113402 | 0 | 12 | 0.006225 | 0.003319 |
| 20 | market_sector | historical_mean | 8931 | -0.045602 | 0.144330 | 1 | 12 | 0.001270 | 0.001253 |
| 20 | market_sector | momentum20 | 8931 | 0.154581 | 0.762887 | 3 | 12 | -0.014857 | 0.005755 |
| 20 | market_sector | reversion5 | 8931 | 0.497024 | 1.000000 | 4 | 12 | -0.052705 | 0.008273 |
| 20 | ridge | zero | 8931 | -0.251598 | 0.041237 | 0 | 12 | 0.062789 | 0.016370 |
| 20 | ridge | historical_mean | 8931 | -0.106956 | 0.134021 | 0 | 12 | 0.057834 | 0.015938 |
| 20 | ridge | momentum20 | 8931 | 0.104972 | 0.649485 | 1 | 12 | 0.041707 | 0.018232 |
| 20 | ridge | reversion5 | 8931 | 0.467510 | 0.989691 | 2 | 12 | 0.003858 | 0.023380 |
| 20 | shallow_boost | zero | 8931 | -0.171814 | 0.113402 | 0 | 12 | 0.006878 | 0.003253 |
| 20 | shallow_boost | historical_mean | 8931 | -0.036392 | 0.164948 | 1 | 12 | 0.001923 | 0.002066 |
| 20 | shallow_boost | momentum20 | 8931 | 0.162027 | 0.742268 | 4 | 12 | -0.014204 | 0.005472 |
| 20 | shallow_boost | reversion5 | 8931 | 0.501454 | 1.000000 | 4 | 12 | -0.052052 | 0.009033 |

| horizon | strongest_simple | complex_models_passing_fold_stock_screen |
| --- | --- | --- |
| 1 | zero | NONE |
| 5 | zero | NONE |
| 10 | zero | NONE |
| 20 | zero | NONE |

Screen requires aggregate improvement, three improving folds and 60% of stocks. Sector/regime consistency and prospective evidence remain required. Prefer simpler within 1% MAE. Overlapping labels and common shocks make pooled rows dependent: horizon-session date blocks are primary support. Block SE is descriptive dispersion, not a calibrated confidence interval. Full stock/industry/fold/regime/trend results are in STRATA.csv.gz. No trading backtest.

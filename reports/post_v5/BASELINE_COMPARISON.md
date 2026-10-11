# Development baseline comparison

Experiment development-20261008T211418-e9aa7e21; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261008T211418-e9aa7e21/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

| horizon | model | n | mae | rmse | direction_accuracy | ic | dates | nonoverlap_blocks | date_rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | elastic_net | 21829 | 0.013076 | 0.018737 | 0.507536 | 0.056805 | 248 | 248 | -0.001484 |
| 1 | historical_mean | 21829 | 0.013136 | 0.018795 | 0.476934 | -0.069612 | 248 | 248 | — |
| 1 | market_only | 21829 | 0.013141 | 0.018797 | 0.476385 | -0.001900 | 248 | 248 | 0.025189 |
| 1 | market_sector | 21829 | 0.013145 | 0.018802 | 0.476293 | -0.000585 | 248 | 248 | -0.019165 |
| 1 | momentum20 | 21829 | 0.013456 | 0.019174 | 0.496908 | -0.003961 | 248 | 248 | 0.000059 |
| 1 | reversion5 | 21829 | 0.014548 | 0.020399 | 0.504879 | 0.012072 | 248 | 248 | 0.015569 |
| 1 | ridge | 21829 | 0.014854 | 0.020268 | 0.488112 | 0.065104 | 248 | 248 | 0.010814 |
| 1 | shallow_boost | 21829 | 0.013144 | 0.018798 | 0.484951 | 0.013127 | 248 | 248 | 0.006435 |
| 1 | zero | 21829 | 0.013058 | 0.018733 | 0.523066 | — | 248 | 248 | — |
| 5 | elastic_net | 21445 | 0.031832 | 0.043532 | 0.522966 | 0.091374 | 244 | 49 | 0.003165 |
| 5 | historical_mean | 21445 | 0.031373 | 0.043019 | 0.488226 | -0.115783 | 244 | 49 | — |
| 5 | market_only | 21445 | 0.031485 | 0.043146 | 0.488226 | -0.000975 | 244 | 49 | 0.039968 |
| 5 | market_sector | 21445 | 0.031445 | 0.043110 | 0.487899 | 0.023319 | 244 | 49 | -0.016652 |
| 5 | momentum20 | 21445 | 0.035074 | 0.047544 | 0.488366 | -0.024613 | 244 | 49 | -0.015606 |
| 5 | reversion5 | 21445 | 0.042807 | 0.056797 | 0.528841 | 0.084584 | 244 | 49 | 0.045059 |
| 5 | ridge | 21445 | 0.044974 | 0.057561 | 0.517230 | 0.066042 | 244 | 49 | 0.019266 |
| 5 | shallow_boost | 21445 | 0.031413 | 0.043097 | 0.492796 | 0.001471 | 244 | 49 | — |
| 5 | zero | 21445 | 0.030917 | 0.042510 | 0.511774 | — | 244 | 49 | — |
| 10 | elastic_net | 20965 | 0.049258 | 0.064302 | 0.485333 | -0.048256 | 239 | 24 | -0.004827 |
| 10 | historical_mean | 20965 | 0.044001 | 0.058229 | 0.485142 | -0.197690 | 239 | 24 | — |
| 10 | market_only | 20965 | 0.044396 | 0.058804 | 0.485142 | -0.106678 | 239 | 24 | 0.124120 |
| 10 | market_sector | 20965 | 0.044402 | 0.058758 | 0.482518 | -0.069298 | 239 | 24 | -0.016961 |
| 10 | momentum20 | 20965 | 0.053062 | 0.070081 | 0.489149 | -0.019554 | 239 | 24 | -0.024137 |
| 10 | reversion5 | 20965 | 0.074124 | 0.099172 | 0.521965 | 0.055979 | 239 | 24 | 0.037336 |
| 10 | ridge | 20965 | 0.071342 | 0.088959 | 0.521917 | 0.066082 | 239 | 24 | 0.010818 |
| 10 | shallow_boost | 20965 | 0.044468 | 0.058641 | 0.497114 | -0.015559 | 239 | 24 | -0.093990 |
| 10 | zero | 20965 | 0.042823 | 0.056828 | 0.514858 | — | 239 | 24 | — |
| 20 | elastic_net | 20003 | 0.071756 | 0.090948 | 0.503924 | 0.165855 | 229 | 12 | -0.030969 |
| 20 | historical_mean | 20003 | 0.062622 | 0.082469 | 0.490026 | -0.254931 | 229 | 12 | — |
| 20 | market_only | 20003 | 0.063972 | 0.084358 | 0.490026 | -0.093893 | 229 | 12 | — |
| 20 | market_sector | 20003 | 0.063898 | 0.084115 | 0.484627 | -0.076695 | 229 | 12 | -0.016271 |
| 20 | momentum20 | 20003 | 0.086176 | 0.112733 | 0.488877 | -0.016578 | 229 | 12 | -0.047004 |
| 20 | reversion5 | 20003 | 0.139178 | 0.187445 | 0.512973 | 0.032790 | 229 | 12 | 0.014479 |
| 20 | ridge | 20003 | 0.096376 | 0.117709 | 0.510973 | 0.328996 | 229 | 12 | 0.016227 |
| 20 | shallow_boost | 20003 | 0.064793 | 0.084860 | 0.480728 | -0.110371 | 229 | 12 | -0.073738 |
| 20 | zero | 20003 | 0.059540 | 0.078611 | 0.509974 | — | 229 | 12 | — |

Errors are cumulative-log-return MAE/RMSE; direction is positive versus nonpositive. Historical mean is fixed once per purged fold. Momentum = ret20 * horizon/20; reversion = -ret5 * horizon/5. Ridge alpha10, elastic-net alpha.001/l1.5, shallow boosting 50iterations/depth2/minleaf100/lr.03/seed42/early_stopping=False. All median/scale fitting is train-only; no search.

| horizon | strongest_simple | simple_mae |
| --- | --- | --- |
| 1 | zero | 0.013058 |
| 5 | zero | 0.030917 |
| 10 | zero | 0.042823 |
| 20 | zero | 0.059540 |

Paired cohort comparisons (positive improvement favors model):

| horizon | model | baseline | n | mae_improvement | stocks_improved_fraction | folds_improved | nonoverlap_blocks | block_delta_mean | block_delta_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | elastic_net | zero | 21829 | -0.001380 | 0.432990 | 1 | 248 | -0.000002 | 0.000033 |
| 1 | elastic_net | historical_mean | 21829 | 0.004556 | 0.886598 | 4 | 248 | -0.000056 | 0.000029 |
| 1 | elastic_net | momentum20 | 21829 | 0.028218 | 0.989691 | 4 | 248 | -0.000373 | 0.000077 |
| 1 | elastic_net | reversion5 | 21829 | 0.101184 | 0.989691 | 4 | 248 | -0.001378 | 0.000160 |
| 1 | market_only | zero | 21829 | -0.006337 | 0.154639 | 1 | 248 | 0.000066 | 0.000039 |
| 1 | market_only | historical_mean | 21829 | -0.000372 | 0.443299 | 2 | 248 | 0.000012 | 0.000020 |
| 1 | market_only | momentum20 | 21829 | 0.023407 | 0.958763 | 4 | 248 | -0.000305 | 0.000080 |
| 1 | market_only | reversion5 | 21829 | 0.096734 | 0.989691 | 4 | 248 | -0.001310 | 0.000164 |
| 1 | market_sector | zero | 21829 | -0.006651 | 0.144330 | 1 | 248 | 0.000071 | 0.000040 |
| 1 | market_sector | historical_mean | 21829 | -0.000684 | 0.463918 | 2 | 248 | 0.000017 | 0.000022 |
| 1 | market_sector | momentum20 | 21829 | 0.023103 | 0.958763 | 4 | 248 | -0.000299 | 0.000081 |
| 1 | market_sector | reversion5 | 21829 | 0.096453 | 0.989691 | 4 | 248 | -0.001305 | 0.000165 |
| 1 | ridge | zero | 21829 | -0.137571 | 0.000000 | 0 | 248 | 0.001631 | 0.000215 |
| 1 | ridge | historical_mean | 21829 | -0.130828 | 0.000000 | 0 | 248 | 0.001576 | 0.000200 |
| 1 | ridge | momentum20 | 21829 | -0.103948 | 0.041237 | 0 | 248 | 0.001260 | 0.000223 |
| 1 | ridge | reversion5 | 21829 | -0.021058 | 0.381443 | 1 | 248 | 0.000255 | 0.000249 |
| 1 | shallow_boost | zero | 21829 | -0.006577 | 0.237113 | 1 | 248 | 0.000053 | 0.000054 |
| 1 | shallow_boost | historical_mean | 21829 | -0.000610 | 0.463918 | 2 | 248 | -0.000001 | 0.000040 |
| 1 | shallow_boost | momentum20 | 21829 | 0.023175 | 0.958763 | 4 | 248 | -0.000318 | 0.000084 |
| 1 | shallow_boost | reversion5 | 21829 | 0.096519 | 0.989691 | 4 | 248 | -0.001323 | 0.000167 |
| 5 | elastic_net | zero | 21445 | -0.029603 | 0.268041 | 1 | 49 | 0.001194 | 0.000665 |
| 5 | elastic_net | historical_mean | 21445 | -0.014627 | 0.381443 | 2 | 49 | 0.000923 | 0.000582 |
| 5 | elastic_net | momentum20 | 21445 | 0.092437 | 0.958763 | 4 | 49 | -0.002502 | 0.000951 |
| 5 | elastic_net | reversion5 | 21445 | 0.256389 | 0.989691 | 4 | 49 | -0.010318 | 0.001426 |
| 5 | market_only | zero | 21445 | -0.018367 | 0.185567 | 1 | 49 | 0.000250 | 0.000414 |
| 5 | market_only | historical_mean | 21445 | -0.003555 | 0.268041 | 2 | 49 | -0.000020 | 0.000163 |
| 5 | market_only | momentum20 | 21445 | 0.102341 | 0.989691 | 4 | 49 | -0.003446 | 0.000675 |
| 5 | market_only | reversion5 | 21445 | 0.264504 | 1.000000 | 4 | 49 | -0.011261 | 0.001197 |
| 5 | market_sector | zero | 21445 | -0.017085 | 0.237113 | 1 | 49 | 0.000242 | 0.000403 |
| 5 | market_sector | historical_mean | 21445 | -0.002291 | 0.422680 | 2 | 49 | -0.000028 | 0.000154 |
| 5 | market_sector | momentum20 | 21445 | 0.103471 | 1.000000 | 4 | 49 | -0.003454 | 0.000678 |
| 5 | market_sector | reversion5 | 21445 | 0.265430 | 1.000000 | 4 | 49 | -0.011269 | 0.001182 |
| 5 | ridge | zero | 21445 | -0.454682 | 0.000000 | 0 | 49 | 0.013493 | 0.002542 |
| 5 | ridge | historical_mean | 21445 | -0.433524 | 0.000000 | 1 | 49 | 0.013223 | 0.002423 |
| 5 | ridge | momentum20 | 21445 | -0.282257 | 0.041237 | 1 | 49 | 0.009797 | 0.002581 |
| 5 | ridge | reversion5 | 21445 | -0.050616 | 0.371134 | 2 | 49 | 0.001982 | 0.002698 |
| 5 | shallow_boost | zero | 21445 | -0.016055 | 0.257732 | 1 | 49 | 0.000245 | 0.000395 |
| 5 | shallow_boost | historical_mean | 21445 | -0.001277 | 0.494845 | 2 | 49 | -0.000026 | 0.000236 |
| 5 | shallow_boost | momentum20 | 21445 | 0.104379 | 0.979381 | 4 | 49 | -0.003452 | 0.000660 |
| 5 | shallow_boost | reversion5 | 21445 | 0.266174 | 1.000000 | 4 | 49 | -0.011267 | 0.001183 |
| 10 | elastic_net | zero | 20965 | -0.150251 | 0.041237 | 0 | 24 | 0.007650 | 0.002487 |
| 10 | elastic_net | historical_mean | 20965 | -0.119462 | 0.030928 | 0 | 24 | 0.007101 | 0.002395 |
| 10 | elastic_net | momentum20 | 20965 | 0.071694 | 0.742268 | 3 | 24 | -0.001150 | 0.003163 |
| 10 | elastic_net | reversion5 | 20965 | 0.335468 | 1.000000 | 4 | 24 | -0.022860 | 0.003832 |
| 10 | market_only | zero | 20965 | -0.036711 | 0.175258 | 1 | 24 | 0.000670 | 0.001192 |
| 10 | market_only | historical_mean | 20965 | -0.008961 | 0.195876 | 1 | 24 | 0.000122 | 0.000394 |
| 10 | market_only | momentum20 | 20965 | 0.163325 | 0.979381 | 4 | 24 | -0.008129 | 0.001798 |
| 10 | market_only | reversion5 | 20965 | 0.401063 | 1.000000 | 4 | 24 | -0.029840 | 0.002440 |
| 10 | market_sector | zero | 20965 | -0.036853 | 0.175258 | 1 | 24 | 0.000662 | 0.001168 |
| 10 | market_sector | historical_mean | 20965 | -0.009099 | 0.309278 | 0 | 24 | 0.000114 | 0.000398 |
| 10 | market_sector | momentum20 | 20965 | 0.163211 | 0.989691 | 4 | 24 | -0.008137 | 0.001784 |
| 10 | market_sector | reversion5 | 20965 | 0.400981 | 1.000000 | 4 | 24 | -0.029848 | 0.002414 |
| 10 | ridge | zero | 20965 | -0.665961 | 0.010309 | 1 | 24 | 0.027299 | 0.006172 |
| 10 | ridge | historical_mean | 20965 | -0.621367 | 0.000000 | 1 | 24 | 0.026751 | 0.005813 |
| 10 | ridge | momentum20 | 20965 | -0.344508 | 0.061856 | 1 | 24 | 0.018499 | 0.006309 |
| 10 | ridge | reversion5 | 20965 | 0.037528 | 0.536082 | 2 | 24 | -0.003211 | 0.006177 |
| 10 | shallow_boost | zero | 20965 | -0.038401 | 0.206186 | 2 | 24 | 0.000933 | 0.001053 |
| 10 | shallow_boost | historical_mean | 20965 | -0.010606 | 0.329897 | 1 | 24 | 0.000385 | 0.000567 |
| 10 | shallow_boost | momentum20 | 20965 | 0.161962 | 0.989691 | 4 | 24 | -0.007866 | 0.001675 |
| 10 | shallow_boost | reversion5 | 20965 | 0.400086 | 1.000000 | 4 | 24 | -0.029577 | 0.002479 |
| 20 | elastic_net | zero | 20003 | -0.205183 | 0.092784 | 0 | 12 | 0.011775 | 0.003522 |
| 20 | elastic_net | historical_mean | 20003 | -0.145870 | 0.061856 | 0 | 12 | 0.010633 | 0.002502 |
| 20 | elastic_net | momentum20 | 20003 | 0.167323 | 0.824742 | 4 | 12 | -0.011417 | 0.004554 |
| 20 | elastic_net | reversion5 | 20003 | 0.484426 | 1.000000 | 4 | 12 | -0.063269 | 0.005114 |
| 20 | market_only | zero | 20003 | -0.074443 | 0.195876 | 1 | 12 | 0.001672 | 0.003377 |
| 20 | market_only | historical_mean | 20003 | -0.021564 | 0.185567 | 1 | 12 | 0.000530 | 0.001171 |
| 20 | market_only | momentum20 | 20003 | 0.257654 | 0.979381 | 4 | 12 | -0.021520 | 0.003975 |
| 20 | market_only | reversion5 | 20003 | 0.540356 | 1.000000 | 4 | 12 | -0.073372 | 0.004151 |
| 20 | market_sector | zero | 20003 | -0.073191 | 0.185567 | 1 | 12 | 0.001508 | 0.003300 |
| 20 | market_sector | historical_mean | 20003 | -0.020374 | 0.268041 | 0 | 12 | 0.000366 | 0.001152 |
| 20 | market_sector | momentum20 | 20003 | 0.258518 | 0.979381 | 4 | 12 | -0.021683 | 0.003919 |
| 20 | market_sector | reversion5 | 20003 | 0.540891 | 1.000000 | 4 | 12 | -0.073535 | 0.004239 |
| 20 | ridge | zero | 20003 | -0.618678 | 0.000000 | 0 | 12 | 0.033025 | 0.011260 |
| 20 | ridge | historical_mean | 20003 | -0.539015 | 0.000000 | 0 | 12 | 0.031883 | 0.011771 |
| 20 | ridge | momentum20 | 20003 | -0.118366 | 0.298969 | 2 | 12 | 0.009834 | 0.012325 |
| 20 | ridge | reversion5 | 20003 | 0.307533 | 0.948454 | 3 | 12 | -0.042018 | 0.012226 |
| 20 | shallow_boost | zero | 20003 | -0.088228 | 0.144330 | 2 | 12 | 0.003630 | 0.002527 |
| 20 | shallow_boost | historical_mean | 20003 | -0.034671 | 0.206186 | 1 | 12 | 0.002488 | 0.001057 |
| 20 | shallow_boost | momentum20 | 20003 | 0.248129 | 0.979381 | 4 | 12 | -0.019562 | 0.003454 |
| 20 | shallow_boost | reversion5 | 20003 | 0.534459 | 1.000000 | 4 | 12 | -0.071414 | 0.004106 |

| horizon | strongest_simple | complex_models_passing_fold_stock_screen |
| --- | --- | --- |
| 1 | zero | NONE |
| 5 | zero | NONE |
| 10 | zero | NONE |
| 20 | zero | NONE |

Screen requires aggregate improvement, three improving folds and 60% of stocks. Sector/regime consistency and prospective evidence remain required. Prefer simpler within 1% MAE. Overlapping labels and common shocks make pooled rows dependent: horizon-session date blocks are primary support. Block SE is descriptive dispersion, not a calibrated confidence interval. Full stock/industry/fold/regime/trend results are in STRATA.csv.gz. No trading backtest.

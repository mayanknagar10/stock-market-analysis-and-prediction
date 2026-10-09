# Target formulation comparison

Experiment development-20261009T083921-00e775cf; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261009T083921-00e775cf/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

| horizon | fold | target | metric_space | n | mae | rmse | ic | train_target_mae | roc_auc | balanced_accuracy | brier | train_rate_brier | train_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | vol_normalized | ex_ante_vol_units | 5629 | 0.907117 | 1.228951 | -0.011950 | 0.792849 | — | — | — | — | — |
| 1 | 1 | market_excess | excess_log_return | 5629 | 0.012198 | 0.018355 | 0.008878 | 0.012084 | — | — | — | — | — |
| 1 | 1 | sector_excess | excess_log_return | 5009 | 0.011507 | 0.017422 | 0.010046 | 0.011801 | — | — | — | — | — |
| 1 | 1 | rank | same_origin_percentile | 5629 | 0.250246 | 0.289177 | -0.000732 | 0.249447 | — | — | — | — | — |
| 1 | 1 | positive_direction | raw_uncalibrated_classifier | 5629 | — | — | — | — | 0.519383 | 0.500000 | 0.238199 | 0.237732 | 0.616973 |
| 1 | 2 | vol_normalized | ex_ante_vol_units | 5000 | 0.891770 | 1.211155 | 0.047097 | 0.792991 | — | — | — | — | — |
| 1 | 2 | market_excess | excess_log_return | 5000 | 0.014616 | 0.020173 | 0.034474 | 0.012087 | — | — | — | — | — |
| 1 | 2 | sector_excess | excess_log_return | 4476 | 0.012577 | 0.017665 | 0.001047 | 0.011778 | — | — | — | — | — |
| 1 | 2 | rank | same_origin_percentile | 5000 | 0.250582 | 0.289299 | 0.026896 | 0.249502 | — | — | — | — | — |
| 1 | 2 | positive_direction | raw_uncalibrated_classifier | 5000 | — | — | — | — | 0.512833 | 0.500000 | 0.238140 | 0.239023 | 0.614363 |
| 1 | 3 | vol_normalized | ex_ante_vol_units | 61 | 0.935318 | 1.155832 | -0.131994 | 0.796585 | — | — | — | — | — |
| 1 | 3 | market_excess | excess_log_return | 61 | 0.010690 | 0.012811 | -0.000159 | 0.012218 | — | — | — | — | — |
| 1 | 3 | sector_excess | excess_log_return | 61 | 0.006248 | 0.008266 | 0.086621 | 0.011829 | — | — | — | — | — |
| 1 | 3 | rank | same_origin_percentile | 61 | 0.482617 | 0.482689 | — | 0.249575 | — | — | — | — | — |
| 1 | 3 | positive_direction | raw_uncalibrated_classifier | 61 | — | — | — | — | 0.511072 | 0.532634 | 0.235758 | 0.234747 | 0.607437 |
| 1 | 4 | vol_normalized | ex_ante_vol_units | 84 | 0.933316 | 1.171054 | 0.207047 | 0.796633 | — | — | — | — | — |
| 1 | 4 | market_excess | excess_log_return | 84 | 0.008863 | 0.011921 | 0.163309 | 0.012215 | — | — | — | — | — |
| 1 | 4 | sector_excess | excess_log_return | 84 | 0.009546 | 0.013010 | 0.032540 | 0.011824 | — | — | — | — | — |
| 1 | 4 | rank | same_origin_percentile | 84 | 0.344328 | 0.386238 | 0.332790 | 0.249715 | — | — | — | — | — |
| 1 | 4 | positive_direction | raw_uncalibrated_classifier | 84 | — | — | — | — | 0.543210 | 0.500000 | 0.231002 | 0.234214 | 0.607057 |
| 5 | 1 | vol_normalized | ex_ante_vol_units | 5629 | 1.660089 | 1.990281 | 0.080908 | 0.796240 | — | — | — | — | — |
| 5 | 1 | market_excess | excess_log_return | 5629 | 0.030164 | 0.044245 | -0.038242 | 0.027534 | — | — | — | — | — |
| 5 | 1 | sector_excess | excess_log_return | 5009 | 0.026252 | 0.039166 | 0.006472 | 0.026743 | — | — | — | — | — |
| 5 | 1 | rank | same_origin_percentile | 5629 | 0.250042 | 0.289128 | -0.000365 | 0.248898 | — | — | — | — | — |
| 5 | 1 | positive_direction | raw_uncalibrated_classifier | 5629 | — | — | — | — | 0.582746 | 0.500000 | 0.257511 | 0.256594 | 0.663157 |
| 5 | 2 | vol_normalized | ex_ante_vol_units | 4620 | 0.871709 | 1.134853 | 0.234576 | 0.802399 | — | — | — | — | — |
| 5 | 2 | market_excess | excess_log_return | 4620 | 0.033266 | 0.044423 | 0.072295 | 0.027559 | — | — | — | — | — |
| 5 | 2 | sector_excess | excess_log_return | 4136 | 0.028491 | 0.038636 | -0.008221 | 0.026695 | — | — | — | — | — |
| 5 | 2 | rank | same_origin_percentile | 4620 | 0.250295 | 0.289489 | 0.031820 | 0.248937 | — | — | — | — | — |
| 5 | 2 | positive_direction | raw_uncalibrated_classifier | 4620 | — | — | — | — | 0.553392 | 0.534171 | 0.248585 | 0.255051 | 0.661779 |
| 5 | 3 | vol_normalized | ex_ante_vol_units | 61 | 1.072873 | 1.256388 | -0.455579 | 0.803925 | — | — | — | — | — |
| 5 | 3 | market_excess | excess_log_return | 61 | 0.047629 | 0.050880 | 0.046166 | 0.027796 | — | — | — | — | — |
| 5 | 3 | sector_excess | excess_log_return | 61 | 0.011478 | 0.014113 | 0.177102 | 0.026800 | — | — | — | — | — |
| 5 | 3 | rank | same_origin_percentile | 61 | 0.486741 | 0.486957 | — | 0.248957 | — | — | — | — | — |
| 5 | 3 | positive_direction | raw_uncalibrated_classifier | 61 | — | — | — | — | 0.670996 | 0.576299 | 0.238347 | 0.251117 | 0.663832 |
| 5 | 4 | vol_normalized | ex_ante_vol_units | 76 | 1.011689 | 1.342249 | 0.036036 | 0.804000 | — | — | — | — | — |
| 5 | 4 | market_excess | excess_log_return | 76 | 0.022349 | 0.028579 | 0.021846 | 0.027797 | — | — | — | — | — |
| 5 | 4 | sector_excess | excess_log_return | 76 | 0.020764 | 0.028723 | -0.060451 | 0.026787 | — | — | — | — | — |
| 5 | 4 | rank | same_origin_percentile | 76 | 0.362962 | 0.401888 | 0.074126 | 0.249102 | — | — | — | — | — |
| 5 | 4 | positive_direction | raw_uncalibrated_classifier | 76 | — | — | — | — | 0.474306 | 0.495833 | 0.252419 | 0.250765 | 0.665107 |
| 10 | 1 | vol_normalized | ex_ante_vol_units | 5629 | 1.974636 | 2.240849 | 0.157371 | 0.786320 | — | — | — | — | — |
| 10 | 1 | market_excess | excess_log_return | 5629 | 0.044910 | 0.061343 | -0.081100 | 0.039221 | — | — | — | — | — |
| 10 | 1 | sector_excess | excess_log_return | 5009 | 0.037383 | 0.052482 | 0.017844 | 0.038267 | — | — | — | — | — |
| 10 | 1 | rank | same_origin_percentile | 5629 | 0.251373 | 0.290598 | -0.032038 | 0.248626 | — | — | — | — | — |
| 10 | 1 | positive_direction | raw_uncalibrated_classifier | 5629 | — | — | — | — | 0.523621 | 0.500000 | 0.277086 | 0.270747 | 0.680289 |
| 10 | 2 | vol_normalized | ex_ante_vol_units | 4145 | 0.760858 | 0.994150 | 0.152080 | 0.794022 | — | — | — | — | — |
| 10 | 2 | market_excess | excess_log_return | 4145 | 0.045819 | 0.060186 | 0.111922 | 0.039239 | — | — | — | — | — |
| 10 | 2 | sector_excess | excess_log_return | 3711 | 0.039366 | 0.052614 | -0.016266 | 0.038134 | — | — | — | — | — |
| 10 | 2 | rank | same_origin_percentile | 4145 | 0.250138 | 0.289516 | 0.051240 | 0.248757 | — | — | — | — | — |
| 10 | 2 | positive_direction | raw_uncalibrated_classifier | 4145 | — | — | — | — | 0.494989 | 0.499661 | 0.276895 | 0.266585 | 0.692514 |
| 10 | 3 | vol_normalized | ex_ante_vol_units | 61 | 1.417798 | 1.563690 | -0.369170 | 0.792788 | — | — | — | — | — |
| 10 | 3 | market_excess | excess_log_return | 61 | 0.084348 | 0.089487 | 0.069857 | 0.039397 | — | — | — | — | — |
| 10 | 3 | sector_excess | excess_log_return | 61 | 0.013907 | 0.016612 | 0.164305 | 0.038199 | — | — | — | — | — |
| 10 | 3 | rank | same_origin_percentile | 61 | 0.484976 | 0.485133 | — | 0.248803 | — | — | — | — | — |
| 10 | 3 | positive_direction | raw_uncalibrated_classifier | 61 | — | — | — | — | 0.447168 | 0.477124 | 0.273073 | 0.255211 | 0.697086 |
| 10 | 4 | vol_normalized | ex_ante_vol_units | 66 | 1.137475 | 1.447889 | 0.026573 | 0.793193 | — | — | — | — | — |
| 10 | 4 | market_excess | excess_log_return | 66 | 0.043509 | 0.051173 | 0.186056 | 0.039413 | — | — | — | — | — |
| 10 | 4 | sector_excess | excess_log_return | 66 | 0.025685 | 0.034797 | 0.246926 | 0.038180 | — | — | — | — | — |
| 10 | 4 | rank | same_origin_percentile | 66 | 0.376344 | 0.408493 | 0.224769 | 0.248943 | — | — | — | — | — |
| 10 | 4 | positive_direction | raw_uncalibrated_classifier | 66 | — | — | — | — | 0.482759 | 0.532153 | 0.247863 | 0.255429 | 0.695738 |
| 20 | 1 | vol_normalized | ex_ante_vol_units | 5629 | 1.116237 | 1.433515 | 0.116933 | 0.791262 | — | — | — | — | — |
| 20 | 1 | market_excess | excess_log_return | 5629 | 0.057047 | 0.075329 | -0.026909 | 0.056895 | — | — | — | — | — |
| 20 | 1 | sector_excess | excess_log_return | 5009 | 0.051859 | 0.070386 | 0.031792 | 0.055522 | — | — | — | — | — |
| 20 | 1 | rank | same_origin_percentile | 5629 | 0.252695 | 0.292020 | -0.056455 | 0.247322 | — | — | — | — | — |
| 20 | 1 | positive_direction | raw_uncalibrated_classifier | 5629 | — | — | — | — | 0.621590 | 0.500000 | 0.308779 | 0.289671 | 0.726296 |
| 20 | 2 | vol_normalized | ex_ante_vol_units | 3195 | 1.051774 | 1.292771 | -0.037150 | 0.793692 | — | — | — | — | — |
| 20 | 2 | market_excess | excess_log_return | 3195 | 0.068553 | 0.087074 | 0.108879 | 0.056682 | — | — | — | — | — |
| 20 | 2 | sector_excess | excess_log_return | 2861 | 0.054753 | 0.070661 | -0.045780 | 0.055095 | — | — | — | — | — |
| 20 | 2 | rank | same_origin_percentile | 3195 | 0.252196 | 0.291763 | -0.017495 | 0.247851 | — | — | — | — | — |
| 20 | 2 | positive_direction | raw_uncalibrated_classifier | 3195 | — | — | — | — | 0.464530 | 0.498026 | 0.304087 | 0.285273 | 0.721115 |
| 20 | 3 | vol_normalized | ex_ante_vol_units | 61 | 1.954724 | 2.051629 | -0.208673 | 0.794775 | — | — | — | — | — |
| 20 | 3 | market_excess | excess_log_return | 61 | 0.111950 | 0.119508 | -0.108144 | 0.056785 | — | — | — | — | — |
| 20 | 3 | sector_excess | excess_log_return | 61 | 0.016903 | 0.019217 | 0.456002 | 0.055032 | — | — | — | — | — |
| 20 | 3 | rank | same_origin_percentile | 61 | 0.488842 | 0.489060 | — | 0.248020 | — | — | — | — | — |
| 20 | 3 | positive_direction | raw_uncalibrated_classifier | 61 | — | — | — | — | 0.401416 | 0.503268 | 0.285707 | 0.260001 | 0.722416 |
| 20 | 4 | vol_normalized | ex_ante_vol_units | 46 | 1.610792 | 1.692345 | 0.073944 | 0.795038 | — | — | — | — | — |
| 20 | 4 | market_excess | excess_log_return | 46 | 0.066254 | 0.069499 | 0.024237 | 0.056773 | — | — | — | — | — |
| 20 | 4 | sector_excess | excess_log_return | 46 | 0.033149 | 0.038721 | -0.338760 | 0.055010 | — | — | — | — | — |
| 20 | 4 | rank | same_origin_percentile | 46 | 0.425962 | 0.432803 | -0.072266 | 0.248168 | — | — | — | — | — |
| 20 | 4 | positive_direction | raw_uncalibrated_classifier | 46 | — | — | — | — | 0.677778 | 0.580556 | 0.242657 | 0.286140 | 0.723877 |

Volatility units, excess-log-return error and same-origin percentile errors are separate metric spaces, never return MAE. Normalize using origin-only daily20 volatility * sqrt(horizon); convert predictions back with this ex-ante scale. Rank target uses eligible stocks of the same origin only; its score is not automatically clipped or reconstructed into returns. Positive direction = expm1(log return) > .003 round-trip cost. Classifier scores are RAW and UNCALIBRATED.

Reconstructed return-space results:

| horizon | model | n | mae | rmse | ic | date_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | decomposition | 9630 | 0.014940 | 0.021190 | 0.005165 | -0.000736 |
| 1 | market_excess | 10774 | 0.015400 | 0.021540 | -0.002859 | 0.011602 |
| 1 | ridge | 10774 | 0.016288 | 0.022357 | 0.021744 | 0.015178 |
| 1 | sector_excess | 9630 | 0.014940 | 0.021190 | 0.005165 | -0.000736 |
| 1 | vol_normalized | 10774 | 0.016418 | 0.022784 | 0.009429 | -0.001213 |
| 5 | decomposition | 9282 | 0.036184 | 0.049635 | 0.078786 | -0.021933 |
| 5 | market_excess | 10386 | 0.039010 | 0.052271 | -0.031092 | 0.018399 |
| 5 | ridge | 10386 | 0.050208 | 0.064619 | 0.098071 | 0.023626 |
| 5 | sector_excess | 9282 | 0.036184 | 0.049635 | 0.078786 | -0.021933 |
| 5 | vol_normalized | 10386 | 0.053692 | 0.070824 | 0.100949 | 0.019279 |
| 10 | decomposition | 8847 | 0.052769 | 0.068930 | -0.006253 | -0.021348 |
| 10 | market_excess | 9901 | 0.059835 | 0.076111 | 0.015618 | 0.024579 |
| 10 | ridge | 9901 | 0.079967 | 0.099379 | 0.085186 | 0.026218 |
| 10 | sector_excess | 8847 | 0.052769 | 0.068930 | -0.006253 | -0.021348 |
| 10 | vol_normalized | 9901 | 0.083397 | 0.107172 | 0.084809 | 0.023793 |
| 20 | decomposition | 7977 | 0.077945 | 0.099997 | -0.069807 | -0.066602 |
| 20 | market_excess | 8931 | 0.078675 | 0.100597 | 0.043254 | 0.007293 |
| 20 | ridge | 8931 | 0.082113 | 0.104133 | 0.063738 | 0.026338 |
| 20 | sector_excess | 7977 | 0.077945 | 0.099997 | -0.069807 | -0.066602 |
| 20 | vol_normalized | 8931 | 0.087243 | 0.110719 | 0.052350 | -0.051402 |

Excess reconstruction adds independently predicted market and sector-minus-market components. Realized future references enter training/validation LABELS ONLY, never predictors or validation-time reconstruction. Market head uses one sample/date; sector-excess head uses train-only industry interactions. Three-head decomposition adds market + sector excess + stock alpha from past features on identical purged folds. Keep nothing automatically; paired negative results remain recorded. No new calibration was fit: held-out ROC-AUC .55 and balanced accuracy .52 with stable date ranking must first be established. Coin-flip Brier=.25 and training-rate Brier are explicit baselines.

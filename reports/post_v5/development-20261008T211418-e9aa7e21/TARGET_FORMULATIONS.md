# Target formulation comparison

Experiment development-20261008T211418-e9aa7e21; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261008T211418-e9aa7e21/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

| horizon | fold | target | metric_space | n | mae | rmse | ic | train_target_mae | roc_auc | balanced_accuracy | brier | train_rate_brier | train_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | vol_normalized | ex_ante_vol_units | 5889 | 0.921357 | 1.246410 | -0.012465 | 0.792039 | — | — | — | — | — |
| 1 | 1 | market_excess | excess_log_return | 5889 | 0.012410 | 0.018680 | 0.004399 | 0.012153 | — | — | — | — | — |
| 1 | 1 | sector_excess | excess_log_return | 5269 | 0.011699 | 0.017708 | 0.011733 | 0.011833 | — | — | — | — | — |
| 1 | 1 | rank | same_origin_percentile | 5889 | 0.250198 | 0.289111 | 0.001758 | 0.249457 | — | — | — | — | — |
| 1 | 1 | positive_direction | raw_uncalibrated_classifier | 5889 | — | — | — | — | 0.511597 | 0.500000 | 0.238375 | 0.237761 | 0.618244 |
| 1 | 2 | vol_normalized | ex_ante_vol_units | 5095 | 0.882790 | 1.203422 | 0.047108 | 0.792607 | — | — | — | — | — |
| 1 | 2 | market_excess | excess_log_return | 5095 | 0.014794 | 0.020236 | 0.008840 | 0.012164 | — | — | — | — | — |
| 1 | 2 | sector_excess | excess_log_return | 4561 | 0.012359 | 0.017496 | 0.001549 | 0.011824 | — | — | — | — | — |
| 1 | 2 | rank | same_origin_percentile | 5095 | 0.246144 | 0.286529 | 0.030238 | 0.249509 | — | — | — | — | — |
| 1 | 2 | positive_direction | raw_uncalibrated_classifier | 5095 | — | — | — | — | 0.520730 | 0.500000 | 0.236356 | 0.238032 | 0.615139 |
| 1 | 3 | vol_normalized | ex_ante_vol_units | 4794 | 0.817232 | 1.064272 | 0.056568 | 0.795250 | — | — | — | — | — |
| 1 | 3 | market_excess | excess_log_return | 4794 | 0.012392 | 0.016546 | 0.014771 | 0.012286 | — | — | — | — | — |
| 1 | 3 | sector_excess | excess_log_return | 4294 | 0.009945 | 0.013792 | -0.010814 | 0.011856 | — | — | — | — | — |
| 1 | 3 | rank | same_origin_percentile | 4794 | 0.250799 | 0.289705 | -0.001343 | 0.249270 | — | — | — | — | — |
| 1 | 3 | positive_direction | raw_uncalibrated_classifier | 4794 | — | — | — | — | 0.526350 | 0.507666 | 0.245532 | 0.246294 | 0.605880 |
| 1 | 4 | vol_normalized | ex_ante_vol_units | 6051 | 0.852121 | 1.163003 | 0.064085 | 0.789993 | — | — | — | — | — |
| 1 | 4 | market_excess | excess_log_return | 6051 | 0.009499 | 0.013726 | -0.006366 | 0.012191 | — | — | — | — | — |
| 1 | 4 | sector_excess | excess_log_return | 5421 | 0.009110 | 0.013061 | 0.021316 | 0.011743 | — | — | — | — | — |
| 1 | 4 | rank | same_origin_percentile | 6051 | 0.249613 | 0.288400 | 0.043755 | 0.249361 | — | — | — | — | — |
| 1 | 4 | positive_direction | raw_uncalibrated_classifier | 6051 | — | — | — | — | 0.525306 | 0.500000 | 0.234887 | 0.236329 | 0.608614 |
| 5 | 1 | vol_normalized | ex_ante_vol_units | 5889 | 1.694017 | 2.023977 | 0.066324 | 0.795928 | — | — | — | — | — |
| 5 | 1 | market_excess | excess_log_return | 5889 | 0.031099 | 0.045077 | -0.051432 | 0.027702 | — | — | — | — | — |
| 5 | 1 | sector_excess | excess_log_return | 5269 | 0.026449 | 0.039228 | 0.016603 | 0.026895 | — | — | — | — | — |
| 5 | 1 | rank | same_origin_percentile | 5889 | 0.249903 | 0.288982 | 0.003125 | 0.249016 | — | — | — | — | — |
| 5 | 1 | positive_direction | raw_uncalibrated_classifier | 5889 | — | — | — | — | 0.582030 | 0.500000 | 0.259033 | 0.256935 | 0.664794 |
| 5 | 2 | vol_normalized | ex_ante_vol_units | 5095 | 0.919576 | 1.192162 | 0.253766 | 0.802231 | — | — | — | — | — |
| 5 | 2 | market_excess | excess_log_return | 5095 | 0.032650 | 0.043642 | 0.083149 | 0.027734 | — | — | — | — | — |
| 5 | 2 | sector_excess | excess_log_return | 4561 | 0.027985 | 0.038125 | -0.011186 | 0.026843 | — | — | — | — | — |
| 5 | 2 | rank | same_origin_percentile | 5095 | 0.250028 | 0.289119 | 0.037589 | 0.249033 | — | — | — | — | — |
| 5 | 2 | positive_direction | raw_uncalibrated_classifier | 5095 | — | — | — | — | 0.484486 | 0.489136 | 0.251430 | 0.253302 | 0.663729 |
| 5 | 3 | vol_normalized | ex_ante_vol_units | 4794 | 1.020016 | 1.245815 | 0.015044 | 0.805432 | — | — | — | — | — |
| 5 | 3 | market_excess | excess_log_return | 4794 | 0.043666 | 0.052339 | 0.018946 | 0.027935 | — | — | — | — | — |
| 5 | 3 | sector_excess | excess_log_return | 4294 | 0.022869 | 0.030971 | -0.007597 | 0.026907 | — | — | — | — | — |
| 5 | 3 | rank | same_origin_percentile | 4794 | 0.250739 | 0.289568 | -0.002556 | 0.249040 | — | — | — | — | — |
| 5 | 3 | positive_direction | raw_uncalibrated_classifier | 4794 | — | — | — | — | 0.533921 | 0.497511 | 0.249631 | 0.249541 | 0.671066 |
| 5 | 4 | vol_normalized | ex_ante_vol_units | 5667 | 1.022006 | 1.310510 | 0.049646 | 0.799917 | — | — | — | — | — |
| 5 | 4 | market_excess | excess_log_return | 5667 | 0.022990 | 0.030727 | 0.007840 | 0.027785 | — | — | — | — | — |
| 5 | 4 | sector_excess | excess_log_return | 5077 | 0.020511 | 0.028124 | -0.011195 | 0.026690 | — | — | — | — | — |
| 5 | 4 | rank | same_origin_percentile | 5667 | 0.249850 | 0.288831 | 0.037718 | 0.249139 | — | — | — | — | — |
| 5 | 4 | positive_direction | raw_uncalibrated_classifier | 5667 | — | — | — | — | 0.500174 | 0.481763 | 0.250378 | 0.251625 | 0.667088 |
| 10 | 1 | vol_normalized | ex_ante_vol_units | 5889 | 1.972334 | 2.239286 | 0.143327 | 0.786832 | — | — | — | — | — |
| 10 | 1 | market_excess | excess_log_return | 5889 | 0.045060 | 0.061673 | -0.092881 | 0.039552 | — | — | — | — | — |
| 10 | 1 | sector_excess | excess_log_return | 5269 | 0.037663 | 0.052808 | 0.010086 | 0.038602 | — | — | — | — | — |
| 10 | 1 | rank | same_origin_percentile | 5889 | 0.251091 | 0.290324 | -0.031249 | 0.248692 | — | — | — | — | — |
| 10 | 1 | positive_direction | raw_uncalibrated_classifier | 5889 | — | — | — | — | 0.536611 | 0.500000 | 0.279994 | 0.271573 | 0.685503 |
| 10 | 2 | vol_normalized | ex_ante_vol_units | 5095 | 0.773074 | 1.005279 | 0.312773 | 0.793899 | — | — | — | — | — |
| 10 | 2 | market_excess | excess_log_return | 5095 | 0.044418 | 0.058609 | 0.130212 | 0.039548 | — | — | — | — | — |
| 10 | 2 | sector_excess | excess_log_return | 4561 | 0.037968 | 0.051057 | -0.019640 | 0.038447 | — | — | — | — | — |
| 10 | 2 | rank | same_origin_percentile | 5095 | 0.249517 | 0.288734 | 0.065396 | 0.248818 | — | — | — | — | — |
| 10 | 2 | positive_direction | raw_uncalibrated_classifier | 5095 | — | — | — | — | 0.373898 | 0.482397 | 0.268026 | 0.256779 | 0.687725 |
| 10 | 3 | vol_normalized | ex_ante_vol_units | 4794 | 1.365701 | 1.580334 | -0.043129 | 0.792437 | — | — | — | — | — |
| 10 | 3 | market_excess | excess_log_return | 4794 | 0.076323 | 0.087231 | 0.108672 | 0.039614 | — | — | — | — | — |
| 10 | 3 | sector_excess | excess_log_return | 4294 | 0.031497 | 0.041755 | -0.019327 | 0.038403 | — | — | — | — | — |
| 10 | 3 | rank | same_origin_percentile | 4794 | 0.250948 | 0.289684 | -0.012143 | 0.248835 | — | — | — | — | — |
| 10 | 3 | positive_direction | raw_uncalibrated_classifier | 4794 | — | — | — | — | 0.502335 | 0.517802 | 0.247784 | 0.245572 | 0.701934 |
| 10 | 4 | vol_normalized | ex_ante_vol_units | 5187 | 1.038474 | 1.315055 | -0.121385 | 0.785605 | — | — | — | — | — |
| 10 | 4 | market_excess | excess_log_return | 5187 | 0.034944 | 0.044538 | 0.041572 | 0.039444 | — | — | — | — | — |
| 10 | 4 | sector_excess | excess_log_return | 4647 | 0.028692 | 0.038539 | -0.001256 | 0.038062 | — | — | — | — | — |
| 10 | 4 | rank | same_origin_percentile | 5187 | 0.249738 | 0.288821 | 0.035888 | 0.248925 | — | — | — | — | — |
| 10 | 4 | positive_direction | raw_uncalibrated_classifier | 5187 | — | — | — | — | 0.573044 | 0.493741 | 0.247270 | 0.253725 | 0.697489 |
| 20 | 1 | vol_normalized | ex_ante_vol_units | 5889 | 1.121935 | 1.438262 | 0.100411 | 0.792274 | — | — | — | — | — |
| 20 | 1 | market_excess | excess_log_return | 5889 | 0.057599 | 0.076424 | -0.029832 | 0.057415 | — | — | — | — | — |
| 20 | 1 | sector_excess | excess_log_return | 5269 | 0.052683 | 0.071062 | 0.007042 | 0.056007 | — | — | — | — | — |
| 20 | 1 | rank | same_origin_percentile | 5889 | 0.252452 | 0.291822 | -0.060641 | 0.247329 | — | — | — | — | — |
| 20 | 1 | positive_direction | raw_uncalibrated_classifier | 5889 | — | — | — | — | 0.621381 | 0.500000 | 0.310059 | 0.290792 | 0.724608 |
| 20 | 2 | vol_normalized | ex_ante_vol_units | 5095 | 1.032347 | 1.287418 | 0.363833 | 0.794320 | — | — | — | — | — |
| 20 | 2 | market_excess | excess_log_return | 5095 | 0.068798 | 0.089018 | 0.228678 | 0.057210 | — | — | — | — | — |
| 20 | 2 | sector_excess | excess_log_return | 4561 | 0.052597 | 0.069136 | -0.035969 | 0.055624 | — | — | — | — | — |
| 20 | 2 | rank | same_origin_percentile | 5095 | 0.249539 | 0.288372 | 0.069099 | 0.247888 | — | — | — | — | — |
| 20 | 2 | positive_direction | raw_uncalibrated_classifier | 5095 | — | — | — | — | 0.400512 | 0.464694 | 0.278150 | 0.260583 | 0.721341 |
| 20 | 3 | vol_normalized | ex_ante_vol_units | 4794 | 1.834924 | 1.986320 | 0.170555 | 0.794486 | — | — | — | — | — |
| 20 | 3 | market_excess | excess_log_return | 4794 | 0.091893 | 0.105535 | 0.097113 | 0.057262 | — | — | — | — | — |
| 20 | 3 | sector_excess | excess_log_return | 4294 | 0.041604 | 0.054758 | 0.055919 | 0.055413 | — | — | — | — | — |
| 20 | 3 | rank | same_origin_percentile | 4794 | 0.250029 | 0.288985 | 0.050257 | 0.247955 | — | — | — | — | — |
| 20 | 3 | positive_direction | raw_uncalibrated_classifier | 4794 | — | — | — | — | 0.512167 | 0.487306 | 0.240032 | 0.239626 | 0.721202 |
| 20 | 4 | vol_normalized | ex_ante_vol_units | 4225 | 1.118487 | 1.367192 | 0.138943 | 0.786988 | — | — | — | — | — |
| 20 | 4 | market_excess | excess_log_return | 4225 | 0.052363 | 0.065616 | 0.090915 | 0.056973 | — | — | — | — | — |
| 20 | 4 | sector_excess | excess_log_return | 3785 | 0.041170 | 0.054523 | 0.005620 | 0.054832 | — | — | — | — | — |
| 20 | 4 | rank | same_origin_percentile | 4225 | 0.248961 | 0.287884 | 0.073276 | 0.247955 | — | — | — | — | — |
| 20 | 4 | positive_direction | raw_uncalibrated_classifier | 4225 | — | — | — | — | 0.613943 | 0.598045 | 0.246206 | 0.256639 | 0.718415 |

Volatility units, excess-log-return error and same-origin percentile errors are separate metric spaces, never return MAE. Normalize using origin-only daily20 volatility * sqrt(horizon); convert predictions back with this ex-ante scale. Rank target uses eligible stocks of the same origin only; its score is not automatically clipped or reconstructed into returns. Positive direction = expm1(log return) > .003 round-trip cost. Classifier scores are RAW and UNCALIBRATED.

Reconstructed return-space results:

| horizon | model | n | mae | rmse | ic | date_rank_ic |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | decomposition | 19545 | 0.013153 | 0.018841 | 0.002270 | 0.008211 |
| 1 | market_excess | 21829 | 0.013704 | 0.019291 | 0.032460 | 0.011931 |
| 1 | ridge | 21829 | 0.014854 | 0.020268 | 0.065104 | 0.010814 |
| 1 | sector_excess | 19545 | 0.013153 | 0.018841 | 0.002270 | 0.008211 |
| 1 | vol_normalized | 21829 | 0.014573 | 0.020307 | 0.042463 | 0.002808 |
| 5 | decomposition | 19201 | 0.031518 | 0.043340 | 0.028230 | -0.002415 |
| 5 | market_excess | 21445 | 0.036509 | 0.048135 | 0.082020 | 0.015919 |
| 5 | ridge | 21445 | 0.044974 | 0.057561 | 0.066042 | 0.019266 |
| 5 | sector_excess | 19201 | 0.031518 | 0.043340 | 0.028230 | -0.002415 |
| 5 | vol_normalized | 21445 | 0.045174 | 0.060445 | 0.057782 | 0.020429 |
| 10 | decomposition | 18771 | 0.044458 | 0.059202 | -0.026645 | 0.010935 |
| 10 | market_excess | 20965 | 0.056162 | 0.071119 | 0.187028 | 0.012787 |
| 10 | ridge | 20965 | 0.071342 | 0.088959 | 0.066082 | 0.010818 |
| 10 | sector_excess | 18771 | 0.044458 | 0.059202 | -0.026645 | 0.010935 |
| 10 | vol_normalized | 20965 | 0.071226 | 0.092811 | 0.060705 | 0.037912 |
| 20 | decomposition | 17909 | 0.064154 | 0.084941 | -0.052836 | 0.000258 |
| 20 | market_excess | 20003 | 0.075449 | 0.094729 | 0.319702 | 0.019402 |
| 20 | ridge | 20003 | 0.096376 | 0.117709 | 0.328996 | 0.016227 |
| 20 | sector_excess | 17909 | 0.064154 | 0.084941 | -0.052836 | 0.000258 |
| 20 | vol_normalized | 20003 | 0.098226 | 0.123673 | 0.305049 | 0.050861 |

Excess reconstruction adds independently predicted market and sector-minus-market components. Realized future references enter training/validation LABELS ONLY, never predictors or validation-time reconstruction. Market head uses one sample/date; sector-excess head uses train-only industry interactions. Three-head decomposition adds market + sector excess + stock alpha from past features on identical purged folds. Keep nothing automatically; paired negative results remain recorded. No new calibration was fit: held-out ROC-AUC .55 and balanced accuracy .52 with stable date ranking must first be established. Coin-flip Brier=.25 and training-rate Brier are explicit baselines.

# V4 versus V5: frozen research comparison

**Result: the frozen research performance gate FAILED. Production promotion remains BLOCKED.**

Candidate: research-20261008T190859. The final period began 2025-10-01 UTC and includes only labels matured by 2026-10-07T16:03:45.369336+00:00. The protocol was locked before evaluation and consumed once. No model decisions were changed after opening it.

This is a retrospective model experiment on revised Yahoo snapshots, six current equities and reconstructed availability/session cutoffs. It is not certified point-in-time evidence or a replay of historical production predictions. Models were created in 2026 with all fitting, blending, calibration and validation labels strictly before the final boundary. Overlapping horizons and correlated stocks make observation counts dependent.

## Frozen decision gates

| Gate | Observed groups | Required | Result |
|---|---:|---:|---|
| MAE better than zero-return | 1/8 | 6/8 | FAIL |
| RMSE better than zero-return | 0/8 | 6/8 | FAIL |
| Brier below 0.25 | 2/8 | 6/8 | FAIL |
| 80% coverage between 70% and 90% | 7/8 | 6/8 | PASS |
| Calibration accepted on development validation | 3/8 | 8/8 | FAIL |

## Full-grid return errors

MAE/RMSE below are in percentage points of cumulative log return, not dollar/rupee price error. V4 refit uses the original 600-tree XGB/LGB architecture and pooled pre-boundary labels (5,580 observations), with training-fitted scaling and equal ensemble weights. Its one-day estimate is compounded as an architecture baseline. The originally shipped synthetic checkpoint is preserved and reported separately in V4_BASELINE.md; no historical training cutoff was available for that checkpoint.

| Family | Sessions | N | V5 MAE | Zero MAE | Drift MAE | V4 compound MAE | V5 RMSE | Zero RMSE | V4 compound RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| india_equity | 1 | 747 | 1.12% | 1.10% | 1.10% | 1.13% | 1.55% | 1.52% | 1.56% |
| india_equity | 5 | 735 | 2.65% | 2.43% | 2.44% | 2.87% | 3.48% | 3.22% | 3.72% |
| india_equity | 10 | 720 | 4.05% | 3.66% | 3.68% | 4.71% | 5.26% | 4.67% | 6.09% |
| india_equity | 20 | 690 | 6.33% | 5.19% | 5.26% | 8.05% | 8.13% | 6.84% | 10.57% |
| us_equity | 1 | 762 | 1.16% | 1.15% | 1.14% | 1.18% | 1.55% | 1.53% | 1.58% |
| us_equity | 5 | 750 | 2.67% | 2.67% | 2.64% | 3.03% | 3.34% | 3.33% | 3.91% |
| us_equity | 10 | 735 | 3.84% | 3.79% | 3.72% | 4.83% | 4.74% | 4.72% | 6.32% |
| us_equity | 20 | 705 | 5.76% | 5.66% | 5.41% | 8.01% | 7.17% | 6.89% | 10.73% |

## Direction, calibration and uncertainty

Return direction tests the sign of the central regressor; classifier direction tests P(up) at 0.5. Brier/log loss/ECE use the accepted calibrated score where available, otherwise the explicitly raw classifier score. Acceptance was decided on development validation, not this period. Development acceptance does not establish calibration under later distribution shift. V4 does not emit calibrated classifier probabilities; its Brier/log loss are unavailable.

| Family | Sessions | Calibration | Return direction | Classifier direction | Brier | Log loss | ECE | 50% coverage | 80% coverage | Mean 80% width | Rank IC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| india_equity | 1 | Rejected; raw | 47.39% | 46.85% | 0.2648 | 0.7239 | 0.1202 | 51.41% | 77.78% | 3.18% | 0.0234 |
| india_equity | 5 | Accepted | 47.35% | 48.30% | 0.2641 | 0.7229 | 0.1207 | 50.07% | 80.82% | 8.65% | -0.0312 |
| india_equity | 10 | Rejected; raw | 45.00% | 43.47% | 0.3702 | 1.0110 | 0.3224 | 31.94% | 77.50% | 12.24% | -0.0159 |
| india_equity | 20 | Accepted | 41.01% | 37.97% | 0.3328 | 0.8708 | 0.3015 | 35.80% | 79.57% | 18.48% | 0.1273 |
| us_equity | 1 | Accepted | 48.43% | 54.72% | 0.2490 | 0.6911 | 0.0345 | 50.26% | 83.20% | 4.07% | -0.0085 |
| us_equity | 5 | Rejected; raw | 55.33% | 54.80% | 0.2531 | 0.7001 | 0.0515 | 46.40% | 91.47% | 11.50% | 0.1151 |
| us_equity | 10 | Rejected; raw | 51.29% | 57.55% | 0.2663 | 0.7352 | 0.1460 | 43.95% | 81.90% | 12.20% | 0.0099 |
| us_equity | 20 | Rejected; raw | 52.48% | 57.73% | 0.2444 | 0.6834 | 0.0862 | 43.12% | 80.43% | 17.55% | -0.0460 |

India 10/20-session direction and probability scores deteriorated substantially. Seven groups meet the broad 80% coverage tolerance, while several 50% intervals undercover; good coverage alone does not demonstrate useful forecast centers. Native five-quantile heads provide 50% and 80% bands; 90% coverage is unavailable. Pinball losses, Pearson IC, bias, class precision/recall and bucket sample counts are retained in the machine report.

## Matched recursive V4 comparison

The actual legacy recursive simulation is run on a fixed, evenly spaced sample of up to 12 origins per stock from the 20-session cohort, using 30 paths and recorded deterministic seeds. V5 below is restricted to exactly those origins. Simulation intervals describe this legacy generator, not calibrated probabilities. Small, dependent samples cannot establish stable superiority.

| Family | Sessions | Matched N | V4 recursive MAE | V5 matched MAE | V4 recursive RMSE | V5 matched RMSE | V4 direction | V5 direction | V4 simulation 80% coverage | V5 matched 80% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| india_equity | 1 | 36 | 1.29% | 1.31% | 1.87% | 2.02% | 61.11% | 55.56% | 77.78% | 77.78% |
| india_equity | 5 | 36 | 2.90% | 2.67% | 3.82% | 3.67% | 52.78% | 66.67% | 66.67% | 72.22% |
| india_equity | 10 | 36 | 4.38% | 4.07% | 5.87% | 5.76% | 44.44% | 47.22% | 69.44% | 75.00% |
| india_equity | 20 | 36 | 6.27% | 7.49% | 8.10% | 9.53% | 52.78% | 41.67% | 58.33% | 72.22% |
| us_equity | 1 | 36 | 1.21% | 1.16% | 1.50% | 1.48% | 52.78% | 47.22% | 83.33% | 80.56% |
| us_equity | 5 | 36 | 2.99% | 2.89% | 3.79% | 3.54% | 52.78% | 44.44% | 72.22% | 88.89% |
| us_equity | 10 | 36 | 4.31% | 4.55% | 5.32% | 5.29% | 58.33% | 33.33% | 66.67% | 77.78% |
| us_equity | 20 | 36 | 4.30% | 4.63% | 5.78% | 5.78% | 69.44% | 52.78% | 83.33% | 97.22% |

## Relative-return heads

Targets here are stock-minus-reference cumulative log returns. Sector availability is limited by the current mappings and source history; India Energy history was insufficient. Auxiliary calibration is independently governed. A high directional percentage may reflect an imbalanced label distribution; balanced accuracy is also reported.

| Family / sessions | Reference | N | MAE | RMSE | Return direction | Classifier direction | Classifier balanced accuracy | Brier | Rank IC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| india_equity_1 | market | 747 | 0.89% | 1.28% | 49.93% | 52.21% | 50.57% | 0.2488 | 0.0072 |
| india_equity_1 | sector | 498 | 0.65% | 0.87% | 51.00% | 53.01% | 50.92% | 0.2477 | 0.0391 |
| india_equity_5 | market | 735 | 1.98% | 2.72% | 48.71% | 52.24% | 51.59% | 0.2551 | 0.0020 |
| india_equity_5 | sector | 490 | 1.49% | 1.97% | 49.59% | 59.80% | 50.00% | 0.2431 | 0.0078 |
| india_equity_10 | market | 720 | 2.95% | 3.97% | 51.39% | 53.19% | 50.59% | 0.2490 | 0.0110 |
| india_equity_10 | sector | 480 | 2.24% | 2.90% | 50.42% | 63.33% | 50.00% | 0.2352 | 0.0130 |
| india_equity_20 | market | 690 | 4.23% | 5.67% | 53.48% | 57.68% | 53.17% | 0.2517 | 0.1642 |
| india_equity_20 | sector | 460 | 3.04% | 4.00% | 49.57% | 71.96% | 50.00% | 0.2179 | -0.0943 |
| us_equity_1 | market | 762 | 1.18% | 1.61% | 49.08% | 49.74% | 49.65% | 0.2534 | 0.0318 |
| us_equity_1 | sector | 762 | 0.92% | 1.35% | 50.13% | 52.36% | 51.97% | 0.2529 | -0.0220 |
| us_equity_5 | market | 750 | 2.74% | 3.60% | 52.93% | 45.33% | 49.22% | 0.2536 | 0.1047 |
| us_equity_5 | sector | 750 | 2.06% | 3.02% | 50.27% | 48.93% | 48.86% | 0.2585 | -0.0203 |
| us_equity_10 | market | 735 | 3.89% | 5.03% | 53.20% | 48.44% | 50.01% | 0.2593 | 0.0964 |
| us_equity_10 | sector | 735 | 2.79% | 3.99% | 46.53% | 49.66% | 49.73% | 0.2548 | -0.0757 |
| us_equity_20 | market | 705 | 5.75% | 7.29% | 44.82% | 47.38% | 45.69% | 0.2644 | -0.0277 |
| us_equity_20 | sector | 705 | 3.69% | 5.17% | 46.10% | 49.22% | 47.90% | 0.2960 | -0.1193 |

## Sector stability

| Family / sessions | Group | N | V5 MAE | Zero MAE | V4 compound MAE | V5 direction | V5 Brier | V5 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| india_equity_1 | Bank | 249 | 1.04% | 1.00% | 1.02% | 42.17% | 0.2761 | 79.12% |
| india_equity_1 | Energy | 249 | 1.01% | 1.00% | 1.02% | 48.59% | 0.2528 | 81.12% |
| india_equity_1 | IT | 249 | 1.32% | 1.29% | 1.35% | 51.41% | 0.2655 | 73.09% |
| india_equity_5 | Bank | 245 | 2.43% | 2.14% | 2.75% | 44.49% | 0.2732 | 86.12% |
| india_equity_5 | Energy | 245 | 2.34% | 2.24% | 2.49% | 53.88% | 0.2567 | 81.63% |
| india_equity_5 | IT | 245 | 3.18% | 2.91% | 3.36% | 43.67% | 0.2625 | 74.69% |
| india_equity_10 | Bank | 240 | 3.88% | 3.44% | 5.00% | 37.08% | 0.3834 | 83.75% |
| india_equity_10 | Energy | 240 | 3.24% | 3.09% | 3.88% | 51.25% | 0.3428 | 82.08% |
| india_equity_10 | IT | 240 | 5.04% | 4.44% | 5.26% | 46.67% | 0.3843 | 66.67% |
| india_equity_20 | Bank | 230 | 6.19% | 4.65% | 8.28% | 34.78% | 0.3513 | 83.48% |
| india_equity_20 | Energy | 230 | 5.01% | 4.00% | 6.37% | 38.70% | 0.3315 | 89.13% |
| india_equity_20 | IT | 230 | 7.78% | 6.90% | 9.49% | 49.57% | 0.3157 | 66.09% |
| us_equity_1 | Energy | 254 | 1.29% | 1.26% | 1.31% | 46.06% | 0.2475 | 80.31% |
| us_equity_1 | Financials | 254 | 1.08% | 1.08% | 1.10% | 50.00% | 0.2492 | 85.83% |
| us_equity_1 | Technology | 254 | 1.10% | 1.10% | 1.13% | 49.21% | 0.2502 | 83.46% |
| us_equity_5 | Energy | 250 | 3.02% | 2.97% | 3.51% | 54.40% | 0.2520 | 89.20% |
| us_equity_5 | Financials | 250 | 2.20% | 2.21% | 2.40% | 57.20% | 0.2463 | 95.60% |
| us_equity_5 | Technology | 250 | 2.78% | 2.84% | 3.18% | 54.40% | 0.2612 | 89.60% |
| us_equity_10 | Energy | 245 | 4.10% | 4.10% | 5.91% | 57.96% | 0.2454 | 78.37% |
| us_equity_10 | Financials | 245 | 3.42% | 3.28% | 3.89% | 48.16% | 0.2872 | 89.39% |
| us_equity_10 | Technology | 245 | 4.01% | 3.99% | 4.70% | 47.76% | 0.2662 | 77.96% |
| us_equity_20 | Energy | 235 | 6.46% | 6.45% | 9.82% | 53.62% | 0.2143 | 72.77% |
| us_equity_20 | Financials | 235 | 4.88% | 4.58% | 6.33% | 50.21% | 0.2592 | 88.51% |
| us_equity_20 | Technology | 235 | 5.95% | 5.94% | 7.86% | 53.62% | 0.2596 | 80.00% |

## Market regime stability

| Family / sessions | Group | N | V5 MAE | Zero MAE | V4 compound MAE | V5 direction | V5 Brier | V5 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| india_equity_1 | bearish_high_vol | 99 | 1.56% | 1.53% | 1.57% | 48.48% | 0.2617 | 71.72% |
| india_equity_1 | bearish_low_vol | 72 | 1.22% | 1.21% | 1.21% | 47.22% | 0.2612 | 75.00% |
| india_equity_1 | bullish_high_vol | 30 | 1.34% | 1.27% | 1.37% | 43.33% | 0.3140 | 66.67% |
| india_equity_1 | bullish_low_vol | 75 | 0.70% | 0.69% | 0.76% | 49.33% | 0.2479 | 86.67% |
| india_equity_1 | sideways_high_vol | 261 | 1.25% | 1.20% | 1.23% | 44.06% | 0.2673 | 74.71% |
| india_equity_1 | sideways_low_vol | 210 | 0.85% | 0.84% | 0.88% | 50.95% | 0.2633 | 83.81% |
| india_equity_5 | bearish_high_vol | 93 | 3.40% | 3.08% | 3.96% | 51.61% | 0.2638 | 67.74% |
| india_equity_5 | bearish_low_vol | 66 | 2.62% | 2.71% | 3.03% | 60.61% | 0.2341 | 71.21% |
| india_equity_5 | bullish_high_vol | 30 | 2.04% | 2.25% | 2.68% | 63.33% | 0.2164 | 90.00% |
| india_equity_5 | bullish_low_vol | 75 | 1.58% | 1.33% | 1.69% | 41.33% | 0.2666 | 96.00% |
| india_equity_5 | sideways_high_vol | 261 | 3.42% | 2.87% | 3.25% | 37.55% | 0.2999 | 74.71% |
| india_equity_5 | sideways_low_vol | 210 | 1.83% | 1.91% | 2.30% | 53.33% | 0.2351 | 90.48% |
| india_equity_10 | bearish_high_vol | 93 | 5.07% | 4.62% | 6.75% | 45.16% | 0.4269 | 73.12% |
| india_equity_10 | bearish_low_vol | 51 | 4.79% | 4.56% | 5.13% | 45.10% | 0.4020 | 62.75% |
| india_equity_10 | bullish_high_vol | 30 | 2.56% | 3.50% | 4.37% | 63.33% | 0.2486 | 86.67% |
| india_equity_10 | bullish_low_vol | 75 | 1.40% | 1.61% | 2.39% | 68.00% | 0.2097 | 100.00% |
| india_equity_10 | sideways_high_vol | 261 | 5.38% | 4.27% | 5.46% | 26.82% | 0.4787 | 67.43% |
| india_equity_10 | sideways_low_vol | 210 | 2.93% | 3.01% | 3.67% | 56.67% | 0.2771 | 86.19% |
| india_equity_20 | bearish_high_vol | 93 | 6.82% | 4.64% | 11.88% | 48.39% | 0.2911 | 92.47% |
| india_equity_20 | bearish_low_vol | 24 | 10.87% | 7.90% | 11.17% | 29.17% | 0.3324 | 62.50% |
| india_equity_20 | bullish_high_vol | 30 | 4.73% | 4.65% | 7.75% | 53.33% | 0.4187 | 86.67% |
| india_equity_20 | bullish_low_vol | 75 | 2.15% | 2.59% | 4.24% | 65.33% | 0.2015 | 100.00% |
| india_equity_20 | sideways_high_vol | 261 | 7.60% | 6.24% | 9.24% | 30.65% | 0.3682 | 67.43% |
| india_equity_20 | sideways_low_vol | 207 | 5.72% | 4.80% | 5.88% | 41.55% | 0.3421 | 82.61% |
| us_equity_1 | bearish_high_vol | 57 | 1.24% | 1.20% | 1.22% | 49.12% | 0.2421 | 82.46% |
| us_equity_1 | bearish_low_vol | 6 | 1.76% | 1.78% | 1.91% | 66.67% | 0.2520 | 83.33% |
| us_equity_1 | bullish_high_vol | 132 | 1.11% | 1.10% | 1.12% | 46.97% | 0.2455 | 84.85% |
| us_equity_1 | bullish_low_vol | 129 | 1.16% | 1.17% | 1.13% | 50.39% | 0.2578 | 84.50% |
| us_equity_1 | sideways_high_vol | 201 | 1.18% | 1.17% | 1.21% | 49.75% | 0.2514 | 84.58% |
| us_equity_1 | sideways_low_vol | 237 | 1.14% | 1.11% | 1.19% | 46.41% | 0.2456 | 80.59% |
| us_equity_5 | bearish_high_vol | 57 | 3.21% | 3.07% | 3.42% | 50.88% | 0.2701 | 92.98% |
| us_equity_5 | bearish_low_vol | 6 | 3.12% | 2.28% | 3.03% | 33.33% | 0.2921 | 100.00% |
| us_equity_5 | bullish_high_vol | 132 | 2.29% | 2.38% | 2.89% | 62.88% | 0.2396 | 95.45% |
| us_equity_5 | bullish_low_vol | 129 | 2.99% | 2.93% | 2.70% | 44.96% | 0.2727 | 88.37% |
| us_equity_5 | sideways_high_vol | 201 | 2.49% | 2.54% | 3.06% | 60.70% | 0.2496 | 95.02% |
| us_equity_5 | sideways_low_vol | 225 | 2.72% | 2.72% | 3.17% | 53.78% | 0.2477 | 87.11% |
| us_equity_10 | bearish_high_vol | 57 | 5.13% | 4.88% | 5.48% | 36.84% | 0.2354 | 78.95% |
| us_equity_10 | bearish_low_vol | 6 | 4.81% | 4.20% | 6.11% | 33.33% | 0.3028 | 66.67% |
| us_equity_10 | bullish_high_vol | 132 | 3.06% | 3.17% | 3.83% | 58.33% | 0.2642 | 88.64% |
| us_equity_10 | bullish_low_vol | 129 | 3.93% | 3.61% | 4.42% | 41.86% | 0.2907 | 83.72% |
| us_equity_10 | sideways_high_vol | 201 | 3.65% | 3.53% | 5.14% | 53.23% | 0.2540 | 84.58% |
| us_equity_10 | sideways_low_vol | 210 | 4.08% | 4.23% | 5.20% | 55.24% | 0.2716 | 75.24% |
| us_equity_20 | bearish_high_vol | 57 | 6.40% | 6.97% | 7.08% | 68.42% | 0.2160 | 89.47% |
| us_equity_20 | bearish_low_vol | 6 | 3.45% | 3.91% | 4.83% | 50.00% | 0.2598 | 100.00% |
| us_equity_20 | bullish_high_vol | 132 | 4.80% | 4.76% | 6.66% | 55.30% | 0.2451 | 89.39% |
| us_equity_20 | bullish_low_vol | 129 | 5.67% | 5.57% | 7.20% | 51.94% | 0.2467 | 74.42% |
| us_equity_20 | sideways_high_vol | 201 | 6.08% | 5.75% | 8.68% | 48.26% | 0.2377 | 78.11% |
| us_equity_20 | sideways_low_vol | 180 | 6.06% | 5.91% | 9.23% | 50.56% | 0.2581 | 77.22% |

## Year stability

| Family / sessions | Group | N | V5 MAE | Zero MAE | V4 compound MAE | V5 direction | V5 Brier | V5 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| india_equity_1 | 2025 | 186 | 0.69% | 0.67% | 0.73% | 50.00% | 0.2560 | 87.63% |
| india_equity_1 | 2026 | 561 | 1.27% | 1.24% | 1.27% | 46.52% | 0.2677 | 74.51% |
| india_equity_5 | 2025 | 186 | 1.64% | 1.47% | 1.85% | 41.94% | 0.2598 | 95.70% |
| india_equity_5 | 2026 | 549 | 2.99% | 2.75% | 3.21% | 49.18% | 0.2656 | 75.77% |
| india_equity_10 | 2025 | 186 | 2.06% | 2.26% | 3.05% | 62.37% | 0.2161 | 95.70% |
| india_equity_10 | 2026 | 534 | 4.75% | 4.14% | 5.30% | 38.95% | 0.4238 | 71.16% |
| india_equity_20 | 2025 | 186 | 3.65% | 3.61% | 4.99% | 60.22% | 0.2348 | 94.09% |
| india_equity_20 | 2026 | 504 | 7.32% | 5.77% | 9.18% | 33.93% | 0.3690 | 74.21% |
| us_equity_1 | 2025 | 192 | 0.93% | 0.92% | 0.92% | 49.48% | 0.2497 | 88.54% |
| us_equity_1 | 2026 | 570 | 1.24% | 1.22% | 1.27% | 48.07% | 0.2487 | 81.40% |
| us_equity_5 | 2025 | 192 | 2.11% | 2.14% | 2.20% | 56.77% | 0.2546 | 96.35% |
| us_equity_5 | 2026 | 558 | 2.86% | 2.85% | 3.32% | 54.84% | 0.2526 | 89.78% |
| us_equity_10 | 2025 | 192 | 2.60% | 2.85% | 3.55% | 61.46% | 0.2363 | 93.23% |
| us_equity_10 | 2026 | 543 | 4.28% | 4.12% | 5.29% | 47.70% | 0.2769 | 77.90% |
| us_equity_20 | 2025 | 192 | 3.80% | 4.39% | 6.25% | 70.31% | 0.2074 | 88.02% |
| us_equity_20 | 2026 | 513 | 6.50% | 6.13% | 8.66% | 45.81% | 0.2582 | 77.58% |

## Stock stability

| Family / sessions | Group | N | V5 MAE | Zero MAE | V4 compound MAE | V5 direction | V5 Brier | V5 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| india_equity_1 | HDFCBANK.NS | 249 | 1.04% | 1.00% | 1.02% | 42.17% | 0.2761 | 79.12% |
| india_equity_1 | RELIANCE.NS | 249 | 1.01% | 1.00% | 1.02% | 48.59% | 0.2528 | 81.12% |
| india_equity_1 | TCS.NS | 249 | 1.32% | 1.29% | 1.35% | 51.41% | 0.2655 | 73.09% |
| india_equity_5 | HDFCBANK.NS | 245 | 2.43% | 2.14% | 2.75% | 44.49% | 0.2732 | 86.12% |
| india_equity_5 | RELIANCE.NS | 245 | 2.34% | 2.24% | 2.49% | 53.88% | 0.2567 | 81.63% |
| india_equity_5 | TCS.NS | 245 | 3.18% | 2.91% | 3.36% | 43.67% | 0.2625 | 74.69% |
| india_equity_10 | HDFCBANK.NS | 240 | 3.88% | 3.44% | 5.00% | 37.08% | 0.3834 | 83.75% |
| india_equity_10 | RELIANCE.NS | 240 | 3.24% | 3.09% | 3.88% | 51.25% | 0.3428 | 82.08% |
| india_equity_10 | TCS.NS | 240 | 5.04% | 4.44% | 5.26% | 46.67% | 0.3843 | 66.67% |
| india_equity_20 | HDFCBANK.NS | 230 | 6.19% | 4.65% | 8.28% | 34.78% | 0.3513 | 83.48% |
| india_equity_20 | RELIANCE.NS | 230 | 5.01% | 4.00% | 6.37% | 38.70% | 0.3315 | 89.13% |
| india_equity_20 | TCS.NS | 230 | 7.78% | 6.90% | 9.49% | 49.57% | 0.3157 | 66.09% |
| us_equity_1 | AAPL | 254 | 1.10% | 1.10% | 1.13% | 49.21% | 0.2502 | 83.46% |
| us_equity_1 | JPM | 254 | 1.08% | 1.08% | 1.10% | 50.00% | 0.2492 | 85.83% |
| us_equity_1 | XOM | 254 | 1.29% | 1.26% | 1.31% | 46.06% | 0.2475 | 80.31% |
| us_equity_5 | AAPL | 250 | 2.78% | 2.84% | 3.18% | 54.40% | 0.2612 | 89.60% |
| us_equity_5 | JPM | 250 | 2.20% | 2.21% | 2.40% | 57.20% | 0.2463 | 95.60% |
| us_equity_5 | XOM | 250 | 3.02% | 2.97% | 3.51% | 54.40% | 0.2520 | 89.20% |
| us_equity_10 | AAPL | 245 | 4.01% | 3.99% | 4.70% | 47.76% | 0.2662 | 77.96% |
| us_equity_10 | JPM | 245 | 3.42% | 3.28% | 3.89% | 48.16% | 0.2872 | 89.39% |
| us_equity_10 | XOM | 245 | 4.10% | 4.10% | 5.91% | 57.96% | 0.2454 | 78.37% |
| us_equity_20 | AAPL | 235 | 5.95% | 5.94% | 7.86% | 53.62% | 0.2596 | 80.00% |
| us_equity_20 | JPM | 235 | 4.88% | 4.58% | 6.33% | 50.21% | 0.2592 | 88.51% |
| us_equity_20 | XOM | 235 | 6.46% | 6.45% | 9.82% | 53.62% | 0.2143 | 72.77% |

## Coverage, limitations and use

Model estimates are available for the eligible rows listed above. Actionable prediction coverage is 0%; research abstention rate is 100%. All research forecasts are LOW confidence. Twenty India row/horizon observations exceed the frozen OOD threshold; none do in the US cohort. That is a distance diagnostic, not proof of domain coverage. No live production track record exists.

The evidence supports software experimentation and auditing, not relying on these forecasts for trading or real-world financial decisions. The balanced improvement criterion is unmet. News/NLP expansion remains gated under spec §10: the core must first beat baselines. PIT certification, production promotion and browser visual/accessibility QA remain outstanding. Future experiments require a new untouched holdout; this period must not be reused for tuning.

Evidence: [machine comparison](validation/FINAL_COMPARISON.json), [row-level predictions](validation/FINAL_COMPARISON.csv), [protocol lock](validation/FINAL_PROTOCOL_LOCK.json), [consumed marker](validation/FINAL_TEST_CONSUMED.json), [original V4 diagnostic](baseline/V4_BASELINE.md).

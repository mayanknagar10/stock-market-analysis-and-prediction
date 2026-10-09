# V5 failure analysis

Experiment development-20261008T211418-e9aa7e21; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261008T211418-e9aa7e21/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

Only frozen DEVELOPMENT.json and its predictions.csv were inspected after verifying every feature/label date preboundary. Native V5 validation (2025-07 onward) is not directly paired with this broader four-fold study. No V5 final report or final CSV was opened/rescored.

| family | horizon | n | mae | zero_mae | improvement_zero | ic | direction_accuracy | brier | coverage_50 | coverage_80 | training_stocks | features | train_metrics_available | calibration_accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| india_equity | 1 | 189 | 0.007495 | 0.007124 | -0.051989 | -0.054622 | 0.439153 | 0.252737 | 0.587302 | 0.857143 | 3 | 110 | False | False |
| india_equity | 5 | 189 | 0.019256 | 0.018355 | -0.049102 | 0.246220 | 0.518519 | 0.261940 | 0.608466 | 0.910053 | 3 | 119 | False | True |
| india_equity | 10 | 189 | 0.029634 | 0.026040 | -0.138000 | 0.229265 | 0.449735 | 0.307301 | 0.449735 | 0.873016 | 3 | 119 | False | False |
| india_equity | 20 | 189 | 0.053051 | 0.036645 | -0.447722 | -0.087968 | 0.391534 | 0.321869 | 0.407407 | 0.867725 | 3 | 110 | False | True |
| us_equity | 1 | 189 | 0.009620 | 0.009572 | -0.004958 | -0.018580 | 0.502646 | 0.245482 | 0.534392 | 0.883598 | 3 | 64 | False | True |
| us_equity | 5 | 189 | 0.024260 | 0.025274 | 0.040119 | -0.004195 | 0.613757 | 0.234176 | 0.523810 | 0.915344 | 3 | 119 | False | False |
| us_equity | 10 | 189 | 0.029475 | 0.032628 | 0.096630 | 0.130652 | 0.624339 | 0.228761 | 0.571429 | 0.862434 | 3 | 119 | False | False |
| us_equity | 20 | 189 | 0.039779 | 0.048873 | 0.186079 | 0.162041 | 0.772487 | 0.193201 | 0.539683 | 0.947090 | 3 | 127 | False | False |

Measured: three stocks per exchange family, many correlated inputs relative to independent dates, horizon-varying IC/direction/error/interval coverage and calibration acceptance. Actual native V5 in-sample train metrics are absent; no native train replay was performed. V5 train/validation overfit magnitude is therefore UNMEASURED. The new study measures its own gaps, not V5 train errors.

Plausible hypotheses, not causal findings: correlated indicators encourage unstable partitions/coefficients; three-stock sector coverage confuses common exposure with transferable alpha; revised histories/current membership inflate replay confidence; overlapping labels/small date-block support weaken calibration evidence. The new measured ablations, sector holdouts, target experiments and gaps test these directions without consumed final outcomes. Historical final gate facts are restricted to published 1/0/2/7/3; they did not select this architecture. No final degradation scalar was required.

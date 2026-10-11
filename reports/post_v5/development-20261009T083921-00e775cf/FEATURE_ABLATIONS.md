# Feature, breadth and stability analysis

Experiment development-20261009T083921-00e775cf; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261009T083921-00e775cf/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

| horizon | model | n | mae | rmse | ic | date_rank_ic | nonoverlap_blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | commodity_only | 10774 | 0.014991 | 0.021186 | 0.007199 | -0.007522 | 248 |
| 1 | decomposition | 9630 | 0.014940 | 0.021190 | 0.005165 | -0.000736 | 248 |
| 1 | fx_only | 10774 | 0.014984 | 0.021191 | -0.056173 | -0.007522 | 248 |
| 1 | global_only | 10774 | 0.015196 | 0.021300 | 0.000284 | 0.007522 | 248 |
| 1 | market_only | 10774 | 0.014946 | 0.021143 | -0.007723 | 0.007522 | 248 |
| 1 | reduced_stable | 10774 | 0.014761 | 0.020983 | 0.069775 | 0.030377 | 248 |
| 1 | ridge | 10774 | 0.016288 | 0.022357 | 0.021744 | 0.015178 | 248 |
| 1 | sector_heldout | 10774 | 0.016310 | 0.022370 | 0.020995 | 0.009794 | 248 |
| 1 | sector_interactions | 10774 | 0.016449 | 0.022521 | 0.015989 | -0.014330 | 248 |
| 1 | sector_only | 10774 | 0.014906 | 0.021113 | -0.056191 | -0.027437 | 248 |
| 1 | sector_specific | 9474 | 0.018549 | 0.024997 | 0.012674 | -0.001284 | 248 |
| 1 | technical_only | 10774 | 0.014985 | 0.021188 | 0.022797 | 0.015424 | 248 |
| 1 | without_commodity | 10774 | 0.015487 | 0.021586 | 0.012740 | 0.015416 | 248 |
| 1 | without_fx | 10774 | 0.016410 | 0.022498 | 0.025431 | 0.017157 | 248 |
| 1 | without_global | 10774 | 0.015048 | 0.021245 | 0.000951 | 0.016869 | 248 |
| 1 | without_market | 10774 | 0.015790 | 0.021924 | -0.003739 | 0.010243 | 248 |
| 1 | without_sector | 10774 | 0.016294 | 0.022360 | 0.022365 | 0.020020 | 248 |
| 1 | without_technical | 10774 | 0.016172 | 0.022212 | 0.023525 | 0.019495 | 248 |
| 5 | commodity_only | 10386 | 0.035874 | 0.049014 | 0.174655 | -0.116597 | 49 |
| 5 | decomposition | 9282 | 0.036184 | 0.049635 | 0.078786 | -0.021933 | 49 |
| 5 | fx_only | 10386 | 0.036085 | 0.049052 | -0.118293 | — | 49 |
| 5 | global_only | 10386 | 0.035862 | 0.049114 | 0.098403 | 0.116597 | 49 |
| 5 | market_only | 10386 | 0.036140 | 0.049320 | 0.054805 | — | 49 |
| 5 | reduced_stable | 10386 | 0.035667 | 0.048919 | 0.118306 | -0.025747 | 49 |
| 5 | ridge | 10386 | 0.050208 | 0.064619 | 0.098071 | 0.023626 | 49 |
| 5 | sector_heldout | 10386 | 0.050218 | 0.064580 | 0.091977 | 0.012280 | 49 |
| 5 | sector_interactions | 10386 | 0.051109 | 0.065812 | 0.087867 | -0.019603 | 49 |
| 5 | sector_only | 10386 | 0.035560 | 0.048705 | 0.001158 | -0.062731 | 49 |
| 5 | sector_specific | 9126 | 0.058458 | 0.074761 | 0.076450 | 0.000905 | 49 |
| 5 | technical_only | 10386 | 0.036350 | 0.049400 | 0.047639 | 0.007112 | 49 |
| 5 | without_commodity | 10386 | 0.041397 | 0.054332 | 0.131360 | 0.019266 | 49 |
| 5 | without_fx | 10386 | 0.048108 | 0.062509 | 0.104607 | 0.022780 | 49 |
| 5 | without_global | 10386 | 0.036073 | 0.048955 | 0.089939 | 0.011998 | 49 |
| 5 | without_market | 10386 | 0.044521 | 0.058178 | 0.057766 | 0.008564 | 49 |
| 5 | without_sector | 10386 | 0.050123 | 0.064510 | 0.099284 | 0.030410 | 49 |
| 5 | without_technical | 10386 | 0.048749 | 0.062936 | 0.110319 | 0.032604 | 49 |
| 10 | commodity_only | 9901 | 0.052400 | 0.067848 | 0.115307 | -0.124120 | 24 |
| 10 | decomposition | 8847 | 0.052769 | 0.068930 | -0.006253 | -0.021348 | 24 |
| 10 | fx_only | 9901 | 0.050605 | 0.065605 | -0.155041 | -0.124120 | 24 |
| 10 | global_only | 9901 | 0.051056 | 0.066985 | 0.063014 | -0.124120 | 24 |
| 10 | market_only | 9901 | 0.052222 | 0.067742 | 0.085241 | — | 24 |
| 10 | reduced_stable | 9901 | 0.054345 | 0.070532 | 0.142300 | -0.090470 | 24 |
| 10 | ridge | 9901 | 0.079967 | 0.099379 | 0.085186 | 0.026218 | 24 |
| 10 | sector_heldout | 9901 | 0.079796 | 0.099214 | 0.085710 | 0.020361 | 24 |
| 10 | sector_interactions | 9901 | 0.081958 | 0.101790 | 0.063328 | -0.034844 | 24 |
| 10 | sector_only | 9901 | 0.050685 | 0.065917 | -0.062589 | -0.040996 | 24 |
| 10 | sector_specific | 8691 | 0.095581 | 0.120110 | 0.017783 | -0.028874 | 24 |
| 10 | technical_only | 9901 | 0.052540 | 0.068455 | 0.023165 | 0.000411 | 24 |
| 10 | without_commodity | 9901 | 0.061008 | 0.078138 | 0.096619 | 0.009353 | 24 |
| 10 | without_fx | 9901 | 0.081459 | 0.101381 | 0.088737 | 0.019174 | 24 |
| 10 | without_global | 9901 | 0.053513 | 0.069461 | 0.040370 | -0.009878 | 24 |
| 10 | without_market | 9901 | 0.058638 | 0.075886 | 0.043675 | 0.019104 | 24 |
| 10 | without_sector | 9901 | 0.079880 | 0.099239 | 0.088022 | 0.031739 | 24 |
| 10 | without_technical | 9901 | 0.075834 | 0.094351 | 0.112229 | 0.007787 | 24 |
| 20 | commodity_only | 8931 | 0.081243 | 0.102706 | -0.101323 | -0.018806 | 12 |
| 20 | decomposition | 7977 | 0.077945 | 0.099997 | -0.069807 | -0.066602 | 12 |
| 20 | fx_only | 8931 | 0.072232 | 0.091987 | -0.172163 | 0.018806 | 12 |
| 20 | global_only | 8931 | 0.066830 | 0.085756 | 0.138486 | — | 12 |
| 20 | market_only | 8931 | 0.078534 | 0.099509 | 0.104378 | — | 12 |
| 20 | reduced_stable | 8931 | 0.088081 | 0.110552 | 0.052546 | -0.150097 | 12 |
| 20 | ridge | 8931 | 0.082113 | 0.104133 | 0.063738 | 0.026338 | 12 |
| 20 | sector_heldout | 8931 | 0.082950 | 0.104885 | 0.060423 | 0.013951 | 12 |
| 20 | sector_interactions | 8931 | 0.085222 | 0.108138 | 0.019322 | -0.051745 | 12 |
| 20 | sector_only | 8931 | 0.073267 | 0.093829 | -0.061565 | 0.037516 | 12 |
| 20 | sector_specific | 7536 | 0.108192 | 0.138287 | -0.017818 | -0.015423 | 12 |
| 20 | technical_only | 8931 | 0.077724 | 0.098936 | 0.043663 | 0.037844 | 12 |
| 20 | without_commodity | 8931 | 0.067957 | 0.087706 | 0.019713 | 0.017541 | 12 |
| 20 | without_fx | 8931 | 0.111114 | 0.133174 | 0.151704 | 0.021115 | 12 |
| 20 | without_global | 8931 | 0.084283 | 0.106311 | -0.184685 | 0.018190 | 12 |
| 20 | without_market | 8931 | 0.069117 | 0.087670 | 0.015153 | 0.033161 | 12 |
| 20 | without_sector | 8931 | 0.082388 | 0.104426 | 0.061275 | 0.007557 | 12 |
| 20 | without_technical | 8931 | 0.081093 | 0.102945 | 0.089974 | 0.006911 | 12 |

Groups: 56 technical stock ratios; domestic market; sector; global equity/volatility/yield; FX USD/INR/DXY; commodities Brent/WTI/gold. Group-only and removed-group ridge share origins. Train-only median imputation excludes all-missing/constant columns (FIT_AUDITS). Context availability is estimated following UTC day, not PIT-certified. For absent official sectors use equal-weight fixed-universe industry DAILY RETURNS excluding forecast stock with minimum2 observed peers. This is NOT an official sector index; survivorship bias remains. Failed/sparse study members remain in ELIGIBILITY.

Reduced train-stable set: train variance filter, greedy absolute training Pearson correlation >.95 pruning; consistent Spearman sign in chronological training halves, minimum half absolute IC .01; cap24 by training strength, fall back first5 decorrelated features if none. No validation selection. Train-only identity vocabularies; sector-specific support >=2000 train rows/252 dates/2stocks.

Paired supported-sector versus pooled:

| horizon | model | baseline | n | mae_improvement |
| --- | --- | --- | --- | --- |
| 1 | sector_specific | ridge_paired_supported_industries | 9474 | -0.139834 |
| 5 | sector_specific | ridge_paired_supported_industries | 9126 | -0.175235 |
| 10 | sector_specific | ridge_paired_supported_industries | 8691 | -0.209984 |
| 20 | sector_specific | ridge_paired_supported_industries | 7536 | -0.328282 |

Unsupported industries/folds:

| horizon | fold | industry | train_n | train_dates | train_stocks | test_n | supported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | Chemicals | 1368 | 684 | 2 | 124 | False |
| 1 | 1 | Construction | 684 | 684 | 1 | 62 | False |
| 1 | 1 | Consumer Durables | 1368 | 684 | 2 | 124 | False |
| 1 | 1 | Consumer Services | 1877 | 684 | 3 | 160 | False |
| 1 | 1 | Realty | 286 | 286 | 1 | 62 | False |
| 1 | 1 | Services | 1368 | 684 | 2 | 124 | False |
| 1 | 1 | Telecommunication | 1368 | 684 | 2 | 124 | False |
| 1 | 2 | Chemicals | 1492 | 746 | 2 | 104 | False |
| 1 | 2 | Construction | 746 | 746 | 1 | 52 | False |
| 1 | 2 | Consumer Durables | 1492 | 746 | 2 | 104 | False |
| 1 | 2 | Realty | 348 | 348 | 1 | 52 | False |
| 1 | 2 | Services | 1492 | 746 | 2 | 104 | False |
| 1 | 2 | Telecommunication | 1492 | 746 | 2 | 104 | False |
| 5 | 1 | Chemicals | 1360 | 680 | 2 | 124 | False |
| 5 | 1 | Construction | 680 | 680 | 1 | 62 | False |
| 5 | 1 | Consumer Durables | 1360 | 680 | 2 | 124 | False |
| 5 | 1 | Consumer Services | 1865 | 680 | 3 | 160 | False |
| 5 | 1 | Realty | 278 | 278 | 1 | 62 | False |
| 5 | 1 | Services | 1360 | 680 | 2 | 124 | False |
| 5 | 1 | Telecommunication | 1360 | 680 | 2 | 124 | False |
| 5 | 2 | Chemicals | 1484 | 742 | 2 | 96 | False |
| 5 | 2 | Construction | 742 | 742 | 1 | 48 | False |
| 5 | 2 | Consumer Durables | 1484 | 742 | 2 | 96 | False |
| 5 | 2 | Realty | 340 | 340 | 1 | 48 | False |
| 5 | 2 | Services | 1484 | 742 | 2 | 96 | False |
| 5 | 2 | Telecommunication | 1484 | 742 | 2 | 96 | False |
| 10 | 1 | Chemicals | 1350 | 675 | 2 | 124 | False |
| 10 | 1 | Construction | 675 | 675 | 1 | 62 | False |
| 10 | 1 | Consumer Durables | 1350 | 675 | 2 | 124 | False |
| 10 | 1 | Consumer Services | 1850 | 675 | 3 | 160 | False |
| 10 | 1 | Realty | 268 | 268 | 1 | 62 | False |
| 10 | 1 | Services | 1350 | 675 | 2 | 124 | False |
| 10 | 1 | Telecommunication | 1350 | 675 | 2 | 124 | False |
| 10 | 2 | Chemicals | 1474 | 737 | 2 | 86 | False |
| 10 | 2 | Construction | 737 | 737 | 1 | 43 | False |
| 10 | 2 | Consumer Durables | 1474 | 737 | 2 | 86 | False |
| 10 | 2 | Realty | 330 | 330 | 1 | 43 | False |
| 10 | 2 | Services | 1474 | 737 | 2 | 86 | False |
| 10 | 2 | Telecommunication | 1474 | 737 | 2 | 86 | False |
| 20 | 1 | Chemicals | 1330 | 665 | 2 | 124 | False |
| 20 | 1 | Construction | 665 | 665 | 1 | 62 | False |
| 20 | 1 | Construction Materials | 1992 | 665 | 3 | 186 | False |
| 20 | 1 | Consumer Durables | 1330 | 665 | 2 | 124 | False |
| 20 | 1 | Consumer Services | 1820 | 665 | 3 | 160 | False |
| 20 | 1 | Realty | 257 | 257 | 1 | 62 | False |
| 20 | 1 | Services | 1330 | 665 | 2 | 124 | False |
| 20 | 1 | Telecommunication | 1330 | 665 | 2 | 124 | False |
| 20 | 2 | Chemicals | 1454 | 727 | 2 | 66 | False |
| 20 | 2 | Construction | 727 | 727 | 1 | 33 | False |
| 20 | 2 | Consumer Durables | 1454 | 727 | 2 | 66 | False |
| 20 | 2 | Consumer Services | 1960 | 727 | 3 | 99 | False |
| 20 | 2 | Realty | 319 | 319 | 1 | 33 | False |
| 20 | 2 | Services | 1454 | 727 | 2 | 66 | False |
| 20 | 2 | Telecommunication | 1454 | 727 | 2 | 66 | False |

Leave-industry-out evaluates every industry with all of its training stocks excluded. It supplements temporal checks; it is not prospective evidence.

Measured train/validation degradation:

| horizon | fold | model | train_mae | validation_mae | ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | ridge | 0.013310 | 0.015664 | 1.176865 |
| 1 | 1 | elastic_net | 0.013456 | 0.013828 | 1.027649 |
| 1 | 1 | shallow_boost | 0.013321 | 0.013993 | 1.050501 |
| 1 | 1 | reduced_stable | 0.013446 | 0.013815 | 1.027481 |
| 1 | 1 | sector_interactions | 0.013305 | 0.015822 | 1.189199 |
| 1 | 2 | ridge | 0.013357 | 0.017121 | 1.281800 |
| 1 | 2 | elastic_net | 0.013486 | 0.016123 | 1.195525 |
| 1 | 2 | shallow_boost | 0.013367 | 0.016221 | 1.213583 |
| 1 | 2 | reduced_stable | 0.013477 | 0.016024 | 1.188926 |
| 1 | 2 | sector_interactions | 0.013354 | 0.017295 | 1.295172 |
| 1 | 3 | ridge | 0.013538 | 0.011664 | 0.861574 |
| 1 | 3 | elastic_net | 0.013658 | 0.007791 | 0.570434 |
| 1 | 3 | shallow_boost | 0.013541 | 0.007878 | 0.581834 |
| 1 | 3 | reduced_stable | 0.013642 | 0.007766 | 0.569292 |
| 1 | 3 | sector_interactions | 0.013536 | 0.011306 | 0.835202 |
| 1 | 4 | ridge | 0.013534 | 0.011846 | 0.875219 |
| 1 | 4 | elastic_net | 0.013653 | 0.008164 | 0.597987 |
| 1 | 4 | shallow_boost | 0.013536 | 0.008075 | 0.596531 |
| 1 | 4 | reduced_stable | 0.013637 | 0.008099 | 0.593882 |
| 1 | 4 | sector_interactions | 0.013532 | 0.011753 | 0.868523 |
| 5 | 1 | ridge | 0.029570 | 0.062180 | 2.102806 |
| 5 | 1 | elastic_net | 0.030195 | 0.035662 | 1.181061 |
| 5 | 1 | shallow_boost | 0.030352 | 0.034727 | 1.144139 |
| 5 | 1 | reduced_stable | 0.030595 | 0.036138 | 1.181176 |
| 5 | 1 | sector_interactions | 0.029500 | 0.063669 | 2.158242 |
| 5 | 2 | ridge | 0.029933 | 0.036086 | 1.205557 |
| 5 | 2 | elastic_net | 0.030516 | 0.035838 | 1.174412 |
| 5 | 2 | shallow_boost | 0.030716 | 0.037011 | 1.204929 |
| 5 | 2 | reduced_stable | 0.031047 | 0.035716 | 1.150376 |
| 5 | 2 | sector_interactions | 0.029875 | 0.036329 | 1.216017 |
| 5 | 3 | ridge | 0.030211 | 0.037099 | 1.227996 |
| 5 | 3 | elastic_net | 0.030767 | 0.016329 | 0.530741 |
| 5 | 3 | shallow_boost | 0.031029 | 0.013020 | 0.419594 |
| 5 | 3 | reduced_stable | 0.031268 | 0.013230 | 0.423127 |
| 5 | 3 | sector_interactions | 0.030172 | 0.035526 | 1.177461 |
| 5 | 4 | ridge | 0.030208 | 0.032513 | 1.076321 |
| 5 | 4 | elastic_net | 0.030756 | 0.017338 | 0.563731 |
| 5 | 4 | shallow_boost | 0.031010 | 0.015429 | 0.497546 |
| 5 | 4 | reduced_stable | 0.031252 | 0.015811 | 0.505934 |
| 5 | 4 | sector_interactions | 0.030168 | 0.031910 | 1.057749 |
| 10 | 1 | ridge | 0.041304 | 0.105372 | 2.551111 |
| 10 | 1 | elastic_net | 0.041947 | 0.065932 | 1.571812 |
| 10 | 1 | shallow_boost | 0.042887 | 0.053193 | 1.240298 |
| 10 | 1 | reduced_stable | 0.043149 | 0.060078 | 1.392331 |
| 10 | 1 | sector_interactions | 0.041174 | 0.107465 | 2.610003 |
| 10 | 2 | ridge | 0.041801 | 0.045929 | 1.098745 |
| 10 | 2 | elastic_net | 0.042320 | 0.045560 | 1.076558 |
| 10 | 2 | shallow_boost | 0.043639 | 0.050142 | 1.149013 |
| 10 | 2 | reduced_stable | 0.043602 | 0.047658 | 1.093040 |
| 10 | 2 | sector_interactions | 0.041693 | 0.047892 | 1.148692 |
| 10 | 3 | ridge | 0.041908 | 0.079368 | 1.893847 |
| 10 | 3 | elastic_net | 0.042395 | 0.025094 | 0.591910 |
| 10 | 3 | shallow_boost | 0.043701 | 0.019841 | 0.454013 |
| 10 | 3 | reduced_stable | 0.043821 | 0.016694 | 0.380950 |
| 10 | 3 | sector_interactions | 0.041859 | 0.076570 | 1.829243 |
| 10 | 4 | ridge | 0.041923 | 0.051429 | 1.226757 |
| 10 | 4 | elastic_net | 0.042390 | 0.025528 | 0.602215 |
| 10 | 4 | shallow_boost | 0.043695 | 0.019727 | 0.451485 |
| 10 | 4 | reduced_stable | 0.043775 | 0.020116 | 0.459533 |
| 10 | 4 | sector_interactions | 0.041873 | 0.050972 | 1.217297 |
| 20 | 1 | ridge | 0.059103 | 0.077275 | 1.307467 |
| 20 | 1 | elastic_net | 0.059749 | 0.088107 | 1.474613 |
| 20 | 1 | shallow_boost | 0.061343 | 0.075735 | 1.234610 |
| 20 | 1 | reduced_stable | 0.063114 | 0.099556 | 1.577399 |
| 20 | 1 | sector_interactions | 0.058777 | 0.080721 | 1.373354 |
| 20 | 2 | ridge | 0.059221 | 0.089056 | 1.503792 |
| 20 | 2 | elastic_net | 0.059723 | 0.078015 | 1.306274 |
| 20 | 2 | shallow_boost | 0.061894 | 0.080642 | 1.302901 |
| 20 | 2 | reduced_stable | 0.062324 | 0.069542 | 1.115812 |
| 20 | 2 | sector_interactions | 0.058946 | 0.091789 | 1.557186 |
| 20 | 3 | ridge | 0.059481 | 0.150338 | 2.527500 |
| 20 | 3 | elastic_net | 0.060026 | 0.043876 | 0.730959 |
| 20 | 3 | shallow_boost | 0.062509 | 0.031025 | 0.496329 |
| 20 | 3 | reduced_stable | 0.062944 | 0.043981 | 0.698722 |
| 20 | 3 | sector_interactions | 0.059317 | 0.147110 | 2.480084 |
| 20 | 4 | ridge | 0.059484 | 0.101435 | 1.705252 |
| 20 | 4 | elastic_net | 0.060024 | 0.054301 | 0.904658 |
| 20 | 4 | shallow_boost | 0.062472 | 0.016290 | 0.260762 |
| 20 | 4 | reduced_stable | 0.062930 | 0.030032 | 0.477225 |
| 20 | 4 | sector_interactions | 0.059322 | 0.097807 | 1.648748 |

Exploratory validation IC leaders (never used to reselect):

| horizon | feature | train_ic | validation_ic | train_residual_ic | validation_residual_ic | degradation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | nasdaq_vol_20 | -0.019640 | 0.087702 | -0.036713 | 0.034488 | -0.068061 |
| 1 | dxy_ret_1d | 0.000098 | 0.083676 | 0.000902 | 0.080341 | -0.083578 |
| 1 | dxy_vol_20 | 0.004806 | 0.069365 | 0.006668 | 0.042795 | -0.064559 |
| 1 | india_vix_drawdown | 0.038911 | 0.066691 | 0.042428 | 0.042008 | -0.027780 |
| 1 | india_vix_ret_5d | -0.003968 | 0.064190 | 0.001262 | 0.047121 | -0.060222 |
| 1 | dxy_drawdown | -0.008979 | 0.063258 | -0.013751 | 0.013089 | -0.054279 |
| 1 | us_yield_vol_20 | 0.024071 | 0.062437 | 0.021609 | -0.021337 | -0.038366 |
| 1 | nikkei_vol_20 | 0.004344 | 0.060100 | 0.014892 | 0.018263 | -0.055756 |
| 5 | nasdaq_vol_20 | -0.033228 | 0.181517 | -0.065179 | 0.112869 | -0.148289 |
| 5 | us_yield_vol_20 | 0.050927 | 0.154233 | 0.052825 | -0.067102 | -0.103306 |
| 5 | nikkei_vol_20 | 0.010950 | 0.137040 | 0.038587 | 0.094344 | -0.126090 |
| 5 | india_vix_ret_20d | 0.036233 | 0.133512 | 0.052958 | 0.097849 | -0.097279 |
| 5 | vix_vol_20 | -0.071076 | 0.131815 | -0.090247 | 0.112711 | -0.060739 |
| 5 | rolling_market_beta | -0.003737 | 0.126172 | 0.000235 | -0.086941 | -0.122436 |
| 5 | atr_pct | 0.028248 | 0.125577 | 0.027428 | 0.065586 | -0.097329 |
| 5 | dxy_vol_20 | 0.027687 | 0.123163 | 0.034711 | 0.079784 | -0.095476 |
| 10 | nasdaq_vol_20 | -0.040835 | 0.216868 | -0.079572 | 0.135893 | -0.176033 |
| 10 | india_vix_trend_50 | 0.086691 | 0.191638 | 0.115022 | 0.154859 | -0.104947 |
| 10 | rolling_market_beta | 0.010142 | 0.191529 | 0.003058 | -0.211414 | -0.181387 |
| 10 | vix_vol_20 | -0.076000 | 0.190869 | -0.095247 | 0.198027 | -0.114869 |
| 10 | hl_spread | 0.027670 | 0.185861 | 0.025526 | 0.144773 | -0.158191 |
| 10 | rolling_market_correlation | 0.007229 | 0.177889 | -0.008645 | 0.187397 | -0.170660 |
| 10 | us_yield_vol_20 | 0.068063 | 0.171871 | 0.063336 | -0.007592 | -0.103808 |
| 10 | india_vix_ret_20d | 0.054154 | 0.153422 | 0.083106 | 0.168102 | -0.099268 |
| 20 | vix_vol_20 | -0.084176 | 0.362770 | -0.121891 | 0.262638 | -0.278594 |
| 20 | nasdaq_vol_20 | -0.070170 | 0.300418 | -0.118154 | 0.260082 | -0.230248 |
| 20 | india_vix_ret_20d | 0.057789 | 0.281203 | 0.070680 | 0.264006 | -0.223414 |
| 20 | rolling_market_beta | 0.021529 | 0.246563 | -0.001403 | -0.320431 | -0.225034 |
| 20 | rolling_market_correlation | 0.018007 | 0.244302 | -0.006857 | 0.273628 | -0.226295 |
| 20 | india_vix_trend_50 | 0.085327 | 0.241735 | 0.112676 | 0.218637 | -0.156409 |
| 20 | us_yield_vol_20 | 0.047426 | 0.225883 | 0.035153 | -0.001727 | -0.178457 |
| 20 | nikkei_vol_20 | 0.002457 | 0.221631 | 0.034733 | 0.226592 | -0.219174 |

Residual IC uses train-fitted market-feature ridge residualization of both signal and target. Full IC in SIGNALS; sample/date/features in ATTEMPTS/FIT_AUDITS; strata in STRATA. Regime rules: past market trend +/-2% bull/bear/range, past rolling volatility median high/low, |past stock ret20|>2% trend/range. Stocks/industries/folds/horizons/regimes reported separately. Complexity sensitivity is fixed linear versus depth2/50 boosting and full versus reduced features; no depth search. Correlated stocks do not multiply effective dates.

# Feature, breadth and stability analysis

Experiment development-20261008T211418-e9aa7e21; input capture f0805fb58dbd033fc42f7f3dafdf1c1315a69c0e039d3e1ae63ace8abe865672. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in development-20261008T211418-e9aa7e21/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.

| horizon | model | n | mae | rmse | ic | date_rank_ic | nonoverlap_blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | commodity_only | 21829 | 0.013193 | 0.018831 | 0.021250 | -0.030089 | 248 |
| 1 | decomposition | 19545 | 0.013153 | 0.018841 | 0.002270 | 0.008211 | 248 |
| 1 | fx_only | 21829 | 0.013301 | 0.018903 | 0.034316 | 0.016184 | 248 |
| 1 | global_only | 21829 | 0.013276 | 0.018917 | 0.034923 | -0.032712 | 248 |
| 1 | market_only | 21829 | 0.013141 | 0.018797 | -0.001900 | 0.025189 | 248 |
| 1 | reduced_stable | 21829 | 0.013050 | 0.018707 | 0.092797 | -0.000066 | 248 |
| 1 | ridge | 21829 | 0.014854 | 0.020268 | 0.065104 | 0.010814 | 248 |
| 1 | sector_heldout | 21829 | 0.014866 | 0.020271 | 0.062484 | 0.002525 | 248 |
| 1 | sector_interactions | 21829 | 0.014901 | 0.020324 | 0.061664 | 0.001210 | 248 |
| 1 | sector_only | 21829 | 0.013136 | 0.018795 | -0.030189 | -0.012103 | 248 |
| 1 | sector_specific | 19549 | 0.016688 | 0.022581 | 0.023546 | 0.006161 | 248 |
| 1 | technical_only | 21829 | 0.013197 | 0.018856 | -0.002078 | 0.007097 | 248 |
| 1 | without_commodity | 21829 | 0.014186 | 0.019633 | 0.075697 | 0.008651 | 248 |
| 1 | without_fx | 21829 | 0.014149 | 0.019823 | -0.004619 | 0.010701 | 248 |
| 1 | without_global | 21829 | 0.013765 | 0.019217 | 0.079379 | 0.008098 | 248 |
| 1 | without_market | 21829 | 0.014446 | 0.019939 | 0.065593 | 0.007581 | 248 |
| 1 | without_sector | 21829 | 0.014863 | 0.020275 | 0.065345 | 0.012016 | 248 |
| 1 | without_technical | 21829 | 0.014384 | 0.019837 | 0.060491 | 0.011080 | 248 |
| 5 | commodity_only | 21445 | 0.031866 | 0.043390 | 0.093713 | 0.055371 | 49 |
| 5 | decomposition | 19201 | 0.031518 | 0.043340 | 0.028230 | -0.002415 | 49 |
| 5 | fx_only | 21445 | 0.032662 | 0.043884 | 0.058552 | — | 49 |
| 5 | global_only | 21445 | 0.031999 | 0.043889 | 0.029892 | -0.064726 | 49 |
| 5 | market_only | 21445 | 0.031485 | 0.043146 | -0.000975 | 0.039968 | 49 |
| 5 | reduced_stable | 21445 | 0.031593 | 0.043255 | 0.102039 | -0.004285 | 49 |
| 5 | ridge | 21445 | 0.044974 | 0.057561 | 0.066042 | 0.019266 | 49 |
| 5 | sector_heldout | 21445 | 0.045043 | 0.057580 | 0.063397 | 0.002082 | 49 |
| 5 | sector_interactions | 21445 | 0.045302 | 0.058129 | 0.060779 | -0.002270 | 49 |
| 5 | sector_only | 21445 | 0.031326 | 0.042940 | -0.001438 | -0.021661 | 49 |
| 5 | sector_specific | 19205 | 0.051701 | 0.067378 | 0.040005 | 0.026786 | 49 |
| 5 | technical_only | 21445 | 0.031880 | 0.043514 | -0.029383 | -0.001414 | 49 |
| 5 | without_commodity | 21445 | 0.038656 | 0.050209 | 0.134544 | 0.015871 | 49 |
| 5 | without_fx | 21445 | 0.040465 | 0.053942 | -0.016048 | 0.017709 | 49 |
| 5 | without_global | 21445 | 0.037937 | 0.048412 | 0.146497 | 0.004256 | 49 |
| 5 | without_market | 21445 | 0.040945 | 0.053010 | 0.079924 | 0.012552 | 49 |
| 5 | without_sector | 21445 | 0.044982 | 0.057552 | 0.066888 | 0.021014 | 49 |
| 5 | without_technical | 21445 | 0.043498 | 0.055881 | 0.061918 | 0.021881 | 49 |
| 10 | commodity_only | 20965 | 0.045351 | 0.059407 | 0.058199 | 0.004159 | 24 |
| 10 | decomposition | 18771 | 0.044458 | 0.059202 | -0.026645 | 0.010935 | 24 |
| 10 | fx_only | 20965 | 0.046783 | 0.060085 | 0.077535 | 0.081016 | 24 |
| 10 | global_only | 20965 | 0.046438 | 0.061417 | -0.101726 | 0.024233 | 24 |
| 10 | market_only | 20965 | 0.044396 | 0.058804 | -0.106678 | 0.124120 | 24 |
| 10 | reduced_stable | 20965 | 0.047620 | 0.062140 | 0.082013 | -0.083363 | 24 |
| 10 | ridge | 20965 | 0.071342 | 0.088959 | 0.066082 | 0.010818 | 24 |
| 10 | sector_heldout | 20965 | 0.071630 | 0.089174 | 0.064968 | 0.005659 | 24 |
| 10 | sector_interactions | 20965 | 0.072050 | 0.089969 | 0.059451 | 0.000996 | 24 |
| 10 | sector_only | 20965 | 0.044033 | 0.058172 | -0.113103 | -0.024156 | 24 |
| 10 | sector_specific | 18775 | 0.079633 | 0.104108 | -0.022504 | 0.006583 | 24 |
| 10 | technical_only | 20965 | 0.045281 | 0.060045 | -0.097461 | -0.021488 | 24 |
| 10 | without_commodity | 20965 | 0.058719 | 0.073631 | 0.103256 | -0.002756 | 24 |
| 10 | without_fx | 20965 | 0.062388 | 0.082221 | -0.107029 | 0.005831 | 24 |
| 10 | without_global | 20965 | 0.059707 | 0.073308 | 0.190611 | -0.015328 | 24 |
| 10 | without_market | 20965 | 0.060038 | 0.075759 | 0.104547 | 0.012660 | 24 |
| 10 | without_sector | 20965 | 0.071199 | 0.088821 | 0.066138 | 0.013262 | 24 |
| 10 | without_technical | 20965 | 0.068716 | 0.085331 | 0.073766 | 0.012675 | 24 |
| 20 | commodity_only | 20003 | 0.066383 | 0.086994 | -0.106454 | 0.018806 | 12 |
| 20 | decomposition | 17909 | 0.064154 | 0.084941 | -0.052836 | 0.000258 | 12 |
| 20 | fx_only | 20003 | 0.070187 | 0.087849 | 0.098962 | 0.013737 | 12 |
| 20 | global_only | 20003 | 0.066201 | 0.085449 | -0.115335 | 0.041563 | 12 |
| 20 | market_only | 20003 | 0.063972 | 0.084358 | -0.093893 | — | 12 |
| 20 | reduced_stable | 20003 | 0.065783 | 0.085497 | 0.058973 | -0.088007 | 12 |
| 20 | ridge | 20003 | 0.096376 | 0.117709 | 0.328996 | 0.016227 | 12 |
| 20 | sector_heldout | 20003 | 0.097122 | 0.118452 | 0.325819 | 0.023491 | 12 |
| 20 | sector_interactions | 20003 | 0.097859 | 0.119302 | 0.319029 | -0.000103 | 12 |
| 20 | sector_only | 20003 | 0.062516 | 0.082110 | -0.181886 | -0.021411 | 12 |
| 20 | sector_specific | 17541 | 0.102384 | 0.133944 | 0.115367 | 0.001154 | 12 |
| 20 | technical_only | 20003 | 0.065346 | 0.085941 | -0.152356 | -0.023087 | 12 |
| 20 | without_commodity | 20003 | 0.079462 | 0.097773 | 0.244845 | 0.004699 | 12 |
| 20 | without_fx | 20003 | 0.084492 | 0.108518 | -0.088012 | 0.009103 | 12 |
| 20 | without_global | 20003 | 0.097070 | 0.115700 | 0.207918 | -0.014853 | 12 |
| 20 | without_market | 20003 | 0.081139 | 0.098818 | 0.270804 | 0.027131 | 12 |
| 20 | without_sector | 20003 | 0.096314 | 0.117633 | 0.327419 | 0.013892 | 12 |
| 20 | without_technical | 20003 | 0.091522 | 0.111975 | 0.336924 | -0.013964 | 12 |

Groups: 56 technical stock ratios; domestic market; sector; global equity/volatility/yield; FX USD/INR/DXY; commodities Brent/WTI/gold. Group-only and removed-group ridge share origins. Train-only median imputation excludes all-missing/constant columns (FIT_AUDITS). Context availability is estimated following UTC day, not PIT-certified. For absent official sectors use equal-weight fixed-universe industry DAILY RETURNS excluding forecast stock with minimum2 observed peers. This is NOT an official sector index; survivorship bias remains. Failed/sparse study members remain in ELIGIBILITY.

Reduced train-stable set: train variance filter, greedy absolute training Pearson correlation >.95 pruning; consistent Spearman sign in chronological training halves, minimum half absolute IC .01; cap24 by training strength, fall back first5 decorrelated features if none. No validation selection. Train-only identity vocabularies; sector-specific support >=2000 train rows/252 dates/2stocks.

Paired supported-sector versus pooled:

| horizon | model | baseline | n | mae_improvement |
| --- | --- | --- | --- | --- |
| 1 | sector_specific | ridge_paired_supported_industries | 19549 | -0.122963 |
| 5 | sector_specific | ridge_paired_supported_industries | 19205 | -0.151712 |
| 10 | sector_specific | ridge_paired_supported_industries | 18775 | -0.117416 |
| 20 | sector_specific | ridge_paired_supported_industries | 17541 | -0.061188 |

Unsupported industries/folds:

| horizon | fold | industry | train_n | train_dates | train_stocks | test_n | supported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | Chemicals | 1368 | 684 | 2 | 124 | False |
| 1 | 1 | Construction | 684 | 684 | 1 | 62 | False |
| 1 | 1 | Consumer Durables | 1368 | 684 | 2 | 124 | False |
| 1 | 1 | Realty | 644 | 644 | 1 | 62 | False |
| 1 | 1 | Services | 1368 | 684 | 2 | 124 | False |
| 1 | 1 | Telecommunication | 1368 | 684 | 2 | 124 | False |
| 1 | 2 | Chemicals | 1492 | 746 | 2 | 106 | False |
| 1 | 2 | Construction | 746 | 746 | 1 | 53 | False |
| 1 | 2 | Consumer Durables | 1492 | 746 | 2 | 106 | False |
| 1 | 2 | Realty | 706 | 706 | 1 | 53 | False |
| 1 | 2 | Services | 1492 | 746 | 2 | 106 | False |
| 1 | 2 | Telecommunication | 1492 | 746 | 2 | 106 | False |
| 1 | 3 | Chemicals | 1600 | 800 | 2 | 100 | False |
| 1 | 3 | Construction | 800 | 800 | 1 | 50 | False |
| 1 | 3 | Consumer Durables | 1600 | 800 | 2 | 100 | False |
| 1 | 3 | Realty | 760 | 760 | 1 | 50 | False |
| 1 | 3 | Services | 1600 | 800 | 2 | 100 | False |
| 1 | 3 | Telecommunication | 1600 | 800 | 2 | 100 | False |
| 1 | 4 | Chemicals | 1698 | 849 | 2 | 126 | False |
| 1 | 4 | Construction | 849 | 849 | 1 | 63 | False |
| 1 | 4 | Consumer Durables | 1698 | 849 | 2 | 126 | False |
| 1 | 4 | Realty | 809 | 809 | 1 | 63 | False |
| 1 | 4 | Services | 1698 | 849 | 2 | 126 | False |
| 1 | 4 | Telecommunication | 1698 | 849 | 2 | 126 | False |
| 5 | 1 | Chemicals | 1360 | 680 | 2 | 124 | False |
| 5 | 1 | Construction | 680 | 680 | 1 | 62 | False |
| 5 | 1 | Consumer Durables | 1360 | 680 | 2 | 124 | False |
| 5 | 1 | Realty | 640 | 640 | 1 | 62 | False |
| 5 | 1 | Services | 1360 | 680 | 2 | 124 | False |
| 5 | 1 | Telecommunication | 1360 | 680 | 2 | 124 | False |
| 5 | 2 | Chemicals | 1484 | 742 | 2 | 106 | False |
| 5 | 2 | Construction | 742 | 742 | 1 | 53 | False |
| 5 | 2 | Consumer Durables | 1484 | 742 | 2 | 106 | False |
| 5 | 2 | Realty | 702 | 702 | 1 | 53 | False |
| 5 | 2 | Services | 1484 | 742 | 2 | 106 | False |
| 5 | 2 | Telecommunication | 1484 | 742 | 2 | 106 | False |
| 5 | 3 | Chemicals | 1600 | 800 | 2 | 100 | False |
| 5 | 3 | Construction | 800 | 800 | 1 | 50 | False |
| 5 | 3 | Consumer Durables | 1600 | 800 | 2 | 100 | False |
| 5 | 3 | Realty | 760 | 760 | 1 | 50 | False |
| 5 | 3 | Services | 1600 | 800 | 2 | 100 | False |
| 5 | 3 | Telecommunication | 1600 | 800 | 2 | 100 | False |
| 5 | 4 | Chemicals | 1690 | 845 | 2 | 118 | False |
| 5 | 4 | Construction | 845 | 845 | 1 | 59 | False |
| 5 | 4 | Consumer Durables | 1690 | 845 | 2 | 118 | False |
| 5 | 4 | Realty | 805 | 805 | 1 | 59 | False |
| 5 | 4 | Services | 1690 | 845 | 2 | 118 | False |
| 5 | 4 | Telecommunication | 1690 | 845 | 2 | 118 | False |
| 10 | 1 | Chemicals | 1350 | 675 | 2 | 124 | False |
| 10 | 1 | Construction | 675 | 675 | 1 | 62 | False |
| 10 | 1 | Consumer Durables | 1350 | 675 | 2 | 124 | False |
| 10 | 1 | Realty | 635 | 635 | 1 | 62 | False |
| 10 | 1 | Services | 1350 | 675 | 2 | 124 | False |
| 10 | 1 | Telecommunication | 1350 | 675 | 2 | 124 | False |
| 10 | 2 | Chemicals | 1474 | 737 | 2 | 106 | False |
| 10 | 2 | Construction | 737 | 737 | 1 | 53 | False |
| 10 | 2 | Consumer Durables | 1474 | 737 | 2 | 106 | False |
| 10 | 2 | Realty | 697 | 697 | 1 | 53 | False |
| 10 | 2 | Services | 1474 | 737 | 2 | 106 | False |
| 10 | 2 | Telecommunication | 1474 | 737 | 2 | 106 | False |
| 10 | 3 | Chemicals | 1598 | 799 | 2 | 100 | False |
| 10 | 3 | Construction | 799 | 799 | 1 | 50 | False |
| 10 | 3 | Consumer Durables | 1598 | 799 | 2 | 100 | False |
| 10 | 3 | Realty | 759 | 759 | 1 | 50 | False |
| 10 | 3 | Services | 1598 | 799 | 2 | 100 | False |
| 10 | 3 | Telecommunication | 1598 | 799 | 2 | 100 | False |
| 10 | 4 | Chemicals | 1680 | 840 | 2 | 108 | False |
| 10 | 4 | Construction | 840 | 840 | 1 | 54 | False |
| 10 | 4 | Consumer Durables | 1680 | 840 | 2 | 108 | False |
| 10 | 4 | Realty | 800 | 800 | 1 | 54 | False |
| 10 | 4 | Services | 1680 | 840 | 2 | 108 | False |
| 10 | 4 | Telecommunication | 1680 | 840 | 2 | 108 | False |
| 20 | 1 | Chemicals | 1330 | 665 | 2 | 124 | False |
| 20 | 1 | Construction | 665 | 665 | 1 | 62 | False |
| 20 | 1 | Construction Materials | 1992 | 665 | 3 | 186 | False |
| 20 | 1 | Consumer Durables | 1330 | 665 | 2 | 124 | False |
| 20 | 1 | Consumer Services | 1974 | 665 | 3 | 186 | False |
| 20 | 1 | Realty | 625 | 625 | 1 | 62 | False |
| 20 | 1 | Services | 1330 | 665 | 2 | 124 | False |
| 20 | 1 | Telecommunication | 1330 | 665 | 2 | 124 | False |
| 20 | 2 | Chemicals | 1454 | 727 | 2 | 106 | False |
| 20 | 2 | Construction | 727 | 727 | 1 | 53 | False |
| 20 | 2 | Consumer Durables | 1454 | 727 | 2 | 106 | False |
| 20 | 2 | Realty | 687 | 687 | 1 | 53 | False |
| 20 | 2 | Services | 1454 | 727 | 2 | 106 | False |
| 20 | 2 | Telecommunication | 1454 | 727 | 2 | 106 | False |
| 20 | 3 | Chemicals | 1578 | 789 | 2 | 100 | False |
| 20 | 3 | Construction | 789 | 789 | 1 | 50 | False |
| 20 | 3 | Consumer Durables | 1578 | 789 | 2 | 100 | False |
| 20 | 3 | Realty | 749 | 749 | 1 | 50 | False |
| 20 | 3 | Services | 1578 | 789 | 2 | 100 | False |
| 20 | 3 | Telecommunication | 1578 | 789 | 2 | 100 | False |
| 20 | 4 | Chemicals | 1660 | 830 | 2 | 88 | False |
| 20 | 4 | Construction | 830 | 830 | 1 | 44 | False |
| 20 | 4 | Consumer Durables | 1660 | 830 | 2 | 88 | False |
| 20 | 4 | Realty | 790 | 790 | 1 | 44 | False |
| 20 | 4 | Services | 1660 | 830 | 2 | 88 | False |
| 20 | 4 | Telecommunication | 1660 | 830 | 2 | 88 | False |

Leave-industry-out evaluates every industry with all of its training stocks excluded. It supplements temporal checks; it is not prospective evidence.

Measured train/validation degradation:

| horizon | fold | model | train_mae | validation_mae | ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | ridge | 0.013373 | 0.016047 | 1.199905 |
| 1 | 1 | elastic_net | 0.013520 | 0.014059 | 1.039884 |
| 1 | 1 | shallow_boost | 0.013377 | 0.014214 | 1.062591 |
| 1 | 1 | reduced_stable | 0.013514 | 0.014038 | 1.038744 |
| 1 | 1 | sector_interactions | 0.013370 | 0.016213 | 1.212709 |
| 1 | 2 | ridge | 0.013432 | 0.016984 | 1.264429 |
| 1 | 2 | elastic_net | 0.013563 | 0.015832 | 1.167333 |
| 1 | 2 | shallow_boost | 0.013451 | 0.015930 | 1.184350 |
| 1 | 2 | reduced_stable | 0.013543 | 0.015763 | 1.163973 |
| 1 | 2 | sector_interactions | 0.013430 | 0.017145 | 1.276607 |
| 1 | 3 | ridge | 0.013585 | 0.015262 | 1.123397 |
| 1 | 3 | elastic_net | 0.013704 | 0.012282 | 0.896217 |
| 1 | 3 | shallow_boost | 0.013578 | 0.012263 | 0.903146 |
| 1 | 3 | reduced_stable | 0.013694 | 0.012245 | 0.894143 |
| 1 | 3 | sector_interactions | 0.013584 | 0.015093 | 1.111098 |
| 1 | 4 | ridge | 0.013511 | 0.011578 | 0.856898 |
| 1 | 4 | elastic_net | 0.013625 | 0.010428 | 0.765332 |
| 1 | 4 | shallow_boost | 0.013512 | 0.010454 | 0.773667 |
| 1 | 4 | reduced_stable | 0.013615 | 0.010443 | 0.767032 |
| 1 | 4 | sector_interactions | 0.013508 | 0.011583 | 0.857448 |
| 5 | 1 | ridge | 0.029733 | 0.064319 | 2.163178 |
| 5 | 1 | elastic_net | 0.030367 | 0.036169 | 1.191062 |
| 5 | 1 | shallow_boost | 0.030518 | 0.035313 | 1.157125 |
| 5 | 1 | reduced_stable | 0.030781 | 0.036500 | 1.185799 |
| 5 | 1 | sector_interactions | 0.029676 | 0.065775 | 2.216419 |
| 5 | 2 | ridge | 0.030112 | 0.037776 | 1.254490 |
| 5 | 2 | elastic_net | 0.030698 | 0.037407 | 1.218556 |
| 5 | 2 | shallow_boost | 0.030884 | 0.038065 | 1.232515 |
| 5 | 2 | reduced_stable | 0.031238 | 0.036990 | 1.184156 |
| 5 | 2 | sector_interactions | 0.030062 | 0.037873 | 1.259814 |
| 5 | 3 | ridge | 0.030461 | 0.044512 | 1.461259 |
| 5 | 3 | elastic_net | 0.031015 | 0.027045 | 0.871989 |
| 5 | 3 | shallow_boost | 0.031263 | 0.027202 | 0.870117 |
| 5 | 3 | reduced_stable | 0.031542 | 0.026864 | 0.851687 |
| 5 | 3 | sector_interactions | 0.030422 | 0.043878 | 1.442297 |
| 5 | 4 | ridge | 0.030330 | 0.031735 | 1.046305 |
| 5 | 4 | elastic_net | 0.030831 | 0.026363 | 0.855072 |
| 5 | 4 | shallow_boost | 0.031064 | 0.024942 | 0.802920 |
| 5 | 4 | reduced_stable | 0.031364 | 0.025641 | 0.817529 |
| 5 | 4 | sector_interactions | 0.030289 | 0.031911 | 1.053545 |
| 10 | 1 | ridge | 0.041622 | 0.105636 | 2.537963 |
| 10 | 1 | elastic_net | 0.042276 | 0.066909 | 1.582675 |
| 10 | 1 | shallow_boost | 0.043154 | 0.054050 | 1.252480 |
| 10 | 1 | reduced_stable | 0.043467 | 0.061076 | 1.405130 |
| 10 | 1 | sector_interactions | 0.041513 | 0.107719 | 2.594842 |
| 10 | 2 | ridge | 0.042092 | 0.046124 | 1.095794 |
| 10 | 2 | elastic_net | 0.042622 | 0.050721 | 1.190026 |
| 10 | 2 | shallow_boost | 0.043892 | 0.051092 | 1.164034 |
| 10 | 2 | reduced_stable | 0.043934 | 0.049920 | 1.136253 |
| 10 | 2 | sector_interactions | 0.041998 | 0.047438 | 1.129528 |
| 10 | 3 | ridge | 0.042223 | 0.084129 | 1.992470 |
| 10 | 3 | elastic_net | 0.042841 | 0.038474 | 0.898052 |
| 10 | 3 | shallow_boost | 0.043920 | 0.036958 | 0.841485 |
| 10 | 3 | reduced_stable | 0.043947 | 0.039961 | 0.909297 |
| 10 | 3 | sector_interactions | 0.042166 | 0.082797 | 1.963610 |
| 10 | 4 | ridge | 0.042066 | 0.045360 | 1.078327 |
| 10 | 4 | elastic_net | 0.042570 | 0.037747 | 0.886700 |
| 10 | 4 | shallow_boost | 0.043542 | 0.034024 | 0.781401 |
| 10 | 4 | reduced_stable | 0.043802 | 0.037162 | 0.848418 |
| 10 | 4 | sector_interactions | 0.042013 | 0.045798 | 1.090078 |
| 20 | 1 | ridge | 0.059589 | 0.078581 | 1.318704 |
| 20 | 1 | elastic_net | 0.060263 | 0.089354 | 1.482750 |
| 20 | 1 | shallow_boost | 0.061774 | 0.078843 | 1.276310 |
| 20 | 1 | reduced_stable | 0.063617 | 0.082528 | 1.297272 |
| 20 | 1 | sector_interactions | 0.059307 | 0.081919 | 1.381283 |
| 20 | 2 | ridge | 0.059719 | 0.086593 | 1.450004 |
| 20 | 2 | elastic_net | 0.060242 | 0.073769 | 1.224546 |
| 20 | 2 | shallow_boost | 0.062399 | 0.077354 | 1.239668 |
| 20 | 2 | reduced_stable | 0.062868 | 0.074323 | 1.182215 |
| 20 | 2 | sector_interactions | 0.059481 | 0.088858 | 1.493901 |
| 20 | 3 | ridge | 0.059989 | 0.149844 | 2.497840 |
| 20 | 3 | elastic_net | 0.060628 | 0.055643 | 0.917773 |
| 20 | 3 | shallow_boost | 0.063097 | 0.049130 | 0.778648 |
| 20 | 3 | reduced_stable | 0.063635 | 0.048883 | 0.768177 |
| 20 | 3 | sector_interactions | 0.059835 | 0.148660 | 2.484497 |
| 20 | 4 | ridge | 0.059755 | 0.072309 | 1.210093 |
| 20 | 4 | elastic_net | 0.060383 | 0.063085 | 1.044753 |
| 20 | 4 | shallow_boost | 0.062826 | 0.047834 | 0.761372 |
| 20 | 4 | reduced_stable | 0.063272 | 0.051320 | 0.811095 |
| 20 | 4 | sector_interactions | 0.059620 | 0.073289 | 1.229283 |

Exploratory validation IC leaders (never used to reselect):

| horizon | feature | train_ic | validation_ic | train_residual_ic | validation_residual_ic | degradation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | dxy_vol_20 | 0.005573 | 0.076285 | 0.007616 | 0.053158 | -0.070712 |
| 1 | india_vix_drawdown | 0.038432 | 0.072036 | 0.040649 | 0.058620 | -0.033604 |
| 1 | india_vix_trend_50 | 0.030002 | 0.064725 | 0.039543 | 0.038375 | -0.034723 |
| 1 | month | 0.015914 | 0.053879 | 0.017616 | 0.072515 | -0.037965 |
| 1 | nikkei_vol_20 | 0.001406 | 0.046965 | 0.011378 | 0.023386 | -0.045558 |
| 1 | nasdaq_vol_20 | -0.020393 | 0.045534 | -0.039283 | -0.031605 | -0.025141 |
| 1 | india_vix_ret_5d | -0.005370 | 0.044783 | -0.001232 | 0.044310 | -0.039413 |
| 1 | is_friday | -0.001150 | 0.044059 | 0.001289 | 0.035146 | -0.042909 |
| 5 | dxy_vol_20 | 0.031650 | 0.171069 | 0.035431 | 0.157015 | -0.139419 |
| 5 | nasdaq_vol_20 | -0.031033 | 0.140349 | -0.068792 | -0.005054 | -0.109315 |
| 5 | india_vix_drawdown | 0.076842 | 0.117240 | 0.092766 | 0.120829 | -0.040398 |
| 5 | month | 0.038191 | 0.114198 | 0.041323 | 0.149437 | -0.076008 |
| 5 | india_vix_trend_50 | 0.069122 | 0.111298 | 0.086070 | 0.067565 | -0.042176 |
| 5 | nikkei_vol_20 | 0.006628 | 0.109233 | 0.034460 | 0.079177 | -0.102605 |
| 5 | us_yield_vol_20 | 0.045101 | 0.096127 | 0.042958 | -0.070638 | -0.051026 |
| 5 | india_vix_ret_5d | 0.038375 | 0.090908 | 0.040216 | 0.063167 | -0.052533 |
| 10 | dxy_vol_20 | 0.057605 | 0.229414 | 0.056284 | 0.202528 | -0.171809 |
| 10 | us_yield_vol_20 | 0.060343 | 0.216522 | 0.050027 | 0.018502 | -0.156179 |
| 10 | gold_vol_20 | 0.052837 | 0.183667 | 0.061114 | 0.208804 | -0.130829 |
| 10 | nasdaq_vol_20 | -0.038173 | 0.163114 | -0.085264 | -0.014771 | -0.124941 |
| 10 | month | 0.048271 | 0.141295 | 0.053488 | 0.172987 | -0.093025 |
| 10 | india_vix_trend_50 | 0.081705 | 0.124375 | 0.107908 | 0.082932 | -0.042669 |
| 10 | vix_trend_50 | -0.067938 | 0.124340 | -0.054584 | 0.027990 | -0.056402 |
| 10 | india_vix_ret_20d | 0.049402 | 0.119909 | 0.073245 | 0.058716 | -0.070507 |
| 20 | us_yield_vol_20 | 0.038886 | 0.217690 | 0.023103 | 0.014515 | -0.178804 |
| 20 | dxy_vol_20 | 0.041570 | 0.202327 | 0.041476 | 0.176684 | -0.160757 |
| 20 | gold_vol_20 | 0.055554 | 0.194340 | 0.068383 | 0.222775 | -0.138787 |
| 20 | india_vix_trend_50 | 0.085692 | 0.146961 | 0.113023 | 0.136408 | -0.061269 |
| 20 | month | 0.039293 | 0.134925 | 0.042766 | 0.159611 | -0.095632 |
| 20 | india_vix_ret_20d | 0.054604 | 0.111535 | 0.069173 | 0.082794 | -0.056931 |
| 20 | nasdaq_vol_20 | -0.071744 | 0.107384 | -0.122740 | 0.007640 | -0.035640 |
| 20 | vix_trend_50 | -0.042456 | 0.100650 | -0.018528 | 0.009698 | -0.058194 |

Residual IC uses train-fitted market-feature ridge residualization of both signal and target. Full IC in SIGNALS; sample/date/features in ATTEMPTS/FIT_AUDITS; strata in STRATA. Regime rules: past market trend +/-2% bull/bear/range, past rolling volatility median high/low, |past stock ret20|>2% trend/range. Stocks/industries/folds/horizons/regimes reported separately. Complexity sensitivity is fixed linear versus depth2/50 boosting and full versus reduced features; no depth search. Correlated stocks do not multiply effective dates.

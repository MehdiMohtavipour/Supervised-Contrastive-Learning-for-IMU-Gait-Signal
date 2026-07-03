# Pairwise ROC AUC Summary

AUC values are oriented so values below 0.5 are flipped for discriminative performance.
`raw_auc` keeps the unflipped direction where the second group in the pair is positive.

## Macro AUC

| feature | feature_label | macro_auc | macro_raw_auc | n_pairs | total_n |
| --- | --- | --- | --- | --- | --- |
| distance_healthy_plus_neuro_minus_ortho | EDGB (This Work) | 0.9285 | 0.6580 | 3 | 532 |
| RF_FreeAcc_Magnitude_u_turn_time | U-turn time | 0.7648 | 0.6575 | 3 | 532 |
| RF_FreeAcc_Magnitude_std_stride_time | Std stride time | 0.6933 | 0.6315 | 3 | 532 |
| RF_FreeAcc_Magnitude_max_stride_time | Max stride time | 0.6911 | 0.6183 | 3 | 532 |
| RF_FreeAcc_Magnitude_gait_cadence | Gait cadence | 0.5955 | 0.4045 | 3 | 532 |
| RF_FreeAcc_Magnitude_avg_stride_time | Avg stride time | 0.5927 | 0.5927 | 3 | 532 |

## Pairwise AUC

| pair | group_a | group_b | positive_class_raw | feature | feature_label | n | n_group_a | n_group_b | raw_auc | auc | score_flipped_for_auc | higher_score_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| healthy_vs_neuro | healthy | neuro | neuro | RF_FreeAcc_Magnitude_avg_stride_time | Avg stride time | 220 | 63 | 157 | 0.5990 | 0.5990 | False | neuro |
| healthy_vs_ortho | healthy | ortho | ortho | RF_FreeAcc_Magnitude_avg_stride_time | Avg stride time | 109 | 63 | 46 | 0.6422 | 0.6422 | False | ortho |
| neuro_vs_ortho | neuro | ortho | ortho | RF_FreeAcc_Magnitude_avg_stride_time | Avg stride time | 203 | 157 | 46 | 0.5370 | 0.5370 | False | ortho |
| healthy_vs_neuro | healthy | neuro | neuro | RF_FreeAcc_Magnitude_gait_cadence | Gait cadence | 220 | 63 | 157 | 0.3722 | 0.6278 | True | healthy |
| healthy_vs_ortho | healthy | ortho | ortho | RF_FreeAcc_Magnitude_gait_cadence | Gait cadence | 109 | 63 | 46 | 0.3521 | 0.6479 | True | healthy |
| neuro_vs_ortho | neuro | ortho | ortho | RF_FreeAcc_Magnitude_gait_cadence | Gait cadence | 203 | 157 | 46 | 0.4892 | 0.5108 | True | neuro |
| healthy_vs_neuro | healthy | neuro | neuro | RF_FreeAcc_Magnitude_max_stride_time | Max stride time | 220 | 63 | 157 | 0.7771 | 0.7771 | False | neuro |
| healthy_vs_ortho | healthy | ortho | ortho | RF_FreeAcc_Magnitude_max_stride_time | Max stride time | 109 | 63 | 46 | 0.6870 | 0.6870 | False | ortho |
| neuro_vs_ortho | neuro | ortho | ortho | RF_FreeAcc_Magnitude_max_stride_time | Max stride time | 203 | 157 | 46 | 0.3908 | 0.6092 | True | neuro |
| healthy_vs_neuro | healthy | neuro | neuro | RF_FreeAcc_Magnitude_std_stride_time | Std stride time | 220 | 63 | 157 | 0.7674 | 0.7674 | False | neuro |
| healthy_vs_ortho | healthy | ortho | ortho | RF_FreeAcc_Magnitude_std_stride_time | Std stride time | 109 | 63 | 46 | 0.7198 | 0.7198 | False | ortho |
| neuro_vs_ortho | neuro | ortho | ortho | RF_FreeAcc_Magnitude_std_stride_time | Std stride time | 203 | 157 | 46 | 0.4074 | 0.5926 | True | neuro |
| healthy_vs_neuro | healthy | neuro | neuro | RF_FreeAcc_Magnitude_u_turn_time | U-turn time | 220 | 63 | 157 | 0.8515 | 0.8515 | False | neuro |
| healthy_vs_ortho | healthy | ortho | ortho | RF_FreeAcc_Magnitude_u_turn_time | U-turn time | 109 | 63 | 46 | 0.7819 | 0.7819 | False | ortho |
| neuro_vs_ortho | neuro | ortho | ortho | RF_FreeAcc_Magnitude_u_turn_time | U-turn time | 203 | 157 | 46 | 0.3390 | 0.6610 | True | neuro |
| healthy_vs_neuro | healthy | neuro | neuro | distance_healthy_plus_neuro_minus_ortho | EDGB (This Work) | 220 | 63 | 157 | 0.0941 | 0.9059 | True | healthy |
| healthy_vs_ortho | healthy | ortho | ortho | distance_healthy_plus_neuro_minus_ortho | EDGB (This Work) | 109 | 63 | 46 | 0.8847 | 0.8847 | False | ortho |
| neuro_vs_ortho | neuro | ortho | ortho | distance_healthy_plus_neuro_minus_ortho | EDGB (This Work) | 203 | 157 | 46 | 0.9950 | 0.9950 | False | ortho |

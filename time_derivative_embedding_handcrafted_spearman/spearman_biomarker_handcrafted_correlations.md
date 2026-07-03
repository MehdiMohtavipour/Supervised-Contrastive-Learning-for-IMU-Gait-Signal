# Spearman Correlation: Proposed Biomarker vs Handcrafted Features

Rows are matched by `Subject_Name` and `Trial_Name`.
Spearman rho is computed between `This work` and each handcrafted feature on the test split.

| Split | Feature | N | Spearman rho | p-value |
| --- | --- | --- | --- | --- |
| test | U-turn time | 266.0000 | -0.2597 | 1.7879e-05 |
| test | Std stride time | 266.0000 | -0.1546 | 0.0116 |
| test | Max stride time | 266.0000 | -0.1394 | 0.0229 |
| test | Avg stride time | 266.0000 | 0.1094 | 0.0750 |
| test | Gait cadence | 266.0000 | -0.0666 | 0.2794 |

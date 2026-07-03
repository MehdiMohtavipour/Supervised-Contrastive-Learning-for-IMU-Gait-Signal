# Pairwise Spearman Correlation: This Work and Handcrafted Features

Rows are matched by `Subject_Name` and `Trial_Name`.
Spearman rho is computed for every pair of variables on the test split.

| Split | Feature 1 | Feature 2 | N | Spearman rho | p-value |
| --- | --- | --- | --- | --- | --- |
| test | Avg stride time | Gait cadence | 266.0000 | -0.9959 | 3.1830e-278 |
| test | Avg angular acceleration (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.9854 | 7.0774e-205 |
| test | Avg angular velocity peak (RF-LF) | Std angular velocity peak (RF-LF) | 266.0000 | 0.9797 | 3.3059e-186 |
| test | Avg acceleration peak (RF-LF) | Std acceleration peak (RF-LF) | 266.0000 | 0.9771 | 1.7672e-179 |
| test | Avg jerk (RF-LF) | Std jerk (RF-LF) | 266.0000 | 0.9764 | 7.7360e-178 |
| test | Max stride time | Std stride time | 266.0000 | 0.9443 | 1.9456e-129 |
| test | Max jerk (RF-LF) | Std jerk (RF-LF) | 266.0000 | 0.9204 | 1.1873e-109 |
| test | Std angular velocity peak (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.9076 | 1.6729e-101 |
| test | Avg acceleration peak (RF-LF) | Avg angular velocity peak (RF-LF) | 266.0000 | 0.9068 | 4.9065e-101 |
| test | Avg acceleration peak (RF-LF) | Std angular velocity peak (RF-LF) | 266.0000 | 0.9021 | 2.2372e-98 |
| test | Std angular velocity peak (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.8998 | 4.1074e-97 |
| test | Std acceleration peak (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.8929 | 1.8871e-93 |
| test | Std acceleration peak (RF-LF) | Std angular velocity peak (RF-LF) | 266.0000 | 0.8908 | 2.0001e-92 |
| test | Std acceleration peak (RF-LF) | Avg jerk (RF-LF) | 266.0000 | 0.8896 | 7.8167e-92 |
| test | Avg angular velocity peak (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.8841 | 3.3553e-89 |
| test | Std acceleration peak (RF-LF) | Avg angular velocity peak (RF-LF) | 266.0000 | 0.8816 | 4.5845e-88 |
| test | Std acceleration peak (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.8793 | 5.0749e-87 |
| test | Avg acceleration peak (RF-LF) | Avg jerk (RF-LF) | 266.0000 | 0.8722 | 6.0230e-84 |
| test | Avg acceleration peak (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.8712 | 1.5693e-83 |
| test | Std acceleration peak (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.8707 | 2.5333e-83 |
| test | Avg angular velocity peak (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.8689 | 1.3521e-82 |
| test | Avg jerk (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.8686 | 1.7920e-82 |
| test | Avg acceleration peak (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.8645 | 7.9621e-81 |
| test | Avg acceleration peak (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.8541 | 6.2743e-77 |
| test | Std jerk (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.8436 | 2.9825e-73 |
| test | Avg jerk (RF-LF) | Max jerk (RF-LF) | 266.0000 | 0.8379 | 2.2161e-71 |
| test | Std acceleration peak (RF-LF) | Std jerk (RF-LF) | 266.0000 | 0.8271 | 5.0321e-68 |
| test | Std angular acceleration (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.8133 | 4.9392e-64 |
| test | Avg angular acceleration (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.7975 | 6.9494e-60 |
| test | Avg acceleration peak (RF-LF) | Std jerk (RF-LF) | 266.0000 | 0.7972 | 8.6923e-60 |
| test | Std angular velocity peak (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.7953 | 2.5140e-59 |
| test | U-turn time | Std angular velocity peak (RF-LF) | 266.0000 | -0.7864 | 3.6171e-57 |
| test | U-turn time | Avg angular velocity peak (RF-LF) | 266.0000 | -0.7834 | 1.8627e-56 |
| test | Std acceleration (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.7805 | 8.6466e-56 |
| test | Avg angular velocity peak (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.7785 | 2.5222e-55 |
| test | Avg jerk (RF-LF) | Std angular velocity peak (RF-LF) | 266.0000 | 0.7774 | 4.3627e-55 |
| test | Std angular velocity peak (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.7771 | 5.0240e-55 |
| test | Std angular acceleration (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.7766 | 6.6566e-55 |
| test | Avg angular acceleration (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.7723 | 5.9590e-54 |
| test | Avg jerk (RF-LF) | Avg angular velocity peak (RF-LF) | 266.0000 | 0.7628 | 6.5182e-52 |
| test | Std acceleration peak (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.7556 | 2.0642e-50 |
| test | Avg angular velocity peak (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.7548 | 2.8718e-50 |
| test | Gait cadence | Std acceleration (RF-LF) | 266.0000 | 0.7425 | 7.4925e-48 |
| test | U-turn time | Avg acceleration peak (RF-LF) | 266.0000 | -0.7416 | 1.1230e-47 |
| test | U-turn time | Std acceleration (RF-LF) | 266.0000 | -0.7375 | 6.6112e-47 |
| test | Avg jerk (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.7372 | 7.3147e-47 |
| test | Avg jerk (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.7258 | 8.4789e-45 |
| test | U-turn time | Std acceleration peak (RF-LF) | 266.0000 | -0.7236 | 2.0727e-44 |
| test | Avg acceleration peak (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.7162 | 3.9385e-43 |
| test | Gait cadence | Std angular acceleration (RF-LF) | 266.0000 | 0.7052 | 2.5744e-41 |
| test | U-turn time | Avg angular acceleration (RF-LF) | 266.0000 | -0.7043 | 3.6271e-41 |
| test | Avg stride time | Std acceleration (RF-LF) | 266.0000 | -0.7031 | 5.6699e-41 |
| test | Gait cadence | Avg angular acceleration (RF-LF) | 266.0000 | 0.6996 | 1.9957e-40 |
| test | Std jerk (RF-LF) | Std angular velocity peak (RF-LF) | 266.0000 | 0.6956 | 8.4800e-40 |
| test | Max jerk (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.6953 | 9.3840e-40 |
| test | Gait cadence | Std acceleration peak (RF-LF) | 266.0000 | 0.6939 | 1.5408e-39 |
| test | U-turn time | Std angular acceleration (RF-LF) | 266.0000 | -0.6880 | 1.2496e-38 |
| test | Avg stride time | Std angular acceleration (RF-LF) | 266.0000 | -0.6805 | 1.6149e-37 |
| test | Std jerk (RF-LF) | Avg angular velocity peak (RF-LF) | 266.0000 | 0.6772 | 4.7653e-37 |
| test | Gait cadence | Std angular velocity (RF-LF) | 266.0000 | 0.6748 | 1.0591e-36 |
| test | Avg stride time | Avg angular acceleration (RF-LF) | 266.0000 | -0.6739 | 1.4499e-36 |
| test | Gait cadence | Avg jerk (RF-LF) | 266.0000 | 0.6675 | 1.1426e-35 |
| test | Avg stride time | Std acceleration peak (RF-LF) | 266.0000 | -0.6558 | 4.3919e-34 |
| test | Avg stride time | Std angular velocity (RF-LF) | 266.0000 | -0.6485 | 3.9288e-33 |
| test | U-turn time | Avg jerk (RF-LF) | 266.0000 | -0.6435 | 1.6964e-32 |
| test | Gait cadence | Avg acceleration peak (RF-LF) | 266.0000 | 0.6377 | 9.1778e-32 |
| test | U-turn time | Std angular velocity (RF-LF) | 266.0000 | -0.6362 | 1.3806e-31 |
| test | Std jerk (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.6334 | 3.0894e-31 |
| test | Std jerk (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.6312 | 5.7031e-31 |
| test | Avg stride time | Avg jerk (RF-LF) | 266.0000 | -0.6262 | 2.2856e-30 |
| test | Std acceleration peak (RF-LF) | Max jerk (RF-LF) | 266.0000 | 0.6224 | 6.2855e-30 |
| test | Gait cadence | Std jerk (RF-LF) | 266.0000 | 0.6163 | 3.2440e-29 |
| test | Avg jerk (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.6046 | 6.6409e-28 |
| test | Gait cadence | Avg angular velocity peak (RF-LF) | 266.0000 | 0.6031 | 9.6527e-28 |
| test | Gait cadence | Std angular velocity peak (RF-LF) | 266.0000 | 0.6020 | 1.2730e-27 |
| test | Avg stride time | Avg acceleration peak (RF-LF) | 266.0000 | -0.5961 | 5.6053e-27 |
| test | U-turn time | Std jerk (RF-LF) | 266.0000 | -0.5926 | 1.3067e-26 |
| test | Max stride time | Avg acceleration peak (RF-LF) | 266.0000 | -0.5921 | 1.4800e-26 |
| test | Max stride time | Avg angular velocity peak (RF-LF) | 266.0000 | -0.5878 | 4.1330e-26 |
| test | Avg acceleration peak (RF-LF) | Max jerk (RF-LF) | 266.0000 | 0.5850 | 8.0935e-26 |
| test | Max stride time | Std acceleration peak (RF-LF) | 266.0000 | -0.5815 | 1.8362e-25 |
| test | Max stride time | Gait cadence | 266.0000 | -0.5798 | 2.7346e-25 |
| test | Avg stride time | Std jerk (RF-LF) | 266.0000 | -0.5740 | 1.0386e-24 |
| test | Avg stride time | Avg angular velocity peak (RF-LF) | 266.0000 | -0.5638 | 1.0157e-23 |
| test | Avg stride time | Std angular velocity peak (RF-LF) | 266.0000 | -0.5633 | 1.1440e-23 |
| test | Avg stride time | Max stride time | 266.0000 | 0.5606 | 2.0624e-23 |
| test | Max stride time | Std angular velocity peak (RF-LF) | 266.0000 | -0.5525 | 1.1674e-22 |
| test | Std jerk (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.5452 | 5.3501e-22 |
| test | Max stride time | Avg angular acceleration (RF-LF) | 266.0000 | -0.5405 | 1.3943e-21 |
| test | Max stride time | Avg jerk (RF-LF) | 266.0000 | -0.5363 | 3.2856e-21 |
| test | Proposed Biomarker | Max jerk (RF-LF) | 266.0000 | 0.5315 | 8.4448e-21 |
| test | Max stride time | Std angular acceleration (RF-LF) | 266.0000 | -0.5191 | 9.4064e-20 |
| test | Mean acceleration (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.5085 | 6.7795e-19 |
| test | Std stride time | Avg angular velocity peak (RF-LF) | 266.0000 | -0.4941 | 8.8688e-18 |
| test | Max stride time | Std jerk (RF-LF) | 266.0000 | -0.4872 | 2.8942e-17 |
| test | Std stride time | Avg acceleration peak (RF-LF) | 266.0000 | -0.4810 | 8.2624e-17 |
| test | U-turn time | Gait cadence | 266.0000 | -0.4806 | 8.8171e-17 |
| test | Max stride time | Std acceleration (RF-LF) | 266.0000 | -0.4750 | 2.2453e-16 |
| test | Max jerk (RF-LF) | Std angular velocity peak (RF-LF) | 266.0000 | 0.4683 | 6.5963e-16 |
| test | Max stride time | Std angular velocity (RF-LF) | 266.0000 | -0.4633 | 1.4747e-15 |
| test | Proposed Biomarker | Std jerk (RF-LF) | 266.0000 | 0.4613 | 2.0014e-15 |
| test | Std stride time | Std acceleration peak (RF-LF) | 266.0000 | -0.4530 | 7.2218e-15 |
| test | Std stride time | Std angular velocity peak (RF-LF) | 266.0000 | -0.4529 | 7.3687e-15 |
| test | Max jerk (RF-LF) | Avg angular velocity peak (RF-LF) | 266.0000 | 0.4433 | 3.1441e-14 |
| test | Gait cadence | Max jerk (RF-LF) | 266.0000 | 0.4399 | 5.1439e-14 |
| test | Avg stride time | U-turn time | 266.0000 | 0.4378 | 6.9779e-14 |
| test | Max stride time | U-turn time | 266.0000 | 0.4245 | 4.6541e-13 |
| test | Std stride time | Avg jerk (RF-LF) | 266.0000 | -0.4232 | 5.5182e-13 |
| test | U-turn time | Max jerk (RF-LF) | 266.0000 | -0.4154 | 1.6098e-12 |
| test | Std stride time | Avg angular acceleration (RF-LF) | 266.0000 | -0.4146 | 1.7978e-12 |
| test | Proposed Biomarker | Avg jerk (RF-LF) | 266.0000 | 0.4073 | 4.7165e-12 |
| test | Avg stride time | Max jerk (RF-LF) | 266.0000 | -0.4004 | 1.1591e-11 |
| test | Std stride time | Gait cadence | 266.0000 | -0.3932 | 2.8906e-11 |
| test | Max jerk (RF-LF) | Std angular acceleration (RF-LF) | 266.0000 | 0.3920 | 3.3386e-11 |
| test | Max jerk (RF-LF) | Avg angular acceleration (RF-LF) | 266.0000 | 0.3837 | 9.2552e-11 |
| test | Std stride time | Std angular acceleration (RF-LF) | 266.0000 | -0.3822 | 1.1121e-10 |
| test | Std stride time | Std jerk (RF-LF) | 266.0000 | -0.3757 | 2.4069e-10 |
| test | Avg stride time | Std stride time | 266.0000 | 0.3747 | 2.7110e-10 |
| test | Max jerk (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.3521 | 3.5153e-09 |
| test | Max stride time | Max jerk (RF-LF) | 266.0000 | -0.3246 | 6.0805e-08 |
| test | Std stride time | U-turn time | 266.0000 | 0.3191 | 1.0434e-07 |
| test | Std stride time | Std acceleration (RF-LF) | 266.0000 | -0.3158 | 1.4248e-07 |
| test | Std stride time | Std angular velocity (RF-LF) | 266.0000 | -0.3099 | 2.4942e-07 |
| test | Proposed Biomarker | U-turn time | 266.0000 | -0.2597 | 1.7879e-05 |
| test | Proposed Biomarker | Avg acceleration peak (RF-LF) | 266.0000 | 0.2383 | 8.6675e-05 |
| test | Std stride time | Max jerk (RF-LF) | 266.0000 | -0.2357 | 1.0385e-04 |
| test | Proposed Biomarker | Std acceleration (RF-LF) | 266.0000 | 0.2076 | 6.5772e-04 |
| test | Proposed Biomarker | Std acceleration peak (RF-LF) | 266.0000 | 0.1918 | 0.0017 |
| test | Proposed Biomarker | Std angular velocity peak (RF-LF) | 266.0000 | 0.1901 | 0.0018 |
| test | Proposed Biomarker | Avg angular velocity peak (RF-LF) | 266.0000 | 0.1682 | 0.0060 |
| test | Std angular velocity peak (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1668 | 0.0064 |
| test | Proposed Biomarker | Std stride time | 266.0000 | -0.1546 | 0.0116 |
| test | Avg acceleration peak (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1410 | 0.0215 |
| test | Avg angular acceleration (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1404 | 0.0220 |
| test | Std acceleration peak (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1395 | 0.0229 |
| test | Proposed Biomarker | Max stride time | 266.0000 | -0.1394 | 0.0229 |
| test | Std angular acceleration (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1390 | 0.0234 |
| test | Avg angular velocity peak (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1315 | 0.0320 |
| test | Avg jerk (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1274 | 0.0378 |
| test | Std jerk (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1230 | 0.0451 |
| test | Std acceleration (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1130 | 0.0657 |
| test | Proposed Biomarker | Avg stride time | 266.0000 | 0.1094 | 0.0750 |
| test | Max jerk (RF-LF) | Mean angular velocity (RF-LF) | 266.0000 | 0.1008 | 0.1008 |
| test | Avg stride time | Mean acceleration (RF-LF) | 266.0000 | -0.0938 | 0.1269 |
| test | U-turn time | Mean angular velocity (RF-LF) | 266.0000 | -0.0911 | 0.1383 |
| test | Gait cadence | Mean acceleration (RF-LF) | 266.0000 | 0.0880 | 0.1523 |
| test | Std acceleration peak (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0846 | 0.1688 |
| test | Mean angular velocity (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | 0.0795 | 0.1961 |
| test | Avg jerk (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0714 | 0.2461 |
| test | Avg acceleration peak (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0699 | 0.2561 |
| test | Proposed Biomarker | Gait cadence | 266.0000 | -0.0666 | 0.2794 |
| test | Std jerk (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0547 | 0.3739 |
| test | Avg angular acceleration (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0540 | 0.3806 |
| test | Std angular velocity peak (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0518 | 0.4002 |
| test | Std angular acceleration (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0497 | 0.4195 |
| test | Mean acceleration (RF-LF) | Std acceleration (RF-LF) | 266.0000 | 0.0494 | 0.4220 |
| test | Proposed Biomarker | Mean acceleration (RF-LF) | 266.0000 | 0.0418 | 0.4969 |
| test | Proposed Biomarker | Avg angular acceleration (RF-LF) | 266.0000 | 0.0386 | 0.5304 |
| test | Max jerk (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0379 | 0.5382 |
| test | U-turn time | Mean acceleration (RF-LF) | 266.0000 | 0.0321 | 0.6018 |
| test | Proposed Biomarker | Std angular acceleration (RF-LF) | 266.0000 | 0.0319 | 0.6049 |
| test | Std stride time | Mean acceleration (RF-LF) | 266.0000 | -0.0294 | 0.6326 |
| test | Proposed Biomarker | Mean angular velocity (RF-LF) | 266.0000 | 0.0292 | 0.6351 |
| test | Std stride time | Mean angular velocity (RF-LF) | 266.0000 | -0.0258 | 0.6754 |
| test | Max stride time | Mean acceleration (RF-LF) | 266.0000 | -0.0247 | 0.6880 |
| test | Avg angular velocity peak (RF-LF) | Mean acceleration (RF-LF) | 266.0000 | 0.0221 | 0.7201 |
| test | Proposed Biomarker | Std angular velocity (RF-LF) | 266.0000 | 0.0159 | 0.7958 |
| test | Max stride time | Mean angular velocity (RF-LF) | 266.0000 | -0.0076 | 0.9018 |
| test | Avg stride time | Mean angular velocity (RF-LF) | 266.0000 | 0.0073 | 0.9052 |
| test | Gait cadence | Mean angular velocity (RF-LF) | 266.0000 | 0.0047 | 0.9394 |
| test | Mean acceleration (RF-LF) | Std angular velocity (RF-LF) | 266.0000 | -0.0022 | 0.9721 |

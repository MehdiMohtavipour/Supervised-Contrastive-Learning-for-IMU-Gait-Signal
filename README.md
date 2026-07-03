Embedding-Distance Gait Biomarker (EDGB)

This repository contains the official implementation of the Embedding-Distance Gait Biomarker (EDGB), a supervised contrastive learning framework for quantitative gait assessment using multi-sensor wearable inertial measurement unit (IMU) signals.

EDGB learns a compact latent representation of gait from acceleration, angular velocity, jerk, and angular acceleration signals acquired from the head, lower back, left foot, and right foot. A clinically interpretable scalar biomarker is then constructed from distances to class-specific embedding prototypes, enabling discrimination between healthy, neurological, and orthopedic gait patterns.

Features
Multi-sensor IMU gait analysis
Supervised contrastive representation learning
Prototype-based digital biomarker construction
Subject-level evaluation to prevent information leakage
Statistical validation and repeatability analysis
Interpretability through correlation with conventional gait measures
Dataset

Experiments are conducted using the publicly available Voisard Gait Dataset:

https://github.com/CyrilVoisard/dataset_gait_1

Citation

If you use this code in your research, please cite our paper:

Supervised Contrastive Learning-based Digital Biomarker Discovery for Wearable IMU Gait Signals.

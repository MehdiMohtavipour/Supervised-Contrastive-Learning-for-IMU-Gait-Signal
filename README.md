# Embedding-Distance Gait Biomarker (EDGB)

EDGB is a deep representation learning framework that transforms multi-sensor wearable inertial measurement unit (IMU) signals into a compact digital biomarker for quantitative gait assessment.

Overview:

Gait impairments are common across neurological and orthopedic disorders and are traditionally assessed using handcrafted spatiotemporal features such as stride time and gait cadence. This repository introduces the Embedding-Distance Gait Biomarker (EDGB), a novel representation-learning approach that:

learns discriminative gait representations using supervised contrastive learning;
integrates information from multiple body-worn IMU sensors;
constructs a scalar biomarker based on distances to class-specific embedding prototypes;
enables quantitative assessment of healthy, neurological, and orthopedic gait patterns.

## Framework

<p align="center"> <img src="gait_framework.PNG" width="900"> </p>

The proposed framework consists of:

Multi-sensor IMU preprocessing and augmentation;
1D CNN encoder for representation learning;
Supervised contrastive training;
Prototype estimation in the embedding space;
Construction of the Embedding-Distance Gait Biomarker (EDGB);
Statistical and clinical validation.

## Main Results<br>
Macro AUC: 92.85%<br>
Effect size (η²): 0.71<br>
ICC(2,1): 0.82<br>
Superior discrimination compared with conventional gait measures.<br>

<p align="center"> <img src="contrastive_time_derivative_embedding_plots/pca_category_all_embeddings.png" width="900"> </p>

## Repository Structure

```text
EDGB/
├── conventional_extracted_features/                         # Conventional gait feature outputs
├── category_signal_sample_plots/                            # Example signal plots for each category
├── contrastive_time_derivative_embedding_plots/              # Embedding visualization plots
├── contrastive_time_derivative_encoder_outputs/              # Trained encoder outputs
├── contrastive_time_derivative_sensor_ablation_study/        # Sensor ablation study outputs
├── ic_reliability_outputs/                                  # ICC reliability results
├── pairwise_roc_biomarker_timing_features/                  # Pairwise ROC and biomarker timing feature results
├── time_derivative_embedding_handcrafted_anova_test/         # ANOVA results on the test set
├── time_derivative_embedding_handcrafted_anova_train/        # ANOVA results on the training set
├── time_derivative_embedding_handcrafted_comparison_plots/   # Comparison plots between embeddings and handcrafted features
├── time_derivative_embedding_handcrafted_comparison_test_only_plots/ # Test-only comparison plots
├── time_derivative_embedding_handcrafted_spearman/           # Spearman correlation results
│
├── README.md                                                # Project documentation
├── ablation_study.py                                        # Ablation study analysis
├── compare_time_derivative_embedding_with_handcrafted_features.py # Compare learned embeddings with handcrafted features
├── compare_time_derivative_embedding_with_handcrafted_features_test_only.py # Test-only comparison analysis
├── compute_embedding_handcrafted_icc_reliability.py          # ICC reliability analysis
├── compute_spearman_biomarker_handcrafted_correlations.py    # Spearman correlation between biomarkers and handcrafted features
├── contrastive_time_derivative_encoder_pretrain.py           # Contrastive encoder pretraining
├── evaluate_time_derivative_embedding_centers.py             # Evaluation of embedding centers
├── gait_framework.PNG                                       # Framework figure used in the README
├── plot_contrastive_time_derivative_embeddings.py            # Plot learned embeddings
├── plot_pairwise_roc_biomarker_timing_features.py            # Plot ROC and biomarker timing results
├── sensor_ablation_study.py                                 # Sensor ablation analysis
├── statistical_test_compare_time_derivative_embedding_with_handcrafted_features_test_only.py # Statistical testing on test set
└── statistical_test_compare_time_derivative_embedding_with_handcrafted_features_train_only.py # Statistical testing on training set
```

## Dataset
Experiments are conducted on the publicly available Voisard Gait Dataset:<br>
Voisard et al., A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts, Scientific Data, 2025.<br>
Repository:<br>
https://github.com/CyrilVoisard/dataset_gait_1<br>
Dataset:<br>
https://doi.org/10.6084/m9.figshare.28806086<br>

## Citation

If you find this repository useful, please cite:

@article{mohtavipour2026edgb,<br>
  title={Supervised Contrastive Learning-Based Digital Biomarker Discovery for Wearable IMU Gait Signals},<br>
  author={Mohtavipour, Seyed Mehdi},<br>
  journal={Preprint},<br>
  year={2026}<br>
}<br>

## Contact

Seyed Mehdi Mohtavipour<br>
Email: mahyar.m1990@gmail.com<br>

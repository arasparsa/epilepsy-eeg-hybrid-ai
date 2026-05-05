# Epilepsy EEG Hybrid AI

**An interpretable hybrid machine learning and deep learning framework for patient-independent epileptic seizure detection across public EEG databases.**

## Project Status

This repository is designed as a research-grade pipeline for public EEG epilepsy datasets. The long-term goal is to produce a reproducible GitHub project and a manuscript suitable for submission to a neuroscience or neural engineering journal.

This project is not intended to be a clinical diagnostic tool. It is a research framework for EEG signal processing, seizure detection, model comparison, and neurophysiological interpretability.

---

## Research Problem

Epileptic seizure detection using EEG is a heavily studied problem, but many published models suffer from weak validation design, especially when EEG windows from the same patient appear in both training and testing sets. This can inflate performance and reduce real-world generalizability.

The central question of this project is:

> Can an interpretable hybrid ML/DL model detect epileptic seizures across independent public EEG datasets under strict patient-independent validation?

---

## Core Hypothesis

A hybrid model that combines classical EEG biomarkers with deep neural embeddings will provide better generalization and interpretability than either handcrafted machine learning features or deep learning features alone.

---

## Main Contributions

1. A reproducible EEG preprocessing pipeline for public epilepsy EEG datasets.
2. Strict patient-independent evaluation to reduce data leakage.
3. Comparison between classical ML, deep learning, and hybrid ML/DL models.
4. Cross-dataset validation using public EEG datasets.
5. Explainability analysis linking model decisions to EEG features, channels, frequency bands, and time windows.

---

## Public Datasets

### 1. CHB-MIT Scalp EEG Database

Primary development dataset.

Planned use:

* Build the first version of the preprocessing pipeline.
* Train baseline ML and DL models.
* Perform patient-independent validation.

Expected data type:

* Pediatric scalp EEG recordings.
* EDF format.
* Annotated seizure onset and offset times.

### 2. Siena Scalp EEG Database

External validation dataset.

Planned use:

* Test whether the model generalizes to an independent adult EEG dataset.
* Evaluate cross-dataset robustness.

Expected data type:

* Adult scalp EEG recordings.
* EDF format.
* 10–20 EEG montage.
* Annotated seizure events.

### 3. TUH EEG Seizure Corpus

Advanced expansion dataset.

Planned use:

* Larger-scale clinical validation.
* Future benchmarking after the CHB-MIT and Siena pipeline is stable.

Expected data type:

* Clinical EEG recordings.
* Expert seizure annotations.
* Multiple seizure types.

---

## Repository Structure

```text
epilepsy-eeg-hybrid-ai/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── config/
│   ├── chbmit.yaml
│   ├── siena.yaml
│   └── tusz.yaml
│
├── data/
│   ├── raw/
│   │   ├── chbmit/
│   │   ├── siena/
│   │   └── tusz/
│   ├── interim/
│   └── processed/
│
├── metadata/
│   ├── chbmit_metadata.csv
│   ├── siena_metadata.csv
│   └── channel_mapping.csv
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_quality_check.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_baseline_ml_models.ipynb
│   ├── 05_deep_learning_models.ipynb
│   ├── 06_hybrid_model.ipynb
│   └── 07_explainability_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── io/
│   │   ├── load_edf.py
│   │   ├── load_annotations.py
│   │   └── build_metadata.py
│   │
│   ├── preprocessing/
│   │   ├── filters.py
│   │   ├── resampling.py
│   │   ├── channel_selection.py
│   │   ├── normalization.py
│   │   └── quality_control.py
│   │
│   ├── segmentation/
│   │   ├── windowing.py
│   │   └── labeling.py
│   │
│   ├── features/
│   │   ├── time_domain.py
│   │   ├── frequency_domain.py
│   │   ├── time_frequency.py
│   │   └── nonlinear.py
│   │
│   ├── models/
│   │   ├── baseline_ml.py
│   │   ├── cnn.py
│   │   ├── cnn_bilstm_attention.py
│   │   ├── transformer.py
│   │   └── hybrid_model.py
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   ├── gradcam.py
│   │   ├── attention_maps.py
│   │   └── band_importance.py
│   │
│   ├── evaluation/
│   │   ├── patient_split.py
│   │   ├── metrics.py
│   │   ├── cross_dataset.py
│   │   └── statistical_tests.py
│   │
│   └── visualization/
│       ├── plot_eeg.py
│       ├── plot_spectrogram.py
│       ├── plot_confusion_matrix.py
│       └── plot_results.py
│
├── scripts/
│   ├── 01_download_instructions.md
│   ├── 02_build_metadata.py
│   ├── 03_preprocess_dataset.py
│   ├── 04_extract_features.py
│   ├── 05_train_baselines.py
│   ├── 06_train_deep_model.py
│   ├── 07_train_hybrid_model.py
│   └── 08_generate_figures.py
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── logs/
│   └── trained_models/
│
├── manuscript/
│   ├── abstract.md
│   ├── introduction.md
│   ├── methods.md
│   ├── results.md
│   ├── discussion.md
│   └── figures/
│
└── tests/
    ├── test_preprocessing.py
    ├── test_segmentation.py
    ├── test_features.py
    └── test_metrics.py
```

---

## Methodological Design

### Step 1: Data Loading

The pipeline will read EEG files in EDF format and extract:

* patient ID
* recording ID
* sampling frequency
* EEG channel names
* seizure onset time
* seizure offset time
* recording duration

Output:

```text
metadata/dataset_metadata.csv
```

---

### Step 2: Channel Harmonization

Because public EEG datasets may use different channel names and montages, a common channel set will be selected where possible.

Initial target channels:

```text
Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
F7, F8, T3/T7, T4/T8, T5/P7, T6/P8, Fz, Cz, Pz
```

If channels are unavailable in a dataset, the pipeline will either:

1. use the intersection of available channels, or
2. run dataset-specific experiments with transparent reporting.

---

### Step 3: Preprocessing

Planned preprocessing steps:

1. Load EDF recording.
2. Select EEG channels only.
3. Resample to a common sampling rate.
4. Apply bandpass filtering.
5. Apply notch filtering if needed.
6. Normalize signal per patient or per recording.
7. Segment into fixed-length windows.
8. Assign seizure or non-seizure labels.

Default parameters:

```yaml
sampling_rate: 256
bandpass_low: 0.5
bandpass_high: 45
notch_frequency: 50_or_60
window_length_seconds: 4
overlap: 0.5
normalization: zscore_per_recording
```

---

### Step 4: Labeling Strategy

Initial binary classification:

```text
0 = non-seizure
1 = seizure
```

Future three-class classification:

```text
0 = interictal
1 = preictal
2 = ictal
```

Important rule:

Windows overlapping seizure onset and offset must be labeled carefully and consistently. Ambiguous boundary windows should either be excluded or assigned based on a predefined overlap threshold.

Default threshold:

```text
A window is labeled as seizure if at least 50% of the window overlaps with an annotated seizure interval.
```

---

### Step 5: Feature Extraction

Classical EEG features:

#### Time-domain features

* mean
* standard deviation
* variance
* skewness
* kurtosis
* zero-crossing rate
* line length
* Hjorth activity
* Hjorth mobility
* Hjorth complexity

#### Frequency-domain features

* delta power
* theta power
* alpha power
* beta power
* gamma power
* relative band power
* spectral entropy
* dominant frequency

#### Time-frequency features

* wavelet energy
* wavelet entropy
* short-time Fourier transform features

#### Nonlinear features

* sample entropy
* approximate entropy
* permutation entropy
* fractal dimension

---

### Step 6: Baseline Models

Baseline machine learning models:

* Logistic Regression
* Support Vector Machine
* Random Forest
* XGBoost
* LightGBM

These models will use handcrafted EEG features.

Purpose:

* establish fair baseline performance
* identify important EEG biomarkers
* compare classical and deep learning approaches

---

### Step 7: Deep Learning Models

Planned deep learning models:

1. 1D CNN
2. CNN + BiLSTM
3. CNN + BiLSTM + Attention
4. Transformer-based EEG model

Input format:

```text
samples × channels × time_points
```

The first deep model should be simple. A complex model should only be introduced after baseline models are stable.

---

### Step 8: Hybrid ML/DL Model

The main proposed model will combine:

1. handcrafted EEG features
2. deep embeddings extracted from CNN or CNN-BiLSTM
3. final classifier such as XGBoost or LightGBM

Conceptual architecture:

```text
Raw EEG Window
      │
      ├── Classical EEG Feature Extraction
      │         └── Feature Vector A
      │
      ├── Deep Neural Network Encoder
      │         └── Embedding Vector B
      │
      └── Concatenate A + B
                │
          XGBoost / LightGBM
                │
          Seizure Prediction
```

---

## Validation Strategy

This project will not use random window-level splitting as the main evaluation method.

Primary validation:

```text
patient-independent split
```

Planned evaluation levels:

1. Within-dataset patient-independent validation.
2. Leave-one-patient-out validation.
3. Cross-dataset validation.
4. External validation from CHB-MIT to Siena.

Example:

```text
Train: CHB-MIT patients 1–17
Validation: CHB-MIT patients 18–20
Test: CHB-MIT patients 21–22
External test: Siena dataset
```

---

## Metrics

Primary metrics:

* sensitivity / recall
* specificity
* precision
* F1-score
* AUROC
* AUPRC
* false alarm rate per hour

Secondary metrics:

* balanced accuracy
* confusion matrix
* patient-level performance
* seizure-level detection rate

Accuracy alone will not be treated as a sufficient performance metric because seizure EEG datasets are usually imbalanced.

---

## Explainability Plan

Explainability will be used to answer whether the model is learning neurophysiologically meaningful patterns or dataset-specific artifacts.

Planned explainability methods:

* SHAP for handcrafted-feature models
* Grad-CAM for CNN-based models
* attention weight visualization
* frequency-band importance analysis
* channel-level importance maps
* time-window importance maps

Key questions:

1. Which EEG frequency bands contribute most to seizure detection?
2. Which channels are most informative?
3. Does the model rely on physiologically plausible EEG activity?
4. Are feature importances stable across patients and datasets?

---

## Planned Figures for Manuscript

1. Study workflow and dataset design.
2. EEG preprocessing and segmentation pipeline.
3. Model architecture.
4. Patient-independent validation design.
5. Performance comparison across models.
6. Cross-dataset generalization results.
7. SHAP feature importance.
8. Grad-CAM or attention visualization.
9. Error analysis by patient and seizure duration.

---

## Reproducibility Principles

This project will follow these principles:

1. No patient leakage between train and test sets.
2. All preprocessing parameters documented.
3. All random seeds fixed.
4. All datasets cited and described transparently.
5. Results reported at both window level and patient level.
6. Negative results and failure cases documented.
7. Code structured for reuse and independent verification.

---

## Installation

```bash
git clone https://github.com/your-username/epilepsy-eeg-hybrid-ai.git
cd epilepsy-eeg-hybrid-ai
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.10+
```

Core packages:

```text
mne
numpy
pandas
scipy
scikit-learn
xgboost
lightgbm
torch
torchvision
matplotlib
seaborn
shap
pyedflib
```

---

## Example Workflow

```bash
python scripts/02_build_metadata.py --dataset chbmit
python scripts/03_preprocess_dataset.py --dataset chbmit
python scripts/04_extract_features.py --dataset chbmit
python scripts/05_train_baselines.py --dataset chbmit
python scripts/06_train_deep_model.py --dataset chbmit
python scripts/07_train_hybrid_model.py --dataset chbmit
python scripts/08_generate_figures.py
```

---

## Expected Manuscript Direction

Working title:

**An Interpretable Hybrid Machine Learning and Deep Learning Framework for Patient-Independent Epileptic Seizure Detection Across Public EEG Databases**

Target article type:

* original research article
* computational neuroscience
* EEG signal processing
* epilepsy detection
* interpretable AI

Potential target journals:

* Journal of Neural Engineering
* NeuroImage: Clinical
* Frontiers in Neuroscience
* Frontiers in Computational Neuroscience
* Epilepsia Open

---

## Current Development Roadmap

### Phase 1: CHB-MIT Pipeline

* Build metadata table.
* Load EDF files.
* Parse seizure annotations.
* Standardize channels.
* Segment EEG windows.
* Train baseline models.

### Phase 2: Deep Learning

* Train 1D CNN.
* Train CNN-BiLSTM.
* Add attention mechanism.
* Compare against baseline models.

### Phase 3: Hybrid Model

* Extract deep embeddings.
* Concatenate handcrafted features and embeddings.
* Train XGBoost or LightGBM classifier.
* Evaluate patient-independent performance.

### Phase 4: External Validation

* Apply trained pipeline to Siena.
* Evaluate cross-dataset performance.
* Analyze failure cases.

### Phase 5: Manuscript Preparation

* Generate figures.
* Write methods.
* Write results.
* Prepare discussion and limitations.

---

## Important Limitations

This project must be honest about its limitations:

1. Public EEG datasets differ in patient age, recording system, montage, and annotation style.
2. Cross-dataset performance may drop substantially.
3. High window-level accuracy does not guarantee clinical usefulness.
4. Seizure detection and seizure prediction are different tasks and must not be mixed.
5. External clinical validation is required before any medical use.

---

## Citation Notice

Users of this repository must cite the original EEG datasets and any related publications according to the dataset providers' instructions.

---

## License

Code license to be decided.

Recommended:

```text
MIT License for code
```

Dataset licenses must follow the original dataset providers' terms.

---

## Contact

Sara Parsa
Ph.D. Candidate in Neuroscience
Research interests: NeuroAI, EEG signal processing, computational neuroscience, epilepsy research, interpretable machine learning.

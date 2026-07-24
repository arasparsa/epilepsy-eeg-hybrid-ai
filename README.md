# Epilepsy EEG Hybrid AI

## Overview
Interpretable hybrid ML/DL framework for seizure detection using public EEG datasets.

### Key Features
- Patient-independent evaluation (no data leakage)
- Hybrid ML + Deep Learning architecture
- Cross-dataset validation (CHB-MIT → Siena)
- Explainable AI (SHAP, Grad-CAM, Attention)

### Goal
Build a research-grade pipeline for EEG-based epilepsy detection and publish a Q1 journal paper.

---
## Research Problem
Epileptic seizure detection using EEG is a heavily studied problem, but many published models suffer from weak validation design, especially when EEG windows from the same patient appear in both training and testing sets. This can inflate performance and reduce real-world generalizability.

The central question of this project is:

> Can an interpretable hybrid ML/DL model detect epileptic seizures across independent public EEG datasets under strict patient-independent validation?

---
## Contributions
1. A reproducible EEG preprocessing pipeline for public epilepsy EEG datasets.
2. Strict patient-independent evaluation to reduce data leakage.
3. Comparison between classical ML, deep learning, and hybrid ML/DL models.
4. Cross-dataset validation using public EEG datasets.
5. Explainability analysis linking model decisions to EEG features, channels, frequency bands, and time windows.

---
## Datasets
### 1. CHB-MIT Scalp EEG Database (training & development)
### 2. Siena Scalp EEG Database (external validation)
### 3. TUH EEG Seizure Corpus (future scaling)

---
## Repository Structure

---
## Installation

```bash
git clone https://github.com/arasparsa/epilepsy-eeg-hybrid-ai.git
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
## Citation Notice

Users of this repository must cite the original EEG datasets and any related publications according to the dataset providers' instructions.

---
## Contact

Sara Parsa
Ph.D. Candidate in Neuroscience
Research interests: NeuroAI, EEG signal processing, computational neuroscience, epilepsy research, interpretable machine learning.

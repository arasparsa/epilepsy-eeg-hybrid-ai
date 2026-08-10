\# CHB-MIT Leakage-Safe Data Pipeline Report



\## 1. Objective



The objective was to construct reproducible PyTorch datasets,

train-only normalization artifacts, and class-imbalance

metadata for the frozen patient-independent nested folds.



\## 2. Dataset input



\- Dataset: CHB-MIT v1.0.0

\- Preprocessing version: preprocessing\_v1

\- Segmentation version: segmentation\_v1

\- Splitting version: splitting\_v1

\- Window shape: 17 × 1024

\- Sampling rate: 256 Hz



\## 3. Signal loading



Window samples were read on demand from continuous FIF files.



A worker-local LRU cache reduced repeated FIF opening without

materializing overlapping windows as separate arrays.



\## 4. Signal units



MNE signal values were converted from volts to microvolts

before normalization.



The model input dtype was float32.



\## 5. Normalization



Per-channel global z-score normalization was used.



Mean and standard deviation were estimated exclusively from

continuous recordings belonging to the training subjects of

each fold.



Validation and test subjects contributed no normalization

statistics.



\## 6. Overlapping-window policy



Normalization statistics were calculated from continuous FIF

signals rather than the overlapping window table.



This prevented duplicated samples from being counted more than

once because of the 50% segmentation overlap.



\## 7. Nested-fold scalers



\- Inner-training scalers: \[FILL]

\- Outer-development scalers: \[FILL]

\- Total scaler artifacts: \[FILL]

\- Channels per scaler: 17

\- Scalers with invalid standard deviation: \[FILL]



\## 8. Class imbalance



The primary baseline strategy was:



\- natural training-window distribution;

\- BCEWithLogitsLoss;

\- positive-class weight calculated from training windows only.



Validation and test distributions were not balanced.



\## 9. Alternative imbalance strategies



The pipeline supports:



\- WeightedRandomSampler;

\- deterministic training-only negative sampling;

\- unweighted binary cross-entropy.



Alternative strategies must be selected inside inner

cross-validation.



\## 10. DataLoader policy



\- Training shuffle: enabled when no sampler is used

\- Validation shuffle: disabled

\- Test shuffle: disabled

\- Training drop\_last: enabled

\- Validation/test drop\_last: disabled

\- Initial Windows worker count: 0

\- FIF cache size per worker: 4



\## 11. Reproducibility



The following random generators were seeded:



\- Python random;

\- NumPy;

\- PyTorch CPU;

\- PyTorch CUDA;

\- DataLoader generator;

\- DataLoader workers.



Deterministic algorithms were requested where supported.



\## 12. Validation



\- Subject statistics complete: \[PASS/FAIL]

\- All scaler arrays finite: \[PASS/FAIL]

\- All scaler standard deviations positive: \[PASS/FAIL]

\- Validation subjects in scaler: 0

\- Test subjects in scaler: 0

\- Nested folds validated: \[FILL]

\- Dataset signal shape valid: \[PASS/FAIL]

\- Dataset labels valid: \[PASS/FAIL]

\- DataLoader batches valid: \[PASS/FAIL]



\## 13. Limitations



\- Global z-score normalization may reduce clinically meaningful

&#x20; absolute-amplitude differences.

\- Training folds have different class ratios.

\- Weighted loss can produce large positive gradients under

&#x20; extreme imbalance.

\- Weighted sampling can repeatedly present the same seizure

&#x20; windows.

\- PyTorch reproducibility is not guaranteed across all software

&#x20; and hardware versions.

\- Opening FIF files through multiple Windows workers may not

&#x20; improve performance.



\## 14. Reproducibility files



\- Configuration:

&#x20; config/chbmit\_data\_pipeline.yaml

\- Subject statistics:

&#x20; scripts/20\_build\_subject\_signal\_statistics.py

\- Fold artifacts:

&#x20; scripts/21\_build\_fold\_data\_artifacts.py

\- Validation:

&#x20; scripts/22\_validate\_chbmit\_data\_pipeline.py

\- Data pipeline version:

&#x20; data\_pipeline\_v1


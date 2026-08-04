\# CHB-MIT Patient-Independent Splitting Report



\## 1. Objective



The objective was to create a frozen, leakage-resistant,

patient-independent nested cross-validation protocol for

seizure-detection model development and evaluation.



\## 2. Grouping unit



All splitting was performed using independent subject IDs.



No subject, case, recording, or window was permitted to occur

in more than one role within a given train-validation-test

partition.



\## 3. Case-to-subject mapping



The original conservative case-level identifiers were replaced

by a verified subject mapping before splitting.



Cases chb01 and chb21 were assigned to the same independent

subject because the official CHB-MIT description reports that

they were collected from the same person.



\## 4. Outer cross-validation



\- Method: StratifiedGroupKFold

\- Outer folds: 5

\- Group: subject\_id

\- Stratification target: binary window label

\- Shuffle: enabled

\- Random seed: 42



Every independent subject appeared exactly once in an outer

test fold.



\## 5. Inner cross-validation



\- Method: StratifiedGroupKFold

\- Inner folds per outer fold: 4

\- Group: subject\_id

\- Purpose:

&#x20; - model selection;

&#x20; - hyperparameter optimization;

&#x20; - early stopping;

&#x20; - classification-threshold selection.



Outer-test subjects were never used in inner-fold decisions.



\## 6. Window eligibility



The master split manifest retained:



\- non-ictal windows;

\- boundary windows;

\- ictal windows;

\- clean-non-ictal indicators;

\- near-seizure indicators.



Boundary windows were not assigned binary metric labels.



\## 7. Dataset summary



\- Cases represented: \[FILL]

\- Independent subjects represented: \[FILL]

\- Recordings represented: \[FILL]

\- Total windows: \[FILL]

\- Non-ictal windows: \[FILL]

\- Boundary windows: \[FILL]

\- Ictal windows: \[FILL]



\## 8. Outer-fold summary



| Fold | Subjects | Cases | Recordings | Non-ictal | Boundary | Ictal |

|---:|---:|---:|---:|---:|---:|---:|

| 0 | | | | | | |

| 1 | | | | | | |

| 2 | | | | | | |

| 3 | | | | | | |

| 4 | | | | | | |



\## 9. Leakage checks



\- Subject overlap: None

\- Case overlap: None

\- Recording overlap: None

\- Window overlap: None

\- chb01/chb21 separated: No

\- Every subject tested exactly once: Yes

\- Every development subject used once as inner validation: Yes



\## 10. Class imbalance



Stratification attempted to reduce fold-level class imbalance,

but subject integrity took priority over exact class-ratio

matching.



No undersampling, oversampling, class weighting, or data

augmentation was applied during split construction.



\## 11. Normalization policy



No normalization parameter was estimated during this phase.



All scalers and transformations will be fitted exclusively on

the training subjects of each inner or outer training fold.



\## 12. Evaluation policy



The outer test fold will remain untouched during:



\- feature selection;

\- architecture selection;

\- hyperparameter optimization;

\- early stopping;

\- sampling-strategy selection;

\- threshold optimization.



\## 13. Limitations



\- The number of independent subjects is limited.

\- Seizure burden varies substantially between subjects.

\- Perfect fold-level class balance is not possible without

&#x20; violating subject independence.

\- Nested cross-validation is computationally expensive.

\- Internal cross-validation does not replace external-dataset

&#x20; validation.



\## 14. Reproducibility



\- Split configuration:

&#x20; config/chbmit\_splitting.yaml

\- Subject mapping:

&#x20; config/chbmit\_case\_subject\_mapping.csv

\- Outer split seed: 42

\- Inner base seed: 4200

\- Splitting version: splitting\_v1

\- Git branch:

&#x20; feature/chbmit-patient-splits


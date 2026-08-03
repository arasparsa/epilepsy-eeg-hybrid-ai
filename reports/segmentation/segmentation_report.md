\# CHB-MIT Segmentation and Window-Labeling Report



\## 1. Objective



The objective was to create a deterministic, sample-aligned

window manifest from the validated continuous preprocessed

CHB-MIT recordings.



\## 2. Input data



\- Dataset: CHB-MIT v1.0.0

\- Preprocessing version: preprocessing\_v1

\- Harmonization policy: harmonization\_v1

\- Channel count: 17

\- Sampling rate: 256 Hz

\- Input format: continuous MNE FIF



\## 3. Window definition



\- Window duration: 4 seconds

\- Window samples: 1024

\- Stride: 2 seconds

\- Stride samples: 512

\- Consecutive-window overlap: 50%

\- Incomplete final windows: discarded

\- Cross-recording windows: prohibited



\## 4. Labeling policy



Each window was assigned one of three labels:



\- non\_ictal: zero seizure overlap;

\- boundary: positive overlap below 50%;

\- ictal: overlap of at least 50%.



Boundary windows remained in the complete manifest but were

not assigned a binary baseline label.



\## 5. Near-seizure metadata



For every window, the distance to the nearest seizure

interval was calculated.



Non-ictal windows at least 60 seconds from every seizure were

marked as clean\_non\_ictal.



No near-seizure window was automatically removed.



\## 6. Storage strategy



Window arrays were not materialized as separate files.



Each window can be reconstructed using:



\- FIF path;

\- start sample;

\- exclusive stop sample;

\- fixed channel ordering.



This design avoids duplicating overlapping EEG samples.



\## 7. Results



\- Recordings segmented: \[FILL]

\- Failed recordings: \[FILL]

\- Total windows: \[FILL]

\- Non-ictal windows: \[FILL]

\- Boundary windows: \[FILL]

\- Ictal windows: \[FILL]

\- Clean non-ictal windows: \[FILL]

\- Cases represented: \[FILL]

\- Seizures represented: \[FILL]



\## 8. Seizure coverage



\- Seizures overlapping at least one window: \[FILL]

\- Seizures generating at least one ictal window: \[FILL]

\- Seizures generating boundary windows only: \[FILL]



\## 9. Validation



\- Unique window IDs: \[PASS/FAIL]

\- Fixed sample count: \[PASS/FAIL]

\- Fixed stride: \[PASS/FAIL]

\- Valid label/fraction relationships: \[PASS/FAIL]

\- Windows inside FIF boundaries: \[PASS/FAIL]

\- Signal-loading sample validation: \[PASS/FAIL]

\- Manual labels reviewed: \[FILL]



\## 10. Class imbalance



The complete window distribution was retained without

undersampling or oversampling.



Balancing strategies will be fitted only to the training set

after patient-independent splitting.



\## 11. Methodological limitations



\- A 4-second window may be too long for some short seizures.

\- A 50% overlap threshold may classify short transition

&#x20; windows as boundary.

\- Overlapping windows are statistically dependent.

\- Window-level metrics do not replace event-level evaluation.

\- The current task is seizure detection, not prediction.

\- Boundary-label handling requires sensitivity analysis.



\## 12. Reproducibility



\- Configuration:

&#x20; config/chbmit\_segmentation.yaml

\- Manifest builder:

&#x20; scripts/14\_build\_chbmit\_window\_manifest.py

\- Validation:

&#x20; scripts/15\_validate\_chbmit\_windows.py

\- Segmentation version:

&#x20; segmentation\_v1

\- Git branch:

&#x20; feature/chbmit-segmentation


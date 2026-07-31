\# CHB-MIT Preprocessing Report



\## 1. Objective



The objective was to produce a reproducible, fixed-channel,

filtered representation of all recordings included by the

frozen harmonization policy.



\## 2. Inputs



\- Dataset: CHB-MIT v1.0.0

\- Inclusion manifest:

&#x20; metadata/harmonization/chbmit\_recording\_inclusion\_manifest.csv

\- Seizure metadata:

&#x20; metadata/chbmit\_seizures.csv

\- Channel policy:

&#x20; harmonization\_v1

\- Primary channel set:

&#x20; primary\_17



\## 3. Channel processing



Each included EDF was:



1\. loaded without modifying the source file;

2\. normalized using validated naming rules;

3\. restricted to the frozen 17-channel montage;

4\. reordered to an identical sequence.



No missing channel was imputed or interpolated.



\## 4. Annotation processing



Validated seizure onset and offset intervals were converted

to MNE ictal annotations.



Annotation counts and temporal boundaries were validated

before and after signal processing.



\## 5. Filtering



\- Filter type: FIR band-pass

\- Lower cutoff: 0.5 Hz

\- Upper cutoff: 45 Hz

\- Phase: zero

\- FIR design: firwin

\- FIR window: Hamming

\- Padding: reflect\_limited



Filtering was applied to continuous recordings before

segmentation.



\## 6. Notch filtering



No separate notch filter was applied because the primary

low-pass cutoff was below the 60-Hz line frequency.



\## 7. Resampling



No resampling was performed. The original CHB-MIT sampling

rate of 256 Hz was retained.



\## 8. Referencing



No rereferencing was performed because the selected CHB-MIT

channels were already bipolar derivations.



\## 9. Normalization



No amplitude normalization was performed in this phase.

Normalization parameters will be fitted using training data

after patient-independent splitting.



\## 10. Artifact policy



Phase-4 QC flags were preserved as metadata and did not cause

automatic channel or recording deletion.



\## 11. Storage



\- Output format: MNE FIF

\- Numerical format: single precision

\- Signal directory:

&#x20; data/interim/chbmit/preprocessing\_v1

\- Raw EDF files modified: No



\## 12. Results



\- Recordings selected: \[FILL]

\- Successfully processed: \[FILL]

\- Failed recordings: \[FILL]

\- Output channel count: 17

\- Output sampling rate: 256 Hz

\- Recordings with ictal annotations: \[FILL]

\- Total ictal annotations: \[FILL]



\## 13. Validation



\- Files with correct channel order: \[FILL]

\- Files with correct sampling rate: \[FILL]

\- Files with preserved duration: \[FILL]

\- Files with valid annotations: \[FILL]

\- Files with finite inspected samples: \[FILL]

\- Manual reviews completed: \[FILL]



\## 14. Limitations



\- The 0.5–45 Hz band excludes high-frequency activity above

&#x20; 45 Hz.

\- Zero-phase filtering is non-causal and is suitable for

&#x20; offline analysis, not real-time deployment.

\- No comprehensive artifact-removal procedure was applied.

\- Bipolar channel geometry limits conventional interpolation

&#x20; and topographic interpretation.

\- Normalization is deferred to the model pipeline.



\## 15. Reproducibility



\- Configuration:

&#x20; config/chbmit\_preprocessing.yaml

\- Runner:

&#x20; scripts/12\_preprocess\_chbmit.py

\- Validation:

&#x20; scripts/13\_validate\_preprocessed\_data.py

\- Git branch:

&#x20; feature/chbmit-preprocessing

\- Preprocessing version:

&#x20; preprocessing\_v1


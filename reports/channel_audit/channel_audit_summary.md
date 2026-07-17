\# CHB-MIT Channel Audit Report



\## 1. Objective



The objective of this audit was to identify channel-name,

channel-count, channel-order, sampling-rate, and montage

heterogeneity across all available CHB-MIT EDF recordings

before preprocessing and model development.



\## 2. Dataset



\- Dataset: CHB-MIT Scalp EEG Database

\- Successfully inspected EDF files: \[FILL]

\- Failed EDF files: \[FILL]

\- Cases represented: \[FILL]

\- Minimum channels per file: \[FILL]

\- Maximum channels per file: \[FILL]

\- Modal channel count: \[FILL]

\- Sampling rates observed: \[FILL]

\- Unique normalized channel labels: \[FILL]

\- Distinct unordered channel signatures: \[FILL]



\## 3. Channel normalization



Only superficial label differences were normalized:



\- leading/trailing whitespace

\- character case

\- spaces around hyphens

\- leading EEG prefix

\- terminal REF or LE suffixes



Original EDF labels were preserved in the metadata.



No anatomical remapping or channel removal was performed.



\## 4. High-coverage channels



Channels present in at least 95% of files:



| Channel | Files | File coverage | Patients | Patient coverage |

|---|---:|---:|---:|---:|

| \[FILL] | | | | |



\## 5. Channel-count heterogeneity



\[Describe the distribution and identify unusual files.]



\## 6. Channel-order heterogeneity



\[Report whether identical channel sets appeared in different orders.]



\## 7. Potential duplicate channels



\[Describe suffix-based duplicates such as channels ending in -0

and report the signal-comparison results.]



\## 8. Candidate channel-set coverage



| Channel set | Channels | Complete files | File coverage | Patients |

|---|---:|---:|---:|---:|

| core\_17 | 17 | | | |

| extended\_18 | 18 | | | |



\## 9. Files requiring review



\[Summarize unusual sampling rates, channel counts, duplicates,

or unreadable EDF files.]



\## 10. Preliminary decision



No final channel set was selected during the audit.



The next phase will compare candidate channel sets based on:



1\. file retention,

2\. patient retention,

3\. seizure retention,

4\. symmetry and neurophysiological coverage,

5\. compatibility with external datasets,

6\. duplicate-channel handling.



\## 11. Reproducibility



\- Audit script: scripts/02\_audit\_chbmit\_channels.py

\- Duplicate check: scripts/03\_check\_duplicate\_channels.py

\- Git branch: feature/chbmit-channel-audit

\- Dataset version: CHB-MIT 1.0.0

\- Raw EDF modification: None


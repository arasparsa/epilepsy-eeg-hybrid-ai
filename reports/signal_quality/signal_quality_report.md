\# CHB-MIT Raw Signal Quality-Control Report



\## 1. Objective



The objective was to quantify raw-signal quality across all

available CHB-MIT EDF recordings before filtering, resampling,

channel removal, rereferencing, or segmentation.



\## 2. Input data



\- Dataset: CHB-MIT

\- Dataset version: 1.0.0

\- Recording metadata: metadata/chbmit\_recordings.csv

\- Seizure metadata: metadata/chbmit\_seizures.csv

\- Raw EDF modification: None



\## 3. Analysis unit



Quality metrics were calculated at three levels:



1\. fixed-duration recording chunks;

2\. individual channels within recordings;

3\. complete EDF recordings.



Chunk duration: 60 seconds  

Minimum final chunk duration: 10 seconds



\## 4. Ictal-awareness



Each QC chunk was labeled as:



\- non-ictal;

\- seizure-boundary;

\- ictal.



Seizure-related physiological changes were not automatically

classified as artifacts.



\## 5. Metrics



\### Data integrity



\- finite/non-finite sample fraction;

\- NaN fraction;

\- positive and negative infinity fraction;

\- zero fraction.



\### Amplitude and dispersion



\- mean;

\- median;

\- standard deviation;

\- variance;

\- RMS;

\- minimum and maximum;

\- peak-to-peak range;

\- 1st–99th percentile robust range;

\- IQR;

\- median absolute deviation;

\- skewness;

\- kurtosis.



\### Temporal continuity



\- first-difference statistics;

\- line length;

\- flat-difference fraction;

\- longest flat run;

\- flatline sample fraction.



\### Spectral properties



\- total spectral power;

\- line-noise power and ratio;

\- high-frequency power and ratio;

\- spectral entropy.



\## 6. Provisional review flags



The first QC version used transparent, configurable thresholds.

Flags indicated the need for review and did not automatically

exclude recordings or channels.



\## 7. Results



\- EDF recordings selected: \[FILL]

\- EDF recordings successfully processed: \[FILL]

\- Failed recordings: \[FILL]

\- Channel-level records: \[FILL]

\- Chunk-channel records: \[FILL]

\- Chunks requiring review: \[FILL]

\- Channels requiring review: \[FILL]

\- Recordings requiring review: \[FILL]



\## 8. Manual review



\- Clean recordings reviewed: \[FILL]

\- Flagged recordings reviewed: \[FILL]

\- Ictal flagged chunks reviewed: \[FILL]

\- Confirmed technical artifacts: \[FILL]

\- Flags explained by ictal activity: \[FILL]

\- Uncertain cases: \[FILL]



\## 9. Important methodological decisions



1\. Raw EDF files were read without modification.

2\. No filtering or rereferencing was performed.

3\. Data were processed in fixed-duration chunks.

4\. Amplitude metrics were reported in microvolts.

5\. EDF headers remained authoritative for sampling rate.

6\. QC flags were not equivalent to exclusion decisions.

7\. Ictal and non-ictal chunks were analyzed separately.



\## 10. Limitations



\- Threshold-based flags are dataset-dependent.

\- CHB-MIT does not provide comprehensive artifact annotations.

\- Bipolar montage limits direct application of some spatial

&#x20; bad-channel methods designed for referential EEG.

\- Flatline and spectral thresholds require manual validation.

\- Extreme physiological seizure activity may resemble artifact.



\## 11. Reproducibility



\- Git branch: feature/chbmit-signal-quality

\- Configuration: config/chbmit\_qc.yaml

\- Metric implementation: src/quality/metrics.py

\- Main runner: scripts/07\_run\_chbmit\_qc.py

\- Validation: scripts/08\_validate\_chbmit\_qc.py


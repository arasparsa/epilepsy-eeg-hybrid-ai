\# CHB-MIT Metadata and Annotation Validation Report



\## 1. Objective



The objective was to parse the official CHB-MIT case summary

files and construct recording-level and seizure-level metadata

tables linked to the EDF header inventory generated during the

channel audit.



\## 2. Annotation source



Seizure onset and offset times were extracted from the official

case-level CHB-MIT summary files. Times were represented in

seconds relative to the beginning of each EDF recording.



\## 3. Dataset version



\- Dataset: CHB-MIT Scalp EEG Database

\- Version: 1.0.0

\- Source: PhysioNet

\- Local raw data modification: None



\## 4. Parser design



The parser supports:



\- indexed seizure annotations;

\- unindexed seizure annotations;

\- multiple seizures in one recording;

\- zero-seizure recordings;

\- clock times crossing midnight;

\- extended clock-hour values greater than 23;

\- preservation of the original source summary file.



\## 5. Output tables



\### Recording-level table



File: metadata/chbmit\_recordings.csv



Each row represents one EDF recording.



\### Seizure-level table



File: metadata/chbmit\_seizures.csv



Each row represents one annotated seizure interval.



\## 6. Dataset statistics



\- Successfully matched EDF recordings: \[FILL]

\- Cases represented: \[FILL]

\- Seizure-containing recordings: \[FILL]

\- Annotated seizures: \[FILL]

\- Total recording duration: \[FILL] hours

\- Total annotated ictal duration: \[FILL] minutes

\- Median seizure duration: \[FILL] seconds

\- Minimum seizure duration: \[FILL] seconds

\- Maximum seizure duration: \[FILL] seconds



\## 7. Validation rules



The following conditions were evaluated:



1\. Every EDF had a corresponding summary recording block.

2\. Every summary recording block had a corresponding EDF.

3\. Reported and parsed seizure counts matched.

4\. Every seizure onset was non-negative.

5\. Every seizure offset exceeded its onset.

6\. Every seizure interval was within EDF duration.

7\. Every seizure identifier was unique.

8\. Recording-level and seizure-level event counts reconciled.

9\. EDF and summary-clock durations were compared.

10\. Potential overlapping seizure intervals were flagged.



\## 8. Validation results



\- Errors: \[FILL]

\- Warnings: \[FILL]

\- Files manually reviewed: \[FILL]

\- Invalid seizure intervals: \[FILL]



\## 9. Important methodological decision



The EDF header duration was treated as the authoritative signal

duration. Summary clock duration was used as an independent

consistency check.



\## 10. Limitations



\- Summary annotations provide seizure onset and offset intervals,

&#x20; but not necessarily detailed seizure-type labels.

\- Case identifiers must not automatically be assumed to represent

&#x20; independent subjects without a verified case-to-subject mapping.

\- Minor duration discrepancies may reflect clock rounding or EDF

&#x20; header conventions.

\- Clinical annotation uncertainty cannot be quantified from the

&#x20; summary files alone.



\## 11. Reproducibility



\- Git branch: feature/chbmit-metadata

\- Parser: src/metadata/chbmit\_summary\_parser.py

\- Parse script: scripts/04\_parse\_chbmit\_summaries.py

\- Metadata builder: scripts/05\_build\_chbmit\_metadata.py

\- Validation script: scripts/06\_validate\_chbmit\_metadata.py


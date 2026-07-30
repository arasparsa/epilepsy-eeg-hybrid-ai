\# CHB-MIT Channel Harmonization Report



\## 1. Objective



The objective was to define a deterministic, reproducible,

and leakage-independent channel inclusion policy for

patient-independent seizure detection.



\## 2. Inputs



\- Phase-2 channel audit

\- Phase-3 recording and seizure metadata

\- Phase-4 raw signal-quality metadata



\## 3. Candidate channel sets



| Set | Role | Channels | Files retained | Cases retained | Seizures retained |

|---|---|---:|---:|---:|---:|

| primary\_17 | Primary | 17 | | | |

| extended\_18 | Sensitivity | 18 | | | |

| temporal\_8 | Ablation | 8 | | | |

| parasagittal\_8 | Ablation | 8 | | | |



\## 4. Primary policy



The primary analysis used the fixed 17-channel bipolar

montage defined in the version-controlled configuration.



\## 5. Canonical naming



Only superficial name normalization and explicitly validated

aliases were permitted. Original EDF channel labels remained

available in the audit metadata.



\## 6. Ordering



Every included recording was reordered to the same explicit

channel sequence. EDF-native ordering was not used as model

input ordering.



\## 7. Missing channels



Recordings missing any primary target channel were excluded

from the primary fixed-input analysis and retained in the

exclusion manifest for possible future channel-masked models.



\## 8. QC integration



Provisional phase-4 QC flags were not automatic exclusions.

Only manually confirmed unusable recordings could be removed.



\## 9. Retention results



\- Recording retention: \[FILL]

\- Recording-hour retention: \[FILL]

\- Case retention: \[FILL]

\- Seizure retention: \[FILL]

\- Ictal-time retention: \[FILL]



\## 10. Limitations



\- A fixed montage excludes recordings missing even one target

&#x20; derivation.

\- The primary montage does not include T8-P8.

\- Reduced montages may lose spatial seizure information.

\- CHB-MIT bipolar channels cannot be treated as independent

&#x20; electrode potentials.

\- External datasets may require a separate harmonization layer.



\## 11. Reproducibility



\- Policy: config/chbmit\_channel\_harmonization.yaml

\- Aliases: config/chbmit\_channel\_aliases.csv

\- Evaluation: scripts/09\_evaluate\_channel\_sets.py

\- Manifest: scripts/10\_build\_inclusion\_manifest.py

\- Validation: scripts/11\_validate\_harmonization.py

\- Git branch: feature/chbmit-channel-harmonization


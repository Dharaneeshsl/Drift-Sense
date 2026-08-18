# Ablation

Each profile recomputes candidates and scoring. Visible-anchor splits are already discriminative; therefore these rows do not isolate periodicity-aware value. The real test is a future adversarial near-decoy set.

| Profile | ID strict | OOD strict | Hard periodic strict | ID mean error | OOD mean error |
|---|---:|---:|---:|---:|---:|
| intensity_only | 18/20 | 8/10 | 2/5 | 0.192 px | 0.252 px |
| +gradient | 19/20 | 8/10 | 2/5 | 0.188 px | 0.249 px |
| +high_pass | 18/20 | 8/10 | 2/5 | 0.166 px | 0.232 px |
| +squared_error_agreement | 18/20 | 8/10 | 1/5 | 0.170 px | 0.235 px |
| +fourier_agreement | 18/20 | 8/10 | 1/5 | 0.169 px | 0.234 px |
| driftsense_fm_full | 18/20 | 8/10 | 1/5 | 0.169 px | 0.234 px |

# OpenSplice Phase-R development result

**Decision:** `phase_r_negative_continue_wider_ladder_confirmation_closed`

All Gate-0, eligibility, completeness, configuration-fingerprint, and hash-tree checks passed. Confirmation remained unopened.

## Eligibility

| Exon | Eligible effects | Frozen effects | Neutrals |
|---|---:|---:|---:|
| BRAF | 6 | 6 | 4 |
| SLC25A48 | 6 | 6 | 4 |

## Frozen ranking

Top candidate: `pre_attention/layer4/S`; Q = 0.0572591; gate pass = False.

| Rank | Candidate | Q | BRAF median B/q | SLC25A48 median B/q | Pass |
|---:|---|---:|---:|---:|:---:|
| 1 | pre_attention/L4/S | 0.0572591 | 0.181562/0.186732 | 0.0576575/0.0572591 | False |
| 2 | post_attention/L3/S | 0.0572591 | 0.181562/0.186732 | 0.0576575/0.0572591 | False |
| 3 | post_mlp/L3/S | 0.0572591 | 0.181562/0.186732 | 0.0576575/0.0572591 | False |
| 4 | pre_attention/L1/S | 0.0553429 | 0.249161/0.24423 | 0.0553429/0.0553429 | False |
| 5 | post_attention/L0/S | 0.0553429 | 0.249161/0.24423 | 0.0553429/0.0553429 | False |
| 6 | post_mlp/L0/S | 0.0553429 | 0.249161/0.24423 | 0.0553429/0.0553429 | False |
| 7 | pre_attention/L3/S | 0.0542345 | 0.211078/0.224053 | 0.055267/0.0542345 | False |
| 8 | post_attention/L2/S | 0.0542345 | 0.211078/0.224053 | 0.055267/0.0542345 | False |
| 9 | post_mlp/L2/S | 0.0542345 | 0.211078/0.224053 | 0.055267/0.0542345 | False |
| 10 | post_attention/L5/S | 0.0537492 | 0.160657/0.151174 | 0.0549065/0.0537492 | False |

The ranking is development-only. The first passing ranked candidate, if one exists, is a locked logit-margin residual hypothesis rather than biological validation. If no candidate passes, confirmation remains closed.

Raw JSON tree SHA-256: `0b24d58491e33f14277c55396614714061d2143ba6dd7641671fab0c854298e9`

# Individual-channel causal validation result

Three model coordinates pass the frozen individual necessity rule:
BRAF E16 channel 3, and SLC25A48 channel 175 at both E2 and E1.
All four parent 8-channel subspaces also show positive, spatially
localized sufficiency for the gene that selected them.

## Advancing individual channels

| Gene | Stage | Channel | Effect median loss | Neutral median loss | Positive effects |
|---|---|---:|---:|---:|---:|
| BRAF | E16 | 3 | 0.01254 | 0.00656 | 4/6 |
| SLC25A48 | E1 | 175 | 0.03987 | 0.00000 | 4/6 |
| SLC25A48 | E2 | 175 | 0.04512 | 0.00000 | 5/6 |

The SLC25A48 result is especially coherent: the same channel number,
175, is necessary at two successive resolutions. Its E2 and E1 parent
subspaces recover median `B=0.01462` and `B=0.02661` by themselves at
the intended site.

## Eight-channel localized sufficiency

| Selected gene | Subspace | Intended median B | Positive effects |
|---|---|---:|---:|
| BRAF | `E32_c0000_0007` | 0.01137 | 5/6 |
| BRAF | `E16_c0000_0007` | 0.00141 | 4/6 |
| SLC25A48 | `E2_c0168_0175` | 0.01462 | 6/6 |
| SLC25A48 | `E1_c0168_0175` | 0.02661 | 5/6 |

Every one of the 160 equal-shape upstream/downstream controls has
exactly zero recovery. The E32 BRAF subspace is localized and
sufficient, but no constituent channel passes the individual
effect-over-neutral rule; its behavior may require within-subspace
synergy or redundancy.

## Claim boundary and next step

All 960 model applies completed and every runtime causal control
passed. Identity and full-route repeats were exact, and confirmation
data remained sealed.

This identifies causal model coordinates, not a motif, RBP or cellular
mechanism. Next test E16 channel 3 and E2/E1 channel 175 by themselves
at intended and shifted positions. If those single-coordinate tests
survive, their sequence preferences can be characterized with
activation optimization, motif comparison and controlled edits.

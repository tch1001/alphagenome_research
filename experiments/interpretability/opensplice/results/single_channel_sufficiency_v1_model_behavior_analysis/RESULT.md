# Single-channel spatial sufficiency result

Two causal AlphaGenome coordinates pass the frozen standalone
sufficiency rule: BRAF E16 channel 3 and SLC25A48 E2 channel 175.
SLC25A48 E1 channel 175 remains necessary but is not consistently
sufficient by itself.

| Selected gene | Coordinate | Effect median B | Neutral median B | Positive effects | Pass |
|---|---|---:|---:|---:|:---:|
| BRAF | `E16_c0003` | 0.00573 | 0.00180 | 4/6 | yes |
| SLC25A48 | `E1_c0175` | 0.02486 | 0.00000 | 3/6 | no |
| SLC25A48 | `E2_c0175` | 0.01533 | 0.00000 | 6/6 | yes |

All 120 upstream/downstream values are exactly zero.

The E2:175 result is the cleanest current feature: it was necessary in
the individual screen, and by itself it transfers positive reciprocal
splice-effect recovery in 6/6 SLC25A48 effects (`median B=0.01533`),
versus zero median in experimental neutrals and zero at shifted sites.
It is also effectively silent in the BRAF effects, supporting an
exon-specific rather than universal splice representation.

BRAF E16:3 passes more modestly (`median B=0.00573`, 4/6 effects,
neutral median `B=0.00180`). E1:175 has a positive median in SLC25A48
but only 3/6 positive effects, indicating context dependence at that
resolution.

All 260 applies completed and every causal runtime control passed.
Confirmation data remained sealed. These remain model features, not
named biological factors. The next step is sequence-level
characterization of E2:175 and E16:3, with E1:175 as a dependency
comparison, followed by controlled motif edits.

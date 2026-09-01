# Independent spatial-analysis audit

This audit reread all 240 raw condition records and recomputed
bidirectional recovery with an algebraically equivalent, sign-reversed
formula. It did not call the main analyzer.

- Raw condition count: 240
- Raw condition tree SHA-256: `540bfbae859aac27de7e201d44d0a2ee3dc2aab563afa9ab58e83c081e5766ce`
- Maximum raw recovery difference: `0.0`
- Maximum aggregation difference from `ANALYSIS.json`: `0.0`
- All shifted controls exactly zero: `true`

| Support | BRAF median B | SLC25A48 median B |
|---|---:|---:|
| V | 0.41409 | 0.39649 |
| A | 0.38700 | 0.65977 |
| D | 0.27905 | 0.00893 |
| S | 0.51613 | 0.66509 |

The independent arithmetic and aggregation match exactly.

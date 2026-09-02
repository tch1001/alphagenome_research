# Independent eight-channel refinement audit

This audit reread all raw active target means, used a sign-reversed
recovery formula, and independently rebuilt every median, cross-gene
ranking, within-gene ranking and individual-channel candidate set. It
did not call the main analyzer.

- Full-route records: 20
- Child records: 400
- Raw active tree SHA-256: `b64d69db65a0cf0954013d38d62780d2a16e6a2a8fc8ec84f05c29e4f71eddb7`
- Maximum raw recovery difference: `0.0`
- Maximum median difference from `ANALYSIS.json`: `0.0`
- All cross-gene and within-gene rankings match exactly: true

Independently selected children:

- BRAF: `E32_c0000_0007`, `E16_c0000_0007`
- SLC25A48: `E2_c0168_0175`, `E1_c0168_0175`

The independent arithmetic, aggregation and selection match exactly.

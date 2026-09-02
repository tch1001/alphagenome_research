# Independent encoder feature-decomposition audit

The audit independently reread all raw component arrays and selected weights. It rebuilt role-aligned allele-difference vectors without importing the main analyzer.

- Raw records: 20
- SLC25A48 effects/neutrals: 6/4
- E1 direct/update/output effect median L2: 0.406511 / 12.877382 / 13.111750
- E1/E2 neutral output median L2: 1.358522 / 1.340567
- E2 effect median amplification: 1.128960 (6/6 amplified)
- Maximum direct-weight/activation error: 0.000976562
- Maximum difference from the main analysis: 0.0

The independently recomputed claims match exactly.

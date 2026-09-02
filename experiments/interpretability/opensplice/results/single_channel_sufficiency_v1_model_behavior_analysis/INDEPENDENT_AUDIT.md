# Independent single-channel sufficiency audit

This audit reread every raw target mean, used an algebraically
equivalent sign-reversed recovery formula, and independently rebuilt
all spatial medians, contrasts and pass decisions. It did not call the
main analyzer.

- Full-route records: 20
- Condition records: 180
- Raw active tree SHA-256: `95cbfff302df5fc171ad239f35f1026b3827662a31c263689c47286889a0391b`
- Maximum raw recovery difference: `0.0`
- Maximum aggregation difference from `ANALYSIS.json`: `0.0`
- All 120 shifted values exactly zero: true

Independently passing coordinates:

- `E16_c0003`
- `E2_c0175`

The independent arithmetic, aggregation and decisions match exactly.

# Independent audit of the exploratory encoder-skip analysis

A separate standard-library implementation reread all 5,120 development
coalition records and recomputed target means directly from raw
relevant/padding logits. It used the permutation definition of Shapley
values rather than the subset-weight formula used by the primary analyzer.

- Primary `ANALYSIS.json` SHA-256: `d6c0b443e0e3b76b08adf2feac4e318902b6db9fd4b37c28cbaec24d950add4b`
- Raw coalition tree SHA-256: `95ddc79c634fecdd4b4e43e090ac760cdc26a268f09fed7cee76843f982e45de`
- Maximum absolute Shapley difference: `1.207e-13`
- Anchor medians, mask selection and effect-versus-neutral medians: exact
- Confirmation artifacts read: no
- Incomplete OOD-anchor artifacts read: no
- Model calls: zero

The independent audit confirms the exploratory mask-110 computational
route and the BRAF neutral-control failure. It does not convert that route
into a biologically specific mechanism.

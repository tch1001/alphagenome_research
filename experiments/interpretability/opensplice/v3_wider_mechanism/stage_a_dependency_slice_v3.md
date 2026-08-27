# Stage-A dependency slice: mandatory closures and whole T/E branches

This implementation is a development-only, dependency-ordered subset of
Stage A in `prospective_protocol_v3.md`. It does not open or accept confirmation
variants and does not replace the remaining Stage-A route census.

## Implemented seams

`WholeSequenceBatchTransfer` transfers a complete live `[batch, sequence,
channel]` tensor between rows of the frozen six-row batch. Full tensors remain
inside one JAX executable. Three separate exact booleans report whether the
natural row equals its same-allele baseline `[0,1,1,1,0,0]`, whether the
effective row equals its own natural value, and whether the effective row
equals the requested intervention donor `[0,1,0,1,1,0]`. The intervention-donor
audit is meaningful only for enabled recipients; in all-false Gate 0, cross
rows 2 and 4 should differ from those opposite-allele donors. Four compact
uint32 reductions of each natural T/E tensor are also returned for cross-call
repeat checks, avoiding host copies of the seven high-resolution encoder skips.

The opt-in `AlphaGenome.forward_trunk_with_stage_a_branches` path exposes:

- **T:** the entire post-transformer 128-bp sequence. The effective tensor is
  used by both the decoder and the 128-bp contribution to `OutputEmbedder`.
- **E:** all seven encoder skips, transferred independently at their point of
  consumption by the decoder.
- **T+E:** both complete branch inputs in the same forward execution.
- **Final A/D closure:** the complete post-GELU 1-bp embedding vectors at the
  strand-aware canonical acceptor and donor.

Normal prediction APIs, `create_model`, parameter/state trees, attention
backends, and checkpoint restoration are unchanged.

## Runner order and failure policy

`run_stage_a_branches_v3.py` first runs the exact current Phase-R factory,
trace selection and all-false intervention twice for all 20 frozen development
rows. It compares that semantic reference to the locked Phase-R identity tree
at an absolute threshold of `2^-8`. Only after every reference passes does it
run the separate Stage-A all-false duplicate/repeat Gate 0 for all 20 rows.
The locked
identity-tree SHA256 is
`ff7182be96e4b5be52e022e613ac16f476651924ff36d6a11b397b95613a3436`
and `PHASE_R_ANALYSIS.json` SHA256 is
`0131d591197fb187b9f291479e028c32c87313e40addd411235cb650df018a21`.
For effects, both executables must also have absolute ALT-REF logit-margin
delta at least `0.01` and the same sign as experimental `delta_logit`.

This dual-reference policy was frozen after the first full-cohort Stage-A
attempt stopped before any branch artifact. `BRAF_e14_T71A` had locked REF/ALT values
`2.546875` and `3.5869140625`; the Stage-A/locked absolute-difference vector was
`[0, 0.0048828125, 0.0048828125, 0.0048828125, 0, 0]`, exceeding `2^-8` on
the three identical ALT rows. The failed process raised before writing its raw
Stage-A values, so the sign of that cross-graph difference was not persisted.
The first variant was exact. That 20-row process produced no closure or
isolated-branch result. A separate earlier one-variant smoke had already
produced final-A/D and joint-T+E closure artifacts, but no isolated T/E,
Shapley or ranking result; its hashes and claim boundary are frozen in
`gate0_dual_reference_amendment_v3_1.md`.

Stage A and Phase R return different trace pytrees and insert different
all-false `where`/gather operations. Under BF16, XLA fusion and reassociation
are therefore not guaranteed to produce bit-identical output logits even when
both graphs are semantic no-ops. Making the new graph structurally resemble
Phase R would still change its outputs and offers no proof that the compiler
will choose the frozen Phase-R arithmetic. Precision/barrier changes would
likewise create a third graph that cannot be compared to the already locked
artifact.

The current Phase-R reference must still pass the unchanged `2^-8` lock and
exact duplicate/repeat audits. The Stage-A graph separately must pass exact
within-graph repeats, same-allele baselines, no-op checks, self transfers and
both target closures. Its difference from the current Phase-R graph is written
as a diagnostic and is not used for eligibility, ranking or thresholding.
This does not assert exact equality between distinct compiled graphs; it
separates semantic/checkpoint identity from the causal graph's internal
contrasts without modifying either frozen artifact or threshold.

For every frozen development effect, regardless of isolated-branch
eligibility, the runner completes both mandatory controls across the entire
cohort before producing isolated branch results for eligible effects:

1. the final A/D embedding patch must make both reciprocal cross-patch targets
   bit-equal to their donor baseline;
2. the joint complete T+E patch must do the same; and only then
3. isolated complete T and E patches are run.

For active whole transfers, natural self-control rows 3 and 5 must exactly
equal their same-allele baselines, while effective recipient rows 2/3/4/5 must
exactly equal their requested intervention donors. Gate 0 requires natural
same-allele equality for rows 2/3/4/5, natural/effective equality at every row,
and an exact target/trace repeat. Donor-match booleans are repeated as audit
values but are not treated as tensor content in the duplicate loop. Target self
controls remain
bit-exact. The four-word repeat fingerprint is deliberately compact: it catches
ordinary tensor drift but collisions are possible, so it is not a
cryptographic proof. The within-call equality booleans are exact.
Per-variant raw T, E, T+E movements, bidirectional recoveries, Shapley values
and interaction are written without clipping. Artifacts are atomic,
fingerprinted and resume-safe.

## Precisely remaining before Stage A is complete

The following protocol sections are intentionally not approximated:

1. **Encoder receptive-field supports (§7.2).** A tested symbolic dependency
   map must account for the initial width-15 convolution, both width-5
   convolutional paths in every residual block, SAME padding, and every
   two-base max-pool. It must freeze `I_b(V)` plus guards at all seven skip
   scales before model values are observed. Simple coordinate division, as in
   the exploratory 51-component runner, is not equivalent.
2. **Skip combinations (§7.2).** Leave-one-scale-out, both cumulative scale
   orders, equal-width ±512-bp supports and 32 frozen random supports require a
   fixed padded multi-stage selection large enough for the largest exact
   support. Whole E is an upper bound, not one of these candidates.
3. **Decoder ancestral supports (§7.3).** The backward dependency of canonical
   A/D bases through seven repeat-by-two upsamplers and the width-5 decoder
   convolutions must be derived and unit-tested. Patching only the bins that
   contain A/D is not sufficient.
4. **Output-embedder addends (§7.3).** `P_dec` and repeated `P_128` must be
   instrumented separately before their addition, RMS normalization and GELU.
   The implemented post-GELU A/D patch is only the mandatory closure control.
5. **Candidate controls and decision (§7.4).** Shifted and random controls,
   cross-exon `B/q/Q` aggregation, the 5% tooling-failure rule and a fail-closed
   route-decision analyzer remain to be implemented.

Until these are complete, this slice can validate instrumentation closure and
partition whole-branch upper bounds, but it cannot select or lock a Stage-A
mechanism.

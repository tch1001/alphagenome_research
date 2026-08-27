# OpenSplice v2 paired-batch development result

**Date:** 2026-08-27

**Code:** `feature/pallas-tiled-attention`, runner
`opensplice-inference-trace-v1.2.0`

**Scope:** frozen development exons BRAF e14 and SLC25A48 e8 only. ELN,
EIF4A2 and DMD confirmation internals were not opened.

## Bottom line

The numerical repair worked, but the preregistered residual-circuit hypothesis
did not pass.

The six-row live paired-batch implementation eliminated the BF16 self-patch
drift that invalidated the earlier cross-execution run. All 12 development
variants, 2,592 intervention groups and 12 identity/repeat audits passed Gate
0. Same-allele output drift was exactly zero in every group.

The best component was the joint variant/acceptor/donor token set (`S`) before
attention at transformer layer 2. Its median bidirectional recovery was only
13.34% in BRAF and 9.95% in SLC25A48. The frozen gate required at least 25% in
both exons plus a positive margin over positional controls. None of the 72
candidate components passed, so no circuit was locked and the confirmation
set was not inspected.

## What was tested

- Context: 16,384 bp, dense attention reference backend.
- Checkpoint: AlphaGenome all-folds snapshot
  `a8f293a76ee73d5b57f3bf2ae146510589fcf187`.
- Variants: six experimentally significant SNVs per development exon.
- Target: mean strand-aware canonical acceptor/donor probability.
- Components: V, A, D and their unique union S at three residual seams across
  transformer layers 0–5.
- Controls: equal-width upstream and downstream token sets.
- Causal metric: minimum of self-corrected REF-to-ALT and ALT-to-REF recovery.
- Selection: per-exon median recovery at least 0.25 and positive control margin.

The model passed the output-direction gate for all 12 development effects.
There were 216 groups per effect: 72 candidate components and 144 matched
positional controls.

## Gate-0 repair and result

The old method captured a BF16 residual in one model call and supplied it to a
later call. Across the full grid, 2,303/2,592 groups showed nonzero self drift,
so that experiment was correctly classified as a tooling failure.

The repaired method puts six aligned sequence rows in one compiled forward:

1. REF baseline/donor;
2. ALT baseline/donor;
3. ALT receiving the live REF residual;
4. ALT receiving the live ALT residual (self control);
5. REF receiving the live ALT residual; and
6. REF receiving the live REF residual (self control).

Donors are gathered from the live residual tensor before any recipient is
written. Nothing is transferred through Python or host memory. The repaired
run produced:

| Check | Result |
|---|---:|
| Development effects passing prediction gate | 12/12 |
| Causal intervention groups complete | 2,592/2,592 |
| Variants with any self-drift failure | 0/12 |
| Maximum same-allele output drift | 0 |
| Identity/repeat audits passing | 12/12 |
| Target repeats bit-exact | 12/12 |
| Trace repeats bit-exact | 12/12 |
| Duplicate REF/ALT rows bit-exact | 12/12 |

This establishes that the negative circuit result is no longer explained by
the earlier execution-boundary drift.

## Frozen ranking result

The leading component was:

| Field | Value |
|---|---:|
| Seam | `pre_attention` |
| Transformer layer | 2 |
| Position set | `S` (unique V/A/D tokens) |
| BRAF median bidirectional recovery | 0.133399 |
| SLC25A48 median bidirectional recovery | 0.099521 |
| Cross-exon Q statistic | 0.099489 |
| Required recovery in each exon | 0.25 |
| Passed | No |

Across all 72 components, the maximum per-exon median recoveries were 0.133399
for BRAF and 0.099521 for SLC25A48. Therefore zero components could reach the
25% gate in both exons, even before considering the control-margin requirement.

## Interpretation boundary

This is a negative result for a specific hypothesis: a small set of 128-bp
residual tokens at the variant and canonical splice sites should carry a
compact, transferable causal state that recovers at least one quarter of the
variant effect across two exons.

It is not evidence that:

- AlphaGenome has no internal splice computation;
- its useful splice information is not mechanistically interpretable;
- no wider, finer, distributed or nonlinear intervention can work; or
- the model's splice predictions are biologically wrong.

Plausible explanations include a distributed representation, dependence on
more surrounding tokens, an encoder- or decoder-local mechanism, a mismatch
between probability-space patching and PSI biology, or information encoded in
directions that whole-token replacement does not isolate.

The selected experimental “neutral” variants are also not reliable model-null
controls: BRAF neutral rows had large AlphaGenome deltas. They were not used to
select the residual component, and no neutral-specificity claim is made.

## Reproducibility

The aggregate is `DEVELOPMENT_ANALYSIS.json`. It binds 2,625 raw JSON artifacts
with SHA-256:

```text
f2ef54d2cda31f55fc810f2579a869555892962b3a71e55fac67642080d9352e
```

The raw baseline and trace JSON files are intentionally gitignored. The frozen
manifest SHA-256 is:

```text
09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6
```

The full development command was:

```bash
experiments/interpretability/opensplice/run_inference_trace.sh \
  --max-variants 20 \
  --output-dir experiments/interpretability/opensplice/results/v2_development_paired_batch \
  --trace-max-variants 12 \
  --trace-layers 0,1,2,3,4,5 \
  --trace-stages pre_attention,post_attention,post_mlp \
  --trace-max-groups-per-variant 216
```

Warm six-row intervention calls took about 0.113 seconds each after a roughly
158-second first compilation on the local RTX 3090.

## Decision

Do not open or run the frozen confirmation internals for this circuit claim.
Any follow-up should be a separately versioned hypothesis with a newly frozen
development/confirmation split—for example finer within-token interventions,
encoder/decoder tracing, or a wider jointly patched sequence window. That new
study must not choose its method by inspecting ELN, EIF4A2 or DMD internals.

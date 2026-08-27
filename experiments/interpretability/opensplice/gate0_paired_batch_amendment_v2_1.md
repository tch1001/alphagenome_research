# OpenSplice protocol amendment v2.1: live paired-batch transfer

**Amends:** `causal_interpretability_protocol_v2.md`, intervention execution
and Gate 0 only

**Date:** 2026-08-27

**Status:** freeze before rerunning development; confirmation internals remain
unopened

## Decision

The completed cross-execution development trace is a **Gate-0 tooling
failure**. It cannot select a circuit and cannot count as a negative biological
or mechanistic result. Rerun the complete, unchanged development grid using a
live paired-batch transfer inside one compiled execution. Do not relax the self
drift threshold and do not narrow the grid around the apparent best component.

This amendment is admissible before confirmation because it repairs the
identity property that Gate 0 was designed to test. It does not change the
manifest, exons, development/confirmation split, target scalar, context,
component grid, controls, selection rule, recovery threshold or claim boundary.

## Audit of the failed run

Audited artifact:

```text
experiments/interpretability/opensplice/results/v2_development_dense
script opensplice-inference-trace-v1.1.0
manifest SHA-256 09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6
checkpoint snapshot a8f293a76ee73d5b57f3bf2ae146510589fcf187
context 16,384 bp; dense attention
```

The run contains 12/12 direction-passing development effects, 2,592 trace
groups (864 candidates and 1,728 matched positional controls), and no
confirmation exon. Independent recomputation found:

| Quantity | Recomputed value |
|---|---:|
| Variants with at least one nonzero self drift | 12/12 |
| Trace groups with nonzero self drift | 2,303/2,592 |
| Median absolute self drift | 0.001953125 |
| Maximum absolute self drift | 0.0048828125 |
| Maximum drift / allele-effect denominator | 15.3846% |

The maximum absolute drift occurs for `SLC25A48_e8_G71T` at
`pre_attention`, layer 0, upstream `D` control. The maximum relative drift is
the distinct `BRAF_e14_A89G` downstream `A` control at `post_attention`, layer
1: `0.00390625 / 0.025390625 = 15.384615%`.

The drift lies on BF16 quantization increments and occurs when a residual is
returned from one model execution and supplied as replacement data to a later
execution. Self-control subtraction removes an additive output offset but does
not establish that the transferred activation is the same computational value
that the recipient would have used internally. When drift is comparable to a
reported recovery, corrected recovery remains untrustworthy.

For completeness, applying the frozen development ranking to the invalid
results nominates `post_mlp`, layer 2, joint set `S`, with
`Q = 0.1024219323`. Its per-exon median bidirectional recoveries are
0.1068840580 (BRAF) and 0.0991689393 (SLC25A48), far below the preregistered
0.25 gate. These values are diagnostic only. They must not be used to select,
prioritize or prune components in the rerun.

The current result tree contains only BRAF and SLC25A48 artifacts. No ELN,
EIF4A2 or DMD internal trace was found during this audit.

## Required paired-batch executable

“Paired batch” means a **live activation permutation at the selected seam in a
single forward execution**. Merely placing REF and ALT in a batch for capture,
returning their traces, and invoking a patch executable later does not satisfy
this amendment.

For each variant and runtime-selected component, construct six sequence rows:

```text
row 0  REF baseline and donor
row 1  ALT baseline and donor
row 2  ALT recipient, donor row 0 (REF -> ALT)
row 3  ALT self-control recipient, donor row 1
row 4  REF recipient, donor row 1 (ALT -> REF)
row 5  REF self-control recipient, donor row 0
```

All six rows enter the same jitted AlphaGenome forward. Immediately before the
chosen residual seam, the live tensors are identical within each same-allele
group. At the seam, first read the unaffected donor tensors from rows 0 and 1,
then use a batch-index map on the live tensor:

```text
baseline rows: rows 0 and 1 receive no replacement
cross rows:    row 2 receives selected positions from row 0;
               row 4 receives selected positions from row 1
self rows:     row 3 receives selected positions from row 1;
               row 5 receives selected positions from row 0
```

The donor values must be gathered from the live, pre-replacement seam tensor
in baseline rows 0 and 1 and scattered into the recipient rows before any
value leaves the compiled computation. Do not source a donor from a cross-
patched recipient, and do not round-trip donors through Python, host memory, a
serialized trace, or a second model invocation. The same construction applies
separately to `pre_attention`, `post_attention`, and `post_mlp` seams.

The component selector must be runtime data: fixed-shape masks specify stage,
layer and trace slots without recompilation. Candidate and positional-control
calls therefore reuse one executable shape and implementation. A static Python
branch or per-component recompilation is not accepted as the primary transfer
path.

### Static implementation audit

At the time of this amendment, the unexecuted v1.2 runner constructs sequence
rows `(REF, ALT, ALT, ALT, REF, REF)` and encodes the four recipient/donor
pairs `(2,0), (3,1), (4,1), (5,0)`, matching the table above. The transfer
primitive gathers every donor from its input residual tensor before applying
updates, so a recipient cannot become another recipient's donor. Its layer,
batch and position arrays are fixed shape. Unit tests cover the donor map, a
BF16 self-copy and simultaneous transfer semantics.

This static review is not a Gate-0 pass. Before launching the full grid, a new
development smoke artifact must additionally record exact output self identity,
the selected pre/post effective residual equality, an all-false-mask result,
and an exact repeat from the already compiled executable. Any mismatch stops
the rerun before component ranking.

The six target means returned by that one execution define all causal metrics:

```text
M_R  = row 0              M_A  = row 1
M_RA = row 2              M_AA = row 3
M_AR = row 4              M_RR = row 5

r_REF_to_ALT = (M_RA - M_AA) / (M_R - M_A)
r_ALT_to_REF = (M_AR - M_RR) / (M_A - M_R)
B             = min(r_REF_to_ALT, r_ALT_to_REF)
```

The public variant API remains the source of the frozen predictive Gate-1
score. The paired identity rows are the causal denominator and must
independently have the experimental sign and `abs(M_A - M_R) >= 0.01`. Each
paired identity allele mean must agree with its public counterpart within one
near-one BF16 ULP (`2^-8 = 0.00390625`); this tolerance is a semantic
cross-executable check, not permission for self drift.

## Amended Gate 0

Before any development component is ranked, all of the following must pass for
every variant and every candidate/control component:

1. `M_RR == M_R` and `M_AA == M_A` bit-for-bit. The original `1e-4` threshold
   remains an outer guard, but BF16 target values make exact equality the
   expected identity result.
2. The live effective residual at every self-patched slot equals the live
   pre-replacement residual bit-for-bit.
3. The identity and self rows have identical DNA, organism index, target
   selection, model parameters and state; only the runtime intervention mask
   differs.
4. An all-false-mask batch gives the same identity target values.
5. Two identical calls to the already compiled executable agree within `1e-6`
   (preferably bit-for-bit).
6. Paired identity values pass the public-equivalence and minimum-denominator
   checks above.
7. No NaN, invalid slot, recompile, precision cast, or failed assertion is
   silently skipped.

One self-identity failure invalidates that component/variant. More than 5% of
components failing Gate 0 invalidates the implementation, as in protocol v2.
Do not replace exact identity with “within one ULP”: one ULP was already as much
as 15.38% of a development denominator.

## Development rerun and confirmation validity

Rerunning development with this implementation preserves a confirmatory claim
provided the following sequence is followed:

1. retain the failed results and this audit; do not overwrite them;
2. version the paired-batch runner and write to a new result directory;
3. rerun all 12 development effects over the full frozen 72-member residual
   grid and both positional controls, not only `S` or the apparent leading
   layers;
4. recompute the frozen `B`, `q`, and `Q` statistics from scratch using only
   paired-batch outputs;
5. apply the unchanged `B >= 0.25` and `q > 0` development gates;
6. if and only if they pass, freeze the one candidate (and any allowed head
   decomposition) in a circuit-lock artifact; and
7. only then execute that locked hypothesis on ELN, EIF4A2 and DMD.

The repair was motivated by an a priori sham failure, not by a desire to
increase recovery, and the confirmation set has not informed it. It therefore
does not consume the confirmation set. Public baseline predictions alone would
not tune an internal component, but under the stricter v2 rule no new
confirmation inference of any kind should be run before the circuit lock.

If any confirmation activation, attribution, layer, head or patch is inspected
before the lock, confirmation is contaminated and becomes retrospective. If
the paired method is chosen or modified after comparing its confirmation
performance with the cross-execution method, the confirmatory claim is also
lost.

## Allowed interpretation after rerun

- Gate 0 passes but no component reaches 0.25: valid negative development
  result for this intervention family; do not open confirmation for a circuit
  claim.
- Gate 0 and the development component gate pass: freeze the component and
  proceed once to confirmation.
- Gate 0 still fails: tooling remains unresolved; report neither a biological
  failure nor a circuit recovery.
- The paired result differs from the old ~10% result: expected under a corrected
  numerical estimand; never choose between them by effect size.

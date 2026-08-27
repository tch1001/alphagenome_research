# OpenSplice v2 frozen causal-interpretability protocol

**Protocol version:** 2.0.0  
**Freeze date:** 2026-08-27  
**Scope:** the 50 variants in `selected_variants_v2.tsv`  
**Status:** frozen; commit with the manifest before local AlphaGenome inference

This document fixes the evaluation before looking at local AlphaGenome
predictions or internal traces. An amendment that changes an exon, variant,
target scalar, component search, threshold or confirmatory analysis must receive
a new protocol version. A failed prediction or causal test must not cause a
variant to be replaced.

The estimand is a **causal component of AlphaGenome's computation of a splice
site prediction**. OpenSplice supplies independent experimental evidence that
the sequence variant changes exon inclusion in its minigene assay. Even a
successful experiment does not, by itself, identify an endogenous biochemical
pathway, an RNA-binding protein, or a physical molecular interaction.

## 1. Frozen cohort and why v2 exists

The canonical manifest is:

```text
experiments/interpretability/opensplice/selected_variants_v2.tsv
SHA-256 09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6
```

It contains 30 experimentally significant SNVs and 20 assay-neutral SNV
controls, ten variants for each of the original five mechanically selected
exons:

| Role | Exons | Significant effects | Neutral controls | Observed effect signs |
|---|---|---:|---:|---|
| Development | BRAF e14, SLC25A48 e8 | 12 | 8 | BRAF: 6 positive; SLC25A48: 6 negative |
| Confirmation | ELN e19, EIF4A2 e4, DMD e78 | 18 | 12 | ELN: 6 positive; EIF4A2 and DMD: 12 negative |
| Total | 5 exons | 30 | 20 | 12 positive, 18 negative |

The split follows frozen exon order: orders 1--2 are development and orders
3--5 are confirmation. No variant from a confirmation exon may be used to
choose a layer, stage, position set, head, context length, output, threshold or
plotting example.

Version 1 required three negative and three positive effects in every exon.
That rule is infeasible: BRAF e14 has no significant negative-effect variant in
the pinned release. Version 2 preserves the original five exons and, using
experimental columns only, selects the six significant SNVs with largest
absolute `delta_logit` in each exon. This revision was made before local
AlphaGenome inference. The loss of within-exon sign balance is real and must be
reported; sign and locus are partly confounded.

### 1.1 Exact v2 eligibility and ordering

A row is eligible only when all of the following hold:

1. it maps exactly to one pinned GRCh38 VCF row;
2. `measured` is true and it is not the WT row;
3. `mut_type == "sub"`, REF and ALT are each one A/C/G/T base, and ALT is not
   multiallelic;
4. all three replicate PSI values, `wt_psi`, `psi`, `delta_psi`,
   `delta_logit`, `se_d` and `padj` are finite; and
5. the uncertainty and adjusted P value are valid (`se_d >= 0`,
   `0 <= padj <= 1`).

For each exon, effect rows satisfy `significant == yes` and `padj < 0.1`. Sort
them by decreasing `abs(delta_logit)`, then `variant_id`, and take six. The
observed sign is whatever the experiment reports; it is not balanced or
resampled.

Neutral candidates satisfy `significant == no` and `padj >= 0.1`. In effect
rank order, match the first four effects to unused neutral SNVs using the best
available tier: same region and mutation type, same region, same mutation type,
then unmatched. Within a tier select the smallest `abs(delta_logit)`, then
`variant_id`.

“Neutral” is operational: nonsignificance plus near-zero ranking is not proof
of equivalence. Report each neutral's `delta_logit`, `se_d`, and whether its
95% normal interval lies entirely within `[-0.5, 0.5]` delta-logit units. This
equivalence flag is a sensitivity stratum and must not be used to replace a
frozen neutral.

## 2. Biological outcome and model target

### 2.1 Experimental outcome

For variant `v`, the experimental effect is the released value

```text
E_v = delta_logit_v = logit(PSI_ALT) - logit(PSI_WT).
```

Use the released error-model estimate rather than recomputing it from rounded
PSI columns. `E_v > 0` means increased inclusion of the tested middle exon;
`E_v < 0` means decreased inclusion. OpenSplice measured this in a massively
parallel minigene assay with three replicate PSI columns. It did not measure
the endogenous transcript at its native locus.

The magnitude of `E_v` and the AlphaGenome scalar below are on different
scales. Their **sign** and rank are comparable; equality of magnitudes is not an
estimand.

### 2.2 Primary AlphaGenome scalar

The primary output is the tissue-agnostic, 1-bp `SPLICE_SITES` probability at
the annotated canonical acceptor and donor of the selected exon. This exactly
matches the OpenSplice authors' AlphaGenome genome-mode reduction rather than
choosing a model output post hoc.

Let `A` and `D` be the strand-aware canonical acceptor and donor coordinates,
and let `p_a(A)` and `p_a(D)` be their correct-strand acceptor and donor
probabilities for allele `a`. Define

```text
M_a       = [p_a(A) + p_a(D)] / 2
DeltaM_v  = M_ALT - M_REF.
```

Coordinate rules are fixed:

| Exon strand | `start_exon` role | `end_exon` role |
|---|---|---|
| `+` | acceptor | donor |
| `-` | donor | acceptor |

The metadata coordinates are 1-based. For an interval with 0-based half-open
start `s`, the 1-bp output index of coordinate `g` is `(g - 1) - s`. Save the
individual acceptor and donor probabilities as well as their mean. Opposing
acceptor/donor changes are not resolved by selecting the endpoint that works;
the predeclared mean remains primary.

An effect variant passes the **model-output eligibility gate** only when:

```text
sign(DeltaM_v) == sign(E_v)  and  abs(DeltaM_v) >= 0.01.
```

All 30 effects remain in the predictive denominator. Only the correctly signed
and sufficiently nonzero subset is eligible for normalized causal recovery.
An insensitive or wrong-sign prediction is a model failure, not a circuit
failure and not an invitation to switch outputs.

### 2.3 Secondary outputs

The following are sensitivity analyses and cannot rescue a failed primary
target:

- the two endpoint deltas separately;
- `SPLICE_SITE_USAGE` at the same strand-aware endpoints for `CL:0002518`
  (kidney epithelial cell, the OpenSplice authors' declared closest available
  ontology to the HEK assay context);
- the two exon-inclusion junctions and the exon-skipping junction, after their
  GENCODE version and exact coordinates are frozen in an inference manifest;
  and
- RNA-seq as qualitative corroboration only.

AlphaGenome does not directly output cassette-exon PSI. In particular, a
maximum merged splicing-impact score is unsigned and unsuitable as the causal
target for measured inclusion direction.

## 3. Input context and checkpoint

### 3.1 Native genome is primary

For every exon and length, use one fixed GRCh38 interval shared by its REF and
all ALT alleles. Center it on
`floor((start_exon + end_exon) / 2)` using the OpenSplice coordinate convention.
Do not recenter on the variant.

- `16,384 bp`: development and pipeline-reproduction context. This matches the
  OpenSplice authors' published genome-mode AlphaGenome script.
- `131,072 bp`: primary confirmatory causal context. All 50 variants receive a
  baseline prediction; every final confirmation result is computed here.
- `131,072 bp` with interval-start shifts of `-64` and `+64 bp`: token-boundary
  sensitivity only. The unshifted interval is always primary.

The component selected at 16,384 bp must be carried unchanged to 131,072 bp.
It is not permissible to report whichever context gives larger recovery.

Use the local `google/alphagenome-all-folds` checkpoint and record its exact
Hugging Face snapshot hash, model source commit, configuration, attention
backend, numerical precision and reference FASTA hash in every result. This
checkpoint is a single defined computational object; an apparent circuit in a
different checkpoint is a replication, not a substitute. If fold-specific
weights and chromosome assignments become available, evaluate the appropriate
held-out fold as a separate robustness study.

### 3.2 Minigene context is secondary

The native sequence is biologically different from the OpenSplice construct.
The minigene sensitivity run must reproduce the pinned upstream construction:
the OpenSplice `nt_seq` in assay/transcript orientation, FAS exon 5/intron 5 on
the left, FAS intron 6/exon 7 on the right, intended middle-exon acceptor at
construct position 216 (0-based), and symmetric `N` padding to 16,384 bp.

This run is closer to the experimental sequence but is out of distribution for
a native-genome model and lacks native chromatin and long-range context. It is
reported beside, never pooled with, genome mode. Interpret outcomes as follows:

| Genome mode | Minigene mode | Allowed interpretation |
|---|---|---|
| pass | pass | model computation is robust to the two sequence contexts |
| pass | fail | native-genome computational result; assay-context mismatch remains |
| fail | pass | minigene/OOD sensitivity result, not endogenous generalization |
| fail | fail | no supported AlphaGenome mechanism for this variant |

## 4. Exact allele interventions

REF and ALT are both valid sequences, so the analysis uses **donor** and
**recipient** rather than implying that one allele is “clean.” Every causal
patch is performed in both directions:

1. REF activation into an ALT recipient (`REF -> ALT`); and
2. ALT activation into a REF recipient (`ALT -> REF`).

For the same selected component `c`, also run REF into REF and ALT into ALT.
These are not optional shams; they estimate intervention/compilation drift in
the exact recipient.

### 4.1 Frozen token mapping

The transformer sequence tower operates on 128-bp tokens. For genomic
coordinate `g` in interval `[s, e)`, define

```text
token(g) = floor(((g - 1) - s) / 128).
```

For each variant define and save:

```text
V = {token(variant position)}
A = {token(canonical acceptor)}
D = {token(canonical donor)}
S = unique(V union A union D)
```

Duplicate token sets are computed once. If `V == A`, `V == D`, or all three
coordinates share a token, an intervention can localize an allele effect to a
128-bp state but cannot establish a variant-to-splice-site route. Route claims
require distinct variant and endpoint tokens.

Two deterministic local positional controls have the same cardinality as the
candidate set. Start with `V - 4` tokens and `V + 4` tokens (512 bp away in
genomic orientation), preserving relative offsets for a multi-token set. If a
control intersects `S` or leaves the interval, move it one further token
outward until valid. Save the resolved indices before any patch result is
viewed.

### 4.2 Development search space

Only the two development exons may be scanned. The exact residual grid is:

- layers 0 through 5;
- `pre_attention_residual`, `post_attention_residual`, and
  `post_mlp_residual`; and
- position sets `V`, `A`, `D`, and `S`.

For a grid member, all positions in its set are replaced jointly at one layer
and stage. No per-variant best layer or window is permitted.

For each member `c`, define its per-variant bidirectional recovery `B_v(c)` as
in Section 5 and

```text
q_v(c) = B_v(c) - max[B_v(left-position control),
                      B_v(right-position control)].
Q(c)   = min(median_BRAF(q_v(c)), median_SLC25A48(q_v(c))).
```

At least three output-eligible effects must exist in each development exon;
otherwise no cross-sign circuit is selected. Choose the single grid member
with largest `Q(c)`, breaking exact ties by stage order
`pre_attention`, `post_attention`, `post_mlp`, then layer number, then position
order `V`, `A`, `D`, `S`. It becomes the one confirmatory residual hypothesis
only if both development-exon medians have `B >= 0.25` and `q > 0`. Otherwise
the residual-circuit discovery is negative and the confirmation set remains
unopened.

After selecting the residual member, head-value-output decomposition is
allowed only at its layer and positions. Evaluate heads 0--7, then joint
subsets in increasing subset size up to four heads. Freeze the lexicographically
first smallest subset whose development median retains at least 80% of the
selected residual member's bidirectional recovery and passes the same local
controls in both development exons. If no subset of at most four passes, report
a causal residual representation but do not claim a compact head circuit.

Pair-bias patches are diagnostic, not a primary hypothesis. These proximal
variants and splice sites frequently occupy the same 2,048-bp pair cell, so a
pair-bias cell cannot identify a variant-to-endpoint edge. Only when the
resolved pair cells are distinct may the frozen diagnostics patch
`A <- V` and `D <- V`, together with reverse (`V <- A/D`), self, and
distance-matched key controls. Pair-bias recovery is a coarse computational
route, not evidence of RNA contact or molecular binding.

Before any confirmation-exon inference, write a circuit-lock artifact that
contains the selected residual member, optional head subset, all resolved
positions and controls, software/checkpoint hashes, and a SHA-256 of this
protocol. Commit or otherwise immutably timestamp that artifact.

### 4.3 Mandatory intervention controls

Every candidate and joint patch has the following matched controls:

1. unpatched REF and ALT baselines;
2. REF-to-REF and ALT-to-ALT self patches;
3. both allele-transfer directions;
4. the two deterministic 512-bp positional controls;
5. 32 same-cardinality random position sets, drawn from tokens outside a
   four-token radius of `S`, using a seed derived as SHA-256 of
   `opensplice-v2|variant_id|component-family`;
6. for residuals, all 17 nonselected layer-stage combinations at the selected
   positions as a component null; for heads, all nonselected heads at the same
   layer and positions;
7. an allocated intervention object with every replacement mask false;
8. exact repeat runs; and
9. the intended target versus the wrong-strand donor/acceptor channels and the
   nearest annotated upstream and downstream non-target exons in the same
   interval. Those exon coordinates must be frozen before traces are viewed.

Random controls are generated once and reused across alleles and contexts.
Random donor loci, zero ablation and global head suppression are not primary
controls: they create a different and often more off-distribution intervention.

## 5. Metrics

Let `M_R` and `M_A` be unpatched REF and ALT scores. Let `M_RR` and `M_AA` be
self-patched scores, `M_RA` the ALT recipient with REF activations, and `M_AR`
the REF recipient with ALT activations. For every component report:

```text
r_REF_to_ALT = (M_RA - M_AA) / (M_R - M_A)
r_ALT_to_REF = (M_AR - M_RR) / (M_A - M_R)
B             = min(r_REF_to_ALT, r_ALT_to_REF)
reciprocity   = abs(r_REF_to_ALT - r_ALT_to_REF)
self_drift    = max(abs(M_RR - M_R), abs(M_AA - M_A))
```

Both recovery terms are positive when the recipient moves toward the allele
that donated the activation. `B` is the primary causal effect. Report raw
scores and raw movements alongside it. Never clip recovery to `[0, 1]`;
overshoot and sign reversal are diagnostic. Do not compute normalized recovery
when `abs(DeltaM) < 0.01`.

For the fixed circuit and each effect variant define

```text
D_v = B_v(candidate) - max[B_v(left control), B_v(right control)].
```

For the four matched neutral pairs per exon, normalized neutral recovery is
undefined if its own allele denominator is small. Instead calculate the maximum
absolute raw cross-patch movement in neutral `n`, scale it by the matched
effect's denominator, and compare it with the matched effect:

```text
N_n|e = max(abs(raw neutral movement in either direction)) / abs(DeltaM_e).
```

Output specificity also uses raw movement. For each direction subtract the
largest absolute wrong-strand or non-target-exon movement from the signed
movement of the intended canonical-site mean; the smaller directional margin
is the per-variant specificity margin.

### 5.1 Required reporting

Report all 50 rows, with no successful-example filtering. Tables must include:

- experimental `delta_logit`, uncertainty, class, region and observed sign;
- `M_REF`, `M_ALT`, `DeltaM`, acceptor and donor components at both context
  lengths;
- output-gate pass/fail and reason for every significant effect;
- every cross/self patch score, both recoveries, `B`, reciprocity, self drift,
  `D_v`, and specificity margin;
- neutral raw movements and matched `N_n|e`;
- results by exon, observed sign and region, not just pooled values; and
- all failures, NaNs, compile variants, repeats, shifts and excluded normalized
  denominators.

Confidence intervals use an exon-block bootstrap (resample exons, then variants
within exon) and are labelled descriptive because only three confirmation
exons exist. Paired effect-neutral comparisons also receive an exact paired
sign/permutation test. No variant-level iid P value is treated as confirmatory,
because variants from one exon share sequence context and sign.

For the random-intervention comparison, compute one pooled median `B` across
the output-eligible confirmation effects for the candidate and for each of the
32 indexed random controls. “Above the 95th percentile” refers to the
candidate statistic against those 32 control statistics; it is not a
per-variant best-control comparison.

## 6. Frozen gates

These gates are conjunctive. Passing a later gate cannot rescue an earlier one.

### Gate 0: provenance and tooling

- the manifest checksum equals
  `09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6`;
- REF bases match the pinned GRCh38 FASTA at every locus;
- the two self-patch drifts are each at most `1e-4`; for output-eligible effects
  they must also be at most 1% of the variant's `abs(DeltaM)`;
- repeated scalar outputs differ by at most `1e-6`; and
- no invalid index, allele mismatch or NaN is silently discarded.

Any provenance failure stops the study. If more than 5% of variants fail an
internal tooling check, the circuit result is invalid rather than a biological
negative.

### Gate 1: predictive adequacy

- apply this gate to the unshifted 131,072-bp native-genome primary target;
- report direction accuracy on all 30 effects;
- require at least 20/30 correct directions overall and at least 13/18 in the
  untouched confirmation exons;
- require at least 4/6 correct in at least two of three confirmation exons and
  no confirmation exon with 0/6 correct; and
- report positive (12) and negative (18) strata separately.

These counts are a preregistered adequacy gate, not an iid binomial claim. A
variant below the `abs(DeltaM) >= 0.01` denominator threshold is still counted
as not eligible.

### Gate 2: held-out causal recovery

On output-eligible confirmation effects, the single frozen residual candidate
must satisfy all of:

- pooled median `B >= 0.25` at 131,072 bp;
- positive per-exon median `B` in all three confirmation exons and median
  `B >= 0.25` in at least two of three;
- pooled median `D_v > 0`; and
- candidate recovery above the 95th percentile of the precomputed matched
  random-intervention null.

If a confirmation exon has no output-eligible effect, cross-exon causal
generalization is not testable and cannot be claimed.

### Gate 3: effect and output specificity

- in at least 10 of the 12 frozen confirmation effect-neutral pairs,
  `B_effect > N_neutral|effect`;
- the pooled median of that paired difference is positive; and
- the intended-output specificity margin is positive for at least 75% of
  output-eligible confirmation effects and has a positive median in every
  confirmation exon.

An effect-neutral pair whose effect fails the model-output eligibility gate
counts as not passing the 10/12 criterion; it is not silently removed.

### Gate 4: robustness and compactness

- the same frozen component has positive per-exon median recovery at 16,384
  and 131,072 bp;
- the direction of candidate recovery and candidate-over-local-control margin
  survives at least two of the three `-64/0/+64 bp` interval starts;
- both allele-transfer directions pass; and
- a compact-head claim additionally requires the frozen at-most-four-head set
  to retain at least 80% of residual recovery on confirmation data.

Failure of the head criterion leaves a potentially distributed residual-level
mechanism; it does not erase an otherwise valid residual result.

## 7. Deletions, insertions and alignment

The v2 causal benchmark is deliberately SNV-only. OpenSplice includes 1-, 3-,
6- and 21-nt deletions but does not assay insertions. Public AlphaGenome
variant-output stitching and OpenSplice's deletion liftover can align final
predictions; they do not make internal token activations homologous.

For a VCF deletion, the anchor base at `POS` remains and the deleted reference
span is `POS + 1` through `POS + len(REF) - 1`. Coordinates downstream of the
deletion shift by `len(REF) - len(ALT)`. None of the assayed deletion lengths is
a multiple of the 128-bp transformer token size, so downstream token contents
and phase differ between alleles. A same-index activation transplant therefore
mixes biological signal with alignment and tokenization artifacts.

Consequently:

- deletion prediction and input-attribution results are a separate sensitivity
  stage;
- direct cross-allele internal patching at or downstream of a deletion is not
  included in any v2 gate;
- an upstream activation patch is allowed only when every patched 128-bp token
  is wholly homologous and sequence-identical outside the intended allele
  difference; and
- any future deletion circuit requires a separately validated allele-coordinate
  map, same-allele self controls, sham indels of matched length, output stitching
  tests and a new protocol version.

The same rule applies to an insertion from another dataset. It must be
left-normalized, have an explicit shared-anchor coordinate map, and remain
outside primary causal claims until internal alignment is validated.

## 8. Leakage boundary

Allowed before circuit lock:

- the experimental allowlist used by the v2 selector;
- exact VCF alleles, exon coordinates, strand and pinned source checksums;
- OpenSplice assay methods and code needed to define the endpoint; and
- the OpenSplice authors' *method* for reducing AlphaGenome canonical-site
  predictions.

Prohibited before circuit lock:

- every released AlphaGenome, SpliceAI, Pangolin or SpliceTransformer score
  column for these variants;
- Supplementary Table 12 predictor results;
- scanning any confirmation-exon activation, attribution, layer, head or patch;
- changing the selected variants, context, target, eligibility threshold or
  control because AlphaGenome performs poorly; and
- choosing examples or figures from confirmation results before the complete
  30-effect/20-neutral table is generated.

The OpenSplice authors already benchmarked AlphaGenome, and their aggregate
choice of 16,384-bp context was informed by experimental correlation. This is
therefore a temporal holdout from AlphaGenome's original training/evaluation,
but not a fully model-blind predictive benchmark. The novel test here is the
pre-frozen **internal causal generalization**, not another claim that
AlphaGenome predicts OpenSplice.

If anyone inspects confirmation-exon internals before the circuit-lock artifact
is frozen, those exons become retrospective. They may still be reported as
tooling results, but a new cross-exon confirmation set and protocol version are
required for a generalization claim.

## 9. Claim boundary

The strongest result supported by all gates is:

> A component of AlphaGenome's internal computation, selected on BRAF and
> SLC25A48, causally mediates correctly predicted canonical splice-site effects
> for frozen SNVs in three untouched OpenSplice exons, beyond matched internal
> and assay-neutral controls.

The following are **tooling or retrospective results**, not generalization:

- successful allele patching on the two development exons;
- a per-variant best layer, head or position;
- a whole-local-residual transplant that does not beat position controls;
- recovery in a confirmation exon after its components were scanned;
- recovery only in minigene mode or only at one convenient context/shift;
- pair-bias recovery when variant and endpoint occupy one coarse pair cell; or
- any result that fails neutral, self-patch or non-target-output controls.

Cross-exon confirmation is computational generalization. It is not proof of a
new biological pathway. OpenSplice validates that the variant changes splicing
in its assay. Assigning an RBP, enhancer/silencer grammar or endogenous
biochemical route additionally requires sequence-level hypotheses and
orthogonal evidence such as motif perturbation, eCLIP/RNA binding, RBP
knockdown, or endogenous base editing. Parallel AlphaGenome output heads must
not be drawn as a biochemical chain.

Because held-out positive effects come from ELN and held-out negative effects
from EIF4A2/DMD, sign generalization is demonstrated across both directions but
positive-direction replication spans only one confirmation exon. That
limitation must accompany the headline result.

## 10. Primary sources

- [OpenSplice preprint](https://doi.org/10.64898/2026.05.22.727141)
- [OpenSplice v5 data release](https://doi.org/10.6084/m9.figshare.32337414.v5)
- [OpenSplice repository at the inspected commit](https://github.com/lehner-lab/OpenSplice/tree/3e4ad8c037c216b952f1a8945f8f498669bff589)
- [OpenSplice AlphaGenome genome-mode SNV script](https://github.com/lehner-lab/OpenSplice/blob/3e4ad8c037c216b952f1a8945f8f498669bff589/benchmarking_predictors/scripts/inference/alphagenome_genome_mode_snvs_inference.py)
- [OpenSplice canonical-site reduction](https://github.com/lehner-lab/OpenSplice/blob/3e4ad8c037c216b952f1a8945f8f498669bff589/benchmarking_predictors/scripts/processing/process_alphagenome_genome_snvs_siteonly_minimal.py)
- [OpenSplice AlphaGenome minigene-mode SNV script](https://github.com/lehner-lab/OpenSplice/blob/3e4ad8c037c216b952f1a8945f8f498669bff589/benchmarking_predictors/scripts/inference/alphagenome_minigene_mode_snvs_inference.py)
- [AlphaGenome output definitions](https://www.alphagenomedocs.com/exploring_model_metadata.html)
- [Official AlphaGenome splicing-scoring workflow](https://www.alphagenomedocs.com/colabs/splicing_variant_scoring.html)
- [AlphaGenome paper](https://doi.org/10.1038/s41586-025-10014-0)

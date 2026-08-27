# OpenSplice v3 prospective protocol: wider and distributed splice mechanisms

**Design date:** 2026-08-28

**Status:** prospective, documentation only; freeze machine-readable manifests
and implementation hashes before new AlphaGenome inference

**Primary scope:** AlphaGenome's computation of the OpenSplice splice-site
target, not a biochemical pathway

**Confirmation status:** blind; no ELN, EIF4A2, or DMD internal result was found
during this design audit

**Source audit commit:** `b1ad1b6e69de1f518c66b73553674c9761fddbed`

## Decision in one paragraph

Version 2 is a valid negative result for a narrow hypothesis, not a negative
result for AlphaGenome interpretability. Whole 128-bp residual vectors at the
variant and canonical splice-site tokens, at three seams in transformer layers
0--5, recovered at most about 10% robustly across the two development exons.
Version 3 will first isolate a newly frozen readout hypothesis by rerunning the
unchanged v2 residual grid against the strand-aware splice-classification
pre-softmax logit margin. Only if that narrow family still fails will v3 census
the two architectural routes into the 1-bp splice head: the transformer trunk
and the seven U-Net encoder skips. It will then complete layers 6--8 and test
fixed wider windows. Only if raw activation patches remain too distributed
will development-only attribution, probes, low-rank directions, and finally a
sparse autoencoder be used to nominate candidates. Every nomination is only a
screen. The same live paired-batch intervention and unchanged `B >= 0.25`
causal gate must validate the final candidate exactly before any
confirmation-exon inference. The original probability result remains a valid
v2 negative and is not reinterpreted by changing the v3 readout.

## 1. Evidence frozen before v3

### 1.1 Bound artifacts

The v3 design inherits the following v2 cohort and result:

| Artifact | SHA-256 |
|---|---|
| `selected_variants_v2.tsv` | `09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6` |
| `causal_interpretability_protocol_v2.md` | `1fc12304df7b912f550f2758ece4d7d0ace0e36f8e2858193e26b4b948735b2a` |
| `gate0_paired_batch_amendment_v2_1.md` | `4757a84ac14a756d2a4173fe46249df9a7f12b1335d5e60d27cc98df4c45b3c1` |
| `DEVELOPMENT_ANALYSIS.json` | `36a7c06bb9bd570787bfaf82c60ea334303855a0dd1fd03e13827dc5278aec6e` |

The audited v2 run used checkpoint snapshot
`a8f293a76ee73d5b57f3bf2ae146510589fcf187`, a 16,384-bp context, and dense
attention. All 12 development effects passed the model-output gate. All 2,592
paired-batch groups and 12 identity/repeat audits passed Gate 0 with exactly
zero same-allele output drift.

The best of the 72 frozen candidates was `pre_attention`, layer 2, position set
`S`. Its median bidirectional recoveries were `0.133399` in BRAF and `0.099521`
in SLC25A48; `Q = 0.099489`. No candidate reached `0.25` in both exons.
Several adjacent seams shared nearly identical values, consistent with a small
state contribution persisting across layers rather than one isolated attention
or MLP event. That is an observation, not proof of a bypass or distributed
code.

No result file whose path names ELN, EIF4A2, or DMD was present under the
OpenSplice result tree during this audit. Confirmation blindness therefore
remains available.

### 1.2 What v2 did and did not reject

Version 2 rejected this claim:

> A whole residual state at `V`, `A`, `D`, or their union in transformer
> layers 0--5 carries at least one quarter of the allele effect in both
> development exons.

It did not test:

- transformer layers 6--8;
- more than the one to three 128-bp tokens in `S`;
- encoder skip states at 1-, 2-, 4-, 8-, 16-, 32-, or 64-bp resolution;
- decoder states or the two addends entering the final 1-bp embedding;
- low-rank channel subspaces, learned feature directions, or SAE features; or
- interactions between the transformer and U-Net skip routes.

## 2. Architectural hypothesis and ranked follow-up

AlphaGenome's local source establishes the following splice path:

```text
DNA
  -> 1/2/4/8/16/32/64-bp encoder states -----------------------+
  -> 128-bp trunk -> 9 transformer blocks -> 128-bp trunk -----+-->
                                                                  decoder
                                                               -> 1-bp state

128-bp transformer output -> projected, repeated 128-bp addend --+
decoder output             -> projected 1-bp addend -------------+-->
                                                     RMSBN -> GELU
                                                     -> linear splice logits
                                                     -> softmax probabilities
```

The decoder receives all seven encoder states. The 1-bp `OutputEmbedder` also
adds a direct projection of the 128-bp transformer embedding to the projected
decoder state. A local splice effect can therefore bypass the transformer
through fine-resolution encoder skips. The v2 ceiling makes that hypothesis
plausible, but only the interventions below can test it.

The prospective priority order is:

0. **Readout isolation.** Freeze the pre-softmax logit margin, audit its
   development baselines, and rerun the unchanged 72-member v2 residual grid.
1. **Encoder/decoder route census.** Establish upper bounds and partition the
   signal between transformer and encoder-skip routes.
2. **Complete and widen exact transformer patches.** Add layers 6--8 and a
   fixed spatial-radius ladder through roughly 4 kb.
3. **Screen raw feature directions.** Use calibrated attribution patching,
   ridge directions, and low-rank allele-difference subspaces, followed by
   exact patches.
4. **Train an SAE only if needed.** Use it at one route-localized state, not as
   an all-layer fishing expedition, and causally validate selected features.

The ladder is adaptive only through the explicit go/no-go rules below. A later
stage cannot revise a prior threshold, target, context, or confirmation set. If
the unchanged residual grid passes under the new readout, stop the wider search
and describe the result as **readout-dependent recovery**, not evidence that a
wider or distributed mechanism was needed.

## 3. Literature boundary and implications

| Work | What is already established | Consequence for v3 |
|---|---|---|
| [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) and [research code](https://github.com/google-deepmind/alphagenome_research) | The model combines a multi-resolution convolutional U-Net, nine transformer blocks, and parallel functional-genomics heads; published interpretation is chiefly output difference and input attribution. | The U-Net bypass and output-embedding merge are AlphaGenome-specific causal targets. Reproducing published output interpretation is not novel. |
| [CREME](https://www.nature.com/articles/s41588-024-01923-3) | Enformer can be interrogated with context, swap, necessity, sufficiency, distance, and interaction tests in input space. | CREME-style sequence tests are baselines for context dependence; they are not internal circuits. |
| [AlphaGenome motif ablation](https://academic.oup.com/bib/article/27/4/bbag422/8753614) | C/EBPbeta motif deletions have been compared with length-matched shams and CEBPB-knockout data; effects are context-dependent and can reverse direction. | Input motif ablation is already demonstrated. Use alignment-preserving edits and do not equate global TF knockout with one motif edit. |
| [SAE-Borzoi workshop paper](https://openreview.net/pdf?id=AlLZnZX01x) and [code](https://github.com/calico/sae-borzoi) | TopK SAEs on early Borzoi convolutional layers yield motif-, RBP-, repeat-, and element-associated features. | Motif-like SAE features in AlphaGenome would be incremental unless they pass target-specific causal interventions and held-out exons. |
| [MINTS public repository and manuscript](https://github.com/ArjunCodess/MINTS) | This non-peer-reviewed current project shows that genomic-transformer probes, QK/OV analysis, attention enrichment, patching, and SAEs can disagree. Its splice labels are decodable, while its completed splice-donor head patch remains modest and its strict CTCF-head claim is negative. | A probe or enriched head is only a nomination. GC, position, raw k-mer, shuffled-label, and matched-background controls are mandatory. |
| [Causal dictionary-learning preprint](https://arxiv.org/abs/2607.19618) | Sparse genomic features have already been composition-matched and causally ablated in DNABERT-2 and Nucleotide Transformer. | “Causal genomic SAE” is not a novelty claim. The AlphaGenome contribution must concern a supervised sequence-to-function route and an experimental held-out splice effect. |
| [Interpretable distillation of splicing models](https://link.springer.com/article/10.1186/s13059-026-04124-9) | Distilled additive motif models closely approximate SpliceAI, Pangolin, and AlphaGenome on synthetic exon scores; the study identifies CpG and stop-codon confounds and an RNA-structure blind spot. | Short motifs may explain much output behavior without being clean biology. CpG, codon, composition, position, and structure annotations must accompany any feature claim. Simple AlphaGenome motif distillation is already done. |
| [AtP*](https://arxiv.org/abs/2403.00745) | Gradient screening is much cheaper than exhaustive patching, but attention saturation and path cancellation create false negatives; QK correction and GradDrop mitigate them. | AtP* may order candidates only after calibration against exact development patches. Exact live patching remains the causal result. |
| [SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015) and [Pangolin](https://pmc.ncbi.nlm.nih.gov/articles/PMC9022248/) | Splice prediction benefits from kilobase-scale context, and tissue-aware models capture context beyond the canonical dinucleotides. | Testing only one 128-bp token is biologically and computationally narrow; the v3 radius ladder is justified. |
| [Enhancer/branchpoint/polypyrimidine crosstalk](https://pmc.ncbi.nlm.nih.gov/articles/PMC1170318/) | Canonical sites, branchpoint and polypyrimidine signals, and exonic regulatory elements can act jointly. | A recovered window or feature should be annotated against multiple cis-regulatory classes rather than forced into a single “splice-site motif” story. |
| [OpenSplice](https://doi.org/10.64898/2026.05.22.727141) and [code](https://github.com/lehner-lab/OpenSplice) | More than 590,000 variants were tested across 608 exons in a minigene assay. | The measurements supply independent variant effects, but do not turn a computational circuit into an endogenous molecular pathway. |

The defensible novelty is a controlled causal route decomposition inside
AlphaGenome, followed by one locked wider or feature-level mechanism tested on
unopened experimental exons. It is not attention visualization, motif
discovery, probe decodability, input ablation, or an SAE by itself.

## 4. Data roles and blindness

### 4.1 Evaluation cohort

Keep the 50-row v2 manifest unchanged:

- development: BRAF e14 and SLC25A48 e8, 12 significant effects and eight
  experimental neutral controls;
- confirmation: ELN e19, EIF4A2 e4, and DMD e78, 18 significant effects and 12
  neutral controls.

Only the 12 output-eligible development effects determine causal gates. The
eight development neutrals retain their v2 specificity role. In particular,
BRAF neutral variants have large AlphaGenome effects and are not model-null
examples.

### 4.2 Optional development-only screening pool

Probes, direction learning, or an SAE feature ranker need more than 12 labelled
examples. Before extracting any new internal activation, create and hash a v3
screening manifest containing all eligible measured SNVs from **BRAF e14 and
SLC25A48 e8 only**, using the exact v2 row-validity rules. This expanded pool:

- may train or calibrate a screen;
- may not change the 12-effect causal denominator;
- may not set the final recovery threshold;
- must group alternate alleles at the same genomic position into the same
  cross-validation fold; and
- is reported as correlated development data, never independent replication.

No experimental or model-derived row from ELN, EIF4A2, or DMD may enter a
screen, hyperparameter choice, early-stop decision, or example figure.

### 4.3 Unsupervised background activations

If an SAE is reached, build its background corpus before activation capture:

1. use fixed 16,384-bp GRCh38 windows centred on GENCODE v46 internal exons;
2. exclude chromosomes 3, 7, and X, thereby excluding every confirmation
   chromosome and BRAF's chromosome from background training;
3. exclude a 1-Mb buffer around all five frozen OpenSplice exons regardless of
   chromosome;
4. cluster overlapping centres within 1 Mb, assign entire clusters to
   train/validation by SHA-256, and use an 80/20 split;
5. select 4,096 windows by the lowest SHA-256 of
   `opensplice-v3|chrom|center|strand`; and
6. freeze the BED/TSV and SHA-256 before capture.

The background corpus supplies unsupervised activation geometry only. It cannot
choose a biological label or final circuit.

### 4.4 Confirmation lock

Before any confirmation-exon baseline or internal inference, write an immutable
`circuit_lock_v3.json` containing:

- all input, cohort, screening-corpus, protocol, code, checkpoint and FASTA
  hashes;
- the one selected candidate and every tensor seam, spatial index, direction or
  feature ID it uses;
- all matched controls and random seeds;
- target scalar, context, metric, gates, and failure policy; and
- a declaration that no ELN, EIF4A2, or DMD internal was inspected.

If confirmation internals are opened before this lock, all later results are
retrospective tooling or case-study results, not held-out confirmation.

## 5. Fixed target, context, and causal metric

Version 3 prospectively changes one scalar before any v3 model inference. Its
primary target is the mean strand-aware canonical splice-classification
**pre-softmax logit margin**:

```text
L_site = logit(relevant acceptor-or-donor class) - logit(background class)
L_exon = [L_acceptor(A) + L_donor(D)] / 2
DeltaL = L_exon_ALT - L_exon_REF
```

Use the actual linear outputs saved under
`splice_sites_classification/logits`; do not invert public probabilities. On
the negative strand, use the negative-strand acceptor and donor classes. The
background is the fifth complement/padding class. The mean of the two frozen
endpoints is primary and cannot be replaced with the endpoint that works.

The logit margin is invariant to a common five-logit shift and avoids
probability saturation, but it is still classification evidence at canonical
bases, not cassette-exon inclusion or PSI. This target decision and reducer
contract are detailed in `../target_readout_protocol_v3.md` and must be hashed
before Phase R below.

The development context remains 16,384 bp with dense attention. The final
confirmation context remains unshifted 131,072 bp, followed by the frozen
`-64/0/+64` shift sensitivity. The component and base-pair window selected at
16,384 bp must map mechanically to the longer context; it cannot be retuned.

The original v2 mean canonical splice-site probability is a mandatory,
non-selecting sensitivity analysis. The two endpoint margins and
probabilities, `CL:0002518` splice-site usage, a prospectively annotated
cassette-junction log-count ratio, minigene context, and RNA-related outputs are
also diagnostics only. They cannot rescue failure of `DeltaL`, nominate a
different circuit, or erase the v2 probability negative. If a v3 candidate
passes only for the logit margin, the result must be called target-dependent.

An effect is eligible for normalized v3 causal recovery only when its paired
identity rows satisfy

```text
sign(DeltaL) == sign(experimental delta_logit) and abs(DeltaL) >= 0.01.
```

Ineligible effects remain failures in the predictive denominator; they are not
replaced. Proceed beyond the development target audit only if at least three of
the six effects in each development exon are eligible.

### 5.1 Phase R: isolate target choice from mechanism choice

Before new route or window instrumentation is ranked:

1. run the duplicate/repeat and endpoint-contract tests in
   `../target_readout_protocol_v3.md` on the frozen 20 development rows;
2. save raw relevant and background logits, both endpoint margins, their mean,
   `DeltaL`, and the corresponding v2 probabilities for every row;
3. require bit-exact duplicate/repeat scalars and at least three eligible
   effects in both BRAF and SLC25A48; and
4. rerun the **unchanged** 72-member v2 residual grid: three seams, layers
   0--5, `V/A/D/S`, the same controls, and the same six-row paired transfer.

Only the scalar reducer changes. Keep the v2 recovery formulas, self controls,
`B`, `q`, `Q`, candidate order, and the per-exon `B >= 0.25`, `q > 0` gates.

- If the unchanged grid passes, lock its first passing candidate under a new
  v3 artifact and stop Stages A--D. Confirmation then tests a logit-margin
  circuit, while the v2 probability result remains negative.
- If the target audit passes but the grid does not, continue to the wider
  mechanism ladder.
- If the target audit fails, stop. A different output cannot be substituted
  after results are visible.

This sequential test prevents a readout improvement from being attributed to
the encoder/decoder census, a wider window, or a learned feature.

### 5.2 Development-only input-space baselines

Before internal screening, partition the 4,352-bp development span around `S`
into fixed 128-bp tiles. For each non-`S` tile, run 32 dinucleotide-preserving
shuffles in both REF and ALT backgrounds and report how the allele effect
changes. Add gradient-times-input and single-base ISM at the actual variant.
These CREME-style tests estimate sequence-context dependence; they neither
identify an internal route nor change the fixed v3 window ladder. Use
length-preserving edits only. Do not run this baseline on confirmation exons
before the circuit lock.

Use the v2 live six-row batch in one compiled execution:

```text
row 0  REF baseline/donor          row 1  ALT baseline/donor
row 2  ALT <- REF                 row 3  ALT <- ALT self
row 4  REF <- ALT                 row 5  REF <- REF self
```

For every candidate retain the v2 definitions of `r_REF_to_ALT`,
`r_ALT_to_REF`, `B`, `q`, `Q`, reciprocity, raw movement, and self drift. Do not
clip over-restoration.

## 6. Gate 0 for every new tensor family

The batch-transfer primitive must operate on a live donor tensor before any
recipient write. Activations may not leave the executable, be rounded through
host memory, or be replayed in a later model call.

For encoder, decoder, output-embedder, projected-subspace, and SAE-feature
interventions, require:

1. REF-to-REF and ALT-to-ALT target values equal their duplicate baseline rows
   bit-for-bit;
2. selected self-patch tensors equal the pre-intervention tensors bit-for-bit;
3. an all-false intervention equals the baseline;
4. an already-compiled exact repeat agrees bit-for-bit for target and selected
   traces;
5. fixed-shape runtime indices do not trigger candidate-specific recompilation;
6. paired identity outputs satisfy the same public-equivalence and denominator
   rules as v2; and
7. no cast, invalid index, NaN, or failed assertion is silently skipped.

One self failure invalidates that candidate/variant. More than 5% of candidates
failing stops the implementation as a tooling failure.

Two additional closure controls are mandatory. For every development effect
and both patch directions:

- patching the complete live donor post-GELU 1-bp embedding at A and D must
  reproduce the donor target bit-for-bit; and
- jointly patching the complete donor transformer output and all seven donor
  encoder skips at the decoder boundary must reproduce the donor target
  bit-for-bit.

Failure of either closure means the new route instrumentation is incomplete;
no component may be ranked.

## 7. Stage A: exact encoder/decoder route census

This stage is diagnostic localization. A whole-state closure is not eligible
as the final mechanism.

### 7.1 Two-branch census at the decoder boundary

For each development effect patch, in both directions:

- `T`: the complete 128-bp transformer output sequence;
- `E`: all seven encoder skip tensors at bin sizes
  `1,2,4,8,16,32,64`; and
- `T+E`: both routes jointly.

Let `m(X)` be the target after donor state from route set `X` enters a recipient.
Alongside ordinary recovery, report the two-route Shapley decomposition:

```text
phi_T = 0.5 * [(m(T)-m(empty)) + (m(T+E)-m(E))]
phi_E = 0.5 * [(m(E)-m(empty)) + (m(T+E)-m(T))]
interaction = m(T+E) - m(T) - m(E) + m(empty)
```

Normalize only after retaining raw movements. This is a computational branch
partition, not an additive biological decomposition.

### 7.2 Encoder skip-scale census

For each skip scale `b`, calculate from the convolution/pooling graph the exact
set `I_b(V)` of bins whose receptive fields contain the variant base. Add one
bin of guard on each side and freeze all indices before patch results. Test:

- each `E_b` alone;
- `E_all_without_b` as a leave-one-scale-out necessity test;
- cumulative fine-to-coarse order `1,2,4,8,16,32,64`; and
- cumulative coarse-to-fine reverse order.

Controls are equal-width regions centred first 512 bp upstream and downstream
of V, moved outward until they do not overlap the candidate support, plus 32
same-width random regions. Candidate and control mappings use the same
receptive-field calculation.

### 7.3 Decoder and output-embedder census

At every decoder output resolution `64,32,16,8,4,2,1`, freeze the exact bins
whose downstream receptive fields can reach A or D. Patch that ancestral
support and a one-bin guard. Use equal-width shifted and random controls.

At the final 1-bp `OutputEmbedder`, separately patch the two pre-normalization
addends at A and D:

- `P_dec`: the projected decoder output;
- `P_128`: the projected and repeated 128-bp embedding; and
- `P_dec + P_128`: the joint state before RMS normalization and GELU.

Report the same two-route Shapley decomposition. The joint addend patch and the
post-GELU 1-bp embedding patch are upper-bound controls, not candidate circuits.

### 7.4 Route decision

- A route is **available** when a non-closure patch in that route has median
  `B >= 0.25` and positive median local-control margin in both development
  exons.
- A route is **present but diffuse** when its whole-route patch reaches `0.25`
  in both exons but no scale/support candidate does.
- A route is **weak under this intervention** when even its whole-route patch
  is below `0.25` in either exon.
- If only `T+E` passes, report a non-separable route interaction and continue
  only with predeclared joint feature screening. Do not call the whole joint
  state a mechanism.

## 8. Stage B: layers 6--8 and wider transformer windows

Let `S` retain its v2 definition. Define the nested transformer window

```text
W_r = {token t: min distance from t to any token in S <= r}
r in {0, 1, 2, 4, 8, 16}
```

In the development artifacts, `S` is two adjacent tokens, so the largest window
is 34 tokens, or 4,352 bp. For every `r`, test all three residual seams and
layers 0--8. This both completes layers 6--8 and reruns the narrow states in a
single versioned grid.

For each candidate width, the two deterministic local controls have identical
shape and relative offsets and are translated upstream and downstream by the
span of `W_r` plus four tokens, then moved outward until disjoint and in bounds.
Also freeze 32 random translations with the same relative offsets, at least four
tokens from `S`.

Patch the complete transformer sequence at `pre_attention` layer 0 and
`post_mlp` layer 8 as branch upper bounds. These full-context patches have no
valid same-width positional control and are diagnostic only.

Within a nested `(stage, layer)` family, choose the **smallest** radius that
passes the development gate; do not choose the radius with the largest raw
recovery. If more than one family passes, rank by `Q`, then stage order, layer,
and radius. A window is eligible for the final lock only if:

- both development-exon medians have `B >= 0.25`;
- both have positive median `q`;
- pooled candidate recovery is above the 95th percentile of the 32 indexed
  random-window statistics; and
- raw intended-output movement exceeds wrong-strand and non-target-exon
  movement under the v2 specificity definition.

If an exact raw route or transformer-window candidate passes, lock the simplest
passing candidate after any predeclared compact decomposition. Stages C and D
may describe it but cannot replace it with a higher-recovery post-hoc story.

## 9. Stage C: probes, attribution, and raw feature directions

Reach this stage only if a route-level upper bound passes but no compact raw
candidate does. Everything in this section is development-only screening until
an exact patch passes Section 11.

### 9.1 Probe tasks and controls

At the single route-localized state, extract REF/ALT activation differences on
the frozen spatial support. Fit:

1. a ridge readout of model `DeltaL` from activation difference;
2. a ridge/rank readout of experimental `delta_logit` and, where both classes
   exist within a split, an effect-sign classifier as biological-alignment
   diagnostics; and
3. splice-role probes for acceptor, donor, exon, and matched background bins.

Cross-validation is grouped by genomic position and exon. Never split alternate
alleles at one position across folds. Report exon-held-out results where the
label is defined, not a random row split. The frozen BRAF effects are all
positive and SLC25A48 effects all negative, so effect-sign prediction is
gene-confounded and cannot pass a gate even if its apparent accuracy is high.

Every probe is compared with:

- raw one-hot/k-mer, GC, CpG, transition/transversion, region, and position
  baselines;
- codon phase and stop-codon creation/destruction for exonic variants;
- branchpoint, polypyrimidine-tract, canonical-site, and published
  enhancer/silencer motif annotations where defined;
- shuffled labels and matched random directions; and
- the scalar AlphaGenome output difference itself.

Probe success establishes decodability only. It never passes a causal gate.
Compute codon, RBP-motif, branchpoint, polypyrimidine, and RNA-structure
annotations in transcript orientation using a frozen transcript and strand
mapping; never score the raw genomic strand indiscriminately.

### 9.2 Attribution-patching screen

For a recipient activation `h_r`, donor activation `h_d`, and primary target
`L_exon`, screen positions/channels using the first-order estimate

```text
a_i = grad_h_i(L_exon recipient) * (h_d_i - h_r_i)
```

Compute both allele directions and aggregate by the same exon-balanced logic as
`Q`. Use dense attention. For query/key candidates apply AtP*'s exact-softmax
QK correction; use GradDrop to diagnose direct/indirect cancellation. A tiled
backend without an audited gradient must not silently substitute for dense.

Before AtP orders a high-dimensional search, calibrate it on the exact Stage B
grid. Require both:

- Spearman correlation at least `0.5` between estimated and exact aggregate
  candidate scores; and
- at least 80% of the exact top decile to occur in the estimated top quartile.

If either fails, AtP is qualitative only. Use exact hierarchical blocks rather
than declaring screened-out units irrelevant.

### 9.3 Frozen raw-direction candidates

At the localized tensor, construct only the following subspaces:

- ridge direction for model `DeltaL`;
- top allele-difference principal subspaces of ranks `1,4,16,64`; and
- top AtP/AtP* coordinate groups of sizes `1,4,16,64`.

Orient differences from lower to higher primary model output, not by BRAF gene
identity. Learn each candidate once on BRAF and test it exactly on SLC25A48,
then reverse the roles. The final development subspace is the deterministic
union of the two reciprocal candidates at the smallest rank that passes in
both directions; cap the union at 64 dimensions. No per-variant direction is
allowed.

Exact subspace transfer must occur live:

```text
h_patched = h_recipient + P * (h_donor - h_recipient)
```

where `P` is the frozen projection. Same-allele transfer must be exactly zero.
Controls include equal-rank random orthonormal subspaces, activation-difference
norm-ranked coordinates, and shuffled-label probe directions.

## 10. Stage D: conditional sparse-autoencoder screen

Reach this stage only if a route is available or diffuse and the raw-direction
stage does not yield a passing candidate. Train at one frozen state and spatial
support selected by the route census.

### 10.1 Frozen SAE family

Train TopK SAEs on the background corpus in Section 4.3 with:

- expansion factors `4` and `8`;
- `k` in `{32, 64, 128}`;
- three fixed seeds derived from the protocol hash; and
- no OpenSplice confirmation activation in training, validation, early
  stopping, feature naming, or examples.

Choose the smallest model on the validation Pareto frontier that meets all of:

- at least 95% activation variance explained and median reconstruction cosine
  at least `0.99`;
- dead-feature fraction at most 10%;
- no more than 10% normalized median change in the primary development target
  when the full state is replaced by its SAE reconstruction; and
- stable causal candidates in at least two of three seeds, matched by decoder
  cosine at least `0.8` and activation correlation at least `0.8`.

If no SAE passes, stop. A lossy dictionary cannot support a causal feature
claim.

### 10.2 Feature nomination and exact intervention

Rank a feature `j` using both allele directions of

```text
AtP_feature_j = grad_h(L_exon) dot d_j * (z_donor_j - z_recipient_j)
```

where `d_j` is its decoder direction. Test joint sets of the first
`1,2,4,8,16` stable features. The live exact intervention leaves recipient
reconstruction error in place:

```text
h_patched = h_recipient
            + sum_j d_j * (z_donor_j - z_recipient_j)
```

Controls are equally active random features, activation-frequency-matched
features, decoder-norm-matched features, shuffled-label features, and equal-size
random subspaces. Patch both allele directions with exact self controls.

MEME/TomTom, RBP motif databases, CpG, stop-codon, repeat, and RNA-structure
associations may name a candidate after its ID is frozen. Annotation never
substitutes for exact causal recovery. A feature aligned with a known confound
is reported as a model shortcut unless independent biology supports it.

## 11. One development lock gate for all candidate families

A Phase-R residual, transformer window, encoder/decoder support, raw subspace,
or SAE feature set becomes the single v3 candidate only if all of the
following hold on the original 12 development effects:

1. Gate 0 passes for every candidate and matched control.
2. Each development exon has median bidirectional `B >= 0.25`.
3. Each development exon has positive median local-control margin `q`.
4. The pooled candidate statistic exceeds the 95th percentile of 32 indexed,
   size-matched random interventions.
5. Both allele directions have positive median recovery in each exon.
6. Intended-output specificity margin is positive in at least 9/12 effects and
   has positive median in each development exon.
7. A learned direction or feature selected on either exon passes exact
   reciprocal testing on the other exon.
8. No result depends on replacing a failed variant, changing the primary
   output, or choosing a per-variant component.

Selection is sequential and parsimonious:

1. stop at and lock the first passing Phase-R residual candidate;
2. otherwise prefer the first passing raw route/window family;
3. otherwise prefer the smallest passing raw-direction subspace;
4. otherwise prefer the smallest passing stable SAE feature set;
5. within a family choose the smallest nested size that passes, then largest
   `Q`, then the lexical order specified in that stage; and
6. lock exactly one candidate.

Closure states, whole-context patches, full output embeddings, direct
output-logit patches, probes, and AtP scores are never lock-eligible.

If no candidate passes, v3 is a negative development result for this entire
predeclared ladder. Keep confirmation unopened.

## 12. Confirmation and claim gates

After the immutable lock, evaluate the candidate once on the frozen ELN,
EIF4A2, and DMD rows. Apply the numerical v2 Gates 1--4 to the frozen v3
`DeltaL` target without weakening:

- at unshifted 131,072 bp, direction accuracy at least 20/30 overall and at
  least 13/18 in confirmation, at least 4/6 in two confirmation exons, and no
  confirmation exon at 0/6; effects with `abs(DeltaL) < 0.01` count as
  ineligible rather than disappearing;
- pooled median confirmation `B >= 0.25`, positive median in every confirmation
  exon, and `>= 0.25` in at least two of three;
- pooled positive control margin and performance above the 95th percentile of
  the frozen random null;
- effect-versus-neutral and intended-output specificity;
- context and `-64/0/+64` shift robustness; and
- compactness only if the locked claim is feature- or subspace-level.

The original canonical-probability delta and its v2 recovery metrics must be
reported for the same locked intervention, but cannot select a candidate or
rescue a failed logit-margin gate. Conversely, a logit-margin pass with
probability recovery below `0.25` is explicitly a target-dependent
computational result, not recovery of the v2 probability mechanism.

Report observed positive and negative signs separately. Because positive sign
occurs in BRAF and ELN while negative sign occurs in SLC25A48, EIF4A2, and DMD,
sign and locus remain partly confounded.

Confirmation success establishes computational generalization across exons for
one AlphaGenome checkpoint and intervention. It does not prove an RBP binding
event, spliceosome assembly step, native-tissue PSI mechanism, or chemical
pathway.

## 13. Mandatory reporting and interpretation

Every result bundle must contain:

- hashes and exact software/checkpoint/FASTA provenance;
- all candidate and control raw scores, both directional recoveries, `B`, `q`,
  `Q`, self drift, reciprocity, and specificity;
- raw `DeltaL` endpoint terms and the mandatory non-selecting v2 probability
  deltas and recoveries for the same interventions;
- route Shapley values and interactions as descriptive branch accounting;
- exact spatial receptive-field/ancestry maps;
- probe performance beside raw-sequence and confounder baselines;
- AtP calibration against exact patches and all known false negatives;
- SAE activation and output fidelity, dead features, seed matching, and
  reconstruction-error effects;
- CpG, GC, codon/stop, region, splice-signal, motif, and RNA-structure strata;
- every output-ineligible or denominator-failing variant; and
- all development attempts, not only the locked candidate.

Allowed conclusions are deliberately tiered:

| Evidence | Allowed statement |
|---|---|
| Closure and Gate 0 only | Tooling implements exact live causal transfer. |
| Development route census | In these two exons, the model effect is localized to or distributed across specified computational branches. |
| Development candidate passes | A candidate computational mechanism is ready for a held-out test. |
| Locked candidate passes confirmation | The same AlphaGenome mechanism generalizes computationally across the three unopened exons. |
| Feature has motif annotation and causal effect | AlphaGenome uses a feature aligned with that sequence pattern; physical RBP binding is still unproven. |
| Feature aligns with CpG/stop/coding confound and disagrees with assay | Evidence for a model shortcut or training-distribution bias, not splice biology. |
| OpenSplice agreement | The model computation is consistent with an independent minigene effect; endogenous native-locus biology remains to be tested. |

The project goal remains interpreting how AlphaGenome arrived at a prediction.
It must not be described as extracting a complete biological or chemical
pathway from the model.

## 14. Pre-inference freeze checklist

Before running v3:

- [ ] Commit or timestamp this protocol and record its SHA-256.
- [ ] Freeze and hash the development screening manifest, if used.
- [ ] Freeze and hash the SAE background manifest, if used.
- [ ] Write exact receptive-field and decoder-ancestry index manifests.
- [ ] Freeze all local and random control positions and seeds.
- [ ] Version new instrumentation without modifying v2 raw artifacts.
- [ ] Pass unit tests for simultaneous donor reads, duplicate positions,
      padding, BF16 self identity, and all-false masks.
- [ ] Pass the two closure controls on development before ranking.
- [ ] Verify that no confirmation baseline, trace, attribution, probe, or SAE
      example has been created.
- [ ] Record the exact adaptive branch taken and every stopped branch.
- [ ] Create `circuit_lock_v3.json` before any confirmation inference.

# OpenSplice v3 target/readout slice

**Status:** development-only experimental plan; no v3 model inference has run

**Date:** 2026-08-28

**Frozen v2 selection:**
`09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6`

## Decision

The primary v3 scalar is the mean canonical **splice-classification logit
margin**:

```text
L_site = logit(relevant donor-or-acceptor class) - logit(padding/background)
L_exon = mean(L_acceptor, L_donor)
DeltaL = L_exon_ALT - L_exon_REF
```

Here “logit” means the actual linear outputs saved by AlphaGenome under
`splice_sites_classification/logits`, before softmax. It does not mean applying
an inverse-logit transform to the public probabilities.

This is the most defensible target that can be used immediately. It preserves
the exact strand-aware 1-bp endpoints, is invariant to an arbitrary common
shift of all five class logits, avoids probability saturation and avoids
AlphaGenome's dynamic splice-junction top-k. The two endpoint margins remain
mandatory diagnostics; their mean is frozen so an endpoint cannot be chosen
after seeing the result.

A three-junction log-count ratio is more directly aligned to the OpenSplice
phenotype in principle, but is not the primary v3 target. It becomes a valid
secondary baseline only after the two inclusion junctions and one skipping
junction are frozen from annotation, before model outputs are viewed, and all
three exact junctions are present for both alleles. Splice-site usage in
`CL:0002518` is a predeclared tissue-aware sensitivity readout, not PSI.

## What the negative v2 result does and does not say

The repaired paired-batch v2 run passed Gate 0 with exactly zero self drift in
all 2,592 intervention groups. Its best residual component recovered a median
0.133399 in BRAF and 0.099521 in SLC25A48, below the frozen 0.25 gate. Thus the
current negative result is valid for whole-token residual transfer when the
scalar target is the mean canonical splice-site **probability**.

It does not test whether the same internal state is clearer against a
pre-softmax classification-evidence target. The v2 probabilities are bounded
and can be saturated at strong canonical sites; probability deltas also have
very different dynamic ranges in the two development exons. v3 tests this
readout hypothesis without rewriting v2 or changing the frozen variants. A
better v3 result would show target dependence, not invalidate v2.

## Source audit

### AlphaGenome semantics

The official source was audited at upstream commit
[`1e55dcffb98ba26b31e74edc5e9f038f54c0e89d`](https://github.com/google-deepmind/alphagenome_research/tree/1e55dcffb98ba26b31e74edc5e9f038f54c0e89d).
The relevant local files are unchanged from that commit.

1. [`heads.py`](https://github.com/google-deepmind/alphagenome_research/blob/1e55dcffb98ba26b31e74edc5e9f038f54c0e89d/src/alphagenome_research/model/heads.py)
   (`SHA-256 11f2f72a5a23bcf8b9eb6a1ffe586d17b254047d9ea988e86f25807e943a52e4`)
   shows that the classification head applies a five-way softmax to its
   internal linear `logits` and trains with cross-entropy from those logits.
   The usage head independently applies sigmoid to each track's internal
   `logits`, casts the public prediction to float16, and trains with binary
   cross-entropy from logits. The junction head produces nonnegative softplus
   counts over donor-by-acceptor pairs and optimizes count and ratio losses.
2. [`io/splicing.py`](https://github.com/google-deepmind/alphagenome_research/blob/1e55dcffb98ba26b31e74edc5e9f038f54c0e89d/src/alphagenome_research/io/splicing.py)
   shows the five classification labels: positive donor, positive acceptor,
   negative donor, negative acceptor and the logical complement/background
   class. This is why the class-minus-padding margin is the identified
   pre-softmax evidence ratio; a relevant-class logit alone is not.
3. The official human metadata
   (`SHA-256 413b4063096bbf7a75a7fdf5560030f8e3866c6af2455fb313e58686ae3d49e8`)
   contains 5 classification tracks, 734 usage tracks (367 tissues times two
   strands) and 367 junction tissue tracks. It has exactly one positive and
   one negative `usage_CL:0002518 total RNA-seq` track, and one
   `junction_CL:0002518 total RNA-seq` tissue.
4. [`variant_scoring/splice_junction.py`](https://github.com/google-deepmind/alphagenome_research/blob/1e55dcffb98ba26b31e74edc5e9f038f54c0e89d/src/alphagenome_research/model/variant_scoring/splice_junction.py)
   (`SHA-256 47f84d74393ee4d830e58c668a9b33e746e2697efac296b586bf1fa26e75978f`)
   shows that public variant scoring normally reports log-offset junction-count
   deltas and that public junction rows use exact genomic junction coordinates.
   A maximum absolute junction change is therefore unsuitable for signed PSI.

### OpenSplice assay semantics

OpenSplice was audited at commit
[`3e4ad8c037c216b952f1a8945f8f498669bff589`](https://github.com/lehner-lab/OpenSplice/tree/3e4ad8c037c216b952f1a8945f8f498669bff589).

1. [`psi_per_barcode.R`](https://github.com/lehner-lab/OpenSplice/blob/3e4ad8c037c216b952f1a8945f8f498669bff589/psi_calculation_pipeline/03_psi_per_barcode/psi_per_barcode.R)
   (`SHA-256 4a5640ed59469b974b8dc858993fb83e25be96bf0a01c525effbc76a0665ff1d`)
   classifies cDNA reads as cassette-exon inclusion, exon skipping or other,
   and defines canonical PSI as inclusion / (inclusion + skipping).
2. [`calculate_psi_with_error_model_locally.R`](https://github.com/lehner-lab/OpenSplice/blob/3e4ad8c037c216b952f1a8945f8f498669bff589/psi_calculation_pipeline/04_psi_per_variant/calculate_psi_with_error_model_locally.R)
   (`SHA-256 3da38f0a7d2ca9d4b86dcb65998295822cf1a51e36ec745756c7743ced74dec2`)
   adds a Jeffreys prior, combines replicates on the logit(PSI) scale and
   applies empirical-Bayes shrinkage.
3. [`00_master_table_creation.R`](https://github.com/lehner-lab/OpenSplice/blob/3e4ad8c037c216b952f1a8945f8f498669bff589/analysis/00_master_table_creation.R)
   (`SHA-256 188706c14fa5aa65e114c8752d0fdcf17381165fd62211cae7ee965c93dfbae1`)
   defines the experimental target as
   `delta_logit = logit_psi_variant - logit_psi_WT`.
4. The pinned replicate configuration uses three `hek` biological replicates.
   The OpenSplice AlphaGenome benchmarking script explicitly calls
   `CL:0002518` kidney epithelial cell the closest available ontology to HEK,
   not an exact assay match. It also uses canonical splice-classification
   probability rather than the usage or junction outputs.

The assay is a minigene inclusion/skipping measurement in HEK context, whereas
the primary AlphaGenome experiment uses native hg38 sequence. No AlphaGenome
single-site score is literally cassette-exon PSI.

## Frozen v3 readouts

### Primary: classification logit margin

For each strand-aware canonical acceptor and donor, subtract the fifth
padding/background logit from the relevant class logit. Average the two
site-level margins and take ALT minus REF. This quantity equals
`log(p_relevant / p_background)` algebraically but is calculated directly from
pre-softmax logits, so it neither saturates nor requires clipping a probability.

Required outputs per allele and variant:

- both raw relevant logits and both padding logits;
- acceptor and donor margins;
- their frozen arithmetic mean; and
- ALT-minus-REF deltas for every quantity.

### Sensitivity 1: `CL:0002518` usage logits

Select exactly one `usage_CL:0002518 total RNA-seq` track with the exon's
strand. Average its pre-sigmoid logits at the same canonical acceptor and donor
and take ALT minus REF. This asks whether the model changes predicted
utilization of the two sites in a vaguely HEK-related context. It does not
encode competition with the exon-skipping junction and cannot be called PSI.

### Sensitivity 2: cassette-junction logit(PSI)

Let `I_up` and `I_down` be predicted counts for the upstream-inclusion and
downstream-inclusion junctions, and `S` the predicted exon-skipping junction
count in `junction_CL:0002518 total RNA-seq`. Define:

```text
I = mean(I_up, I_down)
J = log(I + 1e-7) - log(S + 1e-7)
DeltaJ = J_ALT - J_REF
```

`J` is the logit of `I / (I + S)` and is the closest AlphaGenome readout here
to OpenSplice logit(PSI). It is valid only if all of the following are frozen
before predictions are viewed:

- exact GRCh38/Gencode version and the three junction coordinates;
- the transcript/cassette event used to choose the two flanking exons;
- the `CL:0002518 total RNA-seq` track and `1e-7` offset; and
- a missing-edge policy of “undefined,” never nearest-edge substitution.

All three exact rows must occur once for REF and ALT. Missing, duplicate,
negative or non-finite counts fail closed. Dynamic splice-site top-k membership
must be reported; a missing edge is a coverage limitation, not zero expression.

### Junction-coordinate feasibility audit

The official runtime expects the GRCh38 Gencode v46 asset
`hg38/gencode.v46.annotation.gtf.gz.feather`, but that complete, pinned
annotation was not found in the local experiment inputs. The OpenSplice design
metadata identifies BRAF exon 14 as transcript `ENST00000646891` on the
negative strand (`chr7:140754187-140754233`) and SLC25A48 exon 8 as transcript
`ENST00000510147` on the positive strand
(`chr5:135880773-135880910`). Those cassette-exon coordinates do not by
themselves identify the adjacent exons, so they are insufficient to construct
the three junctions safely.

Consequently, no flanking-junction coordinates are inferred or invented in
v3. Before the junction sensitivity can run, download the comprehensive CHR
GTF from the official [GENCODE release 46
page](https://www.gencodegenes.org/human/release_46.html), record its checksum,
resolve the immediately adjacent exons in those exact transcripts, convert
their introns to 0-based half-open junction keys, and freeze the six resulting
edges in a versioned manifest. The tested junction reducer is ready, but the
biological target remains deliberately unavailable until that prerequisite is
satisfied.

## Development-only execution plan

### Phase A: contract tests

The CPU reference implementation is `target_reducers_v3.py`. Its synthetic
tests lock positive/negative-strand endpoint mapping, paired rather than
Cartesian reduction, common-logit-shift invariance, exact ontology selection,
missing/duplicate junction failure and the cassette logit(PSI) formula.

### Phase B: baseline target audit

1. Create a new output directory named `results/v3_development_targets`; never
   overwrite a v2 JSON.
2. Load only the 20 already-frozen BRAF and SLC25A48 development variants
   (12 effects and 8 neutrals) at the unchanged 16,384-bp native context.
3. In one paired REF/ALT executable, export only:
   `splice_sites_classification/logits` at the two canonical endpoints and the
   padding class, plus the exact `CL:0002518` usage logits. Do not invert the
   public probability tracks.
4. Run duplicate REF and ALT rows and an exact repeated call. Require bit-exact
   duplicate/repeat scalar values before interpreting a delta.
5. Report all 20 variants for the locked primary and both sensitivity
   readouts. For effects, report direction agreement with experimental
   `delta_logit`; for neutrals, report raw deltas and matched-effect ratios.
   Also report Spearman rank correlation within each exon, but do not use it to
   choose a readout.
6. Freeze the three cassette junction coordinates for each development exon
   from pinned Gencode annotation without consulting predictions. Then run the
   junction baseline only if all three exact rows are available for every REF
   and ALT; otherwise report coverage and stop that sensitivity analysis.

The primary logit-margin readout is locked before Phase B. Usage or junction
performance cannot replace it or cause variants, endpoints, tissues or
contexts to be changed.

### Phase C: one frozen residual-grid rerun

Proceed only if at least three effects in each development exon have the
correct `DeltaL` sign and `abs(DeltaL) >= 0.01`, and the paired identity/repeat
audit is exact. Rerun the unchanged 72-member residual grid, V/A/D/S token sets,
upstream/downstream positional controls and six-row live paired transfer. The
classification logit-margin mean replaces only the scalar reducer.

Retain the v2 causal gates without tuning:

- self drift at most `1e-4` and at most 1% of `abs(DeltaL)`;
- `B = min(recovery_REF_to_ALT, recovery_ALT_to_REF)`;
- `q = B_candidate - max(B_upstream, B_downstream)`;
- choose by the same cross-exon Q statistic and tie order; and
- require median `B >= 0.25` and median `q > 0` in both development exons.

If no component passes, v3 is a valid negative for this readout and whole-token
intervention family. If one passes, write and timestamp a new v3 circuit lock
before any confirmation inference. Do not reuse or overwrite a v2 circuit
artifact. Usage and junction targets remain sensitivity analyses at the locked
primary component; they cannot select a different layer or position set.

## Interpretation boundary

The primary margin is evidence that AlphaGenome classifies the canonical bases
as splice sites; it is not exon inclusion itself. The usage readout is
tissue-aware site utilization but lacks explicit exon-skipping competition.
The cassette-junction ratio is closest to PSI but depends on annotation,
dynamic junction coverage, a non-HEK tissue proxy and native rather than
minigene context. Results must retain those names and limitations.

No confirmation data or internals are needed to decide or test this v3
development hypothesis. The confirmation set remains closed unless the new
primary target passes all preregistered development gates.

# OpenSplice circuit-benchmark freeze

This directory implements the OpenSplice stage of the frozen positive-control
protocol. It selects variants using experimental measurements and exact VCF
alleles before any local AlphaGenome inference.

The five exons are the first five entries in the upstream paper's ordered
30-exon curated splice-map list. This mechanical rule was frozen without
consulting any AlphaGenome score. The original v1 rule then attempted to choose
three negative, three positive and four matched neutral variants per exon, but
failed closed because BRAF e14 has no significant negative-effect variant. No
v1 selection was created. `selection_attempt_v1.json` preserves that negative
result without rewriting the original v1 plan or selector.

The feasible v2 protocol keeps the same five exons, restricts the primary
causal benchmark to SNVs, and chooses six significant variants with the largest
absolute experimental effects plus four matched neutral controls per exon.
Full rules and source checksums are in `dataset_plan_v2.json`.

## Data boundary

Three Figshare v5 files are sufficient:

1. `open_splice_experimental_master_unique_sequence_all_predictor_cols.tsv.gz`
   supplies experimental PSI and uncertainty. Although it contains existing
   model scores, `select_holdout.py` has a hard allowlist and cannot use them.
2. `opensplice_predictors_benchmarking_exon_metadata.tsv` supplies strand and
   exon coordinates for later inference.
3. `alphagenome_genome_input_per_exon.zip` supplies exact GRCh38 VCF alleles.

Do not download barcode tables, raw ENA reads, SpliceTransformer inputs or
Supplementary Table 12 for this benchmark. In particular, do not inspect or use
existing AlphaGenome scores to replace a difficult example.

## Reproducing the failed v1 attempt

After downloading and checksum-verifying the three pinned inputs, run:

```bash
python experiments/interpretability/opensplice/select_holdout.py \
  --input /path/to/open_splice_experimental_master_unique_sequence_all_predictor_cols.tsv.gz \
  --frozen-exons experiments/interpretability/opensplice/frozen_exons_v1.tsv \
  --vcf-source /path/to/alphagenome_genome_input_per_exon.zip \
  --dry-run > /tmp/opensplice-selected-preview.tsv
```

Dry-run mode performs the complete attempted selection and VCF join but does
not create a repository artifact. With the pinned release it must fail on the
missing BRAF negative stratum. Do not weaken the v1 rule or create a v1 manifest.

Tests are CPU-only and use synthetic data:

```bash
python -m pytest experiments/interpretability/opensplice/select_holdout_test.py
```

The existing protocol remains authoritative: model circuit analysis is gated
on correct AlphaGenome output direction, failures remain in the denominator,
and final examples are repeated at 131,072 bp. No example may be replaced based
on AlphaGenome performance.

## Frozen v2 inference and tracing

The original balanced-sign v1 rule was infeasible for the predeclared BRAF exon.
The frozen SNV-only v2 manifest therefore takes the six largest absolute
experimental effects and four matched neutral controls per exon, without
requiring three effects in each direction. The tracked inputs are
`selected_variants_v2.tsv` and `frozen_exons_v2.tsv`.

Validate the complete 50-variant 16-kb work plan without loading a checkpoint:

```bash
experiments/interpretability/opensplice/run_inference_trace.sh --dry-run
```

Run one 16-kb baseline variant (no causal tracing):

```bash
experiments/interpretability/opensplice/run_inference_trace.sh \
  --max-variants 1
```

The runner defaults to dense attention for causal measurements because it is
the numerical reference implementation and fits the 16/131-kb contexts used by
this benchmark. `--attention-backend pallas_tiled` is an optional replication,
not a replacement for the dense result.

Tracing is separately bounded and remains disabled by default. For example, add
`--trace-max-variants 1 --trace-max-groups-per-variant 2` to run at most two
stage/layer/region groups on the first effect variant that passes the
output-direction gate. Each group places two baselines, two reciprocal patches,
and two same-allele controls in one six-row execution, transferring live
residuals without a host round trip. Every baseline, Gate 0 audit, and trace
group is atomically written to its own fingerprinted JSON file, so the same
command resumes without repeating completed work. Use
`--confirmation-131kb` only after reviewing the 16-kb gate.

The primary scalar exactly reproduces the upstream genome-mode definition: the
mean of the ALT-minus-REF probabilities at the strand-aware canonical acceptor
and donor. The frozen output gate requires its sign to match experimental
delta-logit-PSI and its absolute magnitude to be at least 0.01. Residual groups
jointly patch V (variant), A (acceptor), D (donor), or S (their unique union),
with same-cardinality controls shifted at least four 128-bp tokens while
preserving relative offsets. The causal path fails closed on non-SNVs because
residual patching requires aligned REF/ALT transformer tokens.

The public predictor and instrumented tracer are separate BF16-compiled graphs.
Their baseline scalar must agree within one BF16 output ULP near probability
one (`2^-8 = 0.00390625`). Before any live patch, a resume-safe Gate 0 artifact
also requires an all-false six-row batch to have exact duplicate-allele rows
and selected traces across two identical executions. Every live group then
requires exact same-allele target identity and exact donor-to-effective-
recipient residual equality at the selected seam.

The v2 manifest is `selected_variants_v2.tsv`; its SHA-256 is
`09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6`.
Full checks are in `selection_validation_v2.json`. The CPU-only v2 tests are:

```bash
python -m unittest \
  experiments/interpretability/opensplice/select_holdout_v2_test.py -v
```

## Frozen development result

The first cross-execution implementation failed Gate 0 because returning BF16
residuals to Python and reinserting them in a later model call introduced
nonzero same-allele drift. That run is retained as a tooling failure and is not
interpreted biologically. The repair and its admissibility before confirmation
are frozen in `gate0_paired_batch_amendment_v2_1.md`.

The v1.2 live paired-batch implementation passed Gate 0 on all 12 development
effects and all 2,592 candidate/control groups. Every same-allele output drift
was exactly zero, and all 12 identity/repeat audits were bit-exact. The best
frozen component was `pre_attention`, layer 2, joint set `S`; median
bidirectional recovery was 0.1334 for BRAF and 0.0995 for SLC25A48. This is
below the preregistered 0.25 threshold, and none of the 72 components passed.

The result is therefore a valid negative for this compact 128-bp residual
intervention family. It does not show that AlphaGenome lacks splice mechanisms
or interpretable representations. The ELN, EIF4A2 and DMD confirmation
internals remain unopened, as required by the frozen protocol. See
`results/v2_development_paired_batch/PAIRED_BATCH_DEVELOPMENT_RESULT.md` and
the machine-readable `DEVELOPMENT_ANALYSIS.json` for the exact result and raw
artifact-tree hash.

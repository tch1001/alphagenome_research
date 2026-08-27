# OpenSplice circuit-benchmark freeze

This directory implements the OpenSplice stage of the frozen positive-control
protocol. It selects 50 variants using experimental measurements and exact VCF
alleles before any local AlphaGenome inference.

The five exons are the first five entries in the upstream paper's ordered
30-exon curated splice-map list. This mechanical rule was frozen without
consulting any AlphaGenome score. The selector then chooses three negative,
three positive and four matched neutral variants per exon. Full rules and
source checksums are in `dataset_plan_v1.json`.

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

## Deterministic dry run

After downloading and checksum-verifying the three pinned inputs, run:

```bash
python experiments/interpretability/opensplice/select_holdout.py \
  --input /path/to/open_splice_experimental_master_unique_sequence_all_predictor_cols.tsv.gz \
  --frozen-exons experiments/interpretability/opensplice/frozen_exons_v1.tsv \
  --vcf-source /path/to/alphagenome_genome_input_per_exon.zip \
  --dry-run > /tmp/opensplice-selected-preview.tsv
```

Dry-run mode performs the complete selection and VCF join but does not create a
repository artifact. It prints the canonical TSV to standard output and its
SHA-256 to standard error. Once provenance and quality checks pass, omit
`--dry-run` and provide `--output` to freeze `selected_variants_v1.tsv`.

Tests are CPU-only and use synthetic data:

```bash
python -m pytest experiments/interpretability/opensplice/select_holdout_test.py
```

The existing protocol remains authoritative: model circuit analysis is gated
on correct AlphaGenome output direction, failures remain in the denominator,
and final examples are repeated at 131,072 bp. No example may be replaced based
on AlphaGenome performance.

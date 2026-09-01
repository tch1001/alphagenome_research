# AlphaGenome interpretability: current scientific status

This is the active research note. Historical run manifests and infrastructure
postmortems remain in the repository for auditability, but they do not define
the research agenda.

## Goal

Identify internal AlphaGenome representations that causally mediate a
biologically measured sequence effect, localize those representations to
specific features or channels, connect them to recognizable sequence programs,
and test the resulting hypothesis on held-out loci.

The object of study is model behavior. Operating-system metadata is not a
scientific endpoint.

## What the experiments currently show

### TAL1 tooling control

For the known `chr1:47239296:C>ACG` TAL1 neo-enhancer example, exchanging the
local early residual representation between REF and ALT nearly exchanges the
predicted allelic effect. Matched distant residual controls are much smaller.

This validates causal patching and localizes dependence near the variant at an
early model stage. It does not identify a specific feature or transport
circuit: individual local attention-head and compact pair-edge interventions
were small, and the same patch also exchanged the nearby `PDZK1IP1` effect.

### OpenSplice transformer-residual result

Across 12 development effects from BRAF and SLC25A48, none of 72 localized
transformer-residual candidates met the preregistered recovery threshold. This
is a useful negative for that intervention family, not evidence that the
transformer is irrelevant or uninterpretable.

### OpenSplice coarse route result

Patching all seven encoder skips transferred substantially more of the
allele-specific canonical splice-site effect than patching the complete
transformer output:

| Route | BRAF median recovery | SLC25A48 median recovery |
|---|---:|---:|
| All encoder skips | 0.71195 | 0.93974 |
| Transformer output | 0.23510 | 0.05716 |

The corresponding median normalized two-player Shapley shares were 0.73761
encoder/0.26239 transformer for BRAF and 0.94129/0.05871 for SLC25A48.

The defensible conclusion is narrow: in this graph, target and development
set, the splice effect transfers much more effectively through the aggregate
encoder-skip route. The result does not yet identify a resolution, position,
channel, motif, RBP or biochemical pathway.

### Exploratory seven-resolution decomposition

A new CPU-only analysis reconstructed all 5,120 records in the already
completed 20-variant by 256-coalition development cube directly from raw
relevant/padding logits. It made no model call and read neither confirmation
nor the incomplete unrelated-donor prefix. A separate permutation-based audit
matched every reported Shapley value to within `1.21e-13`.

With natural transformer output, median normalized Shapley contributions were:

| Skip resolution | BRAF | SLC25A48 |
|---|---:|---:|
| E64 | 0.07876 | 0.03153 |
| E32 | 0.21743 | 0.10381 |
| E16 | 0.18651 | 0.04182 |
| E8 | 0.07276 | 0.09452 |
| E4 | 0.02640 | 0.08200 |
| E2 | 0.04356 | 0.14949 |
| E1 | 0.03106 | 0.42005 |

The smallest coalition satisfying the historical two-gene recovery and
retention rule is `E32+E16+E8+E2+E1`. It reaches median bottleneck recovery
0.62056 in BRAF and 0.77170 in SLC25A48, retaining 86.3% and 82.3% of the
all-skip result. This excludes E64 and E4 but still requires five resolutions.

The profile is strongly exon-dependent: E32/E16 lead in BRAF, whereas E1/E2
lead in SLC25A48. The coalition also fails biological alignment in BRAF:
median absolute movement is 0.57324 for the six effects versus 2.67822 for the
four experimentally neutral variants. SLC25A48 shows the opposite pattern
(4.38184 versus 0.06445). The result therefore identifies a multiscale
computational route, not a specific or general biological mechanism.

Primary result: [exploratory analysis](opensplice/results/v3_3_development_encoder_skip_factorial_exploratory_model_behavior_analysis/RESULT.md).
Arithmetic check: [independent audit](opensplice/results/v3_3_development_encoder_skip_factorial_exploratory_model_behavior_analysis/INDEPENDENT_AUDIT.md).

## Main unresolved question

Within the five-resolution `E32+E16+E8+E2+E1` route, which spatial positions
causally carry the effect? Does a compact region around the variant, acceptor
or donor beat equal-size shifted controls within the same variant? Only after
that should channels or sequence motifs be interpreted.

Prepared next experiment: [spatial encoder-skip design](opensplice/spatial_encoder_skip_experiment.md)
with its [machine-readable coordinate plan](opensplice/spatial_encoder_skip_plan_v1.json)
and [development-only runner](opensplice/run_spatial_encoder_skip_experiment.py).

## Research sequence

1. Spatially restrict `E32+E16+E8+E2+E1` patches to receptive-field supports
   around the variant, acceptor, donor and acceptor/donor union.
2. Compare each support with equal-shape upstream/downstream translations in
   the same variant; retain exact reciprocal, self/no-op and repeat controls.
3. Use neutral variants and non-target outputs as secondary specificity checks,
   recognizing that an experimental neutral need not be AlphaGenome-null.
4. Within spatially localized routes, rank channels/features by causal effect rather
   than activation magnitude alone.
5. Characterize causal features with controlled sequence edits, motif scans,
   maximally activating sequences and, if needed, sparse feature dictionaries.
6. Turn the feature interpretation into explicit predictions—for example,
   which motif-disrupting edits should alter splice-site output—and test them
   on held-out exons before making a biological claim.

## Lean reproducibility policy

Reproducibility safeguards should protect the interpretation without becoming
the project. Active experiments should pin or record:

- model checkpoint and code revision;
- biological dataset, genome build, alleles and target definition;
- intervention sites, donor/recipient construction and controls;
- precision, model backend and random seeds where applicable; and
- raw model outputs needed to independently recompute the claimed statistic.

The following are diagnostics, not default pass/fail gates:

- Linux distribution or kernel patch version;
- hostname, inode numbers and temporary-file names; and
- unrelated cache-directory layout.

An environment difference should stop an experiment only when it changes model
loading, tensor shapes, selected examples, numerical controls, or the measured
model behavior. Exact self-patches, no-op interventions and repeated baselines
remain important because they directly test the causal instrumentation.

## Claim boundary

Causal activation transfer establishes a fact about AlphaGenome's computation,
not automatically about cellular biochemistry. A biological interpretation
requires a localized and specific model feature, a sequence-level hypothesis,
independent measurements, matched controls and held-out replication.

Historical preflight failures and their kernel metadata are retained only as
archived engineering records. They should not be foregrounded in scientific
summaries or used to choose the next interpretability experiment.

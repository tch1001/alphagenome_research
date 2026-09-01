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

## Main unresolved question

Which encoder-skip resolutions and features carry the effect, and do they
encode recognizable splice-regulatory sequence programs that generalize
across loci?

## Research sequence

1. Decompose the seven encoder skips with isolated-skip, leave-one-out and
   coalition interventions, including reciprocal patches and exact self/no-op
   controls.
2. Use neutral variants, unrelated donors, shifted positions and non-target
   outputs to distinguish causal signal from broad representation replacement.
3. For resolutions that replicate across BRAF and SLC25A48, localize the
   effect spatially around the variant, acceptor and donor.
4. Within localized routes, rank channels/features by causal effect rather
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

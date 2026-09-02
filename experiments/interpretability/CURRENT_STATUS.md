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

### Spatial localization of the five-resolution route

The complete 20-variant spatial experiment made 520 model applies with exact
repeats. All live donor-vector, disabled/no-op, same-allele and repeat controls
passed. All 160 equal-shape controls translated at least 512 bp away had
exactly zero recovery.

On the 12 development effects, median bidirectional bottleneck recovery was:

| Support | BRAF | SLC25A48 | Pass both genes |
|---|---:|---:|:---:|
| Variant | 0.41409 | 0.39649 | yes |
| Acceptor | 0.38700 | 0.65977 | yes |
| Donor | 0.27905 | 0.00893 | no |
| Acceptor/donor union | 0.51613 | 0.66509 | yes |

The five-resolution skip route is therefore spatially concentrated in the
variant/splice-site neighborhood rather than diffuse across the sequence.
Donor-only behavior differs sharply between the two exons.

This does not establish separate variant, acceptor and donor modules. Many
selected variants lie at or near a canonical splice site, so their guarded
token supports overlap at several resolutions. The BRAF neutral-control
warning also remains: intended patches move the four experimental neutrals
more than the six effects for every passing support.

Primary result: [spatial model-behavior analysis](opensplice/results/spatial_encoder_skip_v1_model_behavior_analysis/RESULT.md).
Arithmetic check: [independent audit](opensplice/results/spatial_encoder_skip_v1_model_behavior_analysis/INDEPENDENT_AUDIT.md).

### Causal channel localization

A complete 3,520-apply screen withheld each of 172 nonoverlapping 32-channel
blocks from the V-local five-resolution route. All live donor, withheld-value,
same-allele, baseline and non-route controls passed. Identity and full-route
repeats were exact.

The leading cross-gene blocks by the smaller of the two gene-level median
necessity losses were:

| Block | BRAF median loss | SLC25A48 median loss |
|---|---:|---:|
| E1 channels 160-191 | 0.02007 | 0.04814 |
| E32 channels 0-31 | 0.01750 | 0.01567 |
| E16 channels 0-31 | 0.01779 | 0.00963 |

SLC25A48 ranked the same 160-191 channel band highly at every tested
resolution. This aligns with a concrete architectural clue: each
`DownResBlock` carries its existing channel prefix through a zero-padded
residual path while appending 128 channels. Because intervening convolutions
mix channels, this is evidence consistent with a persistent multiscale feature
family, not proof that a feature has invariant identity across layers.

Primary result: [32-channel analysis](opensplice/results/channel_group_screen_v1_model_behavior_analysis/RESULT.md).
Arithmetic check: [independent audit](opensplice/results/channel_group_screen_v1_model_behavior_analysis/INDEPENDENT_AUDIT.md).

### Eight-channel refinement

The five selected parents were each split into four 8-channel children in a
480-apply refinement. All runtime controls again passed. The finer result is
more informative than the coarse shared ranking: all three purportedly shared
parents have different dominant children in the two genes.

| Parent | BRAF dominant child | SLC25A48 dominant child |
|---|---|---|
| E1 160-191 | E1 160-167 | E1 168-175 |
| E32 0-31 | E32 0-7 | E32 16-23 |
| E16 0-31 | E16 0-7 | E16 16-23 |

BRAF preferentially uses channels 0-7 at both E32 and E16. SLC25A48's two
strongest children overall are channels 168-175 at E2 and E1, narrowing its
persistent coarse band to the same eight-channel slice at two resolutions.
The best genuinely shared 8-channel maximin losses are only about 0.001.
The current evidence therefore favors adjacent exon-specific channel programs
over one dominant universal splice channel.

Primary result: [8-channel analysis](opensplice/results/channel_refinement_v1_model_behavior_analysis/RESULT.md).
Arithmetic check: [independent audit](opensplice/results/channel_refinement_v1_model_behavior_analysis/INDEPENDENT_AUDIT.md).

### Individual-channel necessity and localized sufficiency

A 960-apply validation withheld all 32 constituent channels individually and
tested each of the four selected 8-channel subspaces by itself at intended,
upstream and downstream positions. All runtime controls passed, and all 160
shifted sufficiency controls had exactly zero recovery.

Three coordinates pass the frozen rule requiring positive effect median
necessity, an effect median larger than the gene-matched neutral median, and a
positive loss in at least four of six effects:

| Gene | Stage/channel | Effect median loss | Neutral median loss | Positive effects |
|---|---|---:|---:|---:|
| BRAF | E16 channel 3 | 0.01254 | 0.00656 | 4/6 |
| SLC25A48 | E1 channel 175 | 0.03987 | 0.00000 | 4/6 |
| SLC25A48 | E2 channel 175 | 0.04512 | 0.00000 | 5/6 |

The repeated SLC25A48 coordinate is the strongest current mechanistic lead:
channel 175 is necessary at two successive resolutions, and its E1 and E2
eight-channel parent subspaces are independently sufficient at the intended
variant neighborhood but not at either shifted location. This strengthens the
hypothesis of a persistent multiscale model feature.

All four selected parent subspaces pass the localized-sufficiency rule for
their selected gene, although recovery is small (`B=0.00141` to `0.02661`).
The BRAF E32 subspace is localized and sufficient, but no constituent channel
passes the individual effect-over-neutral rule, suggesting within-subspace
synergy or redundancy.

Primary result: [individual-channel analysis](opensplice/results/individual_channel_validation_v1_model_behavior_analysis/RESULT.md).
Arithmetic check: [independent audit](opensplice/results/individual_channel_validation_v1_model_behavior_analysis/INDEPENDENT_AUDIT.md).

## Main unresolved question

Are BRAF E16 channel 3 and SLC25A48 E2/E1 channel 175 sufficient by themselves
at the intended variant neighborhood? Which sequence patterns drive those
coordinates, and do controlled edits support the resulting motif or
splicing-factor hypothesis?

Prepared and completed channel experiments: [32-channel screen design](opensplice/channel_group_screen.md),
[8-channel refinement design](opensplice/channel_refinement.md), and
[refinement plan](opensplice/channel_refinement_plan_v1.json).

Completed spatial experiment: [design](opensplice/spatial_encoder_skip_experiment.md),
[coordinate plan](opensplice/spatial_encoder_skip_plan_v1.json), and
[development-only runner](opensplice/run_spatial_encoder_skip_experiment.py).

## Research sequence

1. Test BRAF E16 channel 3 and SLC25A48 E2/E1 channel 175 by themselves at
   intended and equal-shape shifted positions.
2. Characterize surviving coordinates using sequence preference maps,
   activation optimization and controlled nucleotide edits.
3. Use neutral variants and non-target outputs as secondary specificity checks,
   recognizing that an experimental neutral need not be AlphaGenome-null.
4. Characterize causal features with controlled sequence edits, motif scans,
   maximally activating sequences and, if needed, sparse feature dictionaries.
5. Turn the feature interpretation into explicit predictions—for example,
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

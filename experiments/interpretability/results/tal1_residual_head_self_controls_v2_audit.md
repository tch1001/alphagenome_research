# Canonical v2 audit: TAL1 residual and local-head self controls

## Verdict

The final v2 artifact confirms the central model-level result: the predicted
allelic effect is almost completely and bidirectionally controlled by the
early local residual representation around the TAL1 insertion. The matched
self-corrected recovery is 101.307% REF into ALT and 102.158% ALT into REF.
Same-stage/layer distance controls are zero or negative.

Local head patches identify small, distributed candidates rather than a
dominant route. Promoter L3/H0 remains the strongest bidirectional local patch,
at 2.275% forward and 1.560% reciprocal. At the enhancer, L3/H0 is strongest
forward but L0/H4 is strongest reciprocal; no enhancer head reaches 1% in both
directions.

Per-gene self controls definitively fail the proposed TAL1-specificity check.
The same early residual patch recovers PDZK1IP1 by 101.696% forward and 98.866%
reciprocally relative to its own predicted allelic effect. This is a strong
local-locus intervention, not a TAL1-selective pathway.

Audited artifact:
[`tal1_residual_head_self_controls_v2_131kb_shift959.json`](./tal1_residual_head_self_controls_v2_131kb_shift959.json).

## Independent recomputation

Every stored residual and local-head corrected recovery was independently
recomputed from its donor and self-patch totals. Maximum error was zero at the
stored precision.

| Baseline quantity | Recomputed value |
|---|---:|
| REF TAL1 RNA total | 40,533.8984 |
| ALT TAL1 RNA total | 64,053.5078 |
| ALT - REF | 23,519.6094 |
| ALT / REF | 1.580245 |
| Increase over REF | 58.0245% |
| Scored positions | 15,936 |
| REF and ALT repeat deltas | 0.0, 0.0 |

The artifact contains 108 residual patches and 144 local-head patches. The
single-pair and global head-ablation arrays are empty. Nine unconditional
candidate joint-pair records remain from the shared runner, but the dedicated
pair artifact is the appropriate source for that separate null result.

## Early enhancer residual

The key pre-attention layer-0, five-token enhancer intervention is:

| Quantity | REF into ALT | ALT into REF |
|---|---:|---:|
| Donor-patched total | 39,997.6953 | 64,637.5859 |
| Matched self-patched total | 63,824.6797 | 40,610.4648 |
| Donor effect relative to self | -23,826.9844 | +24,027.1211 |
| Raw recovery | 102.2798% | 102.4834% |
| Corrected recovery | **101.3069%** | **102.1578%** |

The ALT-self graph offset is -228.8281 and the REF-self offset is +76.5664,
so the matched correction remains necessary. After correction, the forward
patch is 612.7695 below the matched REF-self total and the reciprocal patch is
812.9063 above the matched ALT-self total. This mild bidirectional overshoot is
consistent with nonlinear hybrid activation patching.

Matched pre-attention layer-0 controls are:

| Region | Corrected forward | Corrected reciprocal |
|---|---:|---:|
| Enhancer candidate | 101.3069% | 102.1578% |
| Upstream distance control | 0.0000% | 0.0000% |
| Downstream distance control | -0.4098% | -0.2546% |

Using the matched self-patched allele gap as denominator gives 102.640% forward
and 103.502% reciprocal. Across all residual patches, that gap is
98.56--99.54% of the original baseline gap, so the denominator choice changes
the numerical percentage slightly but not the conclusion.

The enhancer trajectory remains bidirectional and structured:

| Enhancer patch | Corrected forward | Corrected reciprocal |
|---|---:|---:|
| Pre-attention L0 | 101.307% | 102.158% |
| Post-attention L0 | 88.655% | 85.629% |
| Pre-attention L2 | 87.319% | 83.502% |
| Pre-attention L4 | 37.015% | 29.904% |
| Pre-attention L7 | 24.428% | 20.117% |
| Pre-attention L8 | 6.611% | 5.771% |

Promoter-window recovery peaks at 19.283% forward and 15.065% reciprocal in
the middle/late tower. This temporal redistribution is compatible with a
long-range model computation, but it is not a proved enhancer-to-promoter
route: the pair state is not patched with the sequence residual, runs are
separate nonlinear hybrids, and the promoter window overlaps the scored
target.

Across all distance controls, the largest corrected forward and reciprocal
effects are 3.051% and 2.436%. Both arise from the upstream window that lies
inside the scored TAL1 interval. For the non-target-overlapping downstream
control, the maxima are 0.486% forward and 0.338% reciprocal; its strongest
bidirectional minimum is 0.338%. The key early enhancer result remains orders
of magnitude larger.

## Local head-value patches

All 144 patches replace one 192-dimensional weighted-value vector at one
sampled token before output projection. Every ALT-self and REF-self local-head
delta is exactly zero.

| Position and head | Corrected REF into ALT | Corrected ALT into REF |
|---|---:|---:|
| Promoter L3/H0 | **2.2746%** | **1.5604%** |
| Promoter L0/H2 | 1.6108% | 1.2946% |
| Promoter L7/H3 | 1.3431% | 0.9299% |
| Promoter L5/H5 | 1.1344% | 0.7199% |
| Enhancer L0/H4 | 1.2922% | **0.4175%** |
| Enhancer L3/H0 | **1.4271%** | 0.3318% |

Only two of 144 patches exceed 1% in both directions: promoter L3/H0 and
promoter L0/H2. L3/H0 is the leading promoter component and the leading
forward enhancer component, but the enhancer reciprocal effect is small and
is surpassed by L0/H4. This is convergent evidence for candidate components,
not a single-head circuit. The run lacks distant-token head controls, joint
head patches, nearby-token windows, and conditional mediation tests.

## Per-gene specificity fails

The v2 specificity block applies the early five-token enhancer residual patch
with separate self controls for each gene:

| Gene | Baseline allelic effect | Corrected REF into ALT | Corrected ALT into REF |
|---|---:|---:|---:|
| TAL1 | +23,519.6094 | 101.3069% | 102.1578% |
| PDZK1IP1 | +20.6162 | **101.6959%** | **98.8662%** |

Thus both predicted gene effects are nearly completely swapped. The stored
PDZK1IP1 absolute patch delta is only 0.0908% when divided by the much larger
TAL1 allelic total, but that is not gene specificity. It merely states that
PDZK1IP1 has a much smaller absolute RNA total and predicted effect.

PDZK1IP1's matched self-patched allele gap is 95.47% of its baseline gap,
versus 98.70% for TAL1. Normalizing by those self gaps gives:

| Gene | REF into ALT | ALT into REF |
|---|---:|---:|
| TAL1 | 102.640% | 103.502% |
| PDZK1IP1 | 106.518% | 103.554% |

The specificity failure is invariant to denominator choice. A valid
specificity experiment needs unaffected and magnitude-matched genes, unrelated
RNA and chromatin tracks, other cell types, and absolute plus proportional or
link-scale effects.

## Defensible claim boundary

Supported:

- the shifted model predicts a 58.0% increase in the selected TAL1 RNA output;
- the early five-token representation around the insertion bidirectionally
  controls almost all of that predicted difference after matched self control;
- the key residual effect is much larger than the tested distance controls;
- promoter L3/H0 and L0/H2 are small, reproducible local model components;
- the intervention affects more than one predicted gene in this locus.

Not supported:

- that the residual feature is specifically a MYB motif representation;
- that any identified head transmits the enhancer signal to the promoter;
- that the intervention is TAL1- or output-specific;
- that a complete computational circuit, physical enhancer-promoter contact,
  or molecular pathway has been recovered.

## Remaining blockers

1. Add length-matched motif-preserving and motif-destroying edits, then narrow
   the whole-residual transplant to tokens and motif-linked feature subspaces.
2. Add local-head controls at nearby and distance-matched tokens, joint head
   patches, and conditional enhancer-to-promoter mediation tests including the
   persistent pair state.
3. Replace the target-overlapping upstream residual control and match controls
   for target distance, sequence, accessibility, and activation norm.
4. Expand self-controlled specificity across unaffected/matched genes, tracks,
   cell types, and fold/crop replicates.
5. Confirm the endpoint with the public 1 Mb stitched-indel path, a
   fold-appropriate checkpoint, and an independent locus.
6. Record the exact runner/source hash and CLI caps and advance the schema
   version; exact device repeats establish numerical replay, not biological
   uncertainty.

The canonical v2 label is: **validated bidirectional multi-gene local-residual
dependence; small distributed head candidates; unresolved feature identity,
transport route, and specificity**.

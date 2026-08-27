# Audit: TAL1 bidirectional pair-bias self controls

## Verdict

The pair-bias intervention is numerically clean and mechanistically negative.
All 432 single-cell self patches, all nine all-head layer self patches, and the
all-layer/all-head self patches reproduce their recipient baselines exactly.
The corrected and raw effects are therefore identical.

No candidate promoter-to-enhancer pair-bias intervention is enriched over
controls or convincingly bidirectional. The best candidate single cell
recovers 0.2302% forward and 0.0750% reciprocally; a distance-control cell has
the stronger bidirectional minimum, 0.1096%. Patching all candidate heads at a
layer or across every layer also fails bidirectionality. These results do not
support the nominated compact pair-bias edge as the transport route.

The artifact's TAL1-versus-PDZK1IP1 specificity block is for the early
five-token **residual** patch, not a pair-bias patch. With the new per-gene
self corrections, that patch swaps both genes' predicted allelic effects and
does not demonstrate TAL1 specificity.

Audited artifact:
[`tal1_pair_self_controls_131kb_shift959.json`](./tal1_pair_self_controls_131kb_shift959.json).

## Baseline and formula checks

| Quantity | Recomputed value |
|---|---:|
| REF TAL1 RNA total | 40,616.3086 |
| ALT TAL1 RNA total | 64,001.1563 |
| ALT - REF | 23,384.8477 |
| ALT / REF | 1.575750 |
| Increase over REF | 57.5750% |
| REF and ALT repeat deltas | 0.0, 0.0 |

All stored single-cell forward and reciprocal corrections were recomputed from
the totals with maximum error zero at stored precision:

- forward: `(REF-donor ALT run - ALT-self ALT run) / (REF - ALT)`;
- reciprocal: `(ALT-donor REF run - REF-self REF run) / (ALT - REF)`.

The expected signature of a causal allele-mediating component is positive in
both directions: REF values should lower the ALT output, and ALT values should
raise the REF output.

## Single edge/head cells

There are 432 patches: 6 edges x 9 layers x 8 heads. Despite the top-level key
`candidate_pair_patches`, five-sixths of the records are controls.

| Statistic | Edge, layer/head | Corrected effect |
|---|---|---:|
| Best candidate forward | promoter-to-enhancer L0/H2 | +0.2302% |
| Its reciprocal effect | promoter-to-enhancer L0/H2 | +0.0750% |
| Best candidate reciprocal | promoter-to-enhancer L0/H4 | +0.1540% |
| Its forward effect | promoter-to-enhancer L0/H4 | -0.4663% |
| Largest forward across all edges | enhancer self L0/H0 | +0.5013% |
| Largest reciprocal across all edges | enhancer self L0/H1 | +0.1948% |
| Best bidirectional minimum overall | downstream-distance L2/H3 | +0.1096% |

The candidate's best bidirectional cell is L0/H2, whose smaller directional
effect is 0.0750%; the matched downstream control L2/H3 is +0.1193% forward
and +0.1096% reciprocal. Nine of 72 candidate cells are positive in both
directions, but all effects are tiny and the strongest matched control is
larger.

Edge-level averages also show no candidate enrichment:

| Edge | Mean corrected forward | Mean corrected reciprocal | Mean absolute forward | Mean absolute reciprocal |
|---|---:|---:|---:|---:|
| Promoter to enhancer | -0.0740% | -0.0029% | 0.0993% | 0.0542% |
| Enhancer to promoter | -0.0645% | -0.0132% | 0.0911% | 0.0504% |
| Enhancer self | -0.0625% | +0.0006% | 0.1096% | 0.0603% |
| Promoter self | -0.0606% | -0.0070% | 0.0655% | 0.0301% |
| Promoter to upstream distance | -0.0639% | -0.0167% | 0.0809% | 0.0518% |
| Enhancer to downstream distance | -0.0869% | -0.0078% | 0.1069% | 0.0518% |

## Joint candidate-edge patches

Patching all eight candidate-edge heads within each layer gives:

| Layer | Corrected forward | Corrected reciprocal |
|---:|---:|---:|
| 0 | +0.4252% | -0.0426% |
| 1 | -0.1085% | -0.2558% |
| 2 | -0.0865% | -0.0580% |
| 3 | +0.2738% | -0.0149% |
| 4 | -0.0646% | -0.0129% |
| 5 | +0.0120% | +0.0385% |
| 6 | -0.1326% | +0.0848% |
| 7 | -0.1319% | +0.0342% |
| 8 | -0.0097% | -0.0011% |

Only layer 5 has the expected sign in both directions, and its smaller effect
is 0.0120%. Patching all eight heads at this edge in every layer is also
directionally inconsistent: +0.5356% forward and -0.1577% reciprocal.

The artifact does not contain all-head or all-layer joint patches for the
reverse, self, or distance-control edges. Thus even the small joint candidate
effects cannot be tested for joint-edge specificity. Their failure to reverse
the REF prediction already rules out a robust bidirectional mediation claim.

## Self-control drift

The self controls are exact:

- all 432 single-cell ALT-self and REF-self deltas are 0.0;
- every all-head-per-layer self total equals its allele baseline;
- both all-layer/all-head self totals equal their allele baselines.

This is expected because compact pair replacement uses the same already-
allocated pair intervention graph as the identity call. It also means the
small pair effects are not caused by the optional residual-patch graph offset
identified in earlier runs. Self correction neither weakens nor strengthens
them.

## TAL1 versus PDZK1IP1 specificity

The specificity block applies the pre-attention layer-0, five-token enhancer
**residual** transplant. It must not be described as pair-edge specificity.
Its per-gene self-corrected results are:

| Gene | Baseline allelic effect | Corrected REF into ALT | Corrected ALT into REF |
|---|---:|---:|---:|
| TAL1 | +23,384.8477 | 102.0166% | 102.6065% |
| PDZK1IP1 | +20.3937 | 109.1630% | 100.4340% |

The new self controls remove the earlier ambiguity: relative to each gene's
own predicted allelic effect, the patch is near-complete in both directions
for both genes. Normalizing PDZK1IP1's absolute patch delta by the much larger
TAL1 effect yields the stored 0.0938%, but that measures absolute spillover in
one track, not gene specificity.

Because PDZK1IP1 has a small effect, denominator choice is visible. The
self-patched allele gap is 99.16% of the baseline gap for TAL1 and 103.28% for
PDZK1IP1. Using each self-patched gap as denominator gives:

| Gene | REF into ALT | ALT into REF |
|---|---:|---:|
| TAL1 | 102.8857% | 103.4806% |
| PDZK1IP1 | 105.6958% | 97.2441% |

The qualitative conclusion is unchanged: both outputs swap with the early
local residual. This validates locus dependence, not TAL1 selectivity. A true
specificity analysis needs unaffected and magnitude-matched genes, unrelated
tracks and cell types, and both absolute and proportional/link-scale metrics.

## Interpretation boundary

Supported:

- pair intervention plumbing is numerically stable under same-allele controls;
- none of the tested single compact pair-bias cells, all-head layer patches,
  or the all-layer candidate patch explains the TAL1 allelic prediction;
- the early enhancer-window residual remains a strong causal locus signal for
  both TAL1 and PDZK1IP1 predictions.

Not supported:

- that the promoter-to-enhancer compact pair-bias edge carries the signal;
- that the pair stream as a whole is irrelevant;
- that the residual effect is specific to TAL1;
- that a biological enhancer-promoter contact or molecular pathway has been
  identified.

A compact pair-bias cell is one head-specific scalar applied to a directional
2,048 bp by 2,048 bp attention tile. It is not an attention probability,
value-vector message, physical contact, or complete pair representation. A
distributed route could use adjacent tiles, Q/K and value content, multiple
edges, or the persistent pair state without being captured by these patches.

## Remaining decisive tests

1. Run bidirectional all-head and all-layer patches for every control edge and
   for adjacent tiles, not only the nominated candidate edge.
2. Patch the underlying pair representation and sequence/value messages, alone
   and jointly, rather than only the derived compact bias scalar.
3. Test conditional mediation: whether a downstream promoter or head patch
   effect depends on restoring the enhancer residual, with matched controls.
4. Add length-matched MYB motif controls, the public 1 Mb stitched-indel
   endpoint, fold-appropriate checkpoints, multiple crops, and independent
   loci.
5. Record source and CLI hashes and bump the schema version when intervention
   fields change; baseline repeats establish numerical replay, not biological
   or checkpoint uncertainty.

The appropriate label is: **clean bidirectional null for the nominated compact
pair-bias edge; strong but multi-gene early residual dependence remains**.

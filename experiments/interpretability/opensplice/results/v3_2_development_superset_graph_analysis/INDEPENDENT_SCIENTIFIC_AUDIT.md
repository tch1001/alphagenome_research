# Independent scientific audit of the OpenSplice v3.2 result

## Scope and method

This is a development-only audit of the completed v3.2.4 offline analysis. It
independently reconstructed endpoint margins, target means, bidirectional
recoveries, Phase-R ranks, Stage-A route summaries and Shapley terms from the
stored `selected_logits` in the raw JSON artifacts. It did not use the emitted
`checks.recovery`, ranking or Shapley values as inputs to those calculations.

The raw manifest exactly enumerates 2,660 JSON artifacts. Every artifact hash
matches, and the independently recomputed tree SHA-256 is
`4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8`.
No confirmation model output, activation or intervention was opened. Later-exon
metadata/label exposure remains disclosed, so this is not complete metadata
blindness.

The audited target is the strand-aware mean of two pre-softmax splice-class
logit margins (relevant acceptor/donor class minus padding class). It is not a
direct PSI readout.

## Target eligibility and controls

- All 20 identity records were structurally valid and repeat-exact.
- All 12 significant effects were eligible: six BRAF and six SLC25A48. Their
  predicted effect exceeded the frozen `0.01` magnitude threshold and matched
  the experimental `delta_logit` direction.
- The eight experimentally nonsignificant variants are behavior controls, not
  AlphaGenome-null controls. All eight had an absolute AlphaGenome target change
  of at least `0.01`.
- BRAF neutral target changes ranged from `-10.248046875` to `-2.421875`, with
  median absolute change `3.9384765625`. SLC25A48 neutral changes ranged from
  `-0.12109375` to `0.015625`, with median absolute change `0.044921875`.

This last observation is important: the neutral labels do not license a null
assumption for this AlphaGenome target, and the neutral variants did not receive
the intervention grid.

## Phase-R residual census

All 2,592 groups were present and valid: 12 effects times 216 groups, with zero
invalid groups. None of the 72 frozen candidates passed the two-exon gate.

The top-ranked candidate was `pre_attention`, layer 4, position set `S`:

| Exon | Median B | Median q |
|---|---:|---:|
| BRAF | 0.18391945133518167 | 0.19012774763540852 |
| SLC25A48 | 0.05679520160680139 | 0.05643523040449758 |

Its frozen `Q = min(median q)` was `0.05643523040449758`. The positive `q`
values mean it outperformed its two matched spatial controls, but both median
recoveries were below the required `B >= 0.25`. In fact, all 72 candidates had
positive median `q` in both exons, while none reached median `B >= 0.25` in both.
The largest median B attained by any candidate was `0.24207613617306725` in
BRAF and `0.05679520160680139` in SLC25A48.

The defensible Phase-R conclusion is therefore negative: no frozen localized
transformer-residual candidate recovered enough of the AlphaGenome target in
both development exons. Positive specificity relative to weak or adverse
controls is not sufficient recovery.

## Stage-A whole-route accounting

All 48 Stage-A groups were valid. Both mandatory closure families closed
exactly at the endpoint-logit level for all 12 effects and both transfer
directions (24 records, 48 directional closure checks). The 24 isolated T/E
records were also valid.

Median bidirectional recovery B was:

| Whole route | BRAF | SLC25A48 |
|---|---:|---:|
| Transformer route T | 0.23509760903151095 | 0.05715517280910521 |
| Encoder route E | 0.711950779808821 | 0.93973634975444 |

The independently recomputed median raw Shapley account was:

| Exon and direction | phi_T | phi_E | interaction | joint movement |
|---|---:|---:|---:|---:|
| BRAF, REF into ALT | -0.255615234375 | -0.590576171875 | 0.041015625 | -0.84814453125 |
| BRAF, ALT into REF | 0.255615234375 | 0.590576171875 | 0.041015625 | 0.84814453125 |
| SLC25A48, REF into ALT | 0.27716064453125 | 4.861328125 | -0.0078125 | 5.166015625 |
| SLC25A48, ALT into REF | -0.27716064453125 | -4.861328125 | -0.0078125 | -5.166015625 |

Every per-variant Shapley efficiency residual was exactly zero. The encoder
route E accounts for substantially more of the whole-route target movement than
T in these two development exons. This is descriptive evidence about a broad
computational route, not evidence that a particular encoder feature, sequence
motif, RBP or biochemical pathway is the mechanism.

## Claim audit

The decisions in `ANALYSIS.json` and `RESULT.md` are methodologically sound:

- `phase_r_negative_stage_a_routes_descriptive_only` matches the raw result;
- “none of the 72 frozen candidates passed” is correct;
- the Stage-A result is correctly limited to descriptive whole-route upper
  bounds and computational Shapley accounting; and
- the documents correctly refuse an RBP, pathway, endogenous-mechanism,
  experimental-replication or confirmation claim.

`RESULT.md` is accurate but terse. A reader-facing summary should also state
that the best Phase-R candidate recovered only about `0.184` and `0.057` in the
two exons, that the apparent Stage-A encoder dominance is not localized, and
that the eight neutral controls were not AlphaGenome-null and did not receive
interventions. Random-region, wrong-strand/non-target-output and neutral-
intervention controls remain prerequisites for a later localized mechanism
claim.

Finally, the compact raw JSON permits independent recomputation of endpoint
algebra, repeat equality and endpoint-level closures. Donor-tensor equality and
internal-seam assertions are represented by runner-emitted booleans and trace
fingerprints rather than full tensors, so this audit verifies their recorded
pass state and provenance but cannot reconstruct those internal tensors from
the compact artifacts alone.

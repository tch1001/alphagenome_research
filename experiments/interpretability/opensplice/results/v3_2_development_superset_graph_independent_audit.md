# Independent arithmetic audit of the OpenSplice v3.2 development result

## Scope

This is a read-only, CPU-only recomputation from the completed development
artifacts for BRAF exon 14 and SLC25A48 exon 8. It did not open or compute any
confirmation model output, activation, or intervention. Later confirmation
metadata/labels had already been exposed after the protocol freeze, as
disclosed by the primary result.

Audited result hashes:

- `ANALYSIS.json`: `a46c827e16fb4e054e7cf702f7147da70e0a3b35f677b430edf64e9a3055013c`
- `RESULT.md`: `e4c8d45c1b35d8934734c4d8c18bd2ca10b78dd0099a9b37c296e60b15b7e52c`
- frozen raw tree: `4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8`

## Independent recomputation

For every identity and active record, the acceptor and donor margins were
recomputed from the persisted relevant-class and padding-class logits using
float32 arithmetic. Their two-endpoint means were then used to recompute every
bidirectional recovery, Phase-R `B`, `q`, and `Q`, and every Stage-A Shapley
term. No analyzer-emitted recovery or summary was used as an input.

The independent results matched `ANALYSIS.json` exactly:

- 20 identities and 12 target-eligible effects, six per development exon;
- all 2,592 Phase-R active groups and the complete 72-candidate ranking;
- all 24 endpoint-level final-embedding and joint-T+E closure checks;
- the isolated whole-T and whole-E route summaries; and
- all per-variant, bidirectional Shapley values and zero efficiency residuals.

## Phase-R result

No transformer-residual candidate passed. The top candidate family was the
combined splice-token set `S`, but its absolute recovery was too small:

| Rank | Seam | Layer | Set | Q | BRAF median B | SLC25A48 median B |
|---:|---|---:|---|---:|---:|---:|
| 1 | pre-attention | 4 | S | 0.05644 | 0.18392 | 0.05680 |
| 2 | post-attention | 3 | S | 0.05644 | 0.18392 | 0.05680 |
| 3 | post-MLP | 3 | S | 0.05644 | 0.18392 | 0.05680 |

All 72 candidates had positive median `q` in both exons, meaning that each
candidate beat its two matched spatial controls on this statistic. That is not
enough: the protocol also required median `B >= 0.25` in both exons. The best
median `B` anywhere was 0.24208 for BRAF and 0.05680 for SLC25A48. Thus the
negative decision was an absolute-recovery failure, especially at SLC25A48,
not a missing-data, invalid-group, or control-subtraction failure.

This rules out the 72 frozen, localized residual-patch candidates under this
target and graph. It does not prove that the transformer is biologically or
computationally irrelevant: its information may be diffuse, represented
outside the chosen token sets/layers, redundant with skip features, or poorly
transferred into a recipient carrying incompatible natural skip tensors.

## Stage-A result

`B` is the smaller of the two reciprocal recoveries for each variant, followed
by the within-exon median. It is an unclipped causal-recovery statistic, not a
probability or percent variance explained.

| Whole route | BRAF median B | SLC25A48 median B | Variants with B >= 0.25 |
|---|---:|---:|---:|
| Encoder skips E (all seven) | 0.71195 | 0.93974 | 12/12 |
| Transformer output T | 0.23510 | 0.05716 | 3/12 |

The whole-E intervention transferred most of the donor allele's target effect
in both directions and for every eligible development variant. Whole T was
borderline and heterogeneous in BRAF, including two negative bottleneck
recoveries, and was small for all six SLC25A48 effects.

The two-player Shapley account corroborated this coarse routing result:

| Exon | Median normalized phi_E | Median normalized phi_T | Median absolute normalized interaction |
|---|---:|---:|---:|
| BRAF | 0.73761 | 0.26239 | 0.04801 |
| SLC25A48 | 0.94129 | 0.05871 | 0.00469 |

The exact joint closures and zero Shapley efficiency residuals show that the
instrumentation and two-route arithmetic closed. The generally small
interaction indicates that T and E were close to additive at this target under
these hybrid patch states. One BRAF variant assigned slightly more than 100%
to E and a negative share to T, which is permitted by an unclipped Shapley
decomposition and illustrates why these values are not probabilities.

## What this establishes—and what it does not

The strongest defensible statement is that, in this instrumented AlphaGenome
graph and these two development exons, the allele-specific canonical
splice-site logit effect is carried much more effectively through the complete
encoder-skip route than through the complete transformer-output route.

It does not yet identify a skip resolution, position, channel, motif, RBP,
spliceosome step, or biochemical pathway. E jointly replaces seven large skip
tensors at 1/2/4/8/16/32/64-bp resolution, so it is a broad upper bound that
can overwrite a great deal of recipient state. The experiment tests causal
transfer under constructed hybrid activations, not necessity in the natural
forward pass. The 12 variants are nested within only two exons, so they are not
12 independent biological loci, and there is no held-out model result here.

## Best next experiment

Prospectively freeze a development-only **seven-skip decomposition** before
opening confirmation:

1. Treat each encoder-skip resolution as a separate player and run the exact
   128 subsets of the seven E routes with the same six-row bidirectional batch.
   This is 1,536 subset calls for 12 variants, before controls, and is comparable
   in scale to the completed 2,592-group Phase-R census.
2. Report exact per-resolution Shapley contributions, interactions, reciprocal
   recovery, and leave-one-skip-out necessity alongside isolated-skip
   sufficiency.
3. At the whole-skip stage include neutral-variant interventions,
   shuffled/unrelated donors, wrong-strand or non-target-output specificity,
   and exact self/no-op checks.
4. If one or a small number of resolutions survive in both development exons,
   freeze a second spatial census within those skips over V/A/D/S and matched
   upstream/downstream plus random-position controls. Only after a localized
   route is locked should channel/feature methods such as sparse dictionaries
   or SAE-style analysis be used to seek motifs or known splicing programs.
5. Lock the candidate and thresholds before a single held-out confirmation
   analysis. Confirmation model outputs, activations, and interventions remain
   unopened at this stage.

This follows the evidence rather than retrying the negative transformer grid:
the current result localizes the promising next search to encoder skips, but
not yet within them.

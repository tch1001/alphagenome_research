# Independent audit of OpenSplice Phase R

**Audit date:** 2026-08-28

**Scope:** development artifacts only; no confirmation file or internal was
opened

**Verdict:** the frozen Phase-R narrow residual hypothesis is negative; proceed
to the wider development ladder and keep confirmation closed

## Recomputed result

I parsed the raw `summary.json`, 20 identity JSONs, and 2,592 group JSONs
without importing the model or the project analyzer. The raw tree contains
2,613 artifacts and independently reproduces its bound SHA-256:

```text
0b24d58491e33f14277c55396614714061d2143ba6dd7641671fab0c854298e9
```

The following checks reproduce exactly:

- frozen 50-row manifest SHA-256
  `09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6`;
- frozen-exon SHA-256
  `b95f8fc540f19222546322bebfb817a6c0f2147dd41325091086833584a09a75`;
- checkpoint snapshot
  `a8f293a76ee73d5b57f3bf2ae146510589fcf187`, dense attention and 16,384 bp;
- prospective-protocol SHA-256
  `852f84c7fdb4f9349ebef388aad4203ccb1fb9fc71abd8c68cad1b38a907c947`;
- target-protocol SHA-256
  `b322d88c5b581da5ed8f8060fd67a24b8a6fea3b728cfe8a104fad02a06aa317`;
- all 2,612 identity/group canonical configuration fingerprints;
- the strand-aware endpoint, output-index and class-track contracts in all 20
  identities; and
- exactly 72 candidates and 144 matched controls for each of the 12 eligible
  effects, with no missing or duplicate grid key.

All six BRAF and all six SLC25A48 effects have the experimental sign and
`abs(DeltaL) >= 0.01`. The six BRAF `DeltaL` values range from `0.4453125` to
`1.329833984375`; the six SLC25A48 values range from `-5.92578125` to
`-1.55078125`. Thus the required at-least-three-per-exon target gate passes
comfortably rather than through borderline denominators.

For every group, I recomputed from the six stored target means:

```text
r_REF_to_ALT = (L_RA - L_AA) / (L_R - L_A)
r_ALT_to_REF = (L_AR - L_RR) / (L_A - L_R)
B = min(r_REF_to_ALT, r_ALT_to_REF)
q = B_candidate - max(B_upstream, B_downstream)
```

The stored recoveries, per-exon medians, all 72 `Q` values, exact tie order and
pass flags match the independent calculation. The top candidate is
`pre_attention/layer4/S`:

| Exon | Median B | Median q | Required B |
|---|---:|---:|---:|
| BRAF | 0.181561845 | 0.186732180 | 0.25 |
| SLC25A48 | 0.057657474 | 0.057259068 | 0.25 |

Its cross-exon `Q` is `0.057259068`. Eleven of 12 individual bottleneck
recoveries and control margins are positive; `BRAF_e14_A117G` is the exception
(`B = -0.0935252`, `q = -0.0863309`). Positive movement in most rows therefore
does not make the cross-exon effect large enough.

No candidate passes. The largest BRAF median `B` anywhere in the grid is
`0.249161426` at `pre_attention/layer1/S` and its equivalent adjacent seams,
but the corresponding SLC25A48 median is only `0.055342868`. The largest
SLC25A48 median anywhere is the top candidate's `0.057657474`. Consequently,
the decision is not a tie-breaking accident or a single candidate narrowly
missing in both exons.

Running `analyze_phase_r_v3.py` again on a temporary raw-only copy reproduced
`PHASE_R_ANALYSIS.json` semantically exactly and reproduced
`PHASE_R_RESULT.md` byte-for-byte. The analyzer's reported decision is correct
for this dataset.

## Comparison with the v2 probability negative

Version 2 used the same 72 residual candidates but normalized movement of the
mean canonical splice-site **probability**. Its top candidate was
`pre_attention/layer2/S`, with median `B = 0.133399` in BRAF and `0.099521` in
SLC25A48 (`Q = 0.099489`). Under the Phase-R logit margin, that same candidate
has median `B = 0.231768` in BRAF but only `0.053825` in SLC25A48
(`Q = 0.053066`).

The Phase-R top candidate also changes unevenly: compared with its v2
probability values, BRAF rises from `0.088933` to `0.181562`, while SLC25A48
falls from `0.098777` to `0.057657`. Across all 72 candidate `Q` values, the
Spearman rank correlation between v2 and Phase R is about `0.4735`. Therefore
the readout materially changes magnitudes and ranking, but it does not solve
the cross-exon bottleneck. The result rules out probability saturation as the
sole explanation for v2's narrow-grid failure.

It does not overturn v2: v2 remains the negative result for probability-space
recovery, while Phase R is a second negative for pre-softmax class-evidence
recovery. Neither scalar is experimental cassette-exon PSI.

## Exact claim boundary

Phase R rejects this prospectively frozen claim:

> At one residual seam in transformer layers 0--5, transferring the complete
> 128-bp residual vectors at `V`, `A`, `D`, or their union `S` recovers at least
> one quarter of the canonical splice-classification logit-margin allele effect
> in both development exons, above matched local controls.

It does **not** test or rule out:

- transformer layers 6--8 or wider than the one-to-two 128-bp tokens in `S`;
- encoder skips, decoder states, output-embedding routes, or interactions
  between the transformer and U-Net branches;
- joint interventions across layers/components;
- low-rank directions, nonlinear features or SAE features inside a state;
- 131-kb native context, minigene context, splice-site usage or cassette
  junction readouts; or
- a biochemical splice pathway. This is a computational activation-transfer
  result for one checkpoint.

The tied top aggregates at `pre_attention/L4`, `post_attention/L3` and
`post_mlp/L3` should be read as a state persisting across adjacent residual
seams, not localization to the layer-4 attention operation.

## Audit limitations and required follow-up discipline

The raw JSON stores target means and pass booleans, not the selected activation
vectors, individual endpoint logits/margins, or probability sensitivity. I can
reconstruct every scalar recovery and verify exact zero target self drift, but
the runner's `trace_repeat_exact` and `donor_vectors_exact` assertions are
code-pinned run-time checks rather than equalities independently recoverable
from these files. This does not change the primary negative, but the final
archive should store endpoint terms and compact hashes of natural/effective
selected vectors.

There is also a small cross-executable baseline difference worth controlling
before comparing Phase R with route-census magnitudes. For the same
`BRAF_e14_A117G` sequence, checkpoint, target and hashed core implementation,
the earlier route smoke reported `L_REF = 2.537109375`, whereas Phase R reports
`2.546875`; `L_ALT = 3.08984375` is identical. The resulting `DeltaL` differs
by `0.009765625` (about 1.8%). Both executions repeat exactly, so this is
consistent with compile/trace-shape numerical sensitivity, although the
artifacts alone do not establish its cause. It is far too small to explain the
SLC25A48 gap to `0.25`, but future route comparisons must use within-executable
identities, retain raw movement, and pass the preregistered closure tests.

One analyzer hardening gap remains. It validates a group's position-set name,
role, and token/slot cardinality, but does not compare the serialized tokens,
slots, and genomic intervals with the linked Gate-0 identity's full
`resolved_position_sets` value. This independent audit made that exact
comparison across all 2,592 groups and found zero mismatches, so the omission
does not affect Phase R. The analyzer should make the comparison itself before
it is reused.

The correct next action is the preregistered development-only wider route
ladder with target and `T+E` closure, matched spatial controls, and no
confirmation access. No Phase-R candidate should be locked or carried into
confirmation.

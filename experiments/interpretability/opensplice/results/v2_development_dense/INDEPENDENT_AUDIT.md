# Independent audit of the v2 development-dense run

Audit date: 2026-08-27

## Scope and result

This audit independently recomputed metrics from the 20 saved 16,384-bp
baseline JSONs and 2,592 saved trace-group JSONs in this directory. It did not
read confirmation results, inspect activation tensors or other model internals,
change model code, or run inference.

The development output gate is encouraging but the intervention result is not
valid yet. All 12 development effects pass the saved sign-and-magnitude output
gate. However, 2,215/2,592 trace groups fail the protocol's absolute self-drift
tolerance. This is far above the protocol's 5% tooling-failure limit. The raw Q
ranking is therefore diagnostic only and must not be used to lock a circuit or
open the confirmation set. Independently, none of the 72 residual candidates
meets the preregistered `median B >= 0.25` requirement in both development
exons.

## Input integrity

- All 20 baseline files and all 2,592 trace files report `status: complete`.
- The trace count is exactly `12 effects * 3 stages * 6 layers * 12 position
  sets = 2,592`. Each effect has 216 groups: 72 candidate groups (`V`, `A`,
  `D`, `S`) and 144 matched upstream/downstream positional-control groups.
- All 2,592 trace fingerprints are unique.
- Recomputed cross-patch recoveries exactly equal the stored
  `self_control_corrected_recovery` values; maximum absolute disagreement is
  0.0.
- Direct instrumented baselines remain within the saved public-equivalence
  tolerance: maximum discrepancy is 0.0029296875 against a tolerance of
  0.00390625; 0/2,592 groups fail this check.

## Recomputed development output gate

The frozen rule is `sign(DeltaM) == experimental sign` and
`abs(DeltaM) >= 0.01`. The primary score is the mean of the canonical acceptor
and donor probabilities. The recomputed result is 12/12 eligible: BRAF 6/6
positive effects and SLC25A48 6/6 negative effects.

| Variant | Experimental sign | Saved `DeltaM` | Eligible |
|---|---:|---:|---:|
| BRAF_e14_A117G | positive | 0.0390625 | yes |
| BRAF_e14_T71A | positive | 0.044921875 | yes |
| BRAF_e14_T71G | positive | 0.052734375 | yes |
| BRAF_e14_A117C | positive | 0.04296875 | yes |
| BRAF_e14_A89G | positive | 0.0234375 | yes |
| BRAF_e14_A77C | positive | 0.044921875 | yes |
| SLC25A48_e8_G70A | negative | -0.309553861618042 | yes |
| SLC25A48_e8_A69C | negative | -0.29101070761680603 | yes |
| SLC25A48_e8_A69T | negative | -0.29686805605888367 | yes |
| SLC25A48_e8_T68G | negative | -0.29046630859375 | yes |
| SLC25A48_e8_G70C | negative | -0.29878687858581543 | yes |
| SLC25A48_e8_G71T | negative | -0.269775390625 | yes |

This is a development-set eligibility result only. It does not evaluate the
protocol's 131,072-bp predictive-adequacy gate on all 30 effects or on the
three untouched confirmation exons.

## Self-drift audit

For each trace group, this audit recomputed:

```text
self_drift = max(abs(M_RR - M_R), abs(M_AA - M_A))
```

The preregistered absolute limit is `1e-4`; output-eligible effects must also
remain within `1% * abs(DeltaM)`.

| Unit | Groups | Fail absolute `1e-4` | Fail relative 1% |
|---|---:|---:|---:|
| All trace groups | 2,592 | 2,215 | 1,257 |
| Candidate groups | 864 | 709 | 412 |
| Local-control groups | 1,728 | 1,506 | 845 |
| BRAF groups | 1,296 | 1,020 | 1,020 |
| SLC25A48 groups | 1,296 | 1,195 | 237 |

Only 377/2,592 groups pass the absolute limit. Across the 5,184 individual
self-patches, REF-to-REF fails the absolute limit in 2,000 groups and
ALT-to-ALT in 996; the corresponding relative-limit counts are 1,223 and 194.
The median group-level self drift is 0.001953125 and the maximum is
0.0048828125.

The nominal top-ranked member, `post_mlp / layer 2 / S`, has zero self drift
for all six BRAF effects but approximately 0.001953125 drift for all six
SLC25A48 effects. Thus 6/12 of even this member's variant traces fail the
absolute tooling gate. Self-control correction is already included in the
recovery formula, but it does not waive Gate 0.

## Protocol Q ranking

For every residual member `c`, this audit recomputed bidirectional recovery
`B`, the two same-cardinality local-control recoveries, per-variant
`q = B_candidate - max(B_upstream, B_downstream)`, the median q within each
development exon, and `Q = min(median_q_BRAF, median_q_SLC25A48)`. Exact-tie
ordering follows the protocol. The leading ten of 72 members are below;
displayed values are rounded to 12 decimal places.

| Rank | Stage | Layer | Positions | Q | Median q BRAF | Median q SLC25A48 | Median B BRAF | Median B SLC25A48 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | post_mlp | 2 | S | 0.102421932268 | 0.106884057971 | 0.102421932268 | 0.106884057971 | 0.099168939289 |
| 2 | pre_attention | 3 | S | 0.100878101584 | 0.148550724638 | 0.100878101584 | 0.127717391304 | 0.099216983806 |
| 3 | pre_attention | 2 | S | 0.099631699436 | 0.148550724638 | 0.099631699436 | 0.127717391304 | 0.098039257934 |
| 4 | post_attention | 0 | S | 0.098340443939 | 0.186781609195 | 0.098340443939 | 0.127717391304 | 0.094986947740 |
| 5 | post_attention | 2 | S | 0.097977335725 | 0.145833333333 | 0.097977335725 | 0.106884057971 | 0.095963906306 |
| 6 | post_mlp | 1 | S | 0.097953337632 | 0.127717391304 | 0.097953337632 | 0.127717391304 | 0.095894291995 |
| 7 | pre_attention | 0 | S | 0.095894334300 | 0.148706896552 | 0.095894334300 | 0.131465517241 | 0.098415251613 |
| 8 | pre_attention | 5 | S | 0.095202398801 | 0.095202398801 | 0.099149904952 | 0.085144927536 | 0.099150099342 |
| 9 | post_attention | 3 | S | 0.094769938627 | 0.104166666667 | 0.094769938627 | 0.085144927536 | 0.097954432850 |
| 10 | post_attention | 4 | S | 0.094303616781 | 0.095202398801 | 0.094303616781 | 0.076149425287 | 0.094770756961 |

Sixty-nine of 72 members have positive median q in both development exons, but
0/72 have median `B >= 0.25` in both. The nominal rank-one member has median B
of 0.1068840579710145 in BRAF and 0.09916893928866433 in SLC25A48, both below
0.25. Consequently the protocol's residual-discovery result is negative on
the current saved outputs even before applying the self-drift invalidation.
No residual hypothesis, head-decomposition search or confirmation run is
licensed by these results.

## Neutral-control specificity warning

The eight development neutral controls have baseline predictions but no trace
groups, so the causal `N_neutral|effect` statistic and Gate 3 cannot be
computed from this directory. A narrower predictive-output warning is still
visible:

- 6/8 assay-neutral controls have `abs(DeltaM) >= 0.01`.
- 4/8 have a larger absolute output change than their frozen matched effect.
- All four such cases are BRAF controls. Their neutral/effect absolute-change
  ratios are 18.624609375, 20.34087073284647, 7.62962962962963 and 9.375.
- In BRAF, the median absolute change is 0.5651779174804688 for the four
  neutrals versus 0.0439453125 for the six effects. All four BRAF neutrals move
  strongly negative even though the six selected BRAF effects move positive.
- In SLC25A48, the median absolute change is 0.01025390625 for neutrals versus
  0.29393938183784485 for effects; only 2/4 neutrals cross 0.01 and all four
  are below 9.5% of their matched effect's magnitude.
- Pooled across the two exons, median absolute change is 0.2158203125 for
  neutrals and 0.1612548828125 for effects. That pooled contrast should not be
  treated as an independent-sample statistic because variants share exon
  context.

These controls were selected for nonsignificance, not demonstrated biological
equivalence, so the warning does not prove that the BRAF predictions are
false. It does show that correct direction on the selected effects alone is
not evidence of output specificity. Neutral causal traces and the frozen
confirmation protocol would be required for a Gate 3 claim, after the tooling
drift is fixed and the development search is rerun under the same frozen rules.

## Allowed conclusion

The saved development baselines reproduce a strong 12/12 output-direction
result for the selected effects. The saved interventions do **not** support an
AlphaGenome circuit claim: Gate 0 fails extensively, the residual recovery
threshold is unmet, and the available neutral baselines raise an exon-specific
BRAF specificity concern. The next justified action is to diagnose and remove
self-patch drift, then rerun this already-frozen development grid. Confirmation
data should remain unopened unless a valid development candidate clears the
preregistered rule.

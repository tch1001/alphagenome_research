# Superseded design note: seven-skip factorial

Status: **superseded before implementation or model execution**. This early
design note does not authorize a model run, define a gate, or supplement the
final protocol. It is retained only to document alternatives considered.

The sole authoritative v3.3 contract is
`encoder_skip_localization_protocol_v3_3.md` (original committed SHA-256
`89a3c5ebf7a6af85de58f37952047694fd14c61ef11e72668ce4392f6077a342`).
Where this note differs, the authoritative protocol controls. In particular,
v3.3 uses 20 variants x 256 main coalitions plus identities and four
cross-exon OOD anchor records per variant (5,220 records); it does not run the
11,264-call control design below. Spatial/random, wrong-strand and non-target
output controls are a separately frozen follow-up after a resolution
nomination. The OOD stress test uses only the cross-exon mapping and four
anchors frozen in the protocol, not the within-exon `+1`/`+3` donor proposal
below. The protocol's minimal-cardinality, `B>=0.25`, and 80%-of-all-E-retention
nomination rule replaces this note's familywise-maximum/5-of-6 rule.

This supersession was recorded after the conflict was identified in review
and before v3.3 code freeze, compilation, GPU execution, or confirmation
access. The rest of the file is historical and must not be implemented.

## 1. Question and claim boundary

The completed v3.2 development result showed strong whole-route recovery from
the seven encoder skips (`E`) and weak-to-moderate recovery from the complete
transformer output (`T`). The next question is:

> Which of the seven skip resolutions causally carry that recoverable target
> movement, and is their contribution isolated, redundant, or interactive?

The seven skip players, in exact model order, are:

```text
E64, E32, E16, E8, E4, E2, E1
```

These are the seven encoder states consumed as decoder skips at 64/32/16/8/4/
2/1-bp resolution. A resolution-level result is still a broad computational
route. It does not identify a position, channel, motif, RBP, molecular pathway,
or endogenous biochemical mechanism.

Only the existing development genes BRAF and SLC25A48 are in scope. No
confirmation model output, activation, intervention, attribution, ranking, or
figure may be opened.

## 2. Factorial design and recommendation for T

### Recommended design: eight players, analyzed with a seven-player primary slice

Use eight binary players:

```text
N = {T, E64, E32, E16, E8, E4, E2, E1}
```

Run all `2^8 = 256` coalitions for every eligible development effect and every
required control. One six-row call contains both reciprocal directions and the
two same-allele self controls. The 12 effect variants therefore require 3,072
factorial calls before controls.

With 12 effects, eight neutrals, and two unrelated donors for each effect, the
full frozen design contains `44 * 256 = 11,264` factorial calls; output controls
are read from those same calls. The 128-coalition fallback would contain 5,632
such calls. Both designs should use one fixed-shape compiled superset rather
than one executable per coalition.

This is preferable to holding T fixed because:

1. the 256-coalition table already contains the complete 128-coalition
   `T=natural` seven-skip experiment;
2. it also contains the `T=donor` slice, so T-by-skip complementarity or
   redundancy cannot be missed;
3. its full coalition is the already validated joint `T+E` closure and should
   exactly reproduce the donor target; and
4. the cost is only twice the seven-player table and is comparable to the
   completed 2,592-group Phase-R census.

The **primary resolution-lock analysis** must use only the `T=natural` slice.
This asks which skips explain the successful whole-E result without allowing T
to subsidize a weak skip coalition. The eight-player Shapley account is a
predeclared secondary robustness analysis that allocates the fully closed
donor-recipient movement and quantifies T-by-skip interactions.

If compute constraints force 128 coalitions, hold T at the recipient's natural
state—not donor T. That design remains valid for localizing E, but its Shapley
sum closes only to the whole-E movement rather than the complete donor target,
and it cannot measure conditional T-by-skip interaction. The current small
two-route interaction makes similar results plausible but is not permission to
assume them.

## 3. Six-row target algebra

For every variant and coalition `S`, persist raw relevant-class and
padding-class logits at both canonical endpoints for the exact six rows:

```text
row 0: REF natural baseline/donor
row 1: ALT natural baseline/donor
row 2: ALT recipient <- REF donor for players in S
row 3: ALT recipient <- ALT self donor for players in S
row 4: REF recipient <- ALT donor for players in S
row 5: REF recipient <- REF self donor for players in S
```

The offline analyzer independently reconstructs float32 endpoint margins and
their two-endpoint mean. Let these means be `L_R`, `L_A`, `L_RA(S)`,
`L_AA(S)`, `L_AR(S)`, and `L_RR(S)`.

Self-corrected raw coalition values are:

```text
v_RA(S) = L_RA(S) - L_AA(S)
v_AR(S) = L_AR(S) - L_RR(S)
```

The corresponding normalized reciprocal recoveries are:

```text
u_RA(S) = v_RA(S) / (L_R - L_A)
u_AR(S) = v_AR(S) / (L_A - L_R)
B(S)    = min(u_RA(S), u_AR(S))
```

`L_AA(S) == L_A` and `L_RR(S) == L_R` are mandatory bit-exact checks for every
coalition. The subtraction remains explicit so a self-control failure cannot be
silently hidden. A zero donor-recipient denominator invalidates the variant for
normalized factorial analysis; it is never replaced by an epsilon.

The empty coalition must equal the natural identity in every row. In the
eight-player design, the full coalition must satisfy endpoint-level joint
closure: row 2 equals row 0 and row 4 equals row 1 in raw selected logits,
margins, totals, and means. The separately frozen final-embedding A/D closure
remains mandatory tooling evidence.

## 4. Exact Shapley estimands

For `n` players, direction `d`, player `i`, and every
`S subseteq N \\ {i}`, compute the raw self-corrected Shapley value:

```text
phi_i^d = sum_S [ |S|! (n-|S|-1)! / n! ]
                  * [v_d(S union {i}) - v_d(S)]
```

Compute this twice:

1. **Primary seven-skip Shapley:** `n=7`, using only coalitions with T natural.
2. **Secondary eight-player Shapley:** `n=8`, using all 256 coalitions.

Normalize only after preserving raw values:

```text
Phi_i^RA = phi_i^RA / (L_R - L_A)
Phi_i^AR = phi_i^AR / (L_A - L_R)
Phi_i^B  = min(Phi_i^RA, Phi_i^AR)
```

Use one frozen subset order, one frozen player order, and `math.fsum` over the
precomputed rational Shapley weights. Persist every marginal term. Require an
absolute raw efficiency residual no larger than `1e-12`:

```text
sum_i phi_i^d = v_d(N) - v_d(empty)
```

Report raw and normalized values for both directions. Negative contributions
and values above one are retained; these are not probabilities.

## 5. Sufficiency, necessity, interaction, and redundancy

For every player and direction, also report:

```text
isolated_i = v({i}) - v(empty)
leave_one_out_i = v(N) - v(N \\ {i})
```

`isolated_i` is a single-route sufficiency diagnostic. `leave_one_out_i` is a
conditional necessity diagnostic in the full route. Neither replaces Shapley.

For each pair `(i,j)`, compute the prospectively defined Shapley interaction
index:

```text
I_ij = sum over S subseteq N \\ {i,j}
       [ |S|! (n-|S|-2)! / (n-1)! ]
       * [v(S union {i,j}) - v(S union {i})
          - v(S union {j}) + v(S)]
```

Preserve both directional raw and normalized `I_ij`. There are 21 skip-skip
pairs in the primary seven-player account and 28 total pairs in the secondary
eight-player account.

Also compute the full Harsanyi dividend table:

```text
delta(S) = sum over A subseteq S (-1)^(|S|-|A|) v(A)
```

Report, by direction and exon:

- median per-player Shapley, isolated, and leave-one-out values;
- median pairwise interaction and its sign consistency;
- interaction mass by order `k = 2..n`, defined as the sum of absolute
  `delta(S)` for `|S|=k`, normalized by the absolute full-coalition movement;
- the isolated redundancy balance
  `sum_i [v({i})-v(empty)] - [v(N)-v(empty)]`, where positive means overlapping
  isolated effects and negative means net complementarity; and
- the smallest coalition recovery surface described below.

Do not infer redundancy merely because Shapley is split across players. Require
agreement among isolated, leave-one-out, pairwise-interaction, and Harsanyi
summaries.

## 6. Controls

All controls use the same frozen subsets, target reducer, raw evidence, and
validity checks. Controls are part of the one prospective run, not added after
seeing the effect factorial.

### 6.1 Neutral variants

Run the full factorial for all eight development neutral variants. They are
experimental behavior controls, not assumed AlphaGenome-null. Persist their
identity effect and raw factorial movements. Normalize by their own model
donor-recipient target difference only when its absolute value is at least
`0.01`; otherwise report raw movement and mark normalized values undefined.

High neutral recovery is evidence that a skip route generically transports a
model-predicted allele difference rather than specifically distinguishing the
development effect class. It cannot be discarded as an inconvenient false
positive.

### 6.2 Unrelated-donor controls

Freeze two within-exon donor derangements before execution: manifest-order
cyclic shifts `+1` and `+3` among the six eligible effects. For each recipient,
run the same factorial using those unrelated live donor states. No donor may
come from the same variant. Normalize movement by the recipient's true
REF-versus-ALT denominator and retain signed and absolute values.

This requires an explicitly tested donor-bank batch or equivalent live-device
mechanism. It may not serialize hidden states to host storage. Wrong-donor
controls must have their own exact donor-index, self/no-op, and repeat audits.

### 6.3 Output-specificity controls

From every call, persist two frozen control readouts without changing the
intervention:

1. the opposite-strand splice-classification margin at the same canonical
   coordinates; and
2. a same-strand, width-matched nearby noncanonical acceptor/donor pair chosen
   from committed development coordinates.

Scale their raw movement by the absolute primary target denominator, never by a
near-zero control-output denominator. The primary circuit must move its target
more strongly than either control output in both directions and both exons.
No post-hoc output choice is allowed.

### 6.4 Exact tooling controls

Every coalition requires:

- bit-exact target and compact-trace repeats;
- rows 0/1 identical to the frozen identity;
- rows 3/5 exact self targets and donor tensors for every enabled player;
- disabled players exactly equal to natural tensors;
- enabled recipient tensors exactly equal to the indexed live donor tensors;
- empty-coalition identity and full-coalition closure as applicable; and
- the same single compiled executable and frozen runtime selector shapes.

## 7. Completeness and invalid/missing handling

No coalition, variant, direction, endpoint, player, or control is dropped.

- One invalid or missing coalition invalidates every Shapley, interaction, and
  Harsanyi summary for that variant because all exact subsets are required.
- One incomplete eligible-effect factorial makes the affected exon and the
  circuit-lock family unselectable. There is no imputation or renormalized
  Shapley weighting.
- One incomplete required neutral, unrelated-donor, or output-control family
  also prevents a circuit lock, although complete effect results may still be
  reported descriptively.
- More than 5% invalid calls in any execution family is additionally reported
  as a family tooling failure, but the stricter zero-missing circuit rule takes
  precedence.
- Invalid raw records must still persist all available endpoint logits,
  selector state, donor indices, exact-check booleans, and failure metadata.

The offline analyzer must verify the raw manifest/hash tree before parsing
scientific values and must recompute all estimands from raw logits rather than
trust runner-emitted summaries.

## 8. Cross-exon summaries and gates

BRAF and SLC25A48 are evaluated separately. Variants are not pooled across
genes, and six variants in one exon are not described as six independent
exons.

For each exon, report all six effect-variant values and the median. A primary
coalition or player must have the expected positive direction in at least five
of six eligible effects per exon; exact ties count as failures. Both exons must
pass every gate. No strong BRAF result may rescue a weak SLC25A48 result.

Pairwise and higher-order interactions are descriptive unless they pass the
control maximum described below. No p-value or confidence interval is used to
claim population-level inference from two exons.

## 9. Multiplicity control

The run evaluates seven skip players, 21 primary pairwise interactions, and 127
nonempty skip coalitions. Do not select a result against only its own negative
control. For each exon and estimand family, compute one **familywise control
maximum** over:

- all seven players or all 127 skip coalitions, as appropriate;
- all eight neutral variants;
- both unrelated-donor derangements; and
- both output-specificity readouts.

For a skip coalition `C`, define:

```text
M_effect,g(C) = median effect-variant B_g(C)
M_control,g   = maximum control statistic over every frozen coalition
q_family,g(C) = M_effect,g(C) - M_control,g
```

Control `B` is used when a valid control donor-recipient denominator exists.
Otherwise use the maximum absolute control movement scaled by the primary
recipient denominator. The exact conversion and sign convention must be frozen
in code before execution.

This max-statistic gate protects the one development selection against the
search over all players/coalitions. It is deliberately conservative and is not
reported as a frequentist p-value. If formal error rates are later desired,
they require a separate sampling design with more independent exons; variants
nested within two exons do not justify asymptotic tests.

An interaction may be called control-surviving only if its absolute median in
both exons exceeds the maximum absolute interaction across all 21 same-family
control pairs and its sign agrees in at least five of six effects per exon.

## 10. Prospective resolution-lock rule

The lock candidates are the 127 nonempty subsets of the seven E resolutions in
the `T=natural` slice. T is never eligible for the encoder-skip resolution lock.

A skip coalition `C` passes only when all of the following hold:

1. every effect and required control factorial is complete and valid;
2. in each exon, median `B(C) >= 0.25` and at least five of six effect variants
   individually have `B(C) >= 0.25`;
3. `q_family,g(C) > 0` independently in both exons;
4. every member skip has positive median bidirectional seven-player Shapley
   contribution in both exons and is positive in at least five of six effects
   per exon;
5. removing every member from `C` produces a positive median loss of coalition
   recovery in both exons, with the sign positive in at least four of six
   effects; and
6. the eight-player sensitivity analysis does not reverse any member's median
   bidirectional Shapley sign in either exon.

Among passing coalitions, lock the smallest cardinality. If several have the
same size, choose the largest
`min(q_family,BRAF, q_family,SLC25A48)`; break an exact remaining tie by the
fixed model order `(E64,E32,E16,E8,E4,E2,E1)`. Every step is automatic and
prospective.

If no coalition passes, the seven-skip census is negative. If the smallest
passing coalition contains more than three resolutions, report a distributed
encoder-route result but do not call it a localized resolution circuit. A lock
of one to three resolutions is only permission for a new, separately frozen
development spatial/channel census with matched and random-position controls.
It is not permission to open confirmation and is not a biological-mechanism
claim.

## 11. Required analyzer outputs

The CPU-only analyzer should emit:

- exact identity/eligibility and all control denominators;
- complete per-variant coalition tables for both directions;
- seven-player and eight-player raw/normalized Shapley tables;
- isolated, leave-one-out, pairwise interaction, Harsanyi, and redundancy
  summaries;
- per-exon medians, sign counts, full-coalition closures, and efficiency audits;
- all neutral, unrelated-donor, and output-control summaries;
- familywise maxima and the result of every lock clause for all 127 coalitions;
- the unique locked coalition or an explicit negative/descriptive-only result;
- exact artifact, code, protocol, reference, checkpoint, compiler, import, and
  executable hashes; and
- the existing disclosure that confirmation metadata/labels were exposed
  post-freeze while confirmation model outputs/activations/interventions remain
  unopened.

The report must show raw values before medians and must never silently clip
recovery, Shapley, or interaction values.

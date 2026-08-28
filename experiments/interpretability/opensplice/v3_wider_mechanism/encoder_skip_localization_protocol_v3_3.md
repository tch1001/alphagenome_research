# Prospective OpenSplice v3.3 encoder-skip localization protocol

Status: **prospective, development-only, docs-only draft**. This document was
written after the completed v3.2 development result and before any v3.3 model
call. It must be committed and included in a separate immutable freeze before
execution. No confirmation model output, activation or intervention may be
opened under this protocol.

## 1. Motivation and disclosed prior evidence

The v3.2 result is already known and motivates this follow-up:

- none of 72 transformer-residual candidates passed the two-exon gate;
- the all-seven encoder-skip intervention had median bidirectional recovery
  `B = 0.711950779808821` in BRAF and `0.93973634975444` in SLC25A48;
- the whole transformer-output intervention had median `B =
  0.23509760903151095` and `0.05715517280910521`, respectively;
- the joint transformer-plus-all-skips intervention closed exactly; and
- all eight experimentally nonsignificant controls had nonzero AlphaGenome
  target changes and therefore are not AlphaGenome-null controls.

This protocol asks which encoder-skip resolutions carry that broad E-route
signal. It does not treat the v3.2 whole-E result as a localized mechanism.

Frozen v3.2 provenance to bind before v3.3:

| Artifact | SHA-256 |
|---|---|
| v3.2 protocol | `1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea` |
| v3.2 raw tree | `4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8` |
| v3.2 `ANALYSIS.json` | `a46c827e16fb4e054e7cf702f7147da70e0a3b35f677b430edf64e9a3055013c` |
| v3.2 `RESULT.md` | `e4c8d45c1b35d8934734c4d8c18bd2ca10b78dd0099a9b37c296e60b15b7e52c` |

The protocol was designed with those development results visible. This is not
an independent replication of v3.2 and is not held-out evidence.

## 2. Frozen biological scope

### 2.1 Cohort and order

Use only orders 0--19 from the committed v2 SNV-only selector. Do not load a
row, path or model artifact for ELN, EIF4A2, DMD or any confirmation partition.

| Order | Role | Variant |
|---:|---|---|
| 0 | BRAF effect | `BRAF_e14_A117G` |
| 1 | BRAF effect | `BRAF_e14_T71A` |
| 2 | BRAF effect | `BRAF_e14_T71G` |
| 3 | BRAF effect | `BRAF_e14_A117C` |
| 4 | BRAF effect | `BRAF_e14_A89G` |
| 5 | BRAF effect | `BRAF_e14_A77C` |
| 6 | BRAF neutral behavior control | `BRAF_e14_T121C` |
| 7 | BRAF neutral behavior control | `BRAF_e14_A69G` |
| 8 | BRAF neutral behavior control | `BRAF_e14_C68T` |
| 9 | BRAF neutral behavior control | `BRAF_e14_G118A` |
| 10 | SLC25A48 effect | `SLC25A48_e8_G70A` |
| 11 | SLC25A48 effect | `SLC25A48_e8_A69C` |
| 12 | SLC25A48 effect | `SLC25A48_e8_A69T` |
| 13 | SLC25A48 effect | `SLC25A48_e8_T68G` |
| 14 | SLC25A48 effect | `SLC25A48_e8_G70C` |
| 15 | SLC25A48 effect | `SLC25A48_e8_G71T` |
| 16 | SLC25A48 neutral behavior control | `SLC25A48_e8_C67T` |
| 17 | SLC25A48 neutral behavior control | `SLC25A48_e8_C67G` |
| 18 | SLC25A48 neutral behavior control | `SLC25A48_e8_T68C` |
| 19 | SLC25A48 neutral behavior control | `SLC25A48_e8_C6A` |

The effect denominator is always all six effects per development exon. An
ineligible or invalid effect is never replaced. The eight neutrals remain in
their exact order and are never relabelled from AlphaGenome behavior.

### 2.2 Model, context and primary target

Keep the v3.2 checkpoint, GRCh38 reference binding, 16,384-bp unshifted
development contexts, dense attention, mixed-precision policy, organism index,
canonical endpoint mapping and splice-classification head. Bind their exact
content and environment manifests in the v3.3 freeze.

The sole selecting target remains the strand-aware mean canonical
acceptor/donor pre-softmax logit margin:

```text
L_site = logit(relevant strand-and-role class) - logit(padding class)
L_exon = mean(L_acceptor, L_donor)
DeltaL = L_exon_ALT - L_exon_REF
```

Raw relevant and padding logits for both endpoints must be persisted. No
probability, PSI, splice-site-usage, junction, RNA, single-endpoint or public
API output may replace or rescue this scalar.

In the new executable, an effect is target-eligible only if

```text
sign(DeltaL) == sign(experimental delta_logit)
and abs(DeltaL) >= 0.01.
```

All 12 effects must remain eligible. A failure stops active work as a target
comparability failure. Do not impose that v3.3 floating-point identity values
equal a different v3.2 executable; all v3.3 baselines, denominators and active
calls must instead be exact within the one v3.3 executable.

## 3. Exact intervention graph

### 3.1 Six-row transfer

Every identity and intervention uses the existing fixed six-row layout:

```text
row 0  REF baseline/donor
row 1  ALT baseline/donor
row 2  ALT <- REF
row 3  ALT <- ALT self
row 4  REF <- ALT
row 5  REF <- REF self

donor rows            [0, 1, 0, 1, 1, 0]
natural identity rows [0, 1, 1, 1, 0, 0]
```

Baseline rows are never patched. Rows 2--5 use the same dynamic route mask;
rows 3 and 5 are mandatory same-allele tensor and target controls.

### 3.2 Seven skip players and transformer context

The seven skip players are the complete encoder intermediates transferred at
decoder lookup immediately before the corresponding `UpResBlock`, in exact
code order:

```text
player index  0    1    2    3   4   5   6
name          E64  E32  E16  E8  E4  E2  E1
bin size bp   64   32   16   8   4   2   1
```

Each enabled player transfers the entire live donor tensor at that resolution.
It does not replace deeper encoder ancestry or the transformer trunk. The
runtime `encoder_skips.transfer_mask` remains shape `[7, 6]`; arbitrary mask
values must reuse one compiled executable and one pytree.

The complete transformer output `T` is retained as an eighth player with mask
shape `[1, 6]`. This is necessary because all-E alone did not close the donor
target in v3.2 whereas `T+all-E` did. A seven-player decomposition with natural
recipient T is still reported, but it is explicitly conditional on T. The
eight-player decomposition allocates the complete joint closure.

All Phase-R residual interventions and the final-embedding intervention stay
false throughout this experiment.

## 4. Exhaustive coalition census

### 4.1 Coalition identifiers and call order

Let `e_mask` be an integer from 0 through 127. Bit `i` selects player `i` in
the exact order `(E64,E32,E16,E8,E4,E2,E1)`; bit 0 is E64. Leading zeros are
retained in the serialized seven-bit representation.

Run two exhaustive blocks for every one of the 20 variants:

```text
coalition_id = 128 * t + e_mask
t = 0: IDs   0..127, natural recipient T
t = 1: IDs 128..255, donor T transferred
```

Thus ID 0 is empty, ID 127 is all E, ID 128 is T alone, and ID 255 is
`T+all-E`. Within each variant, run IDs 0 through 255 in increasing order.
Variants run in frozen order 0 through 19. This is 5,120 six-row coalition
calls, before exact repeats and the separately triggered controls below.

The requested 128-subset encoder census is IDs 0--127. IDs 128--255 are a
predeclared mirrored alternative, not an adaptive retry. They permit skip
effects to be compared with and without transferred T and support a valid
eight-player Shapley account.

Singletons, leave-one-skip-out coalitions, fine-to-coarse cumulative sets and
coarse-to-fine cumulative sets are derived views of these 128 masks. They are
not separate searches and may not replace the exhaustive census to save time.
No greedy or result-dependent subset traversal is allowed.

### 4.2 Required closure anchors

For every effect and neutral, IDs 0, 127, 128 and 255 receive an immediate
exact repeat. ID 0 must equal natural baselines. ID 255 must reproduce the live
donor endpoint logits bit-for-bit in both directions. ID 127 and ID 128 are
upper-bound components and are not required individually to close.

### 4.3 Bounded execution size

The resolution attempt has an exact expected scientific raw-record count:

```text
20 repeated identity records
+ 20 variants * 256 coalition records
+ 20 variants * 4 unrelated-donor anchor records
= 5,220 records
```

Each record contains first/repeat endpoint outputs and compact exactness
evidence, not full skip tensors. At the approximately `0.115` seconds per
already-compiled apply observed in v3.2, the 10,440 first/repeat applies imply
about 20 minutes of model-call time, excluding compilation, validation and
persistence. This estimate is capacity planning only and cannot justify
dropping masks after attempt start. Freeze a conservative wall-time/storage
budget and fail append-only if it is exceeded.

## 5. Frozen estimands

### 5.1 Recovery

For each coalition, use the six target means from the same active call:

```text
L_R  = row 0    L_A  = row 1
L_RA = row 2    L_AA = row 3
L_AR = row 4    L_RR = row 5

r_REF_to_ALT = (L_RA - L_AA) / (L_R - L_A)
r_ALT_to_REF = (L_AR - L_RR) / (L_A - L_R)
B             = min(r_REF_to_ALT, r_ALT_to_REF)
```

Persist raw movements, both unclipped recoveries and B. Never mix a v3.2
denominator into v3.3. Neutrals persist raw movements as primary because an
experimental neutral label is not an AlphaGenome-null assumption; normalized
neutral recovery may be reported as a labelled sensitivity only when the
within-call denominator is nonzero and is never a selection rescue.

### 5.2 Exact Shapley values

Let `m_d(S)` be the raw target for direction `d` under coalition `S`, and let
`N` be the player set. For every player `i`, compute

```text
phi_i = sum over S subset of N\{i} of
        [ |S|! (|N|-|S|-1)! / |N|! ]
        * [m_d(S union {i}) - m_d(S)].
```

Report:

1. seven-player `phi_E64...phi_E1` over IDs 0--127, conditional on natural T;
2. the same seven-player values over IDs 128--255, conditional on transferred
   T; and
3. the eight-player values for `(T,E64,E32,E16,E8,E4,E2,E1)` over all 256
   coalitions.

Retain raw values first. Normalize only by the corresponding within-call
donor-minus-recipient target denominator. For every variant and direction,
require Shapley efficiency to numerical tolerance fixed before execution:

```text
sum(phi_i) == m_d(N) - m_d(empty).
```

The tolerance must be justified from synthetic float32 tests and frozen before
the first model call; it may not be relaxed after seeing a residual.

### 5.3 Interactions

For every pair `i,j`, report the exact pairwise Shapley interaction index

```text
I_ij = sum over S subset of N\{i,j} of
       [ |S|! (|N|-|S|-2)! / (|N|-1)! ]
       * [m(S union {i,j}) - m(S union {i})
          - m(S union {j}) + m(S)].
```

Also persist every Harsanyi dividend

```text
delta(S) = sum over T subset of S of (-1)^(|S|-|T|) m(T)
```

and summarize absolute dividend mass by interaction order 1 through 8. Do not
force pairwise terms to account for higher-order interactions. Shapley and
interaction values are computational routing estimands, not independent
biological effects.

Per exon, report medians across the six effects for raw and normalized Shapley
values, singleton B, leave-one-out decrement, cumulative paths, pairwise
interactions and full coalition B. Preserve all per-variant values.

## 6. Controls

### 6.1 Mandatory internal controls

For every coalition call require:

- row-0/1 raw endpoint logits equal that variant's same-executable identity;
- row-3/5 self tensors and targets equal natural same-allele state exactly;
- every disabled skip equals its natural tensor exactly;
- every enabled skip equals its requested live donor tensor exactly;
- transformer T is exact donor state only when `t=1` and exact natural state
  when `t=0`;
- the all-false call, target repeat and compact trace repeat are exact; and
- the call uses the single recorded executable and fixed pytree.

Compact fingerprints are repeat diagnostics, not substitutes for in-graph
elementwise equality booleans.

### 6.2 Neutral-variant interventions

Run the complete 256-coalition census on all eight frozen neutrals. Do not
select a favorable neutral subset or omit BRAF neutrals because their
AlphaGenome changes are large.

For each coalition and exon report:

- both raw directional movements and their mean absolute magnitude;
- signed agreement with the experimentally measured `delta_logit`;
- optional within-call normalized recovery, clearly labelled as model behavior;
- effect-versus-neutral median raw-movement difference; and
- an exon-stratified rank correlation between coalition movement and
  experimental `delta_logit`, with only ten rows per exon and no inferential
  overclaim.

A computational route can be nominated without treating neutrals as null. A
route is **not biologically aligned** unless, in both exons, the median absolute
candidate-induced movement of the six effects exceeds that of the four
neutrals. This is a predeclared descriptive gate, not a population-level test.

### 6.3 Unrelated/shuffled-donor stress control

There is no clean matched unrelated donor for a whole encoder tensor: another
variant or exon imports many sequence and activation differences. Therefore
this control is mandatory but explicitly an out-of-distribution stress test,
not a null distribution and not a substitute for spatial controls.

Freeze the following cross-exon derangement before any control call:

```text
BRAF effect orders  0..5  <-> SLC25A48 effect orders 10..15 by class rank
BRAF neutral orders 6..9 <-> SLC25A48 neutral orders 16..19 by class rank
```

This stress test requires a separately frozen fixed eight-row control graph:
rows 0--5 retain the recipient six-row layout, while rows 6/7 are the mapped
case's REF/ALT donor sequences. The graph must run the intended same-variant
donor and cross-exon donor calls side by side in the same executable; the
cross-exon call maps row 2 to donor row 6 and row 4 to donor row 7, while rows
3/5 remain recipient self controls. Comparing a v3.3 six-row result with a
different executable is forbidden.

The resolution run may execute this control for the predeclared anchors
`(empty, all-E, T, T+all-E)` before coalition results are analyzed. The single
nominated coalition is tested later inside the separately frozen spatial
control attempt, where intended and unrelated calls again share one executable.
Record raw movement only, donor tensor equality, donor/recipient IDs and all
endpoint logits. Do not calculate a donor-normalized B or use this unmatched
stress test to rescue or reject a candidate. Unexpectedly large movement is a
required warning and strengthens the need for the spatial controls below.

No host-serialized activation, random donor chosen after results, same-exon
reference sequence masquerading as an unrelated donor, channel permutation or
shape-incompatible cross-resolution swap is allowed.

### 6.4 Separately triggered spatial controls

The whole-tensor census can nominate a resolution set but cannot establish a
localized mechanism. If and only if a resolution coalition is nominated,
write and commit a second prospective spatial manifest before any spatial
model call. It must mechanically generate, for every nominated resolution and
all 20 variants, supports for:

- `V`: bins whose encoder receptive fields contain the variant base;
- `A`: bins whose receptive fields contain the canonical acceptor;
- `D`: bins whose receptive fields contain the canonical donor; and
- `S`: the union of A and D supports.

Add exactly one bin of guard on each side of every contiguous component. The
receptive-field calculation, padding/cropping convention and base-to-bin map
must be unit-tested against the actual encoder graph and serialized before
model output is read.

For each candidate support freeze:

1. one equal-shape upstream translation beginning 512 bp away and moved
   outward by the candidate span until disjoint from V/A/D/S;
2. one analogous downstream translation;
3. 32 equal-shape random translations, preserving every relative offset,
   wholly in the valid 16,384-bp context and disjoint from V/A/D/S and the two
   deterministic controls.

Random candidates use only this seed material:

```text
sha256("opensplice-v3.3|" + variant_id + "|" + resolution_bp
       + "|" + support_name + "|random|" + random_index)
```

Map the digest to valid starts by unsigned integer modulo the number of valid
starts. Resolve collisions by appending `|retry|k` with increasing `k`. Freeze
the resulting coordinates and hashes before inference. Random controls are
indexed 0--31 and are never regenerated because a result is inconvenient.

The spatial experiment uses the same donor map, same selected resolution
coalition, same T context and same target. It must not search channels or alter
the coalition.

### 6.5 Wrong-strand and non-target-output controls

The spatial manifest must freeze two additional readouts from the same calls:

- **wrong strand:** the mean canonical A/D classification logit margin at the
  same bases using the opposite-strand acceptor and donor classes and the same
  padding class;
- **non-target exon output:** the same correct-strand logit-margin reducer at
  the nearest other fully in-context canonical exon, excluding the target
  exon and requiring at least 2,048 bp separation from the variant. Choose by
  genomic distance, then coordinate, then Ensembl exon ID. If none exists, the
  spatial manifest fails before inference; do not select another output after
  inspecting predictions.

Raw intended, wrong-strand and non-target endpoint logits must all be saved.
These controls cannot replace the primary target or nominate a candidate.

## 7. Gates and candidate decisions

### 7.1 Gate 0: tooling and target

Before the coalition census:

1. verify all frozen source, protocol, cohort, checkpoint, reference,
   environment, protobuf and compiler inputs at a globally tracked-clean HEAD;
2. require the exact RTX 3090/UUID preflight before append-only attempt start
   and again in the same process before model construction;
3. compile exactly one fixed-shape six-row scientific-coalition executable and
   one separately frozen fixed-shape eight-row unrelated-donor stress
   executable; persist StableHLO/HLO, executable and import-provenance
   fingerprints for both, and permit no other compilation;
4. run all 20 identities twice and require exact target and trace repeats;
5. require all 12 effects target-eligible and retain all eight neutrals;
6. require ID 0 no-op and ID 255 endpoint-level closure for every one of the
   20 variants and both directions; and
7. require exact donor, self and disabled-route tensor assertions.

Any failure stops the run as tooling/target failure. No coalition ranking or
Shapley summary is then valid.

### 7.2 Completeness

Exact Shapley accounting requires a complete cube. Any missing, duplicated,
invalid or non-finite coalition for any effect invalidates the entire effect
family; there is no complete-case filtering, sentinel, imputation or partial
Shapley. Any missing neutral invalidates the biological-alignment control
family. More than one six-row coalition compilation, more than one eight-row
anchor-control compilation, or any candidate-specific executable is a tooling
failure.

### 7.3 Resolution nomination

For every nonempty `e_mask` with natural T (`t=0`), compute per-exon median B
over all six effects and its retention relative to all-E in the same graph:

```text
retention_gene(e_mask) = median_B_gene(e_mask) / median_B_gene(all_E).
```

If all-E median B is nonpositive or non-finite in either exon, retention is
undefined and the skip-only family stops as weak/invalid; do not change the
denominator. ID 255 must be exact closure before the T-dependent retention
family is evaluated.

A skip-only coalition is resolution-available only if, in both exons:

- median `B >= 0.25`; and
- retention is at least `0.80`.

Choose the single nomination by smallest number of enabled skips, then highest
`min(BRAF median B, SLC25A48 median B)`, then increasing `e_mask`. Do not choose
the largest observed recovery or a Shapley-favored exception.

If no skip-only coalition passes, apply the same rule to `t=1`, with retention
relative to ID 255. Such a nomination is labelled **T-dependent joint route**,
not encoder-only. Skip-only candidates always have precedence over T-dependent
candidates. If neither family passes, stop with a negative resolution result;
do not run the spatial phase or open confirmation.

The effect-versus-neutral gate in Section 6.2 is then applied. Failure does not
erase a computational route result, but it prevents a biologically aligned
mechanism nomination.

### 7.4 Spatial candidate gate

For each frozen V/A/D/S spatial candidate, define per effect

```text
q_spatial = B_candidate - max(B_upstream, B_downstream).
```

The candidate passes only if all records are valid and, in both exons:

- median candidate `B >= 0.25`;
- median `q_spatial > 0`;
- candidate median B is strictly greater than the nearest-rank 95th percentile
  of the 32 random-control median-B statistics (sorted index 30, zero-based);
- median absolute intended raw movement is greater than both wrong-strand and
  non-target-exon movement; and
- median absolute effect movement is greater than median absolute neutral
  movement.

If more than one support passes, select the one with the fewest patched tensor
elements, then support order `V,A,D,S`. Do not rank by maximum recovery. The
unrelated/shuffled-donor result is reported as an OOD warning and is not part of
this formal gate.

Passing the resolution gate but no spatial gate means **present but diffuse at
the tested supports**. Passing the spatial gate nominates a development
computational mechanism candidate; it still does not establish an RBP,
spliceosome step or biochemical pathway and does not itself open confirmation.

## 8. Execution order and stop rules

The run is single-use and append-only:

1. committed preflight and freeze validation;
2. all 20 repeated identities and Gate 0 closures;
3. complete effect coalition cubes, in variant then coalition order;
4. complete neutral coalition cubes in frozen order;
5. include the four frozen unrelated/shuffled anchor controls in the same
   append-only raw cohort, using their separately frozen eight-row executable;
6. persist a raw manifest and stop the GPU process;
7. CPU-only analysis computes all frozen estimands and one resolution
   nomination; and
8. if eligible, commit a separately versioned spatial manifest whose one-shot
   control graph includes the nominated intended and unrelated donor calls,
   then perform one spatial attempt.

Do not inspect an intermediate coalition to change order, masks, precision,
tolerance, target, threshold or control. A device, compile, persistence,
numerical or scientific failure consumes the attempt. A new version may repair
an infrastructure defect only after a prospective amendment binds the partial
tree and states the exact permitted change; it may not relax a scientific gate
or cherry-pick a coalition.

Confirmation remains model-output, activation and intervention blind
throughout. The previously disclosed later-exon metadata/label exposure must
remain in every scope statement. No confirmation baseline, attribution, patch,
figure or example selection is allowed before a separately committed circuit
lock names exactly one spatial candidate and all thresholds.

## 9. Required outputs and claim limits

Persist raw endpoint logits, all route masks, donor maps, target means, raw
movements, recoveries, B, exactness booleans, fingerprints, timing, executable
identity and artifact hashes for every call. The CPU analyzer must independently
reconstruct margins, means, the full coalition cube, all Shapley/interaction
values, controls, gates and selection from raw logits.

| Outcome | Maximum claim |
|---|---|
| Gate 0 or cube completeness fails | The v3.3 instrumentation failed; there is no localization result. |
| No resolution coalition passes | None of the exhaustive whole-skip coalitions met the frozen two-exon development gate. |
| Skip-only coalition passes | A subset of encoder-skip resolutions carries a broad development computational route signal under whole-tensor transfer. |
| Only T-dependent coalition passes | The signal is jointly routed through T and the nominated skips; it is not encoder-only. |
| Resolution passes but spatial controls fail | The signal is present but diffuse or nonspecific at the tested supports. |
| Spatial candidate passes | One development computational-mechanism candidate is eligible for a separate lock; random, neutral and output controls passed under this protocol. |

No outcome here establishes a molecular pathway, RBP, spliceosome step,
endogenous necessity, experimental replication or generalization beyond BRAF
e14 and SLC25A48 e8. The 12 variants are nested in two exons and are not 12
independent loci. Whole-tensor transfers construct hybrid states and are route
upper bounds, not natural ablations.

## 10. Pre-run checklist

- [ ] Commit this protocol and record its SHA-256.
- [ ] Freeze exact source, tests, cohort projections, checkpoint/reference,
      environment, compiler options, output paths and analysis code.
- [ ] Unit-test all 256 masks, ID ordering and identical pytree/compile shape.
- [ ] Unit-test the six-row donor/natural map and all enabled/disabled tensor
      equality assertions.
- [ ] Unit-test seven- and eight-player Shapley, pairwise interaction and
      Harsanyi arithmetic on synthetic additive and interacting functions.
- [ ] Freeze Shapley efficiency tolerance before model execution.
- [ ] Verify the full 20-variant call count and append-only stop behavior.
- [ ] Keep confirmation model outputs, activations and interventions unopened.

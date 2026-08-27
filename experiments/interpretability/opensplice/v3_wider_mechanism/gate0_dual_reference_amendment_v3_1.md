# Gate-0 dual-reference amendment v3.1

**Design date:** 2026-08-28

**Status:** prospective development amendment; freeze this document and all
implementation hashes before any v3.1 GPU inference

**Scope:** OpenSplice development exons BRAF e14 and SLC25A48 e8 only

**Confirmation status:** blind; this amendment does not authorize any ELN,
EIF4A2 or DMD baseline, internal, attribution or intervention

## Decision

Stage A may proceed as a separately versioned v3.1 development phase using a
dual-reference Gate 0. This is methodologically acceptable because causal
effects are estimated entirely by exact paired contrasts inside the one
instrumented Stage-A graph. A separately executed, exact Phase-R graph checks
that the current checkpoint, input, target and runtime still reproduce the
locked Phase-R semantics within the already frozen `2^-8` tolerance.

This amendment does **not** make the two compiled graphs numerically equal. It
does not waive, enlarge or reinterpret a failed v3.0.2 threshold. It changes
the object to which that threshold applies and therefore creates a new
development phase. The failed v3.0.2 attempt remains failed and must stay in
the study record.

## 1. Events disclosed before the amendment

### 1.1 Separate closure-only smoke

Before the full Stage-A attempt, a bounded v3.0.2 smoke ran one development
variant, `BRAF_e14_A117G`, and exactly the first two dependency-ordered
components:

1. final post-GELU A/D embedding closure; and
2. joint complete transformer-output plus all-seven-skip (`T+E`) closure.

Both reciprocal closures reached the donor target exactly and their internal
self/donor tensor checks passed. This smoke did **not** run isolated `whole_T`
or `whole_E`, did not calculate a branch partition, did not compare variants or
exons, and did not rank or select a mechanism. It is a tooling observation,
not evidence that either route carries a biological or general computational
effect.

The bounded smoke is fixed by these artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `654bbe6d39ea24b78e75cf25e7908941a973e738fb26f937c11296b13708c2ae` |
| BRAF identity | `682207fe3335b43425a7d9651a5637269184b0476d98984b68f2c7be5a68a66c` |
| A/D closure | `1729c75eacd6b7368e3b85b19a76bf86e0233663078716cdba492a5d0bc15e6a` |
| T+E closure | `1905245d874a3f495835e4c47cfa96592a1946d47402395cb1c746b931ce1e2d` |

### 1.2 Full v3.0.2 attempt and frozen failure

The original v3.0.2 Gate 0 compared the Stage-A graph's six natural identity
targets directly with the locked Phase-R identity at an inclusive absolute
tolerance of

```text
2^-8 = 0.00390625.
```

The full attempt first wrote a passing identity for `BRAF_e14_A117G` and no
active component. On the second development variant, `BRAF_e14_T71A`, the
three ALT-allele rows differed from the lock by
`0.0048828125 = 5/1024`, which exceeds the frozen `4/1024` bound. The process
failed closed before writing the second identity and before any full-run
closure or isolated T/E artifact. The disclosed absolute-difference vector was

```text
[0, 0.0048828125, 0.0048828125, 0.0048828125, 0, 0].
```

This failure is not deleted, rounded to the tolerance, replaced with the
passing smoke, or reclassified using v3.1. In particular:

- v3.0.2 did not pass graph numerical equivalence;
- no v3.0.2 Stage-A route result exists;
- the frozen tolerance is not increased after observing the miss; and
- a later v3.1 result cannot be reported as if v3.0.2 had passed.

The failure does not invalidate the locked Phase-R negative. It also does not
show that Stage-A self transfers drifted; it occurred before active causal
components and concerns two distinct all-false compiled graphs.

## 2. Why the references are separated

Stage A exposes and audits complete T/E tensors and a final-embedding seam.
Phase R exposes selected transformer residual vectors. Even with all transfer
masks false, these programs return different pytrees and introduce different
gathers, comparisons and reductions. Under BF16, graph-dependent fusion and
reassociation can change final logits without a logical data-path mutation.

There are therefore two different validity questions:

1. **Phase-R semantic reproducibility:** does the exact Phase-R factory,
   selection and all-false intervention still reproduce its locked identity
   within the preregistered numerical tolerance?
2. **Stage-A causal validity:** inside the Stage-A graph, are natural
   duplicates, no-op paths, same-allele self controls, donor transfers and
   target closures exact, so patched-versus-self contrasts identify an effect
   of the requested Stage-A state replacement?

Passing the first does not validate an intervention. Passing the second does
not make Stage A bit-identical to Phase R. v3.1 requires both, records their
difference, and keeps their roles distinct.

## 3. Frozen inputs and three references

All v3.1 runs retain:

- selected-variant manifest SHA-256
  `09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6`;
- frozen-exon SHA-256
  `b95f8fc540f19222546322bebfb817a6c0f2147dd41325091086833584a09a75`;
- AlphaGenome checkpoint snapshot
  `a8f293a76ee73d5b57f3bf2ae146510589fcf187`;
- 16,384-bp native-genome SNV context and dense attention;
- the strand-aware mean canonical acceptor/donor class-minus-background
  pre-softmax logit margin; and
- the same six-row DNA and live-donor mapping used in Phase R.

The three target records are named as follows:

| Name | Meaning | Permitted use |
|---|---|---|
| `R_lock` | Immutable saved Phase-R identity tree | Historical semantic lock only |
| `R_current` | Fresh exact Phase-R factory/selection/all-false graph | Must reproduce `R_lock`; semantic reference only |
| `A` | Fresh Stage-A all-false/active graph | Sole source of Stage-A baselines, self controls, causal numerators and denominators |

Neither `R_lock` nor `R_current` supplies a patched counterfactual. Neither may
be used as the primary denominator for a Stage-A recovery, to offset a Stage-A
patched value, or to select between T and E.

## 4. Prospective v3.1 Gate 0

The runner must finish and write all 20 development identities before running
any v3.1 closure or isolated branch. One identity failure stops the phase.

### 4.1 Phase-R semantic reference

For every development row:

1. run the exact `create_splice_classification_logit_margin_route_census_apply`
   factory with the frozen Phase-R trace selection and all-false intervention;
2. run it twice and require bit-exact target and compact trace repeats and
   exact duplicate REF/ALT rows;
3. verify every one of its six target means satisfies
   `abs(R_current - R_lock) <= 2^-8`; and
4. for every frozen effect, require `R_current` and `R_lock` ALT-minus-REF
   deltas to have the experimental sign and absolute magnitude at least
   `0.01`.

The `2^-8` threshold is unchanged. It is applied to the current execution of
the graph that created the Phase-R lock, not transferred to a different
Stage-A graph after v3.0.2 already showed that comparison can fail.

### 4.2 Stage-A within-graph identity

Independently require, in graph `A`:

- bit-exact target and compact-trace repeats;
- natural REF equality across rows `0/4/5` and natural ALT equality across
  rows `1/2/3`;
- all-false natural/effective equality at final A/D, whole T and all seven E
  seams;
- bit-exact target duplicate rows;
- the target mean represents exactly the two frozen canonical endpoints; and
- for causal eligibility, the Stage-A ALT-minus-REF delta has the experimental
  sign and absolute magnitude at least `0.01`.

At least three of six effects in each development exon must remain Stage-A
eligible before isolated branches can be summarized. All ineligible effects
are retained in predictive counts and listed; they are never replaced.

### 4.3 Cross-graph diagnostic

For every row, serialize

```text
A - R_current
abs(A - R_current)
DeltaL_A - DeltaL_R_current
```

and per-exon maxima and distributions. This comparison is diagnostic only. It
does not enter eligibility, recovery, `q`, `Q`, route ranking or a stopping
threshold. A material difference makes every allowed conclusion explicitly
Stage-A-graph-dependent; it cannot be described as recovery in the locked
Phase-R executable.

No offset subtraction, rounding, denominator averaging or choice of the more
favourable graph is allowed. Re-running until a cross-graph value happens to
fall below `2^-8` is also prohibited.

## 5. Mandatory Stage-A closure and execution order

After all 20 identities pass Section 4, execute the following dependency order:

1. final post-GELU A/D embedding closure for every one of the 12 frozen
   development effects;
2. joint complete T+E closure for every one of those 12 effects;
3. isolated whole T and whole E only after both closure families are complete
   and valid across the full effect cohort; and
4. any later localized support or control family only after this dependency
   slice passes its separately frozen requirements.

In both reciprocal directions, each closure must copy the requested live donor
tensor exactly, keep same-allele self tensors and targets exact, and reproduce
the donor target bit-for-bit. Failure is an instrumentation failure, not a
negative route result. More than 5% tooling failures stops the family under the
main v3 protocol; either mandatory closure failure stops this dependency slice.

The closure-only smoke cannot satisfy these cohort-wide requirements.

## 6. Stage-A causal estimands

All primary causal quantities use the six values emitted by graph `A` in the
same active execution:

```text
L_R  = A[row 0]       L_A  = A[row 1]
L_RA = A[row 2]       L_AA = A[row 3]
L_AR = A[row 4]       L_RR = A[row 5]

r_REF_to_ALT = (L_RA - L_AA) / (L_R - L_A)
r_ALT_to_REF = (L_AR - L_RR) / (L_A - L_R)
B             = min(r_REF_to_ALT, r_ALT_to_REF)
```

The numerator is self-corrected inside the same graph. `L_AA == L_A` and
`L_RR == L_R` must still hold bit-for-bit; “self-corrected” is not permission
for self drift.

For route accounting, `empty`, `T`, `E` and `T+E` are all Stage-A target
values. Shapley terms and the nonlinear interaction are computed in raw target
units first, then normalized only by the corresponding Stage-A donor-minus-
recipient effect. They are descriptive computational branch accounting, not
an additive biological decomposition.

Using

```text
(Stage-A patched - Stage-A self) / (R_current donor - R_current recipient)
```

would be a mixed-graph denominator sensitivity, not a causal estimate for the
Phase-R graph. It is not part of v3.1 selection. If reported at all, it must be
labelled diagnostic, shown beside the primary within-A value for every row and
never used to rescue a failed Stage-A gate.

## 7. What this dependency slice may establish

| Evidence | Maximum allowed statement |
|---|---|
| `R_current` reproduces `R_lock` | The current runtime reproduces locked Phase-R identity targets within `2^-8`. |
| Stage-A identity only | The instrumented graph has exact within-graph no-op and repeat behavior. |
| Cohort-wide A/D and T+E closure | The instrumented Stage-A graph exposes causally complete paths to the selected target. |
| Isolated whole T/E development results | Whole-branch upper bounds and a descriptive T/E interaction in this Stage-A graph. |
| Later localized candidate passes full controls | A development computational-mechanism candidate ready for a separately locked held-out test. |

Even a strong isolated E result does not by itself establish a compact
“bypass”: E is all seven complete skip sequences, T/E may interact, and neither
has the localized ancestry-matched controls required by the full Stage-A
protocol. It also does not identify an RBP, spliceosome step, molecular pathway
or endogenous biological mechanism.

v3.1 cannot alter the Phase-R conclusion: the narrow layers-0--5 residual grid
remains negative for both the v2 probability target and the locked Phase-R
logit-margin target.

## 8. Mandatory limitations in every report

Every v3.1 result must state that:

- the amendment was written after v3.0.2 failed its preregistered
  Stage-A-to-lock comparison;
- a separate one-variant closure-only smoke was already observed;
- v3.1 estimates intervention effects in a numerically distinct instrumented
  graph, not patched counterfactuals in `R_lock` or `R_current`;
- BF16 graph dependence can change absolute targets and normalized effect
  magnitudes, so all cross-graph drifts and raw movements are reported;
- complete T/E branches are upper bounds, not compact circuits;
- development data may guide the one final candidate only through the frozen
  ladder; and
- confirmation remained unopened throughout development and amendment.

The amendment improves causal internal validity; it does not restore the
original cross-graph numerical-equivalence claim.

## 9. Freeze and implementation checklist

Before any v3.1 GPU call:

- [ ] Commit or timestamp this amendment and record its SHA-256.
- [ ] Bind that exact amendment SHA-256 in every v3.1 identity/component
      configuration and output summary.
- [ ] Bind the exact `run_stage_a_branches_v3.py`, `run_phase_r_v3.py`, target
      reducer, model instrumentation, checkpoint, FASTA, manifest, exon and
      main-protocol hashes.
- [ ] Use a new output directory; never resume from or overwrite the v3.0.2
      smoke or failed full-attempt directory.
- [ ] Pass CPU contract tests for the frozen six-row donor map, reference-to-
      lock tolerance boundary, cross-graph diagnostic non-gating behavior,
      identity no-op behavior, closure dependency order and confirmation
      allowlist.
- [ ] Run all 20 Stage-A and current-Phase-R identities before any active
      component.
- [ ] Stop on any current-Phase-R-to-lock tolerance failure or Stage-A
      identity failure.
- [ ] Complete both closure families across all 12 development effects before
      isolated T/E.
- [ ] Retain the v3.0.2 failure and closure-only smoke in the final audit tree.
- [ ] Verify no confirmation baseline, activation, attribution or output has
      been created.

The current draft runner imports Phase-R selection/intervention helpers, so its
configuration must explicitly hash `run_phase_r_v3.py`; hashing only the
Stage-A runner and route-census helper is insufficient. The amendment itself
must likewise be bound before the implementation is considered frozen.

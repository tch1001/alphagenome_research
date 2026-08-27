# OpenSplice v3.2 prospective superset-graph protocol

**Design date:** 2026-08-28

**Status:** prospective, development-only amendment. The runner, model
instrumentation, analyzer, tests, this document and a machine-readable freeze
must be committed before any v3.2 GPU or model call.

**Confirmation status:** blind. ELN e19, EIF4A2 e4 and DMD e78 remain excluded
from every v3.2 input, output, path, model call and decision.

## 1. Decision and rationale

Stop trying to reproduce the old Phase-R target lock. The completed exact-source
v3.1.2 diagnostic was internally exact but exceeded the unchanged `2^-8` saved-
target tolerance in 16/20 development identities. Its median per-identity
maximum difference was `0.0078125` and its maximum was `0.03125`. At the same
time, duplicate/repeat controls passed in all 20 rows and all 12 effects retained
the experimental direction with `abs(DeltaL) >= 0.01`.

This pattern establishes cross-executable BF16 numerical disagreement, not a
threshold change. The saved artifacts cannot distinguish differences caused by
the JAX/XLA graph and fusion schedule from cold-compilation autotuning or kernel
selection, so v3.2 does not assign a specific cause.
The old lock remains a failed historical diagnostic and is never reclassified.
Version 3.2 removes cross-executable saved-target equality from the decision
path. It prospectively creates one superset graph that contains both the frozen
Phase-R residual interventions and the already specified Stage-A whole-branch
interventions. Every baseline, donor, self control and causal contrast used by
v3.2 comes from that same graph and compiled executable.

The following frozen conclusions remain unchanged:

- the original probability-target residual census was negative;
- the first logit-margin Phase-R census was negative in its own graph;
- v3.0.2 and v3.1 failed their preregistered cross-graph gates;
- v3.1.1 was an infrastructure-only partial attempt;
- v3.1.2 failed the old target-lock comparison and ran no active patch; and
- the one-variant Stage-A closure smoke is tooling evidence only.

## 2. Frozen data and target

v3.2 uses only the first 20 rows of the existing SNV-only manifest:

| Input | Frozen value |
|---|---|
| `selected_variants_v2.tsv` | `09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6` |
| frozen exon table | `b95f8fc540f19222546322bebfb817a6c0f2147dd41325091086833584a09a75` |
| Development exons | BRAF e14 and SLC25A48 e8 |
| Development rows | orders 0--19: 12 significant effects and 8 neutral controls |
| Checkpoint snapshot | `a8f293a76ee73d5b57f3bf2ae146510589fcf187` |
| Context / attention | 16,384 bp / dense |
| Primary target | strand-aware mean canonical acceptor/donor classification logit minus padding logit |
| Target head | `splice_sites_classification/logits` |

For each allele,

```text
L_site = relevant acceptor-or-donor class logit - padding class logit
L_exon = mean(L_acceptor, L_donor)
DeltaL = L_exon_ALT - L_exon_REF
```

No probability, splice-site-usage, junction, RNA, single-endpoint or historical
target value may replace or rescue this scalar after results are visible.

## 3. Meaning of one superset graph

The v3.2 apply factory must contain all of the following in one JAX program:

1. the unchanged Phase-R residual seams `pre_attention`, `post_attention` and
   `post_mlp` in transformer layers 0--5;
2. the frozen `V`, `A`, `D`, `S` token sets and their upstream/downstream
   width-matched controls;
3. complete Stage-A transformer output `T`;
4. all seven complete encoder skips `E` at 1/2/4/8/16/32/64-bp resolution;
5. joint `T+E`; and
6. the complete final post-GELU 1-bp embedding at canonical A/D for closure.

Intervention family, stage, layer, position slots, direction and route masks are
fixed-shape runtime arrays. They must not be Python/static arguments that create
candidate-specific programs. The same six-row DNA batch and donor map is used
for every call:

```text
row 0 REF baseline/donor       row 1 ALT baseline/donor
row 2 ALT <- REF               row 3 ALT <- ALT self
row 4 REF <- ALT               row 5 REF <- REF self
donor rows [0, 1, 0, 1, 1, 0]
```

The implementation compiles the superset once and reuses that executable for
all 20 identities and every active group. It must persist StableHLO/HLO and an
executable fingerprint. Any additional compile caused by a candidate, variant
or route selector is a tooling failure. Ordinary calls may differ only in
same-shape DNA, target indices and runtime intervention arrays.

All live donor states remain on device. Large T/E tensors are not serialized to
the host and replayed. Exact elementwise donor/self/no-op booleans are computed
inside the executable; compact fingerprints are supporting repeat diagnostics,
not cryptographic proof.

## 4. One-attempt execution order

The entire order is frozen before execution and is not changed after reading an
intermediate value.

1. **Preflight.** Reuse the v3.1.2 environment contract: absent
   `LD_LIBRARY_PATH`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, an external JAX-only
   exact-RTX-3090/UUID gate, and the same-process gate before attempt creation.
   Additionally require `JAX_ENABLE_COMPILATION_CACHE=false` and absent
   `XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`, `JAX_PERSISTENT_CACHE_*` and any
   autotune load/dump/cache environment setting. Record every environment
   variable whose name starts with `XLA`, `JAX`, `CUDA`, `CUDNN`, `CUBLAS` or
   `TRITON`. No persistent executable or autotune artifact may be consumed.
   The launcher must also fail before importing AlphaGenome unless the tracked
   tree is clean and its only untracked files are the following exact generated
   binding exceptions:

   | Binding | Bytes | SHA-256 |
   |---|---:|---|
   | `src/alphagenome_research/protos/calibration_scores_pb2.py` | 2,794 | `4673289dd481fd8c4976f602ab36b07646304107e352e3e6d27b2abe4f9e9ebc` |
   | `src/alphagenome_research/protos/calibration_scores_pb2.pyi` | 1,815 | `329dc390abeb187084fff28fbe6cb6d9868aa8867326bf53f9a52d4c83f527f9` |

   Bind the paths and these exact source/dependency bytes as well:

   | Dependency | Bytes | SHA-256 |
   |---|---:|---|
   | `calibration_scores.proto` | 1,483 | `356f08689a4bafa0761f88f08dac08468a2de2c8aef38dcef093457eceee2f34` |
   | `dna_model.proto` | 15,103 | `d19a7208ec34953ca021efbff32516f1aa277f0477276f7699d9567fd616329a` |
   | `tensor.proto` | 2,856 | `07779023b2868377cbfc3c2ce96cd266ae425a0a1116aea755691c263d6238f7` |
   | imported `dna_model_pb2.py` | 16,279 | `d97564536e77ec09bdf144ba1204d4e08f79095fb9ed6c0cba7b065dc6f252ee` |
   | imported `tensor_pb2.py` | 3,155 | `dea7a5207e82601b6763e95ee4b69356345e95bc2de91632928d6161f873cdb8` |

   The generated header and runtime both report protobuf `7.35.1`. The
   historical generator binary and argument vector are unknown;
   these bytes are intentionally untracked exact build artifacts, not
   reproducibly generated source. Make no regeneration claim. Persist the
   launcher's pre-import attestation, then re-bind every imported module's
   `__file__`, size and hash in the same process before model creation. Direct
   runner invocation without that attestation must fail.
2. **Start and compile.** Create a new append-only v3.2 attempt and compile the
   committed superset graph once. Persist compiler and import provenance.
3. **Identity cohort.** In manifest order 0--19, run the all-false superset
   twice and persist all six targets and compact traces before deciding Gate 0.
4. **Target eligibility.** Compute `DeltaL` for all 12 effects and list all
   eight neutrals. Require at least three eligible effects per development exon.
5. **Frozen Phase-R grid.** For every eligible effect in manifest order, run
   all 216 groups in the existing order: three seams, layers 0--5, `V/A/D/S`,
   and both matched controls. Do not rank or inspect the grid until the raw tree
   is complete.
6. **Stage-A closures.** For all 12 effects, including target-ineligible ones,
   run both reciprocal final-A/D closures, then both reciprocal joint-T+E
   closures. Complete and validate each closure family across the cohort before
   proceeding.
7. **Stage-A isolated branches.** Only after both closure families pass, run
   isolated whole T and whole E, in manifest order, for target-eligible effects.
   Persist raw `empty`, `T`, `E` and `T+E` targets before computing summaries.
8. **Offline analysis.** A CPU-only, development-allowlisted analyzer verifies
   completeness and the artifact hash tree, recomputes Phase-R `B/q/Q`, Stage-A
   recoveries and Shapley accounting, and emits one frozen result. It never
   imports the model or accepts a confirmation-named path.
9. **Stop.** v3.2 never launches confirmation. A later circuit lock or next-
   stage protocol requires a separate committed decision after the complete
   development result is audited.

This order prospectively amends the original “stop Stage A if Phase R passes”
rule for this one integrated diagnostic. Both raw families are collected
automatically so the decision to run Stage A cannot depend on an interim
Phase-R value. Interpretation remains hierarchical: a passing frozen Phase-R
candidate has precedence; Stage-A whole-branch results cannot be chosen instead
because they look more favourable.

## 5. Fail-closed gates

### 5.1 Superset identity and target gate

For every one of the 20 rows require:

- bit-exact target and selected-trace repeats;
- bit-exact natural REF equality across rows 0/4/5 and natural ALT equality
  across rows 1/2/3;
- all-false natural/effective equality at every instrumented seam;
- exact target duplicate rows and exact two-endpoint reducer algebra;
- fixed-shape selector audit and no candidate-specific compilation; and
- no failed donor, index, cast, finite-value or linkage assertion.

Each identity and active artifact must persist, for all six rows and both
canonical endpoints, the selected relevant-class and padding-class raw logits
with shape `[6, 2, 2]`, plus their `[6, 2]` endpoint margins and `[6]` means.
The offline analyzer must independently recompute `L_site`, `L_exon`, the
two-endpoint denominator, strand/class mapping and every baseline/patch target
from these values. A runner-emitted boolean is not sufficient evidence for
this reducer algebra.

There is deliberately no comparison with `R_lock`, v3.1.2 targets or another
compiled graph. There is no numerical tolerance for within-graph identities:
the equality checks above are exact.

A significant effect is output-eligible only when

```text
sign(DeltaL) == sign(experimental delta_logit) and abs(DeltaL) >= 0.01.
```

All six effects per exon stay in the predictive denominator. Ineligible effects
are listed and are not replaced. Fewer than three eligible effects in either
development exon stops all active work as a target/predictive failure. Neutral
controls are retained and reported but are not assumed to be AlphaGenome-null.

### 5.2 Intervention validity

For every enabled Phase-R or Stage-A recipient require:

- rows 0/1 natural baselines equal that variant's frozen superset-identity REF
  and ALT targets bit-for-bit in every active call; this equality must also hold
  across the `empty`, `T`, `E` and `T+E` calls used in one Shapley account;
- its selected effective tensor equals the requested live donor tensor exactly;
- same-allele self tensors and targets equal their natural baselines exactly;
- disabled tensors equal their natural values exactly;
- target and compact-trace repeat checks pass; and
- the call uses the one recorded superset executable.

One failure invalidates that group. More than 5% invalid groups in any family
stops that family as a tooling failure; missing groups are failures, never
dropped observations. Even below that family-level threshold, a Phase-R
candidate is unselectable if its candidate group or either matched-control
group is invalid or missing for any frozen eligible effect. No median, `q` or
`Q` may silently omit such an effect. The isolated Stage-A branch account is
interpretable only if every required `T` and `E` call for every eligible effect
is valid and complete; otherwise it is reported as a failed route family with
no Shapley summary. These stricter decision rules take precedence over the 5%
family threshold and use no sentinel value or post-hoc imputation.

Both Stage-A closure families are stricter: for every one of the 12 effects and
both directions, the patched target must equal the live donor target bit-for-
bit. Any final-A/D or T+E closure failure prevents isolated T/E interpretation.

## 6. Frozen estimands and decisions

For any intervention family, using targets from the same active superset call,

```text
L_R  = row0    L_A  = row1
L_RA = row2    L_AA = row3
L_AR = row4    L_RR = row5

r_REF_to_ALT = (L_RA - L_AA) / (L_R - L_A)
r_ALT_to_REF = (L_AR - L_RR) / (L_A - L_R)
B = min(r_REF_to_ALT, r_ALT_to_REF)
```

Retain raw movements and unclipped recoveries. `L_AA == L_A` and
`L_RR == L_R` remain exact requirements; self correction is not permission for
self drift.

### 6.1 Phase-R ranking

The frozen residual grid contains 72 candidates in stage/layer/position order.
For each effect,

```text
q = B_candidate - max(B_control_upstream, B_control_downstream)
```

For each candidate, compute per-exon median `B` and median `q`, and
`Q = min(per-exon median q)`. Rank descending `Q`; break exact ties by
`pre_attention`, `post_attention`, `post_mlp`, then layer 0--5, then
`V`, `A`, `D`, `S`. A candidate passes only if both exons have at least three
eligible effects, median `B >= 0.25`, and median `q > 0`. The first passing
ranked candidate is the only Phase-R candidate eligible for a later, separately
frozen localization-and-specificity study. It is not yet a circuit lock and
cannot open confirmation. If none passes, the frozen Phase-R family is negative
in the superset graph.

### 6.2 Stage-A branch accounting

For each direction let `m(X)` be the raw target under route set `X`:

```text
phi_T = 0.5 * [(m(T)-m(empty)) + (m(T+E)-m(E))]
phi_E = 0.5 * [(m(E)-m(empty)) + (m(T+E)-m(T))]
interaction = m(T+E) - m(T) - m(E) + m(empty)
```

Normalize only after preserving raw values, using the corresponding within-
superset donor-minus-recipient denominator. Never mix in an old Phase-R or
v3.1.2 denominator.

Whole T and E are route upper bounds without the localized, ancestry-matched
controls required by protocol v3 Sections 7.2--7.4. Report their per-exon median
`B`, reciprocity, raw Shapley terms and interaction. Even `B >= 0.25` in both
exons may only nominate a route for a new prospectively frozen localized census;
it cannot be the final circuit and cannot open confirmation.

The eight development neutral variants are behavior controls only in v3.2 and
do not receive intervention grids. The frozen residual grid has two
width-matched spatial controls per candidate, but v3.2 does not run the earlier
32 random-region controls, wrong-strand/non-target-output patches or a neutral
intervention grid. Therefore even a passing residual candidate is only a
development hypothesis. Before any circuit lock or held-out confirmation, a
separate prospective localized protocol must restore neutral-variant,
random-position and output-specificity controls.

## 7. Leakage and retry rules

- Only BRAF and SLC25A48 orders 0--19 may be loaded. The runner and analyzer
  fail closed on ELN, EIF4A2, DMD, a confirmation partition, or a path component
  containing `confirm`.
- No confirmation baseline, target, attribution, activation, trace, patch,
  ranking, example selection or figure may be computed before a later immutable
  circuit lock.
- **Post-freeze process disclosure:** during v3.2 implementation, an agent
  mistakenly printed later rows of the already frozen selected-variant and exon
  metadata tables while diagnosing the development-projection loader. No
  AlphaGenome prediction, activation, attribution, patch, ranking or result
  artifact for those rows was opened or computed, and the scientific protocol
  had already been frozen. Subsequent code reads committed development-only
  projections. The later exons therefore remain model-output and intervention
  blind, but v3.2 must not claim complete metadata- or label-blindness.
- The v3.2 protocol was designed after seeing the historical-lock failures,
  the earlier development Phase-R negative and the disclosed one-variant
  closure smoke. Those facts must be reported. No confirmation result informed
  this design.
- Existing v3.1.2 targets may be used only as bound provenance. They may not set
  an offset, tolerance, denominator, candidate order, precision or compiler
  option.
- Do not inspect intermediate scientific JSON to alter execution. Raw artifacts
  are append-only and analyzed only after the automated cohort stops.
- Once `ATTEMPT_STARTED.json` exists, a model, compile, persistence, device or
  numerical failure consumes v3.2. Do not delete, resume, overwrite or rerun it.
  A separately versioned infrastructure amendment is permissible only after a
  new prospective document states exactly what failed and binds all partial
  artifacts; it may not adapt to a scientific value.
- Freeze source, protocol, manifest, checkpoint, environment, model tree,
  superset HLO contract, selectors, group order, gates, analyzer and output path
  in a committed manifest before the one attempt.

## 8. Allowed claims

| Outcome | Maximum defensible statement |
|---|---|
| Superset identity or closure fails | The integrated instrumentation failed its tooling gate. No mechanism result exists. |
| Target eligibility fails | The frozen AlphaGenome logit-margin target did not align with enough development effects for this causal benchmark. |
| Phase-R grid passes | One frozen transformer-residual patch family recovers at least one quarter of the target effect beyond two matched spatial controls in both development exons, in this superset graph; it still requires random, neutral and output-specificity controls. |
| Phase-R grid is negative | None of the 72 frozen residual candidates met the two-exon development gate in this superset graph. |
| Stage-A closures pass | The superset graph can causally transfer complete selected paths to the target. This validates tooling, not localization. |
| Isolated T/E results | Descriptive whole-route upper bounds and computational Shapley/interaction accounting in two development exons. |
| A later localized candidate passes all separately frozen random, neutral and output-specificity controls | A development computational-mechanism candidate ready to be locked before held-out confirmation. |

No v3.2 outcome establishes an RBP, spliceosome step, biochemical pathway,
endogenous molecular mechanism or experimental replication. It does not prove
compiler determinism and cannot restore the old cross-executable lock. Until a
separately frozen candidate is tested once on the unopened exons, every result
is development-only.

## 9. Pre-run checklist

- [ ] Commit this protocol and record its SHA-256.
- [ ] Add and test one superset factory with runtime-only fixed-shape selectors.
- [ ] Add a CPU synthetic test for six-row mapping, every intervention family,
      exact donor/no-op/self booleans, closure, formulas and frozen order.
- [ ] Add a development-only analyzer that recomputes all gates and rejects
      confirmation paths/genes.
- [ ] Bind the exact model source, runner, wrapper, tests, analyzer, manifest,
      exons, checkpoint, FASTA/reference, target reducer and protocol in one
      committed freeze.
- [ ] Require one compiled superset executable and persist its compiler
      provenance before active calls.
- [ ] Persist raw relevant/padding endpoint logits and independently recompute
      every endpoint margin and mean in the offline analyzer.
- [ ] Disable the persistent JAX compilation cache and fail closed on ambient
      compiler/autotune cache flags.
- [ ] Confirm both new v3.2 output directories are absent.
- [ ] Run the external and same-process exact-RTX-3090 gates.
- [ ] Run once, audit completely, and stop with confirmation blind.

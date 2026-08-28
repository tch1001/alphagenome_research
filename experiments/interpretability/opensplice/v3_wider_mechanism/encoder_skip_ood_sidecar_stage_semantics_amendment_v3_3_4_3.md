# Prospective v3.3.4.3 amendment: stage-exact source-gate semantics

Status: prospective and unconsumed. This document authorizes no model, GPU,
preflight, analyzer, or scientific run by itself.

Confirmation boundary: later-exon metadata and labels have been exposed, but
later-exon model outputs, activations, and interventions remain unopened and
must remain so.

## 1. Binding, reason, and sole correction

This amendment binds and otherwise preserves the complete prospective
v3.3.4.2 protocol:

- Git commit `f48d6b73839b428fe950b00696548b6410a52659`;
- file
  `experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_nonpublication_terminal_amendment_v3_3_4_2.md`;
- SHA-256
  `1d2109e58d11cb07e99490bfde5fbb5d5ab43bd12e429c28b4ca9dfc0656fb87`.

It also transitively preserves the bound v3.3.4 and v3.3.4.1 commits,
documents, hashes, scientific design, publication primitive, confirmation
boundary, one-shot budgets, exact terminal membership, 58-key terminal
schema, analyzer claim boundary, and no-retry rule.

The sole correction is stage-specific interpretation of two already existing
objects in `NONPUBLICATION_TERMINAL_FAILURE.json`:

1. `same_object_attestation`; and
2. `phase_state.source_program_gate_passed`.

The v3.3.4.2 text correctly permits
`source_program_gate_without_backend_diagnostics` to be a fully derived true
or false object for `diagnostic_failure_record_construction`. Its broader
sentence that set `source_program_gate_passed=false` and
`compiler_record_is_gate_record=null` for every new terminal was incompatible
with that fact and with the inherited diagnostic precedence. This amendment
replaces only those broad values with the exact stage matrix below. It does
not turn a diagnostic failure into a source-program success, permit dispatch,
or change the terminal's incomplete-infrastructure claim.

No v3.3.4, v3.3.4.1, or v3.3.4.2 production, preflight, cache, analysis, or
analysis-attempt path has been launched. Uncommitted implementation drafts are
not protocol consumption and are never runtime inputs.

## 2. Fresh v3.3.4.3 namespace

The only authorized fresh paths are:

```text
experiments/interpretability/opensplice/results/v3_3_4_3_development_ood_sidecar_one_shot
experiments/interpretability/opensplice/results/v3_3_4_3_device_preflight
experiments/interpretability/opensplice/results/v3_3_4_3_preflight_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_3_model_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_3_development_ood_sidecar_analysis
experiments/interpretability/opensplice/results/v3_3_4_3_development_ood_sidecar_analysis_attempt
```

All six must be absent and non-symlinked at protocol freeze, scoped commit,
post-commit audit, and before the one permitted lifecycle. The exact six roles
under each of `v3_3_4`, `v3_3_4_1`, and `v3_3_4_2` are immutable absent
predecessor paths, for 18 exact predecessor absences. No predecessor code or
output is copied, imported, opened, or reused.

All versioned literals become v3.3.4.3: model attempt ID
`v3.3.4.3-development-ood-sidecar-one-shot`, script version `v3.3.4.3`,
analyzer version `v3.3.4.3-structural-analyzer-v1`, fresh script/freeze names,
and external authorization variables `V3343_AUTHORIZED_GIT_HEAD`,
`V3343_AUTHORIZED_FREEZE_SHA256`, and
`V3343_AUTHORIZED_FREEZE_SIZE_BYTES`. The inherited v3.3.4.1 publication
schema and syscall method remain byte-for-byte semantic requirements; only
the versioned roots and source names change.

## 3. Exact stage matrix

The five v3.3.4.2 failure-stage strings and their precedence are unchanged:

```text
stablehlo_text_extraction
pre_backend_hlo_text_extraction
compiled_hlo_text_extraction
source_program_gate_derivation_for_diagnostic_failure
diagnostic_failure_record_construction
```

Every row has lower/compile/success counts `(1,1,1)`, a successful signature
attestation, distinct nonnegative integer `lowered_python_id` and
`compiled_python_id`, true compile-argument identity, true
signature-attestation argument identity, true apply-callable identity, zero
dispatch/raw/apply counts, and `diagnostic_provenance_passed=false`.

### 3.1 Extraction stages

For the first three stages,
`source_program_gate_without_backend_diagnostics=null`, its content binding is
null, terminal import is absent, and
`phase_state.source_program_gate_passed=false`.

The exact same-object graph-prefix values are:

| Failure stage | StableHLO read | Pre-backend HLO read | Compiled-HLO read | Compiler gate identity |
|---|---:|---:|---:|---:|
| `stablehlo_text_extraction` | null | null | null | null |
| `pre_backend_hlo_text_extraction` | true | null | null | null |
| `compiled_hlo_text_extraction` | true | true | null | null |

No graph file is published in any extraction row. A null means the successful
primitive was unavailable at the caught boundary; it is not converted to
false. No source gate is synthesized.

### 3.2 Source-program-gate derivation failure

For `source_program_gate_derivation_for_diagnostic_failure`, all three graph
files and the terminal import inventory exist. Every applicable same-object
primitive is true, including:

```text
stablehlo_read_from_lowered_object=true
pre_backend_hlo_read_from_lowered_object=true
compile_argument_is_lowered_object=true
compiled_hlo_read_from_compiled_object=true
signature_attestation_from_apply_arguments=true
apply_callable_is_compiled_object=true
compiler_record_is_gate_record=true
```

Here `compiler_record_is_gate_record=true` attests that the sole in-memory
compiled object was the exact gate object supplied to the attempted source
derivation. It does not falsely assert that a successful
`COMPILER_PROVENANCE.json` was published.

The source derivation raised, so
`source_program_gate_without_backend_diagnostics=null`, its binding is null,
and `phase_state.source_program_gate_passed=false`. The earlier triggering
diagnostic exception and its exact four-way reason remain separately
preserved as required by v3.3.4.2.

### 3.3 Diagnostic-failure-record construction failure

For `diagnostic_failure_record_construction`, all seven applicable
same-object primitives above are true. The exact independently derived
`source_program_gate_without_backend_diagnostics` and its canonical content
binding are both nonnull.

The phase bit is not hard-coded. It is exactly:

```text
phase_state.source_program_gate_passed ==
source_program_gate_without_backend_diagnostics.source_program_exact
```

Both the true and false cases are legal and must be tested. A true source gate
does not permit dispatch because diagnostic provenance is incomplete and the
required diagnostic-failure record could not be constructed. A false source
gate does not change precedence: the earlier diagnostic exception was reached
first, so the incomplete diagnostic construction remains the terminal reason.
In both cases `diagnostic_provenance_passed=false`, no valid compiler gate
record exists on disk, and the structural claim is unchanged.

### 3.4 Exact unchanged fields and membership

Every other v3.3.4.2 terminal field is unchanged. The terminal still has
exactly 58 keys, the exact extraction or diagnostic-construction membership,
the same trigger/failure separation, source-audit phase values, cache
historical/terminal bindings, prior bindings, publication audit, disclosure,
zero counts, and no-science flags. No new field or status is introduced.

The CPU analyzer must reject:

- null, false, or missing applicable same-object primitives in either
  diagnostic stage;
- a nonnull source gate in an extraction or source-derivation-failure row;
- a null source gate in a diagnostic-record-construction row;
- any mismatch between the derived source-gate boolean and the phase bit;
- a source-gate phase bit of true in either extraction or source-derivation
  row; and
- any attempt to infer the phase bit from terminal status rather than the
  bound source-gate object.

## 4. Analyzer outcome and claim boundary

The exact v3.3.4.2 23-key ANALYSIS schema, status
`complete_incomplete_nonpublication_infrastructure_archive`, decision
`post_compile_nonpublication_failure_no_scientific_analysis`, terminal kind
`nonpublication_terminal_failure`, and
`compiler_state="compiled_without_legal_graph_gate_record"` remain unchanged.

The analyzer independently recomputes the stage matrix before reporting that
outcome. It may disclose whether the already bound source gate was true or
false only as structural provenance. It does not call that a biological
result, does not inspect raw scientific values, and does not permit
normalization, Shapley analysis, interaction, resolution, nomination, or
combination. `control_state_eligible=false` and
`combined_analysis_permitted=false` in every non-publication archive.

## 5. Acyclic freeze and source inventory

The v3.3.4.3 machine freeze retains the exact 84-key structure of v3.3.4.2.
The existing `nonpublication_terminal_contract_v3_3_4_2` key is versioned to
`nonpublication_terminal_contract_v3_3_4_3`; no additional top-level key is
added. Its value binds this amendment's exact stage matrix, source-gate/phase
equality rule, same-object applicability, and analyzer rejection rules.

The `file_sha256` map contains the inherited 96 v3.3.3 rows unchanged plus
exactly these 12 prospective v3.3.4.3 rows, for 108 rows total:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_3.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_3.sh
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_3_test.py
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_3_freeze.py
experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_3.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3_test.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3.sh
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3_test.py
experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_stage_semantics_amendment_v3_3_4_3.md
experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3.py
```

The freeze path is exactly
`experiments/interpretability/opensplice/encoder_skip_ood_sidecar_v3_3_4_3_freeze.json`.
Python, tests, amendment, and freeze have Git mode `100644`; the two shell
wrappers have `100755`. The freeze is tracked but omitted from its own
`file_sha256` map.

The freeze independently binds the v3.3.4.2 commit/path/hash above, the exact
18-path predecessor absence audit, and the fact that no predecessor lifecycle
was consumed. V3.3.4.2 implementation drafts are not imported or
source-inventoried. No inventoried source hard-codes the final freeze hash;
the three external authorization variables are the sole post-commit anchor.

## 6. Required CPU tests and stop rules

Before any scoped commit, tests must cover at least:

1. real runner-shaped injection at every one of the five stages;
2. exact partial/null same-object values for each extraction failure;
3. full true applicable same-object values for both diagnostic stages;
4. source-gate-derivation failure with null gate/binding and false phase bit;
5. diagnostic-record-construction failure with separately frozen true and
   false source-gate objects, exact content bindings, and matching phase bits;
6. rejection of every cross-stage nullability, false primitive, phase/gate
   disagreement, or invented compiler-record file;
7. unchanged six-file and ten-file memberships, exact 58-key schema,
   publication fallbacks, cache/trigger/source/prior bindings, and no-science
   counts;
8. analyzer runner-shaped acceptance of every legal row and tamper rejection
   of every matrix field and source-gate leaf;
9. exact 18-path predecessor absence, six fresh-path absence, 84-key freeze,
   108-row live/Git source inventory, modes, and acyclic external
   authorization; and
10. all inherited publication fault stages, signature validated-prefix cases,
    second-lower/second-compile guarded failures, terminal matrix, dry run,
    confirmation guard, and no-model/no-science analyzer tests.

Any mismatch stops before dispatch. No v3.3.4.3 preflight, GPU, model,
analyzer, or scientific run is authorized until the complete implementation
bundle and freeze are committed, independently audited, clean, and explicitly
approved in a later step.

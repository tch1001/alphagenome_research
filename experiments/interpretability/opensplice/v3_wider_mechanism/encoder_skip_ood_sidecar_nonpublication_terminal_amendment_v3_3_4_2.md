# Prospective v3.3.4.2 amendment: non-publication compiler terminal

Status: prospective and unconsumed. This document authorizes no model, GPU,
preflight, analyzer, or scientific run by itself.

Confirmation boundary: later-exon metadata and labels have been exposed, but
later-exon model outputs, activations, and interventions remain unopened and
must remain so.

## 1. Binding, reason, and narrow scope

This amendment binds and otherwise preserves both prospective predecessors:

- v3.3.4 infrastructure protocol: Git commit
  `f833a8d2108636871abfce8b4cbabe4255536974`, file
  `experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_infrastructure_amendment_v3_3_4.md`, SHA-256
  `38d07c0b612e50aadc64ba18537561cbdb0489b67fd0824cae749bba6214207b`;
- v3.3.4.1 publication protocol: Git commit
  `2b5e3e93a9961ac7cb12c088f6922acc9fdc5dde`, file
  `experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_publication_amendment_v3_3_4_1.md`, SHA-256
  `6abc470f6fb14b70c8930195bb8f26ce730b8c07c636cd842d5451f37d8eb55c`.

The inherited v3.3.4 text requires a best-effort terminal when an ordinary,
non-publication exception prevents obtaining the three compiler graph texts.
The v3.3.4.1 text then gives `TERMINAL_FAILURE.json` one exact schema whose
mandatory `publication_failure` must be a real failure from the frozen
publication primitive. Fabricating a `PublicationError`, putting a null or
synthetic publication object in that schema, or silently leaving the ordinary
exception uncaught would all be false provenance.

This amendment resolves only that conflict. It introduces one differently
named, exactly scoped artifact for a real post-compile, pre-dispatch ordinary
infrastructure failure that makes the normal successful compiler record or
the normal three-graph diagnostic-failure record impossible. It does not
relax or reinterpret any biological, source-program, diagnostic, device,
runtime, cache, same-object, import, confirmation, or structural gate.

In particular, these inherited requirements remain literal implementation
requirements, not clarifications made here:

- a failed signature attestation preserves the actually validated prefix of
  the fixed 32 runtime and frozen container-tag paths; it may use empty
  prefixes only when no tag was validated;
- guarded second-lower and second-compile requests use the inherited failure
  compiler schema, keep actual invocation counts at one, and have null source
  gate and diagnostics; a normal successful compiler record is not a legal
  second-compile terminal; and
- a true publication failure continues to use only the v3.3.4.1
  `TERMINAL_FAILURE.json` schema and claim boundary.

No v3.3.4 or v3.3.4.1 production path has been launched. At protocol freeze,
scoped commit, post-commit audit, and before any v3.3.4.2 preflight or model
process, all twelve predecessor production paths below must be absent and
non-symlinked:

```text
experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_one_shot
experiments/interpretability/opensplice/results/v3_3_4_device_preflight
experiments/interpretability/opensplice/results/v3_3_4_preflight_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_model_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_analysis
experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_analysis_attempt
experiments/interpretability/opensplice/results/v3_3_4_1_development_ood_sidecar_one_shot
experiments/interpretability/opensplice/results/v3_3_4_1_device_preflight
experiments/interpretability/opensplice/results/v3_3_4_1_preflight_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_1_model_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_1_development_ood_sidecar_analysis
experiments/interpretability/opensplice/results/v3_3_4_1_development_ood_sidecar_analysis_attempt
```

## 2. Fresh versioned paths and unchanged scientific design

The only authorized fresh v3.3.4.2 paths are:

```text
experiments/interpretability/opensplice/results/v3_3_4_2_development_ood_sidecar_one_shot
experiments/interpretability/opensplice/results/v3_3_4_2_device_preflight
experiments/interpretability/opensplice/results/v3_3_4_2_preflight_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_2_model_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_2_development_ood_sidecar_analysis
experiments/interpretability/opensplice/results/v3_3_4_2_development_ood_sidecar_analysis_attempt
```

They must all be absent before the one-shot lifecycle begins. The exact
v3.3.4.1 `renameat2(RENAME_NOREPLACE)` publication method, preflight probe,
directory modes, temp/final modes, fsync order, collision preservation,
failure audit, and no-fallback/no-retry rules apply unchanged, with only
versioned root paths and script names changed.

The scientific execution contract is unchanged: the same 20 development
recipients and four anchors in the same order, exactly one eight-row lower and
compile, at most 80 records and 320 applies, zero six-row/identity/main-cube
runs, no old OOD reuse, no confirmation call, and no GPU-side scientific
summary. A complete run still requires all inherited source-program,
diagnostic, cache, invariant, closure, and confirmation gates. Every outcome
has `combined_analysis_permitted=false`.

## 3. Exact non-publication failure boundary

### 3.1 In-memory extraction precedes graph publication

After the sole compile returns, the runner captures these three strings in
memory, in this exact order, before publishing any graph file:

1. StableHLO from the sole lowered object;
2. pre-backend HLO from that same lowered object; and
3. compiled HLO from the sole compiled object.

Only after all three captures succeed may it publish the three graph files in
their inherited order. Therefore an extraction exception has zero graph files,
not an ambiguous partial graph prefix. A failure of the publication helper for
any graph is a real publication failure and follows v3.3.4.1; it is never
converted into the artifact defined here.

When all three graph files exist, entry-ABI and backend-parser errors remain
the inherited `controlled_stop_diagnostic_provenance_failure` and must use
`COMPILER_DIAGNOSTIC_FAILURE.json`. The new terminal is legal only when the
ordinary exception prevents constructing that exact inherited artifact.

### 3.2 Exact stage enum and precedence

`failure_stage` is exactly one of:

```text
stablehlo_text_extraction
pre_backend_hlo_text_extraction
compiled_hlo_text_extraction
source_program_gate_derivation_for_diagnostic_failure
diagnostic_failure_record_construction
```

The first three stages have no graph files and occur before terminal import
capture. `source_program_gate_derivation_for_diagnostic_failure` is legal only
after all three graph files and the terminal import inventory exist, when a
real diagnostic parser/fingerprint/cache exception has already occurred but
the independently required source-program object cannot be derived.
`diagnostic_failure_record_construction` is legal only after those same
artifacts exist and the source-program object was derived, but an ordinary
exception prevents constructing the exact diagnostic-failure object before
its publication call.

The precedence is literal:

1. any `PublicationError` uses the exact v3.3.4.1 publication path;
2. a successfully constructible three-graph diagnostic failure uses the exact
   inherited diagnostic controlled stop;
3. only one of the five ordinary failures above uses this amendment; and
4. no other exception may be relabelled as one of these stages.

The caught exception is never retried. No second graph extraction, lower,
compile, diagnostic derivation, or model dispatch is permitted.

## 4. `NONPUBLICATION_TERMINAL_FAILURE.json`

### 4.1 Exact path, publication, and membership

The artifact path is exactly
`NONPUBLICATION_TERMINAL_FAILURE.json` at the model-run root. It is serialized
with `allow_nan=false` and published once through the supported v3.3.4.1
named-temp/no-replace helper as artifact role
`nonpublication_terminal_failure`. It is a successfully published final, not
a fabricated publication failure.

For an extraction-stage terminal, the preterminal model-run membership is
exactly:

```text
ATTEMPT_STARTED.json
IMPORT_PROVENANCE_PRE_MODEL.json
PROTOBUF_PROVENANCE.json
IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json
compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json
```

There are exactly five preterminal regular files and six after the new
terminal is published. `graph_artifact_bindings={}` and the terminal import
binding is null.

For either diagnostic-construction stage, the preterminal membership is those
five files plus exactly:

```text
compiler/eight_row/graph.stablehlo.mlir
compiler/eight_row/graph.pre_backend.hlo.txt
compiler/eight_row/graph.compiled.hlo.txt
IMPORT_PROVENANCE.json
```

There are exactly nine preterminal regular files and ten after terminal
publication. `graph_artifact_bindings` contains exactly the three sorted
run-root-relative graph paths, each valued by exactly `sha256,size_bytes`.

No row contains `RAW_MANIFEST.json`, `RUN_COMPLETE.json`, a valid raw record,
a failed-current record, or a dispatch-journal event. A pre-existing extra,
symlink, special file, unregistered empty directory, or wrong mode invalidates
the state.

### 4.2 Exact 58-key schema

The artifact has exactly these 58 top-level keys:

```text
schema_version, status, stop_reason, attempt_id, script_version,
amendment_commit, amendment_sha256,
inherited_v3_3_4_commit, inherited_v3_3_4_sha256,
inherited_v3_3_4_1_commit, inherited_v3_3_4_1_sha256,
freeze_sha256, git_head, external_freeze_authorization, runner_pid,
started_at_unix_s, created_at_unix_s, failure_stage, failure,
triggering_diagnostic_failure, triggering_diagnostic_stop_reason, phase_state,
source_input_audit, source_input_audit_content_binding,
program_signature_attestation_binding, same_object_attestation,
same_object_attestation_content_binding, attempt_budget_audit,
compiler_counts, graph_artifact_bindings, import_provenance_phases,
protobuf_provenance_sha256, model_kernel_cache_state,
source_program_gate_without_backend_diagnostics,
source_program_gate_without_backend_diagnostics_content_binding,
prior_v3_3_3_binding, prior_v3_3_3_1_archive_binding,
preterminal_tree_binding, publication_audit,
model_apply_attempt_count, model_apply_success_count, valid_record_count,
raw_record_count, dispatch_started_count, dispatch_completed_count,
six_row_compile_count, identity_rerun_count, main_cube_rerun_count,
old_ood_records_reused, confirmation_model_calls,
confirmation_scope_disclosure,
scientific_summary_computed, donor_normalization_computed,
shapley_or_nomination_computed, interaction_or_resolution_computed,
nomination_performed, combined_analysis_permitted, no_retry
```

The literals are:

```text
schema_version="v3.3.4.2-nonpublication-terminal-v1"
status="incomplete_nonpublication_infrastructure_failure"
stop_reason="post_compile_nonpublication_infrastructure_failure"
attempt_id="v3.3.4.2-development-ood-sidecar-one-shot"
script_version="v3.3.4.2"
model_apply_attempt_count=0
model_apply_success_count=0
valid_record_count=0
raw_record_count=0
dispatch_started_count=0
dispatch_completed_count=0
six_row_compile_count=0
identity_rerun_count=0
main_cube_rerun_count=0
old_ood_records_reused=0
confirmation_model_calls=0
confirmation_scope_disclosure="Later-exon metadata/labels were exposed after protocol freeze; no later-exon model outputs, activations, or interventions are used."
scientific_summary_computed=false
donor_normalization_computed=false
shapley_or_nomination_computed=false
interaction_or_resolution_computed=false
nomination_performed=false
combined_analysis_permitted=false
no_retry=true
```

`failure` has exactly `type,message,traceback` and describes the real ordinary
extraction, source-gate-derivation, or diagnostic-record-construction
exception. `triggering_diagnostic_failure` is null for the three extraction
stages, and `triggering_diagnostic_stop_reason` is also null. For either
diagnostic-construction stage the failure has exactly
`type,message,traceback` and preserves the earlier real parser, fingerprint,
or cache exception that first entered the diagnostic-failure path; it is
distinct from the later `failure`. `triggering_diagnostic_stop_reason` is
exactly one of `diagnostic_parser_failure`, `diagnostic_persistence_failure`,
`cache_signal_unavailable`, or `fingerprint_formula_mismatch`, is captured at
the first failing operation rather than inferred later from free-form text,
and must agree with the inherited literal classifier for that operation.
Times are finite JSON numbers, IDs and
hashes use the exact frozen v3.3.4.2 values, and
`external_freeze_authorization` is the independently validated current
authorization object. `program_signature_attestation_binding` has exactly
`path,sha256,size_bytes` and binds the successful attestation.

`compiler_counts` has exactly
`lower_attempt_count,compile_attempt_count,successful_compile_count` and is
`{"lower_attempt_count":1,"compile_attempt_count":1,
"successful_compile_count":1}`. `attempt_budget_audit` has exactly the six
inherited keys and values `(1,1,1,1,null,false)`. No compiler provenance or
diagnostic-failure binding is present.

`phase_state` has the exact inherited 19-key schema. It records successful
preflight, START, post-START source gate, protobuf, PRE_MODEL, model/reference,
signature, POST_MODEL, lower, and compile phases; dispatch is false. Terminal
import is false for extraction stages and true for diagnostic-construction
stages. Source-program and diagnostic-provenance pass are false.

`source_input_audit` has the exact inherited eight keys and a matching
canonical content binding. For extraction stages, its first seven terms are
independently derived true and
`three_import_inventories_stable_exact=null`. For diagnostic-construction
stages, all eight are independently derived booleans; this terminal is legal
only when all eight are true. No null is converted to false.

`same_object_attestation` always has the inherited exact keys:

```text
lower_call_count, compile_call_count,
stablehlo_read_from_lowered_object,
pre_backend_hlo_read_from_lowered_object,
compile_argument_is_lowered_object,
compiled_hlo_read_from_compiled_object,
signature_attestation_from_apply_arguments,
apply_callable_is_compiled_object,
compiler_record_is_gate_record,
lowered_python_id, compiled_python_id
```

Counts are one; compile-argument, signature-argument, and apply-callable
identity primitives are true. For `stablehlo_text_extraction`, all three graph
read primitives are null; for `pre_backend_hlo_text_extraction`, StableHLO is
true and the other two are null; for `compiled_hlo_text_extraction`, the two
lowered-object reads are true and the compiled-object read is null. For both
diagnostic-construction stages all three graph-read primitives are true.
`compiler_record_is_gate_record` is null in every row because no valid
compiler gate record exists. `lowered_python_id` and `compiled_python_id` are
distinct nonnegative JSON integers and bind the exact objects used by the sole
lower/compile flow. The matching content binding is exact.

`import_provenance_phases` has exactly `pre_model,post_model_precompile,terminal`.
The first two values are exact file bindings. The terminal value is null for
extraction stages and an exact binding for diagnostic-construction stages.
`protobuf_provenance_sha256` binds the persisted protobuf artifact.

`model_kernel_cache_state` has exactly:

```text
pre_import, historical_stage, historical_binding, terminal_live_binding,
cache_hit_evidence, historical_to_terminal_tree_exact,
historical_to_terminal_equality_is_a_gate,
historical_snapshot_not_reauthenticated_as_live_files,
default_user_cache_paths_eligible, cache_outputs_are_diagnostic_only
```

The pre-import, historical, and terminal-live objects use the exact
directory-aware cache binding schema. `historical_stage="post_compile"` and
`historical_binding` is captured immediately after compile and before graph
extraction. `terminal_live_binding` is independently captured immediately
before construction of this terminal. `historical_to_terminal_tree_exact` is
the literal comparison of their tree digests, while
`historical_to_terminal_equality_is_a_gate=false` and
`historical_snapshot_not_reauthenticated_as_live_files=true` preserve the
inherited temporal semantics. Cache evidence is independently captured at the
historical point. `cache_hit_evidence` is nonnull and has the exact inherited
schema for all three extraction stages and for either diagnostic-construction
stage whose triggering reason is not `cache_signal_unavailable`; in those
rows `cache_hit=false`. If and only if
`triggering_diagnostic_stop_reason="cache_signal_unavailable"`,
`cache_hit_evidence=null` and the exact unavailable-evidence exception is the
nonnull `triggering_diagnostic_failure`; no cache evidence is synthesized or
retried. Default paths are ineligible and outputs are diagnostic only. A
nonnull cache-hit result of true uses the inherited cache controlled stop. An
unavailable result uses this amendment's exact diagnostic-construction stage
only when its later source-gate derivation or diagnostic-record construction
also raises the separately preserved ordinary `failure`; otherwise it uses
the constructible inherited diagnostic controlled stop. It may never
dispatch.

`source_program_gate_without_backend_diagnostics` and its content binding are
both null for all three extraction stages and for
`source_program_gate_derivation_for_diagnostic_failure`. For
`diagnostic_failure_record_construction`, the source-gate object is nonnull,
has the exact inherited successful-derivation schema, and its content binding
has exactly `sha256,size_bytes`; the object may record a true or false source
gate, because diagnostic failure has precedence, but every primitive and the
binding must be independently reproducible. The two fields are either both
null or both nonnull.

The two prior bindings are full, independently revalidated v3.3.3 run and
v3.3.3.1 archive objects. `preterminal_tree_binding` has exactly the inherited
lstat-safe `file_count,directory_count,file_bindings,file_tree_sha256,
directory_paths,directory_tree_sha256` schema and excludes the terminal
itself. `publication_audit` is the exact v3.3.4.1 15-key model-run audit at the
preterminal point: no publication failure, no orphan, uncertain final, or
pre-existing entry, and all no-delete/no-reuse/no-retry booleans true.

### 4.3 Failure while publishing this terminal

If publication of `NONPUBLICATION_TERMINAL_FAILURE.json` raises a real
`PublicationError`, the runner invokes the unchanged v3.3.4.1 best-effort
publication-terminal path with that real failure object. The resulting
`TERMINAL_FAILURE.json`, if it can be published, uses only the exact
v3.3.4.1 schema and binds the preterminal prefix plus any orphan,
durability-uncertain final, or pre-existing entry. If terminal publication is
also impossible, the exact filesystem prefix is consumed and unanalyzable;
there is no deletion or retry.

An ordinary error in constructing the non-publication terminal is not a
publication error and must not be fabricated into one. Tests and preflight
must make construction deterministic from already captured primitives. If it
nevertheless occurs, preserve the terminal-less consumed prefix, emit no
completion claim, and prohibit retry; a future prospective protocol would be
required to archive it.

## 5. CPU-only structural analyzer

The v3.3.4.2 analyzer is standalone standard-library code with fresh attempt
and output paths. It does not import or monkeypatch an older analyzer, JAX,
JAXLIB, AlphaGenome, OpenSplice scientific helpers, or model code. It accepts
the new terminal only after independently checking clean HEAD, external freeze
authorization, every frozen source/Git blob, predecessor absence and archive
bindings, preflight/cache trees, exact run membership, all 58 fields and
nested schemas, current file bytes/modes, compiler counts, phase/nullability,
zero journal/raw/apply counts, publication audit, and confirmation isolation.
It never opens a raw scientific value.

For the exact new terminal, the inherited 23-key ANALYSIS schema is retained
unchanged and has:

```text
status="complete_incomplete_nonpublication_infrastructure_archive"
decision="post_compile_nonpublication_failure_no_scientific_analysis"
analysis_version="v3.3.4.2-structural-analyzer-v1"
compiler_state="compiled_without_legal_graph_gate_record"
terminal_kind="nonpublication_terminal_failure"
control_state_eligible=false
```

These values appear in the inherited `terminal_audit`,
`compiler_and_signature_audit`, `control_audit`, and claim-boundary objects at
their exact existing locations. `publication_audit` retains the exact
v3.3.4.1 17-key analyzer schema and independently binds both analysis roots.
Every scientific, normalization, Shapley, interaction, resolution,
nomination, and combination flag is false. RESULT may describe only the real
failure stage, exact provenance/counts, and incomplete-infrastructure claim.

The analyzer uses one append-only invocation, one active START token, exact
START-SHA linkage in ANALYSIS/COMPLETE/FAILURE, and repeats all source/run/path
checks before RESULT, ANALYSIS, and COMPLETE. A terminal-less prefix is
rejected before consuming the analyzer attempt.

## 6. Acyclic freeze and source inventory

The v3.3.4.2 machine freeze is based on the same exact 69-key v3.3.3 base. It
retains the 13 v3.3.4 extension keys and the one v3.3.4.1 publication-contract
key, and adds exactly one top-level key:

```text
nonpublication_terminal_contract_v3_3_4_2
```

The exact top-level count is therefore 84. The new contract key binds the
stage enum, 58-key terminal list, two exact memberships, nested key sets,
literal counts/flags, analyzer outcome, and fallback precedence in Sections
3--5.

The `file_sha256` map contains the inherited 96 v3.3.3 rows unchanged plus
exactly these 12 v3.3.4.2 rows, for 108 rows total:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_2.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_2.sh
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_2_test.py
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_2_freeze.py
experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_2.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_2.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_2_test.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_2.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_2.sh
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_2_test.py
experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_nonpublication_terminal_amendment_v3_3_4_2.md
experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_2.py
```

The freeze path is exactly
`experiments/interpretability/opensplice/encoder_skip_ood_sidecar_v3_3_4_2_freeze.json`.
Python, tests, amendment, and freeze use Git mode `100644`; the two shell
wrappers use `100755`. The freeze itself is tracked but excluded from its own
`file_sha256` map.

The freeze independently binds both predecessor amendment commit/path/hash
triples, the exact twelve-path predecessor absence audit, and the fact that no
v3.3.4/v3.3.4.1 model or analyzer attempt was consumed. It contains no hash
of a predecessor implementation draft.

The dependency remains acyclic: no inventoried source hard-codes the final
freeze hash. After the 12 sources and freeze are committed, an independent
auditor supplies authorized HEAD, freeze SHA-256, and freeze size only through
exactly `V3342_AUTHORIZED_GIT_HEAD`, `V3342_AUTHORIZED_FREEZE_SHA256`, and
`V3342_AUTHORIZED_FREEZE_SIZE_BYTES`. Gate A, Gate B, START, every
terminal, and the analyzer require exact current HEAD, clean tracked diff,
live freeze equal to its Git blob, authorized SHA/size, and all 108 live and
Git-blob hashes equal to `file_sha256`.

## 7. Required CPU tests and stop rules

Before any scoped commit, tests must use isolated temporary roots or mocks and
must cover at least:

1. each of the five exact ordinary failure stages through real runner-shaped
   control flow;
2. all three graph extractions occurring before the first graph publication;
3. zero graph files for each extraction-stage failure and exact three graph
   files for each diagnostic-construction failure;
4. parser/fingerprint/cache failures with a constructible inherited artifact
   routing to the inherited diagnostic controlled stop, not the new terminal;
5. a real publication failure at each graph and new-terminal publication
   routing only to the v3.3.4.1 failure schema;
6. exact six-file and ten-file terminal memberships, modes, directory paths,
   lstat rejection, and tree framing;
7. exact 58-key schema, all nested schemas, stage-dependent nullability,
   signature/source/same-object/cache/prior bindings, and canonical bindings;
8. lower/compile/success counts `(1,1,1)`, zero apply/raw/journal counts, and no
   second extraction/lower/compile/dispatch/retry;
9. exhaustive tamper of every bound source, graph, preflight, cache, prior,
   terminal, stage enum, count, path, hash, size, mode, and claim flag;
10. append-only analyzer START/token/linkage, fresh destination checks, final
    TOCTOU checks, exact analyzer status/decision, and rejection of a
    terminal-less prefix before attempt consumption;
11. JAX/model/scientific/confirmation import and path guards in the analyzer;
12. inherited exact validated-prefix behavior for all signature-attestation
    failure positions and runner-shaped second-lower and second-compile guarded
    failure serializers; and
13. dry-run exact plan of 20 recipients, four anchors, 80 records, 320 applies,
    one eight-row executable, zero six-row/identity/main-cube/confirmation,
    while creating none of the six production paths.

Any schema mismatch, predecessor presence, source drift, cache input, device or
runtime mismatch, unexpected graph membership, nonfinite provenance field,
publication ambiguity, extra path, confirmation access, or scientific field
stops before dispatch. No test, dry run, analyzer, or infrastructure terminal
authorizes a biological claim. Exactly one development-only GPU attempt may be
considered only after the complete v3.3.4.2 bundle is frozen, committed,
independently audited, clean, and explicitly authorized in a later step.

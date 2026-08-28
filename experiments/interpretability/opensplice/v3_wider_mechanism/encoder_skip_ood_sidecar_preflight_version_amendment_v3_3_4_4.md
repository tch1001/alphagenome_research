# OpenSplice encoder-skip OOD sidecar v3.3.4.4: preflight-version repair

Status: prospective infrastructure-only amendment. No v3.3.4.4 code,
preflight, model, GPU, scientific output, or confirmation access is authorized
until this exact document is committed and a separately frozen implementation
passes independent review.

This amendment inherits the scientific design and claim boundary of v3.3.4,
v3.3.4.1, v3.3.4.2, and v3.3.4.3 byte for byte. It repairs one literal
external-preflight version comparison and records the consumed v3.3.4.3
preflight prefix. It does not change the variants, order, readout, model,
checkpoint, reference, compiler/source gates, publication protocol, 80-record
/ 320-apply budget, or structural-only analysis boundary.

## 1. Frozen predecessor and consumed failure

The implementation attempt was made from tracked-clean commit
`ea486661ffe64d5640485ebb2a3ca297e128530a` with freeze:

- path:
  `experiments/interpretability/opensplice/encoder_skip_ood_sidecar_v3_3_4_3_freeze.json`
- SHA256:
  `713790306dd3d88d735229f497587ab6fe611e435eee3f4ef6b862f7baa3cedc`
- size: `174545` bytes
- mode: `100644`

The exact relevant source bindings are:

- external preflight:
  `35025f6a03ce0e6c8f260706c286545590fe8fc4135b0bd8f2aa7fe84c538810`
- launcher:
  `459d3fd7646f495f4fc39e788d35f04d16dc517ee4541229307a749e62c58458`
- bootstrap/publication helper:
  `35f6e5d35e7a2698da65cdc931e07d2a334ab07d22a2d46837f762b0ba5b9a79`
- shell wrapper:
  `a6fffa175bad23089dca62570a9217cbf23a6eaa90afa0cc6bb85b57ce94b4e2`

The v3.3.4.3 freeze stored
`preflight_script_version="opensplice-device-preflight-v3.3.4.3"`, while
`run_device_preflight_v3_3_4_3.py` defined
`SCRIPT_VERSION="v3.3.4.3"`. Its `_validate_authorized_freeze()` compared
those unequal strings directly. In `run_preflight()`, that validator is the
first operation; it runs before external-cache registration, the formal
renameat2 probe, preflight-directory allocation, log creation, or the lazy
JAX/device observation.

The only durable v3.3.4.3 production entry is therefore the launcher-created
external-preflight cache root:

`experiments/interpretability/opensplice/results/v3_3_4_3_preflight_kernel_cache`

It is an exact directory-only tree:

| path relative to cache root | type | mode | st_dev | st_ino | st_nlink | size |
|---|---|---:|---:|---:|---:|---:|
| `.` | directory | `0700` | 66307 | 140791433 | 4 | 4096 |
| `triton` | directory | `0700` | 66307 | 140791434 | 2 | 4096 |
| `xdg` | directory | `0700` | 66307 | 140791435 | 2 | 4096 |

The inherited directory-aware D/F framing gives:

- `directory_paths=[".","triton","xdg"]`
- `directory_count=3`
- `files={}` and `file_count=0`
- tree SHA256:
  `9162636192082efbef80c9b37dd3ebc138aa094f70111874b9dad70e468af1af`
- canonical JSON binding SHA256:
  `d53f56fabd83cb43b79d7a7c73c5b56727d846713e96ea73ef9b26360e18bdea`
- canonical JSON binding size: `745` bytes

The five other v3.3.4.3 production roots are absent: device preflight, model
kernel cache, model run, analysis attempt, and analysis output. All eighteen
v3.3.4, v3.3.4.1, and v3.3.4.2 production roots remain absent. There is no
preflight record or log, publication-probe file, model START, compiler file,
dispatch journal, raw record, terminal, analysis attempt, or analysis output.
There was no JAX observation, device query, model/checkpoint/reference access,
GPU apply, scientific calculation, or confirmation access.

The following traceback was captured by the coordinator from the failed
process and was **not persisted to disk**. No session identifier or wall-clock
timestamp is available beyond the task transcript. The exact UTF-8 block
below, including its final newline, has SHA256
`8b6d0f7575adfc66032ba56d3ab5373f05b0bb1e85b2e39c94e2a82a356e39e9`
and size `953` bytes:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py", line 352, in <module>
    main()
    ~~~~^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py", line 345, in main
    path, passed = run_preflight()
                   ~~~~~~~~~~~~~^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py", line 251, in run_preflight
    freeze = _validate_authorized_freeze()
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py", line 186, in _validate_authorized_freeze
    raise ValueError(f'v3.3.4.3 preflight freeze mismatch: {name}.')
ValueError: v3.3.4.3 preflight freeze mismatch: preflight_script_version.
```

This prefix is consumed. It must not be deleted, renamed, modified, reused as
cache input, or treated as a completed preflight. The v3.3.4.3 launcher,
preflight, model run, and analyzer must not be invoked again.

## 2. Authorized repair and fresh paths

One new version, v3.3.4.4, may repair only the version-field mismatch. It must
use distinct constants:

- `SCRIPT_VERSION="v3.3.4.4"` for the model-run/attempt version;
- `PREFLIGHT_SCRIPT_VERSION="opensplice-device-preflight-v3.3.4.4"` for the
  external-preflight executable and freeze field.

The preflight validator must compare the freeze's
`preflight_script_version` to `PREFLIGHT_SCRIPT_VERSION`, never to
`SCRIPT_VERSION`. The preflight record's `script_version` is the exact
preflight-script literal; model START and model terminals use the model-run
literal. Tests and the freeze bind both strings independently. Before any
v3.3.4.4 cache or other production root is allocated, stdlib-only Gate A must
also prove this three-way equality:

```text
freeze.preflight_script_version
== bootstrap.PREFLIGHT_SCRIPT_VERSION
== the direct literal PREFLIGHT_SCRIPT_VERSION assignment in the frozen
   run_device_preflight_v3_3_4_4.py source
== "opensplice-device-preflight-v3.3.4.4"
```

Gate A verifies the producer literal by an exact source hash plus a
stdlib-only AST check; it does not import JAX or the producer. The child
preflight repeats the same three-way check before cache registration or any
other operation. This check is not deferred until after external-cache
allocation.

The one authorized v3.3.4.4 attempt uses these six fresh roots:

```text
results/v3_3_4_4_preflight_kernel_cache
results/v3_3_4_4_device_preflight
results/v3_3_4_4_model_kernel_cache
results/v3_3_4_4_development_ood_sidecar_one_shot
results/v3_3_4_4_development_ood_sidecar_analysis_attempt
results/v3_3_4_4_development_ood_sidecar_analysis
```

All six must be absent before implementation freeze, post-commit dry-run, and
the sole launch. The v3.3.4.3 consumed cache root is not a cache source for
v3.3.4.4. No v3.3.4.4 process may open any file from it as compiler input.

## 3. Exact predecessor-prefix attestation

The v3.3.4.4 freeze adds exactly one top-level object named
`prior_v3_3_4_3_consumed_preflight_prefix`. It has the following exact
13-key schema and values:

```text
status, predecessor_commit, predecessor_freeze,
failure_stage, failure_type, failure_message,
traceback_provenance, cache_tree_binding, directory_lstat_rows,
other_predecessor_paths_absent, no_jax_or_model_access,
no_gpu_or_confirmation_access, immutable_and_not_cache_input
```

`status="consumed_external_preflight_freeze_validation_failure"` and
`predecessor_commit` is the exact commit in Section 1. `predecessor_freeze`
has exactly:

```text
path, sha256, size_bytes, git_mode,
top_level_key_count, file_sha256_count, source_row_count
```

Its literal values are the absolute freeze path, SHA/size/mode in Section 1,
and `(84,108,108)` for the three counts. `failure_stage` is
`preflight_freeze_validation`, `failure_type` is `ValueError`, and
`failure_message` is
`v3.3.4.3 preflight freeze mismatch: preflight_script_version.`.
`traceback_provenance` has exactly:

```text
storage, sha256, size_bytes, session_id,
captured_at_unix_s, wall_clock_timestamp_available
```

Its values are
`("coordinator_captured_not_persisted",8b6d0f7575adfc66032ba56d3ab5373f05b0bb1e85b2e39c94e2a82a356e39e9,953,null,null,false)`.
No claim that these unpersisted bytes were emitted as a durable artifact is
permitted.

`cache_tree_binding` has exactly:

```text
cache_role, cache_root, triton_cache_dir, xdg_cache_home,
directory_paths, directory_count, files, file_count, tree_sha256,
default_user_cache_paths_eligible,
diagnostic_outputs_only_no_cache_input
```

Its values are the absolute root/child paths in Section 1,
`cache_role="external_preflight"`, the exact directory/file values and tree
digest there, and `(false,true)` for the final two booleans.
`directory_lstat_rows` is a three-element list in exact order
`.,triton,xdg`. Every row has exactly
`path,entry_type,mode,st_dev,st_ino,st_nlink,size_bytes`, with the literal
values in Section 1. Integers are nonnegative JSON integers (never booleans),
`entry_type="directory"`, and modes are four-character strings.

Across this prefix object, statuses, paths, hashes, failure text, entry types,
and Git/filesystem modes are JSON strings; sizes, counts, device/inode/link
values are JSON integers and never booleans; the three no-access/immutability
fields and every `absent` field are JSON booleans. The only nulls are the two
explicit unavailable traceback metadata values. Every digest is 64 lowercase
hexadecimal characters and every absolute path is normalized and contained
under the frozen repository/result roots.

`other_predecessor_paths_absent` is an exact 23-key map. Its keys are:

```text
v3_3_4.device_preflight, v3_3_4.external_cache,
v3_3_4.model_cache, v3_3_4.model_run,
v3_3_4.analysis_attempt, v3_3_4.analysis_output,
v3_3_4_1.device_preflight, v3_3_4_1.external_cache,
v3_3_4_1.model_cache, v3_3_4_1.model_run,
v3_3_4_1.analysis_attempt, v3_3_4_1.analysis_output,
v3_3_4_2.device_preflight, v3_3_4_2.external_cache,
v3_3_4_2.model_cache, v3_3_4_2.model_run,
v3_3_4_2.analysis_attempt, v3_3_4_2.analysis_output,
v3_3_4_3.device_preflight, v3_3_4_3.model_cache,
v3_3_4_3.model_run, v3_3_4_3.analysis_attempt,
v3_3_4_3.analysis_output
```

Each value has exactly `path,absent`; `path` is the normalized absolute path
already frozen by the predecessor/current path contracts and `absent=true`.
The v3.3.4.3 external cache is deliberately not in this absent map. The three
final flags `no_jax_or_model_access`, `no_gpu_or_confirmation_access`, and
`immutable_and_not_cache_input` are all true.

The associated prefix content binding has exactly `sha256,size_bytes` and is
computed from exactly:

```python
json.dumps(
    prefix, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False, allow_nan=False,
).encode('utf-8')
```

Before allocating a v3.3.4.4 path, stdlib-only Gate A must rehash the exact
v3.3.4.3 commit/freeze/source bytes, lstat the consumed cache root and all
three rows without following links, reproduce the directory-aware tree and
canonical binding digests, reject every extra/missing/symlink/special entry,
and reprove all named absences. It then repeats the same predecessor check
after v3.3.4.4 START as part of Gate B.

The exact predecessor-prefix object and its canonical content binding must be
persisted in the successful v3.3.4.4 external-preflight record, model START,
every model terminal, and the structural analyzer result. The analyzer must
independently revalidate the live prefix and its frozen literal before reading
any v3.3.4.4 compiler/raw artifact. A drift consumes the new attempt and fails
closed; it never licenses cleanup or retry.

### 3.1 Revised exact record schemas

The two new field names are always:

```text
prior_v3_3_4_3_consumed_preflight_prefix
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

They are nonnull and exact at every phase below. Their addition supersedes
only the inherited top-level key counts; every inherited key and nested
schema remains unchanged.

The external-preflight pass and persisted-failure record has exactly 20 keys:

```text
amendment_sha256, atomic_publication_probe, created_at_unix_s,
external_freeze_authorization, external_cache_post_observation,
external_cache_hit_evidence, failure, freeze, freeze_sha256, logs,
no_jit_or_array_kernel, no_model_or_biological_access, observation,
original_protocol_sha256, preflight_attempt_number, script_version,
status, warnings,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

Pass/fail nullability otherwise remains inherited. A failure before the
preflight root is allocated cannot fabricate this record; it preserves a
terminal-less consumed v3.3.4.4 cache prefix and forbids retry.

Model `ATTEMPT_STARTED.json` has exactly 36 keys:

```text
status, attempt_id, script_version, amendment_sha256, amendment_commit,
original_protocol_sha256, freeze_path, freeze_sha256, git_head,
external_freeze_authorization, runner_pid, parent_pid, started_at_unix_s,
successful_preflight, same_process_preflight,
same_process_preflight_content_binding, fresh_paths, budgets,
execution_contract, source_inventory_attestation, prior_v3_3_3_binding,
prior_v3_3_3_1_archive_binding, source_input_audit,
source_input_audit_content_binding, program_signature_contract,
cache_isolation_contract, confirmation_scope_disclosure,
confirmation_model_calls, scientific_summary_computed,
donor_normalization_computed, shapley_or_nomination_computed,
interaction_or_resolution_computed, nomination_performed,
combined_analysis_permitted,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

`POST_START_PROVENANCE_FAILURE.json` has exactly 22 keys:

```text
status, stop_reason, message, failure, attempt_id, script_version,
amendment_sha256, freeze_sha256, git_head, external_freeze_authorization,
runner_pid, source_inventory_failure, model_constructed, model_apply_count,
source_input_audit, source_input_audit_content_binding,
confirmation_model_calls, scientific_summary_computed,
combined_analysis_permitted, failed_at_unix_s,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

Every ordinary `RUN_COMPLETE.json`, including complete and controlled-stop
rows, has exactly the inherited ordered 64 keys below plus the two new keys,
for 66 total:

```text
status, stop_reason, message, failure, attempt_id, script_version,
amendment_sha256, amendment_commit, original_protocol_sha256,
freeze_sha256, git_head, external_freeze_authorization, runner_pid,
started_at_unix_s, completed_at_unix_s, phase_state, terminal_detail,
budgets, source_input_audit, source_input_audit_content_binding,
same_object_attestation, same_object_attestation_content_binding,
program_signature_attestation_binding, source_program_gate,
compiler_binding, compiler_artifact_bindings, attempt_budget_audit,
diagnostic_provenance_complete, compiled_backend_diagnostic_only,
backend_diagnostics, diagnostic_comparisons, dispatch_journal, raw_manifest,
preterminal_tree_binding, valid_record_count, failed_current_binding,
model_apply_attempt_count, model_apply_success_count,
expected_model_apply_count, eight_row_lower_attempt_count,
eight_row_compile_attempt_count, eight_row_successful_compile_count,
six_row_compile_count, identity_rerun_count, main_cube_rerun_count,
old_ood_records_reused, confirmation_model_calls,
all_80_recipient_anchors_complete, id0_all20, id255_all20,
import_provenance_phases, protobuf_provenance_sha256,
model_kernel_cache_final, prior_v3_3_3_binding,
prior_v3_3_3_1_archive_binding, confirmation_scope_disclosure,
publication_audit, scientific_summary_computed,
donor_normalization_computed, shapley_or_nomination_computed,
interaction_or_resolution_computed, nomination_performed,
combined_analysis_permitted, no_retry,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

Publication `TERMINAL_FAILURE.json` has exactly 33 keys:

```text
schema_version, status, stop_reason, attempt_id, script_version,
external_freeze_authorization, runner_pid, publication_failure,
preterminal_tree_binding, source_input_audit,
source_input_audit_content_binding, same_object_attestation,
same_object_attestation_content_binding, phase_state,
model_apply_attempt_count, model_apply_success_count, valid_record_count,
failed_current_binding, temporary_orphan_bindings,
durability_uncertain_final_bindings, preexisting_entry_states,
no_new_entry_failure, confirmation_model_calls,
scientific_summary_computed, donor_normalization_computed,
shapley_or_nomination_computed, interaction_or_resolution_computed,
nomination_performed, combined_analysis_permitted, no_retry,
created_at_unix_s, prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

`NONPUBLICATION_TERMINAL_FAILURE.json` has the inherited 58-key order plus
the two new fields, for exactly 60 keys:

```text
schema_version, status, stop_reason, attempt_id, script_version,
amendment_commit, amendment_sha256, inherited_v3_3_4_commit,
inherited_v3_3_4_sha256, inherited_v3_3_4_1_commit,
inherited_v3_3_4_1_sha256, freeze_sha256, git_head,
external_freeze_authorization, runner_pid, started_at_unix_s,
created_at_unix_s, failure_stage, failure,
triggering_diagnostic_failure, triggering_diagnostic_stop_reason,
phase_state, source_input_audit, source_input_audit_content_binding,
program_signature_attestation_binding, same_object_attestation,
same_object_attestation_content_binding, attempt_budget_audit,
compiler_counts, graph_artifact_bindings, import_provenance_phases,
protobuf_provenance_sha256, model_kernel_cache_state,
source_program_gate_without_backend_diagnostics,
source_program_gate_without_backend_diagnostics_content_binding,
prior_v3_3_3_binding, prior_v3_3_3_1_archive_binding,
preterminal_tree_binding, publication_audit, model_apply_attempt_count,
model_apply_success_count, valid_record_count, raw_record_count,
dispatch_started_count, dispatch_completed_count, six_row_compile_count,
identity_rerun_count, main_cube_rerun_count, old_ood_records_reused,
confirmation_model_calls, confirmation_scope_disclosure,
scientific_summary_computed, donor_normalization_computed,
shapley_or_nomination_computed, interaction_or_resolution_computed,
nomination_performed, combined_analysis_permitted, no_retry,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

The structural analyzer's `ANALYSIS_ATTEMPT_STARTED.json` has exactly 16 keys:

```text
status, analysis_version, attempt_id, acknowledgement, git_head,
freeze_sha256, external_freeze_authorization, analyzer_binding,
test_binding, run_root, run_terminal_binding, fresh_output_dir,
old_analyzer_destinations_absent, started_at_unix_s,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

`ANALYSIS.json` has exactly 25 keys:

```text
status, decision, analysis_version, analysis_attempt_start_binding,
run_binding, preflight_binding, model_cache_binding, source_and_prior_audit,
compiler_and_signature_audit, dispatch_journal_audit, raw_prefix_audit,
control_audit, terminal_audit, publication_audit, confirmation_boundary,
claim_boundary, scientific_summary_computed, donor_normalization_computed,
shapley_or_nomination_computed, interaction_or_resolution_computed,
nomination_performed, combined_analysis_permitted, completed_at_unix_s,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

Within its `source_and_prior_audit`, the inherited
`current_108_source_rows_exact` key is prospectively renamed
`current_120_source_rows_exact`, and a new
`prior_v3_3_4_3_consumed_preflight_prefix_exact` key is added. That nested
object therefore has exactly 12 keys:

```text
current_120_source_rows_exact, historical_96_source_rows_exact,
git_head_exact, tracked_clean, external_freeze_authorization_exact,
prior_v3_3_3_exact, prior_v3_3_3_1_exact,
old_analyzer_paths_absent, pre_start_exact, post_start_exact, final_exact,
prior_v3_3_4_3_consumed_preflight_prefix_exact
```

All twelve values are independently derived true in a successful analyzer
record. This is the only revised analyzer nested keyset; other inherited
nested analyzer schemas remain unchanged.

`ANALYSIS_FAILURE.json` has exactly 17 keys:

```text
status, attempt_id, analysis_attempt_start_binding, type, message,
traceback, raw_values_read, scientific_analysis_performed, output_dir_state,
publication_failure, temporary_orphan_bindings,
durability_uncertain_final_bindings, preexisting_entry_states,
no_new_entry_failure, failed_at_unix_s,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding
```

The analyzer's existing `ANALYSIS_COMPLETE.json` remains its exact nine-key
link record because its `analysis_binding` already binds the 25-key result;
no unbound prefix claim is made there.

There is deliberately **no standalone CPU prefix-archive invocation** and no
additional archive attempt/output path. The committed amendment is the
human-readable failure record; the v3.3.4.4 freeze is the exact machine
archive, and every later preflight/model/analyzer artifact rebinds it as
specified above. The inapplicable v3.3.4.3 analyzer and all of its destinations
remain uninvoked and absent. No result is called a completed standalone
archive.

## 4. Freeze and source dependency graph

The v3.3.4.4 freeze is derived from the exact v3.3.4.3 freeze. It changes only
the version/path fields, adds the predecessor-prefix object, and adds the
versioned implementation/amendment sources. The expected schema is:

- 85 top-level keys;
- 120 `file_sha256` entries and 120 source-inventory rows: the inherited 108
  plus exactly eleven v3.3.4.4 source/test/shell files and this amendment;
- exactly two new shell wrappers at Git mode `100755`; the other nine new
  implementation/test files and this amendment at `100644`;
- the new freeze itself tracked at `100644` and omitted from its own content
  inventory to keep the dependency graph acyclic.

The exact twelve new inventory paths are:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_4.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_4.sh
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_4_test.py
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_4_freeze.py
experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_4.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_4.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_4_test.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_4.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_4.sh
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_4_test.py
experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4.py
experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_preflight_version_amendment_v3_3_4_4.md
```

No inventoried source hard-codes the final freeze digest. After a scoped
commit, an external authorization supplies and binds the actual commit,
freeze SHA256, and freeze size. Gate A, external preflight, START, Gate B,
every terminal, and the analyzer bind that authorization exactly.

Except for the explicitly revised top-level record keysets and the one
analyzer nested audit keyset in Section 3.1,
the implementation bundle retains the exact v3.3.4.3 nested schemas,
publication helper, source-program gate, canonical 32-path signature adapter,
cache isolation, import/protobuf inventories, one-lower/one-compile budget,
dispatch ledger, failure prefixes, terminal matrix, and structural analyzer.

## 5. Attempt order and stop rules

The only legal production order is:

1. globally tracked-clean HEAD and external freeze authorization;
2. stdlib-only Gate A, including the exact consumed-prefix audit and the
   three-way preflight-version proof, before any v3.3.4.4 allocation;
3. allocate a fresh v3.3.4.4 external cache;
4. launch exactly `preflight_0000` in a distinct process;
5. repeat the exact three-way preflight-version proof before child cache
   registration;
6. run the inherited formal renameat2 publication probe;
7. run the JAX-only GPU/device observation and persist the exact five-file
   preflight tree;
8. allocate a distinct, empty v3.3.4.4 model cache;
9. run the same-process GPU gate; persist START once;
10. repeat Gate B and predecessor-prefix checks;
11. import the frozen runner, lower once, compile once, and if all gates pass,
    run the unchanged 80-record / 320-apply sidecar;
12. invoke the versioned CPU structural analyzer at most once after an exact
    terminal exists.

Any failure consumes the relevant v3.3.4.4 prefix. There is no deletion,
resume, overwrite, cache reuse, second preflight, second lower/compile, or
retry-until-match. If source-program equality fails, the run controlled-stops
at zero applies. Compiled backend HLO remains diagnostic rather than an
equality gate. Publication failures follow the inherited v3.3.4.1 schemas;
ordinary graph/diagnostic construction failures follow v3.3.4.2/v3.3.4.3.

## 6. Scientific contract and claim boundary

The scientific contract is unchanged:

- development-only frozen 20 recipients in their existing order;
- anchors `0,127,128,255` for every recipient;
- exactly 80 records and at most 320 applies;
- exactly one eight-row executable, zero six-row executable, zero main-cube
  rerun, zero identity rerun, and zero reused v3.3.2 OOD records;
- identical intended/unrelated/repeat donors, invariant-row, trace,
  fingerprint, ID0 and ID255 closure checks;
- zero confirmation model calls, activations, interventions, or outputs.

The GPU process computes and persists structural evidence only. It does not
compute donor normalization, Shapley values, interactions, resolution ranks,
nomination, or a combined analysis. The CPU analyzer is also structural-only.
Even a complete v3.3.4.4 run supports only a tooling/localization observation;
biological or mechanistic claims require a separately prospective scientific
protocol and confirmation remains blind.

## 7. Required tests and launch gate

Before a scoped implementation commit, CPU tests must prove:

1. the v3.3.4.3 mismatch is reproduced and the two v3.3.4.4 version literals
   are independently validated before any allocation and again in the child;
   substituting `SCRIPT_VERSION` for `PREFLIGHT_SCRIPT_VERSION` fails;
2. every consumed-prefix file/path/lstat/hash/size/mode/device/inode/link/tree
   field rejects tampering, extra entries, missing entries, symlinks, special
   files, and directory replacement;
3. all six v3.3.4.4 roots are fresh and the five absent v3.3.4.3 roots plus
   eighteen earlier roots remain absent;
4. preflight validation occurs before cache registration, publication probe,
   preflight allocation, JAX import/device observation, and model access;
5. all exact revised record keysets/counts in Section 3.1 are serializer- and
   analyzer-tested; preflight/START/Gate B/terminal/analyzer predecessor-prefix
   bindings are exact and their live rechecks precede scientific reads;
6. the complete inherited publication-fault, terminal-matrix, source-gate,
   signature, cache, import/protobuf, dispatch-prefix, and analyzer suites
   remain green;
7. runner-shaped complete and every controlled-stop path retain exact count,
   membership, no-retry, no-science, and no-confirmation predicates;
8. freeze generation is deterministic at 85/120/120 and all paths, bytes,
   sizes, and Git modes are exact;
9. the committed wrapper dry-run creates none of the six new roots and reports
   20 recipients, 80 records, 320 applies, one eight-row compile, zero six-row
   compiles, and zero confirmation calls.

Only after exact stopped hashes, independent read-only review, a scoped commit,
tracked-clean post-commit bootstrap, and a successful committed dry-run may
the coordinator authorize **one** v3.3.4.4 development-only GPU launch.

# Prospective v3.3.4.1 amendment: no-replace named-temp publication

Status: prospective and unconsumed. This document authorizes no model, GPU,
preflight, analyzer, or scientific run by itself.

Confirmation boundary: later-exon metadata and labels have been exposed, but
later-exon model outputs, activations, and interventions remain unopened and
must remain so.

## 1. Binding, reason, and narrow scope

This amendment binds and otherwise preserves the complete prospective v3.3.4
protocol:

- Git commit `f833a8d2108636871abfce8b4cbabe4255536974`;
- file `experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_infrastructure_amendment_v3_3_4.md`;
- SHA-256
  `38d07c0b612e50aadc64ba18537561cbdb0489b67fd0824cae749bba6214207b`.

It supersedes only the append-only publication primitive, the versioned path
namespace, and the structural provenance needed to audit that primitive. The
following v3.3.4 requirements are unchanged:

- the same 20 frozen development recipients and four anchors in the same
  recipient-major, anchor-minor order;
- exactly one eight-row executable, zero six-row executables, no identity or
  main-cube rerun, and no old OOD result reuse;
- at most one lowering and one compilation attempt, with no compilation-cache
  input and no retry;
- the exact 32-path, tuple-to-list-only program-signature adapter and the exact
  canonical 2,877-byte payload with SHA-256
  `d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300`;
- exact StableHLO, pre-backend HLO, entry ABI, checkpoint, reference, runtime,
  device, toolchain, protobuf, import-inventory, source-input, same-object,
  diagnostic-provenance, cache-isolation, and confirmation gates;
- exactly 80 valid records and 320 started and completed model calls for a
  complete sidecar;
- the exact four-call dispatch journal, failed-current lossless encoding,
  invariant-row, repeat, intended-donor, unrelated-donor, ID0, and ID255
  structural controls;
- no GPU-side scientific summary, normalization, Shapley value, interaction,
  resolution result, rank, or nomination; and
- `combined_analysis_permitted=false` for every terminal and analyzer result.

This is an infrastructure repair, not a relaxation of any scientific gate.

### 1.1 Why v3.3.4 must not be launched

An informal, non-persisted host diagnostic was performed during the task on
2026-08-28 in Asia/Singapore. No exact wall-clock timestamp was recorded, so
none is asserted here. The diagnostic used `/usr/bin/python3`, `ctypes`, and a
`tempfile.TemporaryDirectory(prefix='v334-atomic-probe.')` and observed:

- Linux `6.8.0-136-generic #136~22.04.1-Ubuntu`, x86-64;
- glibc `2.35-0ubuntu3.14`;
- `stat -f -c %T` reported `ext2/ext3` for both `/tmp` and
  `/home/degen2/alphafold-stuff/alphagenome_research`;
- `os.open(root, O_RDWR|O_TMPFILE, 0600)` succeeded;
- the diagnostic wrote the exact five bytes `anon\n`, called `fsync`, and set
  mode `0400`;
- `libc.linkat(anonymous_fd, b'', directory_fd,
  b'exact-empty-path', AT_EMPTY_PATH=0x1000)` returned `-1` with errno `2`,
  `ENOENT`; the destination remained absent;
- the diagnostic-only `/proc/self/fd/<fd>` plus
  `AT_SYMLINK_FOLLOW=0x400` form succeeded, but that fallback remains
  forbidden;
- a same-directory named temporary file published with
  `libc.renameat2(directory_fd, b'named.tmp', directory_fd,
  b'named.final', RENAME_NOREPLACE=1)` succeeded; a collision returned `-1`
  with `EEXIST`, while source and destination bytes remained unchanged; and
- the temporary context removed the diagnostic files after the diagnostic.

This diagnostic was not a frozen v3.3.4 external preflight and is not evidence
that v3.3.4 was consumed. It has no durable artifact hash and must never be
represented as though it did. It establishes only the prospective motivation
for this amendment. A formal v3.3.4.1 external preflight must independently
test the replacement primitive.

At amendment freeze, scoped commit, post-commit audit, and immediately before
any v3.3.4.1 invocation, all six v3.3.4 production paths must be absent and
non-symlinked:

```text
experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_one_shot
experiments/interpretability/opensplice/results/v3_3_4_device_preflight
experiments/interpretability/opensplice/results/v3_3_4_preflight_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_model_kernel_cache
experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_analysis
experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_analysis_attempt
```

Any presence consumes or contaminates the predecessor state and is a blocker;
it cannot be silently removed. No v3.3.4 production launch is permitted.

## 2. Fresh v3.3.4.1 namespace and one-shot lifecycle

The prospective identifiers are exactly:

```text
script_version=opensplice-encoder-skip-ood-sidecar-v3.3.4.1
preflight_script_version=opensplice-device-preflight-v3.3.4.1
analysis_version=opensplice-encoder-skip-ood-sidecar-analysis-v3.3.4.1
attempt_id=opensplice-v3.3.4.1-development-ood-sidecar-one-shot
publication_schema_version=v3.3.4.1-named-temp-renameat2-noreplace-v1
```

The only production paths are:

```text
model_run=experiments/interpretability/opensplice/results/v3_3_4_1_development_ood_sidecar_one_shot
external_preflight=experiments/interpretability/opensplice/results/v3_3_4_1_device_preflight
external_cache=experiments/interpretability/opensplice/results/v3_3_4_1_preflight_kernel_cache
model_cache=experiments/interpretability/opensplice/results/v3_3_4_1_model_kernel_cache
analysis_output=experiments/interpretability/opensplice/results/v3_3_4_1_development_ood_sidecar_analysis
analysis_attempt=experiments/interpretability/opensplice/results/v3_3_4_1_development_ood_sidecar_analysis_attempt
```

The only external freeze-authorization environment names are:

```text
V3341_AUTHORIZED_GIT_HEAD
V3341_AUTHORIZED_FREEZE_SHA256
V3341_AUTHORIZED_FREEZE_SIZE_BYTES
```

Any other `V3341_AUTHORIZED_*` name is forbidden. No v3.3.4 authorization
variable may substitute for a v3.3.4.1 variable.

The lifecycle remains one shot:

1. a standard-library Gate A proves a clean tracked HEAD, exact frozen source
   bytes and prior archives, external freeze authorization, and absence of all
   six v3.3.4 and all six v3.3.4.1 paths;
2. the launcher creates the fresh external cache and performs exactly one
   external allocation, `preflight_0000`;
3. only a complete passing five-file preflight permits allocation of the
   distinct fresh model cache;
4. the same process performs the JAX-only same-device gate, publishes START,
   repeats the standard-library source/prior Gate B, and only then imports the
   scientific runner;
5. the runner performs the unchanged model lifecycle; and
6. the CPU-only structural analyzer is separately prospective, has fresh
   attempt/output paths, and may be invoked at most once after an eligible
   immutable terminal.

No path may be resumed, reused, cleaned, or retried.

## 3. Exact no-replace publication primitive

### 3.1 One implementation and allowed calls

There is one standard-library publication helper, defined in the versioned
bootstrap module and used by the preflight, launcher, model runner, and CPU
analyzer. No caller implements a second serializer. The helper uses only a
named temporary file and same-directory `renameat2(RENAME_NOREPLACE)`.

The following alternatives are forbidden:

- `O_TMPFILE`, `linkat(AT_EMPTY_PATH)`, or `/proc/self/fd` publication;
- `os.rename`, `os.replace`, ordinary `renameat`, hard-link-plus-unlink, copy,
  truncate, overwrite, or delete-and-recreate fallbacks;
- cross-directory or cross-filesystem publication;
- retrying after a temporary-name collision, final-name collision, short
  write failure, durability failure, or any syscall error; and
- deletion or reuse of a published final, a failed temporary file, or a
  durability-uncertain final.

Every final file is published from bytes already fully serialized in memory,
except preflight stdout/stderr streams, whose anonymous-in-memory replacement
is the exact open named temporary descriptor described in Section 3.4.

### 3.2 Path and temporary-name grammar

The caller passes an already frozen root role, a directory beneath that root,
an ASCII final basename, exact payload bytes, and a monotonically increasing
publication ordinal. The final basename must match `[A-Za-z0-9_.-]+`, contain
no slash, and be neither `.` nor `..`. Directory traversal and absolute final
paths are forbidden.

The helper generates exactly one nonce using `secrets.token_hex(16)`. It must
be 32 lowercase hexadecimal characters. It makes exactly one temporary-name
attempt, with basename:

```text
.v3341.tmp.{runner_pid_decimal}.{publication_ordinal:06d}.{nonce32hex}
```

The process-local ordinal counter is initialized to zero. For an artifact, the
helper selects the current value and increments the counter before its first
filesystem syscall; therefore the first emitted or failed attempt has ordinal
zero. Every failed attempt consumes its ordinal. The counter is never reset or
reused within the process. An `O_EXCL` collision is a consumed failure; no
second nonce or ordinal is attempted for the same artifact.

All operations use basenames relative to one already-open parent-directory
file descriptor. The parent is opened with
`O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, must be a regular directory
within the frozen root, must have mode `0700`, and must be on the expected
device. The temporary and final names share that exact parent descriptor.

Every required production directory is created once with mode `0700`, is
immediately `fstat`-validated through its parent descriptor, and is followed
by `fsync` of that parent. `EEXIST` is failure even when the existing entry is
a directory. Empty or partially populated directories are preserved after a
failure and make the version consumed; they are never removed and recreated.

### 3.3 Required syscall sequence

For a byte artifact, the exact sequence is:

1. reject a pre-existing final entry of any type using a no-follow directory
   lookup; `ENOENT` is the only accepted absence result;
2. call `openat(parent_fd, temp_basename,
   O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, 0600)` exactly once;
3. `fstat` the returned descriptor and require a regular file, link count one,
   mode `0600`, and the same device as the parent;
4. write the complete payload with a short-write loop; a zero-progress write
   is failure;
5. call `fsync(temp_fd)`;
6. call `fchmod(temp_fd, 0400)` and then `fsync(temp_fd)` again;
7. while retaining the open descriptor, seek to offset zero, read back all
   bytes, and require exact size and SHA-256 equality with the serialized
   payload;
8. call
   `renameat2(parent_fd, temp_basename, parent_fd, final_basename,
   RENAME_NOREPLACE=1)` exactly once;
9. require the temporary basename to be absent and the final basename to be a
   regular, non-symlinked file with the same `st_dev` and `st_ino` as the open
   descriptor, link count one, mode `0400`, and exact size and SHA-256;
10. call `fsync(parent_fd)`; and
11. repeat the final no-follow type, inode, mode, size, and hash checks before
    closing both descriptors.

The implementation must set and inspect `ctypes` argument and result types and
capture errno immediately after `renameat2`. The only successful return is
zero. `EEXIST` proves no replacement but is still a consumed publication
failure. `ENOSYS`, `EINVAL`, `EXDEV`, `ENOENT`, `EPERM`, and every other errno
are failures. There is no fallback.

The helper returns an in-memory success object with exactly:

```text
schema_version, method, root_role, final_relative_path,
temp_basename, publication_ordinal, runner_pid, nonce_hex,
sha256, size_bytes, mode, st_dev, st_ino, st_nlink,
file_fsync_before_rename, file_fsync_after_fchmod,
rename_noreplace_succeeded, parent_fsync_succeeded,
post_publish_revalidation_exact
```

`method` is exactly `named_temp_renameat2_noreplace`; `mode` is the JSON string
`"0400"`; `st_dev`, `st_ino`, and `st_nlink` are non-negative JSON integers,
with `st_nlink=1`; all five success booleans are true. This object is not embedded in the
artifact being published and therefore creates no self-hash cycle. The
downstream manifest or terminal binds the final file; the frozen source hash
and formal preflight bind the publication implementation.

### 3.4 Streaming stdout and stderr

Each preflight log opens one named temporary descriptor through steps 1--3,
duplicates that descriptor onto the appropriate stream, and captures bytes.
The ordinary step-4 payload write is omitted for a streaming log; captured
bytes must not be appended or duplicated. Before restoring the original
stream descriptor, the implementation flushes the corresponding Python
stream. It then restores the stream, calls `fsync(temp_fd)`, calls
`fchmod(temp_fd, 0400)`, calls `fsync(temp_fd)` again, seeks to zero, reads all
captured bytes once to compute exact size and SHA-256, and performs ordinary
steps 8--11 without reopening or closing the temp descriptor before rename.
The two log temporaries have separate ordinals and nonces. A log-finalization
failure consumes the version and preserves every entry already published or
left temporary.

### 3.5 Publication failure state

A publication exception carries an exact `publication_failure` object:

```text
schema_version, method, root_role, artifact_role,
final_relative_path, temp_relative_path, publication_ordinal,
runner_pid, failure_stage, errno, error_type, message,
rename_noreplace_attempted, rename_noreplace_succeeded,
parent_fsync_attempted, parent_fsync_succeeded,
temp_state, final_state, created_at_unix_s
```

`failure_stage` is exactly one of:

```text
parent_open, parent_validation, final_preexistence,
temp_open, temp_validation, write, first_file_fsync,
fchmod, second_file_fsync, readback, rename_noreplace,
post_rename_validation, parent_fsync, final_revalidation
```

`errno` is an integer or null when no syscall errno applies. Attempt/success
fields are booleans. `temp_state` and `final_state` each have exactly:

```text
state, entry_type, mode, size_bytes, sha256, st_dev, st_ino, st_nlink
```

`state` is `absent`, `present`, or `unreadable`. When absent, all other fields
are null. When present, `entry_type` is exactly `regular`, `directory`,
`symlink`, `fifo`, `socket`, `block`, `character`, or `other`; regular files
have mode as a four-character octal JSON string, size as a non-negative JSON
integer, SHA-256 as 64 lowercase hexadecimal characters, and device, inode,
and link count as non-negative JSON integers. For a present non-regular entry,
mode/device/inode/link-count remain exact, while size and SHA-256 are null.
For `unreadable`, all fields except `state` are null. No symlink is followed.

There are exactly three preserved failure classes:

- **no-new-entry/pre-existing-entry:** failure occurs before this publication
  attempt creates a temporary or final entry. The temporary state is absent
  or the exact unchanged pre-existing temp entry; the final state is absent,
  unreadable, or the exact unchanged pre-existing final entry.
  `parent_open`, `parent_validation`, `final_preexistence`, and a
  `temp_open` error before successful exclusive creation use this class. A
  temp-name `EEXIST` binds the exact pre-existing temp entry; it is never
  treated as an orphan created by this attempt;
- **temporary orphan:** rename did not succeed and the uniquely named
  temporary remains; the final is absent or is the unchanged pre-existing
  collision target; and
- **durability-uncertain final:** rename succeeded but parent fsync or final
  revalidation failed; the final is preserved and must never be rewritten.

The caller never unlinks any class. If a terminal artifact can still be
published with a later ordinal, it embeds the exact failure object and binds
all observed orphans/finals. If terminal publication also fails, the existing
filesystem prefix is preserved unchanged and this analyzer version must not
invent a terminal.

## 4. Formal v3.3.4.1 external preflight

The external preflight remains JAX-only and performs no JIT, array kernel,
model construction, checkpoint read, reference read, OpenSplice scientific
helper import, or confirmation access. Before importing JAX, it performs the
following standard-library publication probe in the fresh external cache:

1. first require and bind that the new external cache is empty; this is the
   only pre-import cache-input/cache-hit snapshot, and later probe files are
   outputs rather than cache inputs;
2. publish `atomic_publication_probe_v3_3_4_1.txt` with exact bytes
   `opensplice-v3.3.4.1-renameat2-noreplace-probe-v1\n` using Section 3;
3. open a second unique temporary with exact bytes
   `opensplice-v3.3.4.1-collision-probe-v1\n` and attempt to publish it to the
   same final;
4. require `renameat2` to return `-1/EEXIST`;
5. require the first final's inode, mode, size, and bytes to remain unchanged;
6. preserve the collision temporary as a deliberate preflight-cache orphan;
7. call `fsync` on the collision temporary's parent directory and revalidate
   both the final and orphan; and
8. bind the successful final and collision orphan in the preflight record and
   post-observation external-cache tree.

The exact preflight `atomic_publication_probe` object has:

```text
schema_version, method, supported, successful_final_binding,
collision_errno, collision_no_replace_exact,
collision_temp_binding, destination_unchanged,
temp_orphan_preserved, parent_fsync_exact
```

For a pass, `supported`, `collision_no_replace_exact`,
`destination_unchanged`, `temp_orphan_preserved`, and `parent_fsync_exact` are
true; `collision_errno=17`; bindings contain root-relative path, SHA-256,
size, mode, device, inode, and link count.
`parent_fsync_exact=true` means both the successful-final parent fsync and the
post-`EEXIST` collision-orphan parent fsync succeeded and both entries passed
their following revalidation.

Both `successful_final_binding` and `collision_temp_binding` have exactly:

```text
path, sha256, size_bytes, mode, st_dev, st_ino, st_nlink
```

`path` is external-cache-root-relative POSIX text; `sha256` is 64 lowercase
hexadecimal characters; `size_bytes,st_dev,st_ino,st_nlink` are non-negative
JSON integers; `mode` is the JSON string `"0400"`; and `st_nlink=1`.

The collision temporary is an intentionally induced, passing preflight
diagnostic, not a model-run publication failure. It is explicitly identified
by `atomic_publication_probe`, is never reused or deleted, and is excluded
from model cache-hit input. Any other temporary orphan makes the preflight
fail.

The exact complete preflight root is still five files:

```text
.allocation.lock
.preflight_0000.reserved
preflight_0000.json
preflight_0000.stdout.log
preflight_0000.stderr.log
```

All three nonempty files use Section 3. The preflight JSON's exact top-level
key set is the exact v3.3.4 set plus exactly one new key,
`atomic_publication_probe`:

```text
amendment_sha256, atomic_publication_probe, created_at_unix_s,
external_freeze_authorization, external_cache_post_observation,
external_cache_hit_evidence, failure, freeze, freeze_sha256, logs,
no_jit_or_array_kernel, no_model_or_biological_access, observation,
original_protocol_sha256, preflight_attempt_number, script_version,
status, warnings
```

It changes only version strings, fresh paths, authorization, and the exact
probe object defined above. `status` is exactly `pass` or `fail`. A complete
`pass` requires the publication probe, device/runtime gate, cache routing, and
five-file tree all to pass. The launcher independently rehashes the five-file
tree and external-cache tree before allocating the model cache.

If the primitive fails before a complete five-file preflight can be
published, the reserved preflight/cache prefixes and publication failures are
preserved, the wrapper exits nonzero, the model cache and model-run paths stay
absent, and v3.3.4.1 is consumed with no retry. No weaker serializer may be
used merely to record the failure.

## 5. Model-run publication, orphan binding, and terminals

### 5.1 Dispatch durability is unchanged

Each `dispatch_started` event must be successfully published before its model
call. Each returned call output must be losslessly held until its
`dispatch_completed` event is successfully published. The started count is
the durable model-apply-attempt count; the completed count is the durable
successful-return count. The exact `4*k+d` arithmetic, failed-current
semantics, and raw exclusion rules remain those of v3.3.4.

No call is retried after a started event, a completed-event publication
failure, a raw publication failure, or any other failure.

### 5.2 Terminal publication audit

Every v3.3.4.1 `RUN_COMPLETE.json` adds the exact top-level object
`publication_audit`:

```text
schema_version, method, successful_final_count_before_terminal,
successful_final_bindings_before_terminal,
temporary_orphan_count, temporary_orphan_bindings,
durability_uncertain_final_count, durability_uncertain_final_bindings,
preexisting_entry_count, preexisting_entry_states, no_new_entry_failure,
publication_failure, no_published_final_deleted,
no_temp_or_final_reused, no_publication_retry
```

Bindings are run-root-relative objects keyed by sorted POSIX path and valued
by exactly `sha256,size_bytes,mode,st_dev,st_ino,st_nlink`. Counts equal map
lengths. `preexisting_entry_states` is keyed by sorted run-root-relative path
and valued by the exact Section 3.5 entry-state object. The three `no_*`
booleans are true. `no_new_entry_failure` is false and
`publication_failure` is null for
every `RUN_COMPLETE.json`. A publication failure never writes RUN_COMPLETE;
it follows the `TERMINAL_FAILURE.json` path below. This narrowly supersedes
the inherited v3.3.4 `controlled_stop_provenance_publication_failure`
RUN_COMPLETE outcome. The normal non-publication controlled-stop rows remain
unchanged. Consequently every RUN_COMPLETE has both orphan/uncertain counts
zero, the pre-existing count zero, and all three maps empty; the deliberately
retained external-preflight collision temp belongs to the external cache and
is not a model-run orphan.

The terminal itself cannot bind its own bytes. `publication_audit` therefore
describes the exact preterminal tree. The CPU analyzer later binds the terminal
and whole run tree without weakening this distinction.

`TERMINAL_FAILURE.json`, when it can be published, has exactly:

```text
schema_version, status, stop_reason, attempt_id, script_version,
external_freeze_authorization, runner_pid, publication_failure,
preterminal_tree_binding, source_input_audit,
source_input_audit_content_binding, same_object_attestation,
same_object_attestation_content_binding, phase_state,
model_apply_attempt_count, model_apply_success_count,
valid_record_count, failed_current_binding,
temporary_orphan_bindings, durability_uncertain_final_bindings,
preexisting_entry_states, no_new_entry_failure,
confirmation_model_calls, scientific_summary_computed,
donor_normalization_computed, shapley_or_nomination_computed,
interaction_or_resolution_computed, nomination_performed,
combined_analysis_permitted, no_retry, created_at_unix_s
```

`status="incomplete_publication_failure"` and
`stop_reason="artifact_publication_failure"`. Every scientific/count flag is
false or the exact journal-derived prefix count; confirmation calls are zero;
`combined_analysis_permitted=false`; `no_retry=true`. For the first failure
class in Section 3.5, `no_new_entry_failure=true`, the orphan/uncertain maps
are empty, and `preexisting_entry_states` is empty or contains only the exact
blocking parent/final/temp entry. For the other two classes,
`no_new_entry_failure=false`; all maps exactly match the failure object and
current no-follow tree.

The normal exhaustive v3.3.4 terminal matrix remains in force with versioned
strings. Publication introduces only these additional outcomes:

| State | Required preserved membership | Claim |
|---|---|---|
| failure before this attempt creates an entry; terminal succeeds | exact normal prefix, optional unchanged pre-existing-entry state, `TERMINAL_FAILURE.json`; no temp/final attributed to the failed attempt | incomplete infrastructure archive only |
| failure after temp creation and before rename succeeds; terminal succeeds | exact normal prefix, temporary orphan, `TERMINAL_FAILURE.json` | incomplete infrastructure archive only |
| rename succeeds; parent fsync/revalidation fails; terminal succeeds | exact normal prefix, durability-uncertain final, `TERMINAL_FAILURE.json` | incomplete infrastructure archive only |
| terminal publication also fails | exact filesystem prefix, including terminal temp or durability-uncertain final | consumed and unanalyzable by v3.3.4.1; future prospective audit required |
| no publication failure | exact v3.3.4 scientific/structural terminal | same v3.3.4 structural claim boundary |

There is no complete-science interpretation in any row.

## 6. CPU-only structural analyzer

The v3.3.4.1 analyzer remains standalone standard-library code. It must not
import JAX, JAXLIB, AlphaGenome, model code, OpenSplice scientific helpers, or
an older analyzer. It never opens confirmation model outputs, activations, or
interventions and never computes a scientific estimator.

Before reading any raw artifact, it independently validates:

- clean tracked HEAD and the acyclic external freeze authorization;
- all frozen source bytes and Git blobs;
- the never-launched absence of all six v3.3.4 paths;
- every immutable predecessor tree;
- exact v3.3.4.1 preflight, external-cache, model-cache, START, compiler,
  signature-attestation, source-input, same-object, import/protobuf, journal,
  manifest, terminal, and publication schemas;
- every final file as regular, non-symlinked, mode `0400`, exact size/hash,
  and exact bound device/inode/link count where bound;
- every temporary orphan and durability-uncertain final without following a
  symlink; and
- active internal analysis START token and SHA before raw entry.

The analyzer attempt/output lifecycle uses Section 3. It may emit exactly:

| Model state | ANALYSIS `status` | ANALYSIS `decision` |
|---|---|---|
| eligible complete 80/320 terminal | `complete_structural_sidecar_audit` | `structurally_complete_no_scientific_analysis` |
| exact inherited `POST_START_PROVENANCE_FAILURE.json` | `complete_controlled_stop_structural_archive` | `controlled_stop_post_start_provenance_failure` |
| eligible non-publication RUN_COMPLETE controlled stop | `complete_controlled_stop_structural_archive` | exact RUN_COMPLETE `status` |
| exact `TERMINAL_FAILURE.json`, no new entry and no pre-existing entry bound | `complete_incomplete_publication_archive` | `publication_failed_no_new_entry_no_scientific_analysis` |
| exact `TERMINAL_FAILURE.json`, no new entry and a pre-existing entry bound | `complete_incomplete_publication_archive` | `preexisting_entry_preserved_no_scientific_analysis` |
| exact `TERMINAL_FAILURE.json` with temporary orphan | `complete_incomplete_publication_archive` | `temporary_orphan_preserved_no_scientific_analysis` |
| exact `TERMINAL_FAILURE.json` with durability-uncertain final | `complete_incomplete_publication_archive` | `durability_uncertain_final_preserved_no_scientific_analysis` |

For a non-publication controlled stop, the only legal decisions are exactly:

```text
controlled_stop_import_provenance_failure
controlled_stop_protobuf_provenance_failure
controlled_stop_cache_hit
controlled_stop_model_setup_failure
controlled_stop_signature_attestation_failure
controlled_stop_lower_failure
controlled_stop_compile_failure
controlled_stop_attempt_budget_violation
controlled_stop_same_object_provenance_failure
controlled_stop_source_program_mismatch
controlled_stop_diagnostic_provenance_failure
controlled_stop_partial_dispatch
controlled_stop_four_call_invalid
```

Their `compiler_state`, `terminal_kind`, and `control_state_eligible` values
are exactly the corresponding v3.3.4 table, except that the superseded
publication-RUN_COMPLETE row is illegal. Every new publication archive has
`terminal_kind="terminal_failure"` and `control_state_eligible=false`.

A run without an eligible `RUN_COMPLETE.json` or exact
`TERMINAL_FAILURE.json` is rejected before consuming the analyzer attempt.
Publication failure after analyzer START is append-only and writes
`ANALYSIS_FAILURE.json` only if that artifact can be published. It never
deletes a partial RESULT, ANALYSIS, final, or temp. No result from this analyzer
permits scientific interpretation or combination.

The exact `ANALYSIS_ATTEMPT_STARTED.json` key set remains the v3.3.4 set with
versioned values. The authoritative inherited v3.3.4 ANALYSIS key set is its
literal 22-key list; the contradictory inherited prose count of 23 is
superseded. v3.3.4.1 ANALYSIS adds exactly `publication_audit`, for this exact
23-key top-level list:

```text
status, decision, analysis_version, analysis_attempt_start_binding,
run_binding, preflight_binding, model_cache_binding,
source_and_prior_audit, compiler_and_signature_audit,
dispatch_journal_audit, raw_prefix_audit, control_audit,
terminal_audit, publication_audit, confirmation_boundary,
claim_boundary, scientific_summary_computed,
donor_normalization_computed, shapley_or_nomination_computed,
interaction_or_resolution_computed, nomination_performed,
combined_analysis_permitted, completed_at_unix_s
```

The freeze and tests must bind that exact count and list. The publication
audit has the 15 Section 5.2 keys plus
exactly `analysis_attempt_tree_binding` and `analysis_output_tree_binding`,
for 17 keys. Each tree binding has exactly:

```text
root_role, file_count, directory_count, file_bindings,
file_tree_sha256, directory_paths, directory_tree_sha256
```

`file_bindings` is keyed by sorted root-relative POSIX path and each value has
exactly `sha256,size_bytes,mode,st_dev,st_ino,st_nlink`.
`directory_paths` is the sorted root-relative POSIX array including `.` for
the root; all directories are mode `0700`, regular directories, and
non-symlinked. File-tree framing is sorted path, one NUL, and 32 raw hash
bytes. Directory-tree framing is sorted path, one NUL, and ASCII mode.

Within analyzer publication-audit maps, the map value is an exact two-key
object, `analysis_attempt,analysis_output`; each nested value is the ordinary
sorted root-relative map from Section 5.2. Counts are the sum of both nested
map lengths. This is the sole legal way to distinguish the two roots; an
unqualified path or a flattened cross-root map is forbidden.

Inside ANALYSIS, `analysis_attempt_tree_binding` covers exactly the attempt
START and `analysis_output_tree_binding` covers exactly the already published
RESULT. Inside ANALYSIS_COMPLETE, the attempt binding still covers the
pre-COMPLETE START-only tree and the output binding covers exactly RESULT plus
ANALYSIS. This ordering avoids every self-hash cycle.

`ANALYSIS_COMPLETE.json` has the exact v3.3.4 key set plus exactly
`publication_audit`. `ANALYSIS_FAILURE.json` has the exact v3.3.4 key set plus
exactly these five keys:

```text
publication_failure
temporary_orphan_bindings
durability_uncertain_final_bindings
preexisting_entry_states
no_new_entry_failure
```

`publication_failure` is null and the four state fields are empty/false unless
the failure is at a publication boundary. The two file-binding maps use the
exact file-binding schema above; `preexisting_entry_states` uses Section
3.5's entry-state schema. No other analyzer schema extension is allowed.

For ANALYSIS_FAILURE, inherited `output_dir_state` is prospectively replaced
by exactly:

```text
state, published_prefix, published_final_bindings,
temporary_orphan_bindings, durability_uncertain_final_bindings,
preexisting_entry_states, file_tree_sha256,
entry_state_tree_sha256, directory_paths, directory_tree_sha256
```

`state` is exactly `absent`, `published_prefix`, or
`publication_failure_prefix`. `published_prefix` is exactly `[]`,
`["RESULT.md"]`, or `["RESULT.md","ANALYSIS.json"]`, and includes only
successfully published durable finals. A temporary or durability-uncertain
entry is represented only in its dedicated map. The tree hashes cover every
present regular final or temporary without following a symlink.
`entry_state_tree_sha256` covers the union of all four state/binding maps by
sorted root-qualified path, one NUL, and the SHA-256 of the canonical
allow-nan-false JSON entry-state object. It therefore binds a non-regular or
unreadable pre-existing entry without hashing through it.

ANALYSIS, ANALYSIS_COMPLETE, and ANALYSIS_FAILURE bind the exact
`ANALYSIS_ATTEMPT_STARTED.json` SHA-256 and size. The analyzer repeats global
clean/source/prior/run/publication checks before RESULT, before ANALYSIS, and
before COMPLETE. Any TOCTOU difference consumes the one analysis attempt.

## 7. Acyclic machine freeze

The v3.3.4.1 freeze is generated from the exact 69-key v3.3.3 base. It carries
the same 13 extension key names specified by v3.3.4, with v3.3.4.1 values, and
adds exactly one new top-level key:

```text
publication_contract_v3_3_4_1
```

Its exact top-level key count is therefore 83. Its `file_sha256` map contains
the exact inherited 96 v3.3.3 rows unchanged and exactly these 12 prospective
v3.3.4.1 rows, for 108 rows total:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_1.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_1.sh
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_1_test.py
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_1_freeze.py
experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_1.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_1.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_1_test.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_1.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_1.sh
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_1_test.py
experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_publication_amendment_v3_3_4_1.md
experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_1.py
```

The freeze path is exactly
`experiments/interpretability/opensplice/encoder_skip_ood_sidecar_v3_3_4_1_freeze.json`.
The amendment, freeze, Python files, and Python tests have Git mode `100644`;
the two shell wrappers have Git mode `100755`. The freeze itself is tracked
and externally authorized but is not included in its self-referential
`file_sha256` map.

The v3.3.4 implementation drafts are not runtime inputs and are not imported,
copied, or source-inventoried. The new implementation may be derived from
reviewed logic, but its complete bytes are independently frozen under the
v3.3.4.1 paths.

`publication_contract_v3_3_4_1` has exactly:

```text
schema_version, method, temp_name_regex, nonce_bytes,
open_flags, initial_mode, sealed_mode, rename_flags,
same_directory_required, keep_fd_open_through_rename,
file_fsync_count, parent_fsync_required,
post_publish_inode_revalidation_required,
no_replace, no_fallback, no_retry,
temporary_orphan_preservation_required,
durability_uncertain_final_preservation_required,
successful_publication_object_keys,
publication_failure_object_keys,
entry_state_object_keys,
external_preflight_probe_contract
```

The scalar and array values are exactly:

```text
schema_version="v3.3.4.1-named-temp-renameat2-noreplace-v1"
method="named_temp_renameat2_noreplace"
temp_name_regex="^\\.v3341\\.tmp\\.[1-9][0-9]*\\.[0-9]{6}\\.[0-9a-f]{32}$"
nonce_bytes=16
open_flags=["O_RDWR","O_CREAT","O_EXCL","O_NOFOLLOW","O_CLOEXEC"]
initial_mode="0600"
sealed_mode="0400"
rename_flags=["RENAME_NOREPLACE"]
same_directory_required=true
keep_fd_open_through_rename=true
file_fsync_count=2
parent_fsync_required=true
post_publish_inode_revalidation_required=true
no_replace=true
no_fallback=true
no_retry=true
temporary_orphan_preservation_required=true
durability_uncertain_final_preservation_required=true
successful_publication_object_keys=["schema_version","method","root_role","final_relative_path","temp_basename","publication_ordinal","runner_pid","nonce_hex","sha256","size_bytes","mode","st_dev","st_ino","st_nlink","file_fsync_before_rename","file_fsync_after_fchmod","rename_noreplace_succeeded","parent_fsync_succeeded","post_publish_revalidation_exact"]
publication_failure_object_keys=["schema_version","method","root_role","artifact_role","final_relative_path","temp_relative_path","publication_ordinal","runner_pid","failure_stage","errno","error_type","message","rename_noreplace_attempted","rename_noreplace_succeeded","parent_fsync_attempted","parent_fsync_succeeded","temp_state","final_state","created_at_unix_s"]
entry_state_object_keys=["state","entry_type","mode","size_bytes","sha256","st_dev","st_ino","st_nlink"]
```

`external_preflight_probe_contract` has exactly:

```text
final_basename="atomic_publication_probe_v3_3_4_1.txt"
final_sha256="97de696d85c4c0c98438dd74b9cb30d44eced002afa43e7f3e01f32f8a94518a"
final_size_bytes=49
collision_sha256="9f26f62f146a1efd26974bb9eb6b07ec8cd0f87e61713b188b1b4ddee55f7eb9"
collision_size_bytes=39
collision_errno=17
collision_temp_preserved=true
parent_fsync_exact_required=true
```

Every other literal value comes from Sections 3--4. The freeze also updates the
existing dispatch-journal, preflight, terminal, analyzer, and source-inventory
contracts to bind this amendment's exact schemas. It binds the v3.3.4
amendment commit, path, SHA-256, and the six-path absence audit as a
prerequisite, but contains no v3.3.4 draft implementation hash.

The freeze JSON cannot contain its own hash. None of the 108 inventoried files
hard-codes the final freeze hash. After the 12 source files and freeze are
committed, an independent post-commit auditor computes the exact HEAD, freeze
SHA-256, and freeze size and supplies them only through the three
`V3341_AUTHORIZED_*` variables. Runtime code independently requires:

```text
authorized HEAD == current HEAD
tracked diff is empty
live freeze bytes == git show HEAD:<freeze path>
live freeze SHA/size == authorized SHA/size
all 108 live source hashes == freeze file_sha256
all 108 git-show source hashes == freeze file_sha256
```

This is the only freeze-hash anchor and is acyclic.

## 8. Required tests and stop rules

Before scoped commit, all tests are CPU-only and use isolated temporary roots
or mocks; no production preflight, JAX model, GPU, raw score, or confirmation
artifact may be invoked. Tests must cover:

1. exact open/write/fsync/fchmod/readback/renameat2/fsync/revalidate syscall
   order and success evidence;
2. short writes and zero-progress writes;
3. every failure stage in Section 3.5, including errno capture;
4. ordinal zero for the first attempt, monotonic consumption after success or
   failure, and no reuse/reset;
5. parent-open/validation, final-preexistence, and temp-open failures with the
   exact no-new-entry/pre-existing-entry membership and empty orphan/uncertain
   maps;
6. existing-final `EEXIST`, unchanged destination bytes/inode, and preserved
   temporary source;
7. temporary-name collision with no retry;
8. final/temp symlink, FIFO, socket, directory, traversal, and special-entry
   rejection without following links;
9. same-directory and same-device enforcement;
10. mode `0600` before sealing and `0400` after sealing;
11. no-new-entry, pre-existing-entry, temporary-orphan, and
    durability-uncertain-final classification, binding,
    preservation, and terminal membership;
12. no deletion, overwrite, reuse, or fallback API, including source scans for
    `O_TMPFILE`, `/proc/self/fd`, `linkat`, `os.rename`, and `os.replace` in the
    v3.3.4.1 publication path;
13. the formal successful probe and the exact unsupported/failure prefixes;
14. preflight log streaming without a duplicate payload write, through the
    same helper, and the exact five-file pass tree;
15. Gate A, preflight, distinct cache, same-process gate, START, Gate B, and
   scientific-import ordering;
16. one-shot failure for every preexisting v3.3.4.1 path and blocker failure
   for every present v3.3.4 predecessor path;
17. exact 83-key freeze, 108-row source inventory, source modes, Git blobs,
   external authorization, and fixed-point absence;
18. all frozen v3.3.4 source-program, 32-path, same-object, import/protobuf,
   diagnostic, cache, dispatch, raw, and terminal tests unchanged;
19. runner-shaped `k=0..80`, `d=0..4`, all controlled stops, all publication
   failure boundaries, and full 80/320 completion;
20. exact dispatch-started-before-call and completed-after-return durability;
21. no call after any failed publication and no second lower/compile/dispatch;
22. analyzer rejection of terminal-less prefixes, exact archive of all three
    publication-failure classes, all five failure-key extensions and exact
    `output_dir_state` memberships, exact controlled-stop status/decision,
    internal token enforcement, TOCTOU checks, and append-only output failure;
23. import guards for JAX/JAXLIB/AlphaGenome/model/older analyzers in the CPU
   analyzer; and
24. zero confirmation access, zero scientific estimator, false scientific
    flags, and `combined_analysis_permitted=false` in every outcome.

Any failed test, schema mismatch, unsupported `renameat2(RENAME_NOREPLACE)`,
pre-existing path, source drift, freeze-authorization mismatch, cache input,
publication orphan not exactly bound, durability-uncertain final not exactly
bound, or scientific/provenance gate failure is a stop. Thresholds, expected
effects, controls, and scientific gates cannot be changed in response.

## 9. Allowed claim boundary

Before a successful external preflight, the only claim is that v3.3.4 was not
launched and the host could not satisfy its prospectively frozen anonymous
link primitive in an informal diagnostic.

After a successful v3.3.4.1 preflight but before a model terminal, the only new
claim is that this host demonstrated the exact named-temp
`renameat2(RENAME_NOREPLACE)` publication and collision behavior.

After an eligible model terminal and CPU structural audit, claims remain
limited to provenance, structural controls, exact execution counts, and the
frozen v3.3.4 claim boundary. Even an exact 80/320 pass does not establish a
biological mechanism, model faithfulness, causal localization, or utility for
biologists. A separate, prospectively frozen CPU scientific analysis is
required for any scientific conclusion.

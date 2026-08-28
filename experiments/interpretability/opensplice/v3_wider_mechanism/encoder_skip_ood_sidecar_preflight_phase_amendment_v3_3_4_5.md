# OpenSplice encoder-skip OOD sidecar v3.3.4.5: completed-preflight phase repair

Status: prospective infrastructure-only amendment. No v3.3.4.5 code,
preflight, model, GPU, analysis, or confirmation access is authorized until
this exact document is committed and a separately frozen implementation has
passed independent review.

This amendment inherits the scientific design and claim boundary of
v3.3.4 through v3.3.4.4 byte for byte. It repairs one parent-side validation
phase error after a successful external preflight and archives the consumed
v3.3.4.4 prefix. It does not change the variants, recipient/donor order,
readout, model, checkpoint, reference, compiler/source gates, publication
protocol, 80-record / 320-apply budget, controls, or structural-only analysis.

## 1. Frozen predecessor and consumed v3.3.4.4 prefix

The sole v3.3.4.4 invocation used tracked-clean commit
`6858bbcdd869ac9ae93064910227003a911d0bd1` and the exact freeze:

- path:
  `experiments/interpretability/opensplice/encoder_skip_ood_sidecar_v3_3_4_4_freeze.json`
- SHA256:
  `73b26eddf5578ef0847ac69c279c262e6f43102127bbe4299bbdab7e52227e30`
- size: `187923` bytes
- Git mode: `100644`
- exact counts: `85` top-level keys, `120` `file_sha256` entries, and
  `120` source-inventory rows.

The v3.3.4.4 protocol amendment is commit
`84c07067613c3a206772c47e59d34b7d49886c7d`, path
`experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_preflight_version_amendment_v3_3_4_4.md`, and SHA256
`a188c252fb8ef01bcb48123aa57673eb1bb6eb720f8a9b68e35cafda0793e523`.
The exact relevant v3.3.4.4 source hashes are:

- launcher `4cdeee9df8b14043633383b99cdf8d88bdf1c3a0a3e4146a3c40adf9e78991f9`;
- bootstrap/publication helper
  `8e2559d5dae96f6d9ddaa752e2aa4de3829ec85115e78fba356e6fe8c8abccb8`;
- external preflight
  `190b4ede0be981269ff2345931577ad2d8d901d38617268168a8cf02424f6907`;
- model runner
  `55619f20485e59ee0f08eba0b71fcb03126a74f1287f424d420a83f288de2ef7`;
- shell wrapper
  `6913faaad359d18b72c7ec7f9323feb1e9263e4bfaefbb9199c05b1ebe1785b1`;
- structural analyzer
  `ab8e12b22de770b0cca9ebe38e2818b0d03dfb7b2150e39fc7a36a330063dc14`.

### 1.1 Successful external preflight

The external preflight completed successfully. Its root is:

`experiments/interpretability/opensplice/results/v3_3_4_4_device_preflight`

The root is a non-symlink directory with mode `0700`, `st_dev=66307`,
`st_ino=140791442`, `st_nlink=2`, and size `4096`. It contains exactly five
regular non-symlink files and no subdirectories:

| relative path | mode | st_dev | st_ino | st_nlink | size | SHA256 |
|---|---:|---:|---:|---:|---:|---|
| `.allocation.lock` | `0600` | 66307 | 140791443 | 1 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.preflight_0000.reserved` | `0400` | 66307 | 140791444 | 1 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.json` | `0400` | 66307 | 140791447 | 1 | 27062 | `a240bf223dd62ebc53b84da35bb614df7987254c3694d7f07aae9785adec3801` |
| `preflight_0000.stderr.log` | `0400` | 66307 | 140791446 | 1 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.stdout.log` | `0400` | 66307 | 140791445 | 1 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Using the inherited relative-path NUL raw-SHA file-tree framing, the five-file
tree SHA256 is
`f009ba6fe67a715301b443940876be8f85998a50f71f320d0dc5e3dd52dfd6e5`.
Under the inherited directory-aware `D\0<relative>\0` /
`F\0<relative>\0<raw-SHA>` framing, with the root represented as `.`, the
same immutable tree has SHA256
`1a343cacb96cc1a1c88735c6a3bb8edfb0b71c1df89e924f52e099c30e3217f5`.
The canonical JSON serialization of the complete parsed 20-key preflight
record has SHA256
`9b1a5e3bbc9845d04430c259ed39db0f39f31b56128f022443382b91e6027285`
and size `22193` bytes.

The record has `status="pass"`, `failure=null`,
`preflight_attempt_number=0`,
`script_version="opensplice-device-preflight-v3.3.4.4"`, and
`freeze_sha256` equal to the v3.3.4.4 freeze above. Its external process PID
is `2696297`. It observed exactly one GPU device, default backend `gpu`, device
kind `NVIDIA GeForce RTX 3090`, UUID
`GPU-64111645-1e42-a96d-f192-4abbec4b8090`, and compute capability `8.6`.
It has `no_jit_or_array_kernel=true`,
`no_model_or_biological_access=true`, and external cache hit `false`.

This was a real JAX/device GPU preflight. Therefore v3.3.4.5 must not claim
that v3.3.4.4 had no JAX or GPU access. It may claim only that there was no
array/JIT kernel, model/checkpoint/reference construction, biological model
access, or scientific computation in that preflight.

### 1.2 External-preflight cache and publication probe

The external cache root is:

`experiments/interpretability/opensplice/results/v3_3_4_4_preflight_kernel_cache`

Its exact directory lstat rows are:

| relative path | type | mode | st_dev | st_ino | st_nlink | size |
|---|---|---:|---:|---:|---:|---:|
| `.` | directory | `0700` | 66307 | 140791437 | 4 | 4096 |
| `triton` | directory | `0700` | 66307 | 140791438 | 2 | 4096 |
| `xdg` | directory | `0700` | 66307 | 140791439 | 2 | 4096 |

It contains exactly two regular non-symlink files:

| relative path | mode | st_dev | st_ino | st_nlink | size | SHA256 |
|---|---:|---:|---:|---:|---:|---|
| `.v3344.tmp.2696297.000001.55167cfd266423a5ba861df8ca40686d` | `0400` | 66307 | 140791441 | 1 | 39 | `a1e62f4f34497aa5e72ece0670f1d865cd6eaacdcdfbacb00c39648d9e83f14f` |
| `atomic_publication_probe_v3_3_4_4.txt` | `0400` | 66307 | 140791440 | 1 | 49 | `47efa8c868d4d9455730ad1e89d6e44afee44172f0d2af7521d8574b7d85ecc9` |

The exact inherited directory-aware cache binding has
`directory_paths=[".","triton","xdg"]`, `directory_count=3`,
`file_count=2`, and tree SHA256
`3a294e09038311b8bad85836c6983da31f50fdeef3365b844e8842922d33acba`.
Its canonical JSON binding is SHA256
`88c2a4cde3a9881f76dc719b48dbd7f051b5841843dc5439a8e7d2349aabbc46`
and size `1033` bytes. The exact 10-key publication-probe object has canonical
SHA256
`a25798b7c788ce614d10a7cc0d07f1795ebaf6fd9e928dcef78e096323d9bf70`
and size `738` bytes. It reports the inherited
`named_temp_renameat2_noreplace` method, success, collision `errno=17`, exact
no-replace behavior, unchanged destination, preserved collision orphan, and
exact parent fsync.

### 1.3 Parent-side failure and exact boundary

After launching the child, the parent still had:

```text
ALPHAGENOME_V3_3_4_4_CACHE_ROLE=external_preflight
ALPHAGENOME_V3_3_4_4_CACHE_ROOT=<the v3.3.4.4 external cache root>
```

This is expected: `_allocate_cache(..., "external_preflight")` sanitized and
routed the parent before spawning the child, and the child inherited that
environment. The child completed the valid preflight. The parent then called
`_validate_preflight_record()`, which called the role-polymorphic
`validate_preflight_state_for_role()`. Because the ambient role was still
`external_preflight`, that function selected its entry-time absence branch and
rejected the correctly existing completed preflight directory. This is a
parent validation-phase routing bug, not a preflight failure or cache hit.

The exact failure is:

- stage: `parent_completed_external_preflight_validation`;
- type: `FileExistsError`;
- message:
  `Preflight directory exists before the external_preflight process.`

The following traceback was captured by the coordinator and was not persisted
to disk. No session identifier or wall-clock timestamp is available. The exact
UTF-8 block below, including its final newline, has SHA256
`03cf721c145a2d70764455c8ab197482aed52a062d10c1ca29818bb0c1c8c3d3`
and size `1168` bytes:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_4.py", line 707, in <module>
    main()
    ~~~~^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_4.py", line 627, in main
    preflight_record, successful_preflight = _validate_preflight_record(path)
                                             ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_4.py", line 133, in _validate_preflight_record
    state = bootstrap.validate_preflight_state_for_role()
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4.py", line 2579, in validate_preflight_state_for_role
    raise FileExistsError(
        f'Preflight directory exists before the {role} process.'
    )
FileExistsError: Preflight directory exists before the external_preflight process.
```

The v3.3.4.4 model-cache, model-run, analysis-attempt, and analysis-output roots
are absent. No model cache was allocated; no same-process model gate, model or
scientific-module import, checkpoint/reference construction, lower, compile,
model START, apply, dispatch journal, raw record, terminal, analysis, or
confirmation access occurred. Model apply and raw-record counts are zero.

Both durable v3.3.4.4 roots are consumed and immutable. They must not be
deleted, renamed, modified, resumed, or used as cache input. The v3.3.4.4
launcher, preflight, runner, and analyzer must never be invoked again.

## 2. Exact v3.3.4.4 consumed-prefix object

The v3.3.4.5 freeze adds exactly one new top-level object named
`prior_v3_3_4_4_consumed_preflight_prefix`. The inherited
`prior_v3_3_4_3_consumed_preflight_prefix` remains unchanged. The new object
has exactly these 18 keys:

```text
status, predecessor_commit, predecessor_freeze,
failure_stage, failure_type, failure_message, traceback_provenance,
root_cause, external_preflight_archive, external_cache_archive,
other_v3_3_4_4_paths_absent, no_model_cache_or_start,
no_model_or_biological_access, no_array_jit_or_model_kernel,
no_scientific_or_confirmation_access, immutable_and_not_cache_input,
claim_boundary, access_boundary
```

The scalar values are exactly those in Section 1. `status` is
`consumed_successful_external_preflight_then_parent_role_routing_failure`.
`predecessor_commit` and `predecessor_freeze` bind the commit/freeze in
Section 1; `predecessor_freeze` has exactly `path,sha256,size_bytes,git_mode,
top_level_key_count,file_sha256_count,source_row_count` with counts
`(85,120,120)`.

`traceback_provenance` has exactly:

```text
storage, sha256, size_bytes, session_id, captured_at_unix_s,
wall_clock_timestamp_available
```

Its values are
`("coordinator_captured_not_persisted",03cf721c145a2d70764455c8ab197482aed52a062d10c1ca29818bb0c1c8c3d3,1168,null,null,false)`.

`root_cause` has exactly:

```text
parent_ambient_cache_role, parent_ambient_cache_root, called_validator,
selected_branch, rejected_state, required_validator,
failure_before_model_cache_allocation, failure_before_model_start,
launcher_source_binding, bootstrap_source_binding
```

The respective semantic values are `external_preflight`, the absolute
v3.3.4.4 external-cache root,
`validate_preflight_state_for_role`, `external_preflight_entry_absence`,
`completed_preflight_directory_present`,
`validate_completed_external_preflight_state`, `true`, and `true`.
The two source bindings each have exactly `path,sha256,size_bytes` and bind,
respectively, the absolute v3.3.4.4 launcher at
`4cdeee9df8b14043633383b99cdf8d88bdf1c3a0a3e4146a3c40adf9e78991f9`
/ `27372` bytes and bootstrap at
`8e2559d5dae96f6d9ddaa752e2aa4de3829ec85115e78fba356e6fe8c8abccb8`
/ `143561` bytes.

`external_preflight_archive` has exactly:

```text
root, directory_lstat_rows, directory_count, directory_paths,
file_count, files, file_tree_sha256, record_binding,
directory_aware_tree_sha256, record_canonical_binding, record_semantics
```

Its directory/file rows and both tree hashes are the literal Section 1.1
values. `directory_lstat_rows` is a one-item JSON list in the exact order
`["."]`; its value has exactly
`path,entry_type,mode,st_dev,st_ino,st_nlink,size_bytes` and the literal root
row in Section 1.1. Every `files` value has exactly
`path,sha256,size_bytes,mode,st_dev,st_ino,st_nlink`.
`record_binding` has exactly `path,sha256,size_bytes,mode` and the raw record
values in Section 1.1. `record_canonical_binding` has exactly
`sha256,size_bytes` and is
`{"sha256":"9b1a5e3bbc9845d04430c259ed39db0f39f31b56128f022443382b91e6027285","size_bytes":22193}`.
`record_semantics` has exactly:

```text
status, failure, preflight_attempt_number, script_version, freeze_sha256,
external_pid, jax_default_backend, jax_gpu_device_count, device_kind,
device_uuid, compute_capability, no_jit_or_array_kernel,
no_model_or_biological_access, external_cache_hit
```

with the literal Section 1.1 values.

`external_cache_archive` has exactly:

```text
root, directory_lstat_rows, cache_tree_binding, cache_tree_content_binding,
atomic_publication_probe, atomic_publication_probe_content_binding
```

The directory rows are a three-item JSON list in exact order
`[".","triton","xdg"]`. Each row has exactly
`path,entry_type,mode,st_dev,st_ino,st_nlink,size_bytes`.
`cache_tree_binding` has exactly:

```text
cache_role, cache_root, triton_cache_dir, xdg_cache_home,
directory_count, directory_paths, file_count, files, tree_sha256,
default_user_cache_paths_eligible,
diagnostic_outputs_only_no_cache_input
```

Its role is `external_preflight`; its three paths are the absolute Section 1.2
root and its `triton`/`xdg` children; its counts, relative directory list,
two-file SHA/size map, and tree SHA are the literal Section 1.2 values; and its
last two booleans are `false,true`. `cache_tree_content_binding` has exactly
`sha256,size_bytes` and is
`{"sha256":"88c2a4cde3a9881f76dc719b48dbd7f051b5841843dc5439a8e7d2349aabbc46","size_bytes":1033}`.

`atomic_publication_probe` has exactly:

```text
schema_version, method, supported, successful_final_binding,
collision_errno, collision_no_replace_exact, collision_temp_binding,
destination_unchanged, temp_orphan_preserved, parent_fsync_exact
```

Its schema version is
`v3.3.4.4-named-temp-renameat2-noreplace-v1`, method is
`named_temp_renameat2_noreplace`, `supported=true`, `collision_errno=17`, and
the final three booleans plus `collision_no_replace_exact` are all `true`.
Each final/collision binding has exactly
`path,sha256,size_bytes,mode,st_dev,st_ino,st_nlink` and the corresponding
literal Section 1.2 file row. `atomic_publication_probe_content_binding` has
exactly `sha256,size_bytes` and is
`{"sha256":"a25798b7c788ce614d10a7cc0d07f1795ebaf6fd9e928dcef78e096323d9bf70","size_bytes":738}`.

`other_v3_3_4_4_paths_absent` has exactly the sorted roles
`analysis_attempt,analysis_output,model_cache,model_run`. Each value has
exactly `path,absent`, with its absolute frozen v3.3.4.4 path and `true`.

`access_boundary` has exactly:

```text
external_preflight_device_observation_only,
external_gpu_device_observation_occurred, no_jit_or_array_kernel,
no_model_or_biological_access, model_cache_allocated,
same_process_preflight_reached, model_constructed, model_apply_count,
scientific_raw_record_count, confirmation_model_calls
```

The values are exactly
`(true,true,true,true,false,false,false,0,0,0)` in that order. This object
makes the permitted JAX/GPU observation explicit while preventing a false
claim that v3.3.4.4 never touched the GPU.

The four outer boundary booleans
`no_model_cache_or_start,no_model_or_biological_access,
no_array_jit_or_model_kernel,no_scientific_or_confirmation_access` and
`immutable_and_not_cache_input` are all JSON `true`. `claim_boundary` is a
JSON string with the exact literal:

```text
A JAX-only external GPU/device preflight passed; no model, model cache, START, apply, raw scientific record, analysis, or confirmation access occurred.
```

For canonical reconstruction, `predecessor_freeze.path`,
`external_preflight_archive.root`, `external_cache_archive.root`, and
`external_preflight_archive.record_binding.path` are normalized absolute
paths rooted at `/home/degen2/alphafold-stuff/alphagenome_research`.
Each `external_preflight_archive.files[*].path` is instead normalized relative
to `external_preflight_archive.root`. No other path-base interpretation is
permitted.

Canonical JSON means UTF-8, sorted keys, separators `(',',':')`,
`ensure_ascii=false`, and `allow_nan=false`. Under that framing the exact
18-key prefix has SHA256
`efcb6d8946666d104d7458c0f13cc8f53e6dfaa1a30a2e83744f48641978f3c7`
and size `8653` bytes. Gate A, the child preflight, Gate B, every terminal,
and the analyzer independently reconstruct and compare both the object and
this binding. Reading the two old roots for provenance is allowed; opening
them as compiler, model, JAX, or cache input is forbidden.

This committed amendment plus the future v3.3.4.5 freeze is the sole
v3.3.4.4 prefix archive. No standalone v3.3.4.4 analyzer attempt or output is
authorized, because there is no v3.3.4.4 model START or model terminal to
analyze. Those two old analyzer paths must remain absent permanently.

## 3. Authorized repair and fresh one-shot paths

One new version, v3.3.4.5, may repair only the parent completed-preflight
validation phase. Its exact version literals are:

```text
SCRIPT_VERSION="v3.3.4.5"
PREFLIGHT_SCRIPT_VERSION="opensplice-device-preflight-v3.3.4.5"
ATTEMPT_ID="v3.3.4.5-development-ood-sidecar-one-shot"
```

The external record uses the long preflight literal; START, runner, terminal,
and analyzer records use the short model-run literal. Before any v3.3.4.5
production allocation, Gate A repeats the inherited exact source-hash/AST
three-way proof:

```text
freeze.preflight_script_version
== bootstrap.PREFLIGHT_SCRIPT_VERSION
== the AST-extracted value of the sole direct PREFLIGHT_SCRIPT_VERSION
   assignment in the frozen v3.3.4.5 preflight producer
```

All three values must equal exactly
`"opensplice-device-preflight-v3.3.4.5"`.

The child repeats that proof before cache registration. External post-commit
authorization uses only `V3345_AUTHORIZED_GIT_HEAD`,
`V3345_AUTHORIZED_FREEZE_SHA256`, and
`V3345_AUTHORIZED_FREEZE_SIZE_BYTES`; the wrapper rejects any other
`V3345_AUTHORIZED_*` variable.

The attempt uses six fresh roots:

```text
results/v3_3_4_5_preflight_kernel_cache
results/v3_3_4_5_device_preflight
results/v3_3_4_5_model_kernel_cache
results/v3_3_4_5_development_ood_sidecar_one_shot
results/v3_3_4_5_development_ood_sidecar_analysis_attempt
results/v3_3_4_5_development_ood_sidecar_analysis
```

All six must be absent before freeze, post-commit dry-run, and launch. There is
one launch only. No failure, controlled stop, partial prefix, or successful
run may be deleted or retried.

The parent lifecycle is exact:

1. stdlib Gate A verifies tracked-clean HEAD, the v3.3.4.5 freeze and sources,
   both consumed-prefix objects/bindings, all prior archives, and all six fresh
   paths before any v3.3.4.5 production allocation;
2. allocate/sanitize only the fresh v3.3.4.5 external-preflight cache;
3. invoke exactly one fresh external preflight child;
4. the child repeats the source/freeze/prefix gates before cache registration,
   probe, preflight-root allocation, or lazy JAX import;
5. after the child exits zero, call the new pure
   `validate_completed_external_preflight_state()` before allocating the model
   cache;
6. only after that validator succeeds, allocate/sanitize the distinct fresh
   v3.3.4.5 model cache, perform the same-process JAX-only device gate, publish
   START, repeat Gate B, and import the unchanged runner.

`validate_completed_external_preflight_state()` must not select behavior from
the ambient cache-role string. It takes or closes over the exact fresh
external-preflight root and record path and validates the completed state:
one non-symlink `0700` root, exactly the five inherited lifecycle/log/record
paths with their required modes, the exact file-tree framing, one 22-key pass
record, null failure, matching external authorization, both consumed-prefix
objects/bindings, exact publication probe/live bindings, exact external-cache
post-observation and no-hit evidence, and the external child PID distinct from
the parent. It also independently asserts that the parent ambient routing is
still `cache_role="external_preflight"` and points to the fresh external
cache. This routing assertion is evidence, not phase dispatch.

The child entry-time validator remains distinct and must continue to reject a
pre-existing v3.3.4.5 preflight root before child allocation. The parent must
not spoof or set `cache_role="model"` merely to pass preflight validation.
The model role/root may be installed only by allocation of the fresh model
cache after completed-preflight validation succeeds.

If the v3.3.4.5 child preflight or completed-state validator fails, the fresh
prefix is consumed without START. No model cache/run/analyzer may be created,
and no retry is allowed. A later versioned amendment would be required.

## 4. Unchanged scientific execution contract

The v3.3.4.5 model path is scientifically identical to v3.3.4.4:

- exactly the frozen 20 development recipients and four anchors
  `(0,127,128,255)` in recipient-major, anchor-minor order;
- exactly 80 records and four calls per record, hence 320 applies;
- exactly one eight-row executable and zero six-row executables;
- zero identity reruns, zero main-cube reruns, and zero reused old OOD records;
- the same intended/unrelated donor maps, repeat calls, invariant rows
  `(0,1,3,5,6,7)`, ID0/ID255 controls, closures, source-program gate,
  diagnostic-only compiled backend, cache isolation, dispatch ledger,
  failed-current semantics, publication protocol, and terminal matrix;
- no confirmation model outputs, activations, interventions, or paths;
- `combined_analysis_permitted=false` and structural-only offline analysis.

No v3.3.4.4 preflight observation may be reused as the v3.3.4.5 successful
preflight, and neither v3.3.4.4 root is eligible cache input. The v3.3.4.5
preflight and model processes must generate fresh evidence.

## 5. Additive record and analyzer schemas

Every v3.3.4.5 preflight, START, post-START failure, RUN_COMPLETE,
publication terminal, nonpublication terminal, analysis START, ANALYSIS, and
ANALYSIS_FAILURE retains the complete v3.3.4.4 schema and adds exactly:

```text
prior_v3_3_4_4_consumed_preflight_prefix
prior_v3_3_4_4_consumed_preflight_prefix_content_binding
```

The first is the exact 18-key object in Section 2 and the second is exactly
`{"sha256":"efcb6d8946666d104d7458c0f13cc8f53e6dfaa1a30a2e83744f48641978f3c7","size_bytes":8653}`.
The inherited v3.3.4.3 pair remains byte-identical and separately validated.
The revised exact top-level key counts are:

| record | v3.3.4.4 | v3.3.4.5 |
|---|---:|---:|
| external preflight record | 20 | 22 |
| ATTEMPT_STARTED | 36 | 38 |
| POST_START_PROVENANCE_FAILURE | 22 | 24 |
| RUN_COMPLETE | 66 | 68 |
| publication TERMINAL_FAILURE | 33 | 35 |
| NONPUBLICATION_TERMINAL_FAILURE | 60 | 62 |
| ANALYSIS_ATTEMPT_STARTED | 16 | 18 |
| ANALYSIS | 25 | 27 |
| ANALYSIS_FAILURE | 17 | 19 |

`ANALYSIS.source_and_prior_audit` retains its 12 inherited keys and adds
exactly `prior_v3_3_4_4_consumed_preflight_prefix_exact=true`, for 13 keys.
The v3.3.4.5 analyzer rehashes both consumed prefixes before analysis START,
after START before any run/raw read, and at each prepublication TOCTOU gate.
It never calls the v3.3.4.4 analyzer, never creates an old analyzer path, and
never computes scores, normalization, Shapley values, interactions,
resolution rankings, or nominations.

The nested `START.successful_preflight` object is also corrected and frozen.
The v3.3.4.4 launcher would have emitted the eight inherited fields plus the
v3.3.4.3 prefix pair, while its analyzer still expected only the eight base
fields. v3.3.4.5 must not preserve that latent mismatch. Its exact 12 keys are:

```text
artifact_binding, root_file_count, root_file_tree_sha256,
external_pid, status, external_freeze_authorization,
external_cache_post_observation, external_cache_hit_evidence,
prior_v3_3_4_3_consumed_preflight_prefix,
prior_v3_3_4_3_consumed_preflight_prefix_content_binding,
prior_v3_3_4_4_consumed_preflight_prefix,
prior_v3_3_4_4_consumed_preflight_prefix_content_binding
```

The launcher serializer and analyzer validator use this identical literal
keyset. The top-level START count remains `38`; this correction is inside the
already-counted `successful_preflight` value.

## 6. Prospective freeze and acyclic source inventory

The v3.3.4.5 freeze is generated once from committed predecessor bytes and
stopped new sources. Starting from the v3.3.4.4 freeze, it replaces
version-specific paths/constants and the v3.3.4.4 nonpublication contract,
adds the one new top-level consumed-prefix object, and adds exactly these 12
new inventory paths:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_5_freeze.py
experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_5.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_5.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_5_test.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_5.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_5.sh
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_5_test.py
experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_preflight_phase_amendment_v3_3_4_5.md
experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py
```

The two shell wrappers have Git mode `100755`; every other new inventory path
has mode `100644`. The freeze itself has mode `100644` and is deliberately not
in its own `file_sha256` or source-row inventory. Exact prospective counts are:

```text
top-level freeze keys = 86
file_sha256 entries = 132
source-inventory rows = 132
```

The generator must rebuild byte-identically twice. Every one of the 132 rows
binds normalized repo-relative path, SHA256, size, and Git mode; Gate A and the
analyzer compare live bytes to both the freeze and `git show <authorized
HEAD>:<path>`. No inventoried source hard-codes the final freeze SHA. The
external post-commit authorization supplies HEAD, freeze SHA, and freeze size
at launch, preserving the inherited acyclic dependency.

The v3.3.4.5 freeze/additive contract must not remove or rewrite the frozen
v3.3.4.3 consumed-prefix object, the v3.3.4.4 raw/scientific contract, or any
prior archive binding. It binds the full v3.3.4.4 preflight/cache prefix even
though those artifacts are untracked production evidence.

## 7. Required tests and stop rules

Before commit, and again against committed bytes where applicable, CPU-only
tests must cover:

1. exact reconstruction and canonical binding of every leaf/key in both
   consumed-prefix objects; every old file/hash/size/mode/inode/link, directory
   row, cache binding, record semantic, probe binding, four-path absence, and
   source/freeze binding must fail independently on tamper, symlink, extra,
   missing, or special entry;
2. the exact traceback SHA/size and explicit unpersisted provenance;
3. child entry validation rejects an existing fresh preflight root before
   allocation;
4. parent completed-state validation succeeds with ambient role
   `external_preflight`, verifies that exact routing, and never dispatches to
   the entry-time absence branch;
5. wrong ambient role/root, wrong phase, wrong child PID, non-pass record,
   non-null failure, wrong 22-key record, tree/mode/hash drift, cache hit, or
   changed probe fails before model-cache allocation;
6. call-order/AST and runner-shaped integration prove Gate A, fresh external
   allocation, child, completed-state validation, fresh model allocation,
   same-process gate, START, Gate B, and runner import in that order;
7. model role is never installed before successful completed-state validation,
   and no role spoof can make a malformed preflight pass;
8. all additive keysets and exact counts in Section 5, including every normal
   and controlled terminal and analyzer outcome, plus exact 12-key
   `START.successful_preflight` serializer/analyzer parity;
9. deterministic freeze generation at exactly `86/132/132`, exact 12-path
   delta, modes, live hashes, and absence of the replaced old version-specific
   nonpublication key;
10. full inherited publication fault matrix, signature 32-path adapter,
    source/same-object/compiler gates, cache isolation, dispatch counts
    `d=0..4`, 80-record completion, invariant/closure/repeat/donor controls,
    and confirmation-path guards;
11. dry-run with exact external authorization creates none of the six fresh
    paths and leaves no temporary cache residue;
12. the structural analyzer directly rejects unacknowledged invocation,
    confirmation paths, old analyzer destinations, missing active START token,
    source/prefix/run TOCTOU, and any attempt to emit scientific or combined
    analysis.

The implementation is a controlled stop if any provenance, preflight,
publication, source-program, compiler, cache, closure, repeat, invariant,
finiteness, or count gate fails. It never retries. If all 80 records complete,
the only permitted claim is that the frozen development-only structural OOD
controls passed under the exact v3.3.4.5 provenance. No biological mechanism,
causal localization, pathway discovery, confirmation result, or combined
analysis may be claimed without a separate prospective protocol.

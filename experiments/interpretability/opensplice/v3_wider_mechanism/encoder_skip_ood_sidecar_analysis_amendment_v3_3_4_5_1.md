# OpenSplice v3.3.4.5.1 CPU analyzer directory-membership amendment

Status: prospective analyzer-only amendment, frozen before any v3.3.4.5.1
implementation or invocation.

Model-output scope: development-only structural provenance. No confirmation
model output, activation, or intervention may be opened. No model, JAX, GPU,
lowering, compilation, or intervention call is authorized by this amendment.

## 1. Immutable inputs and claim boundary

The single v3.3.4.5 GPU invocation is complete and immutable at Git commit
`0da8f47ea6e576a72a1cda204ce868ef79cc2ce5`, with freeze SHA-256
`2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366`
and freeze size 204,697 bytes. The v3.3.4.5 amendment is commit
`001b4453833cb3c57991187c96416fccd22e4928`, SHA-256
`e64af0ba8ad6436530a1bd0da2807f3a9bd6ef874306255e1f9683e1731574c8`.
The frozen 132 source rows must remain byte-exact live and at commit
`0da8f47ea6e576a72a1cda204ce868ef79cc2ce5`; the tracked worktree and index
must remain clean.

The model run is an exact structural controlled stop:

- run-root file count 12, directory paths
  `['.', 'compiler', 'compiler/eight_row']`, file-tree SHA-256
  `960faf1675caaa0f3c9798f7b943998e91650b2a9359b26ca31ce2d417c2ce0b`,
  and directory/file tree SHA-256
  `5331e5041b557a4324ba57d57179dc2bfd8e6ab981f0cdede5d4f126e94c2041`;
- `ATTEMPT_STARTED.json` is 116,707 bytes, SHA-256
  `c211bf46f9fd55689da21d02d3b7859f08cd14f27a0419c872047f4a1f3f3f13`;
- `RAW_MANIFEST.json` is 1,562 bytes, SHA-256
  `3ee95b22d483c7c4f234fbb75281e05e84f0be263b1ee670a94b2cd442d61136`;
- `RUN_COMPLETE.json` is 43,760 bytes, SHA-256
  `fdbd0a1dc7d24145f88c5a009cc80d8904e57920e0c9584426e791373fae6d8f`;
- the five-file compiler subtree, rooted at the model-run `compiler` directory
  with relative directory paths `['.', 'eight_row']`, has file-tree SHA-256
  `b1094dfaddb0e8c6672b09a18e124af2a20a1a91c7ca817911c2f3fe4c0220a3`
  and directory/file tree SHA-256
  `bb042bf9a2cb34c61aae121733edce583cc2d747de1913b1d74f00b7a8de200c`;
- the five-file device-preflight tree has file-tree SHA-256
  `ae277eafa4f7f20bfa74c3a0a1bbaa0f51468cac945d29d0c49cab699738ecfd`
  and directory/file tree SHA-256
  `cc106b406da58ddd95611aef7e471f5a5cefd96e302ebb91ea4ef9e28a618c87`;
  `preflight_0000.json` is 50,472 bytes, SHA-256
  `cefe7d95c67868668f912575c8e00fecc9813ec011d395b8332f65d0a2c7d785`;
- the two-file/three-directory external cache has file-tree SHA-256
  `3bd7b53ba7ab1dae7161999ff907137f82ee6d7f322512a3221646f66bb1e975`
  and directory/file tree SHA-256
  `d040af81aa50fbe28e0523747355f84d851f36f39e586294b24dd994f69f66a0`;
- the one-file/four-directory model cache has file-tree SHA-256
  `487c67a6dbb251aca190ac9eda5d2425c3584febc9ad63e60d0812c7f2fb69ea`
  and directory/file tree SHA-256
  `51fe59713c301342bf5bb161f26b9e4ee6828e508b96e4dbd21c6efcdde1115e`;
- terminal status is `controlled_stop_diagnostic_provenance_failure`, stop
  reason is `diagnostic_persistence_failure`, and failure type/message are
  `DiagnosticPersistenceFailure` / `'eight_row_compiler'`;
- lower attempts, compile attempts, and successful compiles are each one;
  dispatch starts, dispatch completions, apply attempts, apply successes, and
  valid raw records are each zero;
- the raw, started-journal, and completed-journal maps are empty and each tree
  SHA-256 is the empty SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- `all_80_recipient_anchors_complete`, `id0_all20`, and `id255_all20` are
  false; no six-row compile, identity rerun, main-cube rerun, old-record reuse,
  or confirmation call occurred; and
- every scientific-summary, normalization, Shapley/nomination,
  interaction/resolution, nomination, and combined-analysis flag is false.

The source-program gate, the exact 32-path tuple-to-list signature adapter,
StableHLO, pre-backend HLO, entry ABI, source inputs, and same-object
attestation passed. Compiled HLO remains diagnostic only. This amendment does
not convert the controlled stop into a successful 80-record/320-apply run and
does not authorize a control, mechanistic, biological, or confirmation claim.

The v3.3.4.5 model run, preflight, external cache, model cache, source files,
freeze, and original analyzer bytes must never be edited, deleted, replaced,
or rerun. A future model attempt requires a separate prospective protocol.

## 2. Consumed v3.3.4.5 analyzer invocation

Exactly one invocation of the committed wrapper
`analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh` with argument
`--acknowledge-structural-only-v3-3-4-5` exited 1 after 1.95131362 seconds.
The coordinator tool output had chunk ID `77e144`, no session ID, and no
captured wall-clock timestamp. Standard output was empty (zero bytes, SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
Standard error was 1,587 UTF-8 bytes including its final newline, SHA-256
`0158926b7b41b6636bfacd2acdcf268bae7f7082b9f935edb483aa184bdd6967`.
Neither stream was persisted to the filesystem. The text below is an honest
copy of coordinator-captured, otherwise unauthenticated stderr; its hash is
evidence of the copied text, not evidence of a durable original log.

The new freeze and analyzer bind an exact `consumed_analyzer_failure` object
with these 15 sorted keys and literal semantics:

```text
captured_at_unix_s = null
chunk_id = "77e144"
command = [<absolute committed wrapper path>, "--acknowledge-structural-only-v3-3-4-5"]
destination_states = {<four absolute paths listed below>: "absent"}
exit_code = 1
failed_before_start = true
failure = {"message":"v3.3.3 cache contains an extra/empty directory.","stage":"precheck_prior_v3_3_3_cache","type":"AnalysisError"}
no_jax_model_raw_or_confirmation_access = true
retry_permitted = false
session_id = null
status = "consumed_pre_start_failure"
stderr = {"final_newline":true,"persisted_to_filesystem":false,"sha256":"0158926b7b41b6636bfacd2acdcf268bae7f7082b9f935edb483aa184bdd6967","size_bytes":1587,"source":"coordinator_captured_unpersisted_tool_output"}
stderr_text = <the exact 1,587-byte block below>
stdout = {"persisted_to_filesystem":false,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size_bytes":0}
wall_time_seconds = 1.95131362
```

`command[0]` and every `destination_states` key are normalized absolute paths.
The object's content binding is canonical UTF-8 JSON with sorted keys, compact
separators, `ensure_ascii=true`, and `allow_nan=false`; its SHA-256 and byte
length are computed prospectively by the freeze generator rather than guessed
in this amendment.

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 8513, in <module>
    main()
    ~~~~^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 8452, in main
    precheck = _analysis_attempt_precheck(run_dir, bundle_root=bundle_root)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 7854, in _analysis_attempt_precheck
    _validate_freeze_v3345(run_dir, bundle_root=bundle_root)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 5162, in _validate_freeze_v3345
    prior333 = _validate_prior_v3_3_3()
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 1201, in _validate_prior_v3_3_3
    cache_paths = _strict_tree(
        _PRIOR_CACHE_DIR, set(_PRIOR_CACHE_FILES), 'v3.3.3 cache'
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 1076, in _strict_tree
    raise AnalysisError(f'{label} contains an extra/empty directory.')
AnalysisError: v3.3.3 cache contains an extra/empty directory.
```

The failure occurred inside the provenance-only precheck, before
`ANALYSIS_ATTEMPT_STARTED.json`, before either analysis directory was
allocated, before `analyze()` was called, and before any raw record access.
The following paths remain absent and must remain permanently absent:

- `results/v3_3_4_5_development_ood_sidecar_analysis_attempt`;
- `results/v3_3_4_5_development_ood_sidecar_analysis`;
- `results/v3_3_3_development_ood_sidecar_analysis_attempt`; and
- `results/v3_3_3_development_ood_sidecar_analysis`.

The successful v3.3.3.1 analysis attempt/output archives remain immutable.
The consumed v3.3.4.5 analyzer invocation must not be retried, and its source,
test, wrapper, and freeze remain unchanged at these exact hashes:

- analyzer `9320184c53ed6bc3b246443314d84c1f1543bbbf77aa10e3fff982bd5c18913a`;
- test `dcede28da855e3784a86453fb8f1cdeb3b94d326bc249023f3e72b82316a0fe5`;
- shell `cea01ac69a8468f54e4bfb8a453709494449ef886e7dc6ae28e073a79fa2855c`;
- freeze `2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366`.

## 3. Exact cause and the only permitted repair

The immutable v3.3.3 model-cache root has mode `0700` and exactly these four
mode-`0700` directory paths, in POSIX order:

```text
.
triton
xdg
xdg/matplotlib
```

It has exactly one regular file:
`xdg/matplotlib/fontlist-v3.11.0.json`, mode `0600`, size 163,240, SHA-256
`a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125`.
Its file-only tree SHA-256 is
`d1d11bc6dc48b302cf675fb48727bd6ededec09142429eaa9e368f7631463717`.
Its directory/file tree, framed by sorted `D\0<relative>\0` rows followed by
sorted `F\0<relative>\0<32 raw SHA-256 bytes>` rows, is
`a1dafb75097282b3d28174e4bb6c79a81d2f1c8c25cd8c64a1ea84642ba7f43a`.

The replacement validator freezes these exact no-follow lstat rows. `sha256`
is JSON null for directories and the exact lowercase digest for the file.

| path | entry_type | mode | size_bytes | st_dev | st_ino | st_nlink | sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| `.` | `directory` | `0700` | 4096 | 66307 | 140791354 | 4 | null |
| `triton` | `directory` | `0700` | 4096 | 66307 | 140791357 | 2 | null |
| `xdg` | `directory` | `0700` | 4096 | 66307 | 140791358 | 3 | null |
| `xdg/matplotlib` | `directory` | `0700` | 4096 | 66307 | 140791359 | 2 | null |
| `xdg/matplotlib/fontlist-v3.11.0.json` | `regular` | `0600` | 163240 | 66307 | 140791361 | 1 | `a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125` |

The file-only tree is rooted at the v3.3.3 cache root and framed by sorted
`<root-relative POSIX file path>\0<32 raw SHA-256 bytes>` rows. The D/F tree
uses the same root and the framing stated above. The exact five-row lstat
array, file-binding map, both tree digests, counts, and path arrays are one
content-bound object; a same-byte replacement with a different inode fails.

The old analyzer's `_strict_tree` derives allowed directories only from the
parents of expected files. That produces `.`, `xdg`, and `xdg/matplotlib`, so
it rejects the legitimate frozen empty `triton` directory. The immediately
following old code separately names all four exact directory paths and binds
the exact directory/file tree. The failure is therefore an analyzer
directory-schema integration bug, not cache drift.

The only permitted semantic repair is an explicit directory-aware no-follow
tree validator for this v3.3.3 cache. It must accept exactly the four directory
paths and one file above. It must independently lstat every entry; require the
root and directories to be real mode-`0700` directories and the file to be a
real mode-`0600` regular file; recompute its size, SHA-256, file-only tree, and
directory/file tree; and reject every missing, extra, symlink, hard-link
anomaly, special, wrong-mode, wrong-size, or wrong-byte entry. Empty
directories are not ignored generically: `triton` is accepted only because it
is an explicitly frozen member. No file or directory may be created, removed,
renamed, chmodded, normalized, or copied to make validation pass.

Every other v3.3.4.5 analyzer validator, order constraint, model-run binding,
publication rule, terminal classifier, source/signature gate, raw boundary,
confirmation guard, and claim boundary remains semantically unchanged. The
new implementation must be a separate standalone analyzer; it must not call
the old analyzer's `main()` or `analyze()`, monkeypatch the old module, or
modify the old analyzer. No generic relaxation of strict-tree validation is
allowed.

## 4. Fresh namespace, source authority, and exact freeze

The only five authorized new implementation paths are:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5_1_test.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.sh
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.py
experiments/interpretability/opensplice/encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.json
```

The analyzer/test/generator modes are Git `100644`; the shell is `100755`;
the freeze is `100644`, must be tracked at the clean launch HEAD, and must
match its external authorization. It is excluded from its own source
inventory.
That inventory has exactly 137 sorted unique rows partitioned as follows:

- the inherited 132 rows must match live bytes and Git blobs at model commit
  `0da8f47ea6e576a72a1cda204ce868ef79cc2ce5`;
- this amendment must match live bytes and its eventual dedicated docs commit;
- the analyzer, test, shell, and generator must match live bytes and their
  eventual source-authority implementation commit; and
- all 137 rows must also match the clean current launch HEAD before START and
  at every repeated gate.

No new row is claimed to exist at an earlier commit. Each source row has
exactly `path,sha256,size_bytes,git_mode,authority_commit`; `path` is
repository-relative POSIX text, SHA-256 is 64 lowercase hex, `size_bytes` is a
nonnegative JSON integer, and `git_mode` is `100644` or `100755`. The source
inventory has exactly eight keys:
`row_count,rows,authority_partitions,source_authority_head,source_authority_tree_exact,all_rows_authority_exact,all_rows_live_at_generation_exact,tree_sha256`.
`authority_partitions` has exactly the keys
`inherited_132,amendment,new_implementation_4`; each value has exactly
`authority_commit,row_count,paths`, with a full 40-hex commit, integer count,
and sorted repository-relative path array. Their counts are 132, one, and four
and their path arrays are disjoint. `source_authority_head` is the full 40-hex
source-authority implementation commit that contains the amendment and all
137 inventoried source rows but not the freeze. The three audit values are
true. `tree_sha256` is the SHA-256 of the canonical UTF-8 JSON serialization
of the sorted `rows` array.

The dependency graph is exactly acyclic and uses three commits. First, this
amendment is committed alone at the docs authority commit. Second, exactly the
four new inventoried analyzer/test/shell/generator files are committed at the
source-authority implementation commit; that clean commit contains all 137
inventoried rows and no v3.3.4.5.1 freeze. The generator runs against that
commit and writes its 40-hex identity to `source_authority_head`. Third, a
freeze-only launch commit adds exactly the generated freeze. The freeze does
not contain or predict that later launch commit. A post-commit external audit
then supplies the actual launch HEAD and freeze SHA/size to the wrapper. Before
START and at every repeated source gate, the analyzer requires the launch HEAD
to be tracked-clean and independently requires each of the 137 live rows to
equal both its partition authority blob and the same path's Git blob at the
externally authorized launch HEAD. Thus `source_authority_head` is historical
source authority, never a mislabeled current/launch HEAD.

The freeze has exactly these 20 top-level keys:

```text
schema_version
analysis_version
attempt_id
acknowledgement_token
amendment_path
amendment_sha256
amendment_commit
prior_model_head
prior_model_freeze_binding
source_inventory_contract
immutable_model_artifact_contract
consumed_analyzer_failure
consumed_analyzer_failure_content_binding
prior_cache_contract
prior_cache_contract_content_binding
analysis_attempt_dir
analysis_dir
publication_contract
record_contracts
claim_boundary
```

The literal identities are
`schema_version="v3.3.4.5.1-analysis-freeze-v1"`,
`analysis_version="v3.3.4.5.1-structural-analyzer-v1"`,
`attempt_id="v3.3.4.5.1-development-ood-sidecar-structural-analysis"`, and
`acknowledgement_token="--acknowledge-structural-only-v3-3-4-5-1"`.
`prior_model_freeze_binding` is an absolute three-key file binding
`path,sha256,size_bytes`. A content binding always has exactly
`sha256,size_bytes` and binds canonical compact sorted UTF-8 JSON with
`ensure_ascii=true` and `allow_nan=false`.

`immutable_model_artifact_contract` has exactly eight keys:
`run_root_binding,compiler_tree_binding,preflight_tree_binding,external_cache_tree_binding,model_cache_tree_binding,run_terminal_binding,raw_manifest_binding,old_analyzer_bundle`.
Each tree binding has exactly eight keys:
`root,file_count,directory_count,file_bindings,file_tree_sha256,directory_paths,directory_tree_sha256,directory_file_tree_sha256`;
`root` is normalized absolute, file-map keys are root-relative POSIX paths,
and every file-map value is exactly
`sha256,size_bytes,mode,st_dev,st_ino,st_nlink`. The old analyzer
bundle has exactly `git_head,analyzer,test,shell,freeze`, with the last four
values absolute file bindings.

`prior_cache_contract` has exactly ten keys:
`root,file_count,directory_count,directory_paths,lstat_rows,file_bindings,file_tree_sha256,directory_file_tree_sha256,exact_membership,no_follow`.
It contains the exact Section 3 values. Each of its five ordered `lstat_rows`
has exactly `path,entry_type,mode,size_bytes,st_dev,st_ino,st_nlink,sha256`.
`file_bindings` is a sorted one-row map whose value is exactly
`sha256,size_bytes,mode,st_dev,st_ino,st_nlink`.
`exact_membership` and `no_follow` are true.

`record_contracts` has exactly nine keys:
`start_keys,analysis_keys,complete_keys,failure_keys,publication_success_keys,publication_failure_keys,publication_audit_keys,output_state_keys,failure_phase_values`.
Each `*_keys` value is the POSIX-sorted array of the exact names defined below;
the presentation order below is not the serialized order. `claim_boundary`
has exactly ten true/false values:
`structural_only=true,no_biological_claim=true,no_scientific_summary=true,no_normalization=true,no_shapley=true,no_interaction=true,no_resolution=true,no_nomination=true,combined_analysis_permitted=false,future_protocol_required=true`.

`publication_contract` has exactly these 23 keys:
`schema_version,method,temp_name_regex,nonce_bytes,open_flags,initial_mode,sealed_mode,rename_flags,same_directory_required,keep_fd_open_through_rename,file_fsync_count,parent_fsync_required,post_publish_inode_revalidation_required,no_replace,no_fallback,no_retry,root_roles,success_keys,failure_keys,audit_keys,output_state_keys,entry_state_keys,failure_stages`.
Its literal values are the Section 5 schema/method/regex and stage/key arrays;
nonce bytes are 16, open flags are the five flags stated there, initial/sealed
modes are `0600`/`0400`, rename flags are `["RENAME_NOREPLACE"]`, root roles
are `["analysis_attempt","analysis_output"]`, file-fsync count is two, and
all seven required/no-replace/no-fallback/no-retry booleans are true.

A post-commit external authorization binds the freeze path/SHA/size and clean
launch HEAD. The inventoried sources accept and validate that authorization;
they do not contain the launch commit or hard-code the final freeze SHA,
preserving the three-commit acyclic build above.

The fresh append-only destinations are exactly:

```text
experiments/interpretability/opensplice/results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1_attempt
experiments/interpretability/opensplice/results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1
```

Both must be absent before the sole invocation. The four old analyzer
destinations in Section 2 remain permanently absent and are rechecked before
START, post-START, before each output publication, and before COMPLETE.
The shell accepts exactly these arguments in this order and no environment
substitute or bypass:

```text
--acknowledge-structural-only-v3-3-4-5-1
--authorized-git-head <40 lowercase hex>
--authorized-freeze-sha256 <64 lowercase hex>
--authorized-freeze-size-bytes <nonnegative decimal integer>
```

The three authorization values come from an external post-commit read-only
audit and must equal the current launch HEAD and live/tracked freeze bytes
before any attempt-root allocation.

## 5. Standalone append-only publication contract

The old v3.3.4.5 publication helper is hard-wired to old destinations and must
not be imported, called, copied at runtime, or monkeypatched. The new analyzer
contains its own stdlib/ctypes implementation with exactly two public internal
operations:

```text
ensure_publication_directory(root_role, first_final_relative_path, first_artifact_role) -> normalized absolute Path
publish_bytes(root_role, final_relative_path, payload, artifact_role) -> publication_success
```

`root_role` is exactly `analysis_attempt` or `analysis_output` and maps only to
the two fresh paths above. The helper opens and validates the fixed parent
no-follow, proves the final root absent, calls `mkdir` once at mode `0700`,
fsyncs the parent, opens/revalidates the root no-follow, proves it empty, and
thereafter binds `st_dev,st_ino` plus an open directory fd. Reallocation,
role/root mismatch, symlink, inode change, nonempty initial root, or a second
registration fails closed.
Root allocation errors raise the same local `PublicationError`, using the
planned first artifact/path, null temp path/ordinal, and absent file-entry
states; the separately captured output-state object records whether a root was
created before failure.

For every file, the helper uses a same-directory randomized name matching
`^\.v33451\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$`, opened with
`O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC` at mode `0600`. It writes all
bytes, fsyncs, fchmods to `0400`, fsyncs again, seeks and reads back exact
bytes/hash, then calls libc `renameat2(...,RENAME_NOREPLACE)`, fsyncs the parent,
and revalidates the final no-follow inode, mode `0400`, `st_nlink=1`, size, and
SHA-256 before closing. The fd stays open through rename. No `/proc/self/fd`,
`linkat`, ordinary rename, overwrite, delete, cleanup, fallback, or retry is
permitted. Operational failure raises a new local `PublicationError` carrying
the exact 19-key publication-failure object; no old exception/helper is used.

A publication-success object has exactly these 19 keys:

```text
schema_version,method,root_role,final_relative_path,temp_basename,
publication_ordinal,runner_pid,nonce_hex,sha256,size_bytes,mode,st_dev,st_ino,
st_nlink,file_fsync_before_rename,file_fsync_after_fchmod,
rename_noreplace_succeeded,parent_fsync_succeeded,
post_publish_revalidation_exact
```

The schema/method are
`v3.3.4.5.1-named-temp-renameat2-noreplace-v1` and
`named_temp_renameat2_noreplace`; the five operation booleans are true,
ordinal/PID/device/inode/link/size are nonnegative JSON integers excluding
booleans, mode is `0400`, and nonce is 32 lowercase hex.

A publication-failure object has exactly these 19 keys:

```text
schema_version,method,root_role,artifact_role,final_relative_path,
temp_relative_path,publication_ordinal,runner_pid,failure_stage,errno,
error_type,message,rename_noreplace_attempted,rename_noreplace_succeeded,
parent_fsync_attempted,parent_fsync_succeeded,temp_state,final_state,
created_at_unix_s
```

`failure_stage` is one of
`root_parent_open,root_parent_validation,root_final_preexistence,root_mkdir,root_parent_fsync,root_revalidation,parent_open,parent_validation,final_preexistence,temp_open,temp_write,file_fsync_before_rename,fchmod,file_fsync_after_fchmod,readback,rename_noreplace,parent_fsync,post_publish_revalidation`.
`temp_relative_path` and `publication_ordinal` are null only for a `root_*`
stage; otherwise the temp path matches the frozen grammar and ordinal is a
nonnegative integer. `errno` is null or a nonnegative integer.
An entry state has exactly
`state,entry_type,mode,size_bytes,sha256,st_dev,st_ino,st_nlink`; absent fields
other than `state="absent"` are null, while present entries bind all fields.
Failures before a temp is created create no entry; pre-rename failures after
creation preserve the temporary orphan; a successful rename followed by
parent-fsync/revalidation failure preserves a durability-uncertain final; an
EEXIST collision preserves exact preexisting source/destination states. No
entry created by an invocation is deleted or reused.

A publication-audit object has exactly these 15 keys:
`schema_version,method,successful_final_count_before_terminal,successful_final_bindings_before_terminal,temporary_orphan_count,temporary_orphan_bindings,durability_uncertain_final_count,durability_uncertain_final_bindings,preexisting_entry_count,preexisting_entry_states,publication_failure,no_new_entry_failure,no_publication_retry,no_published_final_deleted,no_temp_or_final_reused`.
Every map is POSIX-sorted and root-role qualified as
`analysis_attempt/<relative>` or `analysis_output/<relative>`.
Successful, temporary-orphan, and durability-uncertain binding-map values have
exactly `sha256,size_bytes,mode,st_dev,st_ino,st_nlink`; preexisting-state map
values use the eight-key entry-state schema.

An output-state object has exactly these 13 keys:
`state,root_role,root_lstat,regular_final_bindings,temporary_orphan_bindings,durability_uncertain_final_bindings,preexisting_entry_states,directory_paths,directory_tree_sha256,directory_file_tree_sha256,file_tree_sha256,entry_state_tree_sha256,publication_failure`.
`state` is `absent`, `published_prefix`, or `publication_failure_prefix`.
`root_lstat` is null only for an absent root, otherwise an eight-key entry
state. File trees use normalized root-relative POSIX paths and the Section 3
file-only framing. `directory_tree_sha256` frames each sorted directory as
`D\0<relative>\0`; `directory_file_tree_sha256` appends the sorted Section 3
`F` rows after those `D` rows. Empty directories are preserved.

If START directory allocation or START publication fails, no START/terminal is
invented; the invocation is consumed and partial entries are preserved. After
START, any failure triggers one attempt to publish `ANALYSIS_FAILURE.json` in
the attempt root. Failure of that publication is terminal-less, preserves the
exact partial state, and is never recursively retried.

The lifecycle membership table is authoritative. Its two `successful final`
columns list only entries in `regular_final_bindings`; they deliberately
exclude physical regular files recorded in
`durability_uncertain_final_bindings` or `preexisting_entry_states`, and they
exclude temporary orphans. Every row additionally permits exact failed-entry
maps in either allocated root when the named publication primitive reached
that state. Such entries are preserved, never promoted to successful finals,
and remain part of whole-tree validation.

| outcome | attempt-root successful finals | output-root successful finals | terminal |
|---|---|---|---|
| pre-START gate failure | root absent | root absent | none |
| START allocation/publication failure | none | root absent | none; exact attempt-root orphan/uncertain/preexisting maps if created |
| post-START failure, failure record published | `ANALYSIS_ATTEMPT_STARTED.json`, `ANALYSIS_FAILURE.json` | absent or an exact successful `RESULT.md`/`ANALYSIS.json` prefix | `ANALYSIS_FAILURE.json`; exact failed-entry maps allowed in both roots |
| failure-record publication failure | `ANALYSIS_ATTEMPT_STARTED.json`, no successful failure final | absent or an exact successful `RESULT.md`/`ANALYSIS.json` prefix | none; exact failed-entry maps allowed in both roots, including failed `ANALYSIS_FAILURE.json` |
| success | `ANALYSIS_ATTEMPT_STARTED.json`, `ANALYSIS_COMPLETE.json` | `RESULT.md`, `ANALYSIS.json` | `ANALYSIS_COMPLETE.json` |

No unbound filesystem entry is legal. A physical regular failed entry is
legal only if it appears in exactly one of the temporary-orphan,
durability-uncertain-final, or preexisting-entry maps and is excluded from
`regular_final_bindings`. Whole-tree validation enumerates every lstat entry,
requires the disjoint union of successful and failed-entry maps to equal the
physical tree exactly, and rejects an extra, missing, duplicated, or
misclassified entry. This includes a durability-uncertain
`ANALYSIS_COMPLETE.json` after `complete_publication`, and failed/preexisting
RESULT or ANALYSIS entries. Every created root and partial entry is described
by an output-state object; pre-START gate failure is the only case with both
roots necessarily absent.

## 6. Literal record schemas

`ANALYSIS_ATTEMPT_STARTED.json` has exactly these 22 keys:

```text
status,schema_version,analysis_version,attempt_id,acknowledgement,git_head,
external_freeze_authorization,freeze_binding,analyzer_binding,test_binding,
shell_binding,generator_binding,amendment_binding,run_terminal_binding,
source_inventory_attestation,immutable_input_audit,consumed_analyzer_failure,
consumed_analyzer_failure_content_binding,prior_cache_audit,
prior_cache_audit_content_binding,fresh_paths,started_at_unix_s
```

Status/schema are `started` / `v3.3.4.5.1-analysis-start-v1`.
Acknowledgment equals the frozen token. The authorization has exactly
`git_head,freeze_path,freeze_sha256,freeze_size_bytes,live_equals_git_show,tracked_clean,authorization_source`.
The final two booleans are true and `authorization_source` is exactly
`external_post_commit_audit`.
All named bindings are absolute `path,sha256,size_bytes` objects.
`source_inventory_attestation` has exactly 11 keys:
`row_count,rows,authority_partitions,source_authority_head,launch_git_head,source_authority_tree_exact,all_rows_authority_exact,all_rows_live_exact,all_rows_launch_head_exact,launch_head_tracked_clean,tree_sha256`.
Its rows, partitions, authority head, and tree digest equal the frozen source
contract; `launch_git_head` equals the external authorization; all five audit
booleans are true. This runtime object deliberately differs from the
eight-key generation-time source contract because only the post-freeze launch
commit can be the current HEAD.
`immutable_input_audit` has exactly 13 true booleans:
`inherited_132_live_exact,inherited_132_historical_exact,amendment_live_historical_exact,new_implementation_4_live_historical_exact,source_authority_exact,launch_head_clean,current_137_live_launch_exact,immutable_model_run_exact,immutable_preflight_cache_exact,consumed_failure_exact,prior_cache_exact,old_destinations_absent,new_destinations_fresh`.
The two embedded evidence objects equal their frozen objects and bindings. `fresh_paths`
has exactly six normalized absolute-path keys
`old_v3345_attempt,old_v3345_output,old_v333_attempt,old_v333_output,new_attempt,new_output`,
all with value `absent`; both `new_destinations_fresh` and this map explicitly
record the immediately pre-allocation state rather than the state at later
START validation. Timestamp is finite JSON number.

`ANALYSIS.json` has exactly these 26 keys:

```text
status,decision,analysis_version,analysis_attempt_start_binding,run_binding,
preflight_binding,external_cache_binding,model_cache_binding,
source_and_prior_audit,consumed_analyzer_failure_audit,prior_cache_audit,
compiler_and_signature_audit,dispatch_journal_audit,raw_prefix_audit,
control_audit,terminal_audit,publication_audit,confirmation_boundary,
claim_boundary,scientific_summary_computed,donor_normalization_computed,
shapley_or_nomination_computed,interaction_or_resolution_computed,
nomination_performed,combined_analysis_permitted,completed_at_unix_s
```

The only allowed status/decision are
`complete_controlled_stop_structural_archive` /
`controlled_stop_diagnostic_provenance_failure`; analysis version is frozen.
`analysis_attempt_start_binding` is the absolute three-key file binding
`path,sha256,size_bytes`. `run_binding`, `preflight_binding`,
`external_cache_binding`, and `model_cache_binding` are each the exact
eight-key tree binding defined in Section 4; they are not three-key file
bindings. `source_and_prior_audit` has exactly these 13 true keys:
`inherited_132_live_exact,inherited_132_historical_exact,amendment_live_historical_exact,new_implementation_4_live_historical_exact,source_authority_exact,launch_head_clean,current_137_live_launch_exact,immutable_model_run_exact,immutable_preflight_cache_exact,consumed_failure_exact,prior_cache_exact,old_destinations_absent,active_attempt_exact`.
The consumed failure and prior-cache audits equal the frozen objects.
`compiler_and_signature_audit` has exactly
`terminal_status,stop_reason,source_program_exact,signature_adapter_exact,stablehlo_exact,prebackend_exact,entry_abi_exact,successful_compile_count,diagnostic_provenance_complete,compiled_backend_diagnostic_only`,
with literal values
`controlled_stop_diagnostic_provenance_failure,diagnostic_persistence_failure,true,true,true,true,true,1,false,true`.
`dispatch_journal_audit` has exactly
`expected_record_count,expected_apply_count,valid_record_count,started_count,completed_count,raw_tree_sha256,started_tree_sha256,completed_tree_sha256`,
with 80, 320, three zero counts, and three empty SHA-256 values.
`raw_prefix_audit` has exactly
`manifest_binding,status,artifact_count,failed_current_binding,raw_directory_absent,journal_directories_absent`,
with status `empty_controlled_stop`, zero, null, true, true.
`control_audit` has exactly
`control_state_eligible,all_80_complete,id0_all20,id255_all20,six_row_compile_count,identity_rerun_count,main_cube_rerun_count,old_records_reused`,
all false/zero. `terminal_audit` has exactly
`status,stop_reason,terminal_linkage_exact,count_arithmetic_exact,no_retry,failure_exact,publication_exact,imports_protobuf_exact`.
Its first two values are the exact model status/reason above and its six
booleans are true.
`confirmation_boundary` has exactly
`confirmation_paths_opened,confirmation_model_calls,later_exon_metadata_label_exposure_disclosed,model_outputs_activations_interventions_blind,no_confirmation_scientific_access`,
with false, zero, true, true, true. Claim boundary equals the frozen ten-key
object. All six top-level science/combined flags are false.
On the successful path its `publication_audit` binds exactly the already
published START and RESULT finals (count two) and no orphan, uncertain,
preexisting, or failure entry; ANALYSIS cannot claim its own later publication.

`ANALYSIS_COMPLETE.json` has exactly these 11 keys:
`status,schema_version,analysis_version,attempt_id,start_binding,analysis_binding,result_binding,attempt_tree_before_complete,output_tree_complete,publication_audit,completed_at_unix_s`.
Status/schema are `complete` / `v3.3.4.5.1-analysis-complete-v1`.
The START, `ANALYSIS.json`, and `RESULT.md` bindings are exact. The attempt tree
before COMPLETE contains only START; the output tree contains exactly RESULT
and ANALYSIS. Tree bindings use the exact eight-key schema in Section 4.
Its publication audit binds exactly those three already published finals
(START, RESULT, ANALYSIS), with all failure-state maps empty/null.

`ANALYSIS_FAILURE.json` has exactly these 13 keys:
`status,schema_version,analysis_version,attempt_id,start_binding,failure,failure_phase,raw_access_reached,analysis_output_state,attempt_output_state,publication_audit,old_destinations_absent,failed_at_unix_s`.
Status/schema are `failure` / `v3.3.4.5.1-analysis-failure-v1`; `failure` has
exactly `type,message,traceback`; `raw_access_reached` must remain false and
`old_destinations_absent` true. `failure_phase` is one of
`post_start_source_gate,post_start_prior_cache_gate,model_input_rehash,structural_terminal_audit,result_publication,analysis_publication,final_toctou,complete_publication`.
Both output-state fields use the 13-key Section 5 schema.

All key lists above are authoritative and are stored as POSIX-sorted arrays of
those exact names in `record_contracts`; their counts are respectively START 22,
ANALYSIS 26, COMPLETE 11, FAILURE 13, publication success 19, publication
failure 19, publication audit 15, and output state 13.

## 7. One-shot order, outputs, and tests

Before allocating the new attempt directory, a stdlib-only precheck must:

1. prove forbidden JAX/jaxlib/AlphaGenome/model modules are absent;
2. validate the phase-partitioned 137-row source authority and clean current
   HEAD exactly as Section 4 specifies;
3. rehash all exact v3.3.4.5 model-run/preflight/cache/compiler/terminal/import/
   protobuf artifacts and both older consumed prefixes without opening raw;
4. validate the old analyzer bundle, consumed failure object, all old/new
   destination states, and exact directory-aware v3.3.3 cache; and
5. recompute every frozen content binding.

The destination assertion in this pre-START gate is exactly: all four old
destinations are absent and both new destinations are absent. Only then may it
allocate the new attempt root and publish START. The persisted `fresh_paths`
map and `new_destinations_fresh=true` are historical evidence of this
pre-allocation observation; they are never reused as a live post-START absence
predicate.

After START, the analyzer repeats every stdlib source-authority, launch-HEAD,
prior-run, preflight, cache, consumed-failure, content-binding, and old-four-
destinations-absent gate. Its phase-specific new-destination predicate is now
exactly: the attempt root exists with mode `0700` and contains only
`ANALYSIS_ATTEMPT_STARTED.json` mode `0400`, whose live SHA/size, attempt ID,
acknowledgement token, and external authorization equal the active in-memory
binding; the output root remains absent. It must not assert that the new
attempt root is absent. Only after that repeated gate may it call the internal
analysis function, which requires an unforgeable module-private token plus the
exact live START SHA and rejects direct calls.

Immediately before RESULT, the same post-START predicate and all non-
destination gates repeat. Immediately before ANALYSIS, the attempt prefix is
still exactly the singleton START and the output prefix is exactly RESULT,
both with their frozen modes and live bindings. Immediately before COMPLETE,
the attempt prefix is exactly the singleton START and the output prefix is
exactly RESULT plus ANALYSIS. Error handling uses only the exact partial
prefixes and publication states in Section 5; it never substitutes the
pre-START all-absent predicate after allocation. At every phase the four old
destinations remain absent. The implementation is standalone and never
imports/calls/patches the old analyzer or publication helper.

The successful output root contains exactly `RESULT.md` and `ANALYSIS.json`;
the successful attempt root contains exactly START and COMPLETE. The analyzer
does not open any raw record: it validates the empty manifest and absent raw/
journal directories structurally, with a read sentinel proving no raw access.
Any nonempty raw state fails closed. The model run, preflight, caches, and old
analyzer bytes are immutable read-only inputs; no model/GPU path is allocated.

Before commit or invocation, CPU-only tests must cover:

- the exact live v3.3.3 cache passing, including every frozen lstat field and
  the empty `triton` directory;
- removal/tamper of every expected directory and file, every extra entry
  including empty dirs, symlinks, hard-link count, FIFO/special entries, modes,
  devices, inodes, sizes, hashes, file tree, D/F tree, framing, and root base;
- generic strict-tree rejection of an unlisted empty directory;
- every old analyzer bundle/failure field, stdout/stderr hash/size/nullability,
  exact traceback, destination state, and content binding;
- separate old-132/docs/new-4 historical authority, source-authority commit,
  current-137 live/launch-HEAD equality, mode/hash/size/path tamper, freeze
  key/count, and the exact docs -> source-authority -> freeze-only launch
  acyclic rebuild;
- real no-side-effect precheck on the immutable current filesystem;
- publication success plus a real injected failure at every named stage,
  EEXIST, pre-temp no-entry, temp orphan, uncertain final, inode/root TOCTOU,
  START failure, result/analysis failure, failure-terminal failure, and no
  cleanup/fallback/retry;
- literal serializer parity for all 20 freeze keys and every record/nested key
  list/count/type/status/nullability/tree/membership above;
- pre/post-START ordering, active-token/direct-call rejection, source drift
  before model-run/raw reads, old destination mid-analysis appearance, and
  final TOCTOU;
- the exact apply-zero diagnostic-persistence result and prohibition of any
  raw/scientific/confirmation access; and
- AST/import checks forbidding JAX, model code, old analyzer/helper imports or
  monkeypatches, model/GPU execution, and changed claim flags.

The stopped bytes, tests, freeze, modes, clean HEAD, live dry precheck, and
fresh paths require independent read-only audit before commit and again after
commit before exactly one CPU-only wrapper invocation. A failed invocation is
never retried. A successful analyzer authorizes only later independent audit
and scoped archival of exact immutable artifacts; it never authorizes science,
confirmation access, or another model run.

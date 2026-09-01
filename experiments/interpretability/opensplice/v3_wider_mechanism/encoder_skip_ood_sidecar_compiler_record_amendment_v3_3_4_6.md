# OpenSplice v3.3.4.6 compiler-record provenance amendment

Status: prospective infrastructure-only amendment. This document must be
committed before any v3.3.4.6 implementation freeze, preflight, model import,
GPU work, output allocation, or analysis attempt.

## 1. Scope and scientific boundary

This amendment authorizes at most one fresh development-only OOD-sidecar
attempt. It does not authorize a rerun or continuation of the immutable v3.3
256-coalition main cube, any six-row intervention, any identity rerun, any old
record reuse, or any confirmation-exon model call. The sidecar remains exactly
the inherited 20-recipient by four-anchor design: 80 records, 320 counted model
applies, one eight-row executable, anchors `[0,127,128,255]`, zero six-row
compiles, zero main-cube applies, and zero confirmation calls.

The only prospective model-run change is the source of the historical
v3.3 compiler record passed to the already-frozen compiler-provenance
serializer. All model inputs, checkpoint bytes, cases, target scalar, route
masks, row order, intervention tensors, source-program gate, compiler-attempt
budget, diagnostic derivations, cache isolation, OOD predicates, controls,
analysis thresholds and claim boundary remain unchanged from v3.3.4.5 and its
incorporated predecessors.

The v3.3.4.5 attempt and v3.3.4.5.1 structural archive are immutable. They
must not be deleted, renamed, modified, retried, imported as cache input, or
used as a resumable output prefix.

## 2. Frozen predecessor facts and cause

The v3.3.4.5 implementation commit is
`0da8f47ea6e576a72a1cda204ce868ef79cc2ce5`. Its freeze is 204,697 bytes,
mode `100644`, SHA-256
`2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366`,
with exact 86 top-level keys and 132 source rows.

Its one authorized GPU attempt stopped after one lower and one successful
compile but before dispatch. `RUN_COMPLETE.json` is 43,760 bytes, SHA-256
`fdbd0a1dc7d24145f88c5a009cc80d8904e57920e0c9584426e791373fae6d8f`,
with status `controlled_stop_diagnostic_provenance_failure`, reason
`diagnostic_persistence_failure`, and persisted failure
`DiagnosticPersistenceFailure: 'eight_row_compiler'`. Its empty
`RAW_MANIFEST.json` is 1,562 bytes, SHA-256
`3ee95b22d483c7c4f234fbb75281e05e84f0be263b1ee670a94b2cd442d61136`.
The exact run contains 12 regular files and directories
`.,compiler,compiler/eight_row`; its file-only and directory/file tree digests
are respectively
`960faf1675caaa0f3c9798f7b943998e91650b2a9359b26ca31ce2d417c2ce0b`
and
`5331e5041b557a4324ba57d57179dc2bfd8e6ab981f0cdede5d4f126e94c2041`.
There were zero dispatch-ledger entries, model applies, raw records, identity
reruns, main-cube reruns, old-record reuse and confirmation calls. Every
scientific, normalization, Shapley, interaction, resolution, nomination and
combined-analysis flag is false.

The v3.3.4.5.1 amendment is commit
`564a01dc2981d57c8f8298f3efca5b22fcb381e0`, document SHA-256
`16af8ccb65f3e08739c3792c5c9ab3affcb19a3ca9993904260729a898afd5c4`.
Its source-authority commit is
`dfa56d90c035c3aa370c65b79197820dd5787c92`; its freeze-only launch commit is
`eeadee88b747acc75e9437b5f2d1e7e3aab9701c`. The analysis freeze is 85,875
bytes, mode `100644`, SHA-256
`3c5405e8d9aadbe8f594fbc262a155a669bfbf67301898dee687aeeb2e286d9f`,
with exact 20 top-level keys and 137 source rows. The final archive commit is
`c292622e5732329cbee50575381682519017ac68`; its four exact payloads are:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `ANALYSIS_ATTEMPT_STARTED.json` | 67,913 | `ada53945670529d24c396514e31ed5155c4c3da0c2d88d2d86d93cfc4bbfc9c1` |
| `ANALYSIS_COMPLETE.json` | 4,672 | `1512aefe1613a81ed7532a1c66cb270a005aeb3f1d9a6c82a0558bd702b78277` |
| `ANALYSIS.json` | 18,633 | `5090675905789aff6a290dcbeeabefe3a8a2938ebc95ecdddd296a3f0ca31a6f` |
| `RESULT.md` | 504 | `5d2f30d217c3324d79097e256ffa4ab52d3469ba6453c59ac866191ab8bfdffd` |

That archive independently proves the controlled stop and the zero-science
boundary. It is not a biological result.

The exact software defect is in the normal post-compile serializer call. The
v3.3.4.5 runner supplies
`launch['v3_3_3_run']['eight_row_compiler']` as the first historical compiler
argument to `_compiler_artifacts`. `validate_v3_3_3_run()` deliberately returns
an apply-zero structural summary and has no `eight_row_compiler` key. The
lookup therefore raises `KeyError('eight_row_compiler')`; the broad diagnostic
construction boundary correctly wraps it as `DiagnosticPersistenceFailure`
and emits the observed controlled stop.

The serializer's first historical argument is named
`original_v3_3_compiler`, and its `comparisons.v3_3` branch compares against
the original v3.3 compiler provenance. The exact required object already
exists in the Gate-A record at
`launch['gate_a']['original_run']['eight_row_compiler']`, returned by
`validate_original_run()`. Its second historical/source-gate argument remains
exactly `launch['gate_a']['v3_3_2_run']['eight_row_compiler']`.

## 3. Exact repair contract

The successful post-compile call must pass these objects in this order:

```text
_compiler_artifacts(
    lowered,
    compiled,
    compile_seconds,
    launch['gate_a']['original_run']['eight_row_compiler'],
    launch['gate_a']['v3_3_2_run']['eight_row_compiler'],
    adapted_signatures,
    ... unchanged keyword arguments ...,
)
```

The production v3.3.4.6 runner must contain no lookup of
`v3_3_3_run.eight_row_compiler`. It must not add that key to the v3.3.3
summary, reinterpret the v3.3.3 structural stop as an original compiler
record, catch or suppress the lookup, synthesize a compiler object, weaken any
comparison, or classify a resulting provenance mismatch as success.

The first compiler record must be the exact mapping object authenticated by
Gate A as `original_run.eight_row_compiler`; the second must be the exact
mapping object authenticated as `v3_3_2_run.eight_row_compiler`. Tests must
inject distinct sentinel mappings and prove positional identity at the real
post-compile call site. The source-program gate remains anchored to the second
object exactly as in v3.3.4.5.

Any genuine graph extraction, entry-ABI, backend-diagnostic, persistence,
cache-signal, source-gate, publication or compiler-attempt failure retains the
inherited exact controlled-stop or terminal-failure route. This amendment does
not turn any failure into a pass.

## 4. Fresh namespace and one-shot lifecycle

The exact fresh paths are:

```text
results/v3_3_4_6_preflight_kernel_cache
results/v3_3_4_6_device_preflight
results/v3_3_4_6_model_kernel_cache
results/v3_3_4_6_development_ood_sidecar_one_shot
results/v3_3_4_6_development_ood_sidecar_analysis_attempt
results/v3_3_4_6_development_ood_sidecar_analysis
```

All six must be absent, including symlinks and special entries, before the
freeze is generated, before external launch authorization, and immediately
before their phase-specific allocation. Every v3.3.4.5 and earlier production
path retains its inherited immutable present/absent state. No path is reused.

The ordering remains:

1. stdlib-only Gate A rehashes the committed freeze, source inventory,
   predecessor archives, exact current HEAD and all freshness predicates;
2. the fresh external cache is allocated and the external device preflight is
   run and durably validated;
3. the fresh model cache is allocated and sanitized;
4. the same-process device/runtime gate runs;
5. the fresh model-run root and atomic START are published;
6. Gate B rehashes all pre-model evidence;
7. only then may scientific imports, checkpoint/model construction, one lower,
   one compile and the source-program/diagnostic gates occur;
8. only a complete passing compiler record may precede dispatch;
9. the exact 80 records and 320 counted applies run once in frozen order;
10. one terminal is published; no retry or resume is permitted.

If any step consumes a v3.3.4.6 path, the attempt is consumed regardless of
whether a terminal can be published. There is no retry after success, failure,
controlled stop, crash, timeout or terminal-less prefix.

## 5. Predecessor archive binding

The v3.3.4.6 freeze must contain one exact top-level object named
`prior_v3_3_4_5_controlled_stop_archive`. It binds:

- the v3.3.4.5 implementation commit, freeze path/SHA/size/mode and exact
  132-row source contract;
- the exact six-root v3.3.4.5 filesystem state, including preflight/cache/run
  directory-aware trees and both analysis-path absences at the time of the
  old attempt;
- the exact 12-file run membership, START, terminal, empty manifest, compiler
  diagnostic artifact, zero ledgers/raw counts and every no-science flag;
- the v3.3.4.5.1 amendment/source/freeze/archive commits and bindings;
- the exact 2+2 v3.3.4.5.1 archive membership, modes, four payload hashes,
  START/ANALYSIS/COMPLETE linkage and structural-only decision; and
- `old_model_or_analyzer_retry_permitted=false` and
  `old_paths_are_cache_inputs=false`.

The object has a canonical JSON content binding (`sort_keys=true`, compact
separators, UTF-8, `allow_nan=false`) stored beside it in the freeze under the
exact key
`prior_v3_3_4_5_controlled_stop_archive_content_binding`. The external
preflight record, START, every post-START terminal and each analyzer
START/result/failure must copy that exact binding under the same key. The
nested `START.successful_preflight` object retains its inherited exact 12-key
schema; the copied binding is a START sibling, not a thirteenth nested key.
Every record also binds the exact authorized freeze SHA. This is a transitive
archive binding, not permission to import or execute old analyzers.

The repaired runtime schemas otherwise retain the exact v3.3.4.5 key sets and
nullability. Adding the one copied predecessor content binding makes the exact
top-level counts: external preflight 23, START 39, post-START provenance
failure 25, common `RUN_COMPLETE` 69, publication terminal 36,
nonpublication terminal 63, analysis START 19, successful ANALYSIS 28 and
analysis failure 20. The implementation must display and serializer-test the
literal sorted key list for each record; these prose counts are not a license
to infer missing keys.

## 6. Freeze and commit graph

The prescribed graph is acyclic:

1. commit this amendment alone;
2. from that exact docs commit, commit exactly the eleven new v3.3.4.6
   implementation/test/shell/generator sources, with two wrappers mode
   `100755` and all other source files mode `100644`;
3. generate the freeze from that clean source-authority commit;
4. commit exactly the freeze as the sole child delta used for launch;
5. externally authorize the exact launch HEAD, freeze SHA and size.

The exact new implementation paths are the v3.3.4.6 versions of:

```text
analyze_encoder_skip_ood_sidecar.py
analyze_encoder_skip_ood_sidecar.sh
analyze_encoder_skip_ood_sidecar_test.py
generate_encoder_skip_ood_sidecar_freeze.py
launch_encoder_skip_ood_sidecar.py
run_device_preflight.py
run_device_preflight_test.py
run_encoder_skip_ood_sidecar.py
run_encoder_skip_ood_sidecar.sh
run_encoder_skip_ood_sidecar_test.py
validate_encoder_skip_ood_sidecar_bootstrap.py
```

with the literal suffix `_v3_3_4_6` before each extension. The freeze excludes
itself. Its source inventory is derived from the authenticated 132 rows of the
v3.3.4.5 freeze plus this amendment and the eleven listed sources: exactly 144
sorted unique rows. The freeze has exactly 88 top-level keys: the inherited
v3.3.4.5 86-key structure after replacing versioned current-path/source/schema
contracts, plus `prior_v3_3_4_5_controlled_stop_archive` and
`prior_v3_3_4_5_controlled_stop_archive_content_binding`. The two older
consumed-preflight prefix keys remain unchanged and present. The generator must
test the literal set rather than relying on this arithmetic.

All 144 rows bind path, SHA-256, size, Git mode and historical authority
commit. At launch they must equal both live bytes and `git show` bytes/modes at
the authorized HEAD. The docs commit must be the sole parent of the
source-authority commit with exactly eleven additions; the freeze-only launch
commit must be the sole child with exactly one addition.

## 7. Required CPU tests and audit gates

Before freeze generation, isolated CPU tests must cover at least:

- the exact real-call-site compiler mapping repair, with distinct sentinel
  objects and rejection of swapped, missing, copied, synthesized or v3.3.3
  compiler inputs;
- a runner-shaped successful compiler-record construction that passes source,
  signature, entry-ABI, backend-diagnostic, cache and same-object gates and
  reaches the pre-dispatch boundary without `DiagnosticPersistenceFailure`;
- every inherited compiler/diagnostic controlled-stop route and publication
  fallback, proving the repair does not relabel real failures;
- exact 80-record/320-apply ordering, one compile, zero six-row/main-cube/old
  reuse, zero confirmation and no retry;
- exact v3.3.4.5 and v3.3.4.5.1 predecessor object reconstruction, every leaf
  tamper, canonical binding, filesystem mode/type/inode/link/hash/membership
  tamper and old-path preservation;
- Gate-A/Gate-B ordering, six fresh paths, cache-role isolation, completed
  preflight validation, same-process gate and no scientific import before
  START/Gate B;
- exact literal record keysets and nullability for every successful and
  terminal path, including the new predecessor content binding;
- source-inventory 144-row membership/order/path/SHA/size/mode/authority
  tamper, exact docs-to-source and source-to-freeze commit deltas, and
  byte-identical deterministic freeze rebuild;
- named-temp/no-replace publication success and every inherited failure stage,
  root/path/final/temp TOCTOU, special entries, short writes, orphan/uncertain
  state, terminal-less unbindable state and no fallback; and
- CPU analyzer guards proving no JAX, jaxlib, AlphaGenome model, confirmation,
  raw-record or scientific value is opened unless the complete sidecar and all
  structural gates make combined analysis eligible.

An external postcommit dry run must report exact version, HEAD/freeze
authorization, 20 recipients, anchors `[0,127,128,255]`, 80 records, 320
applies, one eight-row compile, zero six-row/main-cube/identity/confirmation
work and `production_paths_created=false`.

Only after independent exact-byte audits of the committed source and freeze,
a clean tracked/index state, byte equality for the full source inventory, and
six absent current paths may the one GPU wrapper invocation be authorized.

## 8. Analysis and claim rule

The frozen v3.3.4.6 analyzer may combine the immutable v3.3 main cube with the
new sidecar only if all 80 sidecar records and 320 applies are complete and
valid, all source/compiler/cache/publication gates pass, all required ID0,
ID255, no-op, donor, self, repeat, unrelated-donor and OOD predicates pass,
and the inherited analysis contract marks the combined archive eligible.

If eligible, it may compute only the preregistered skip-resolution Shapley,
interaction, resolution and control summaries from the already-frozen v3.3
main cube plus the new OOD controls. It must not change thresholds, choose a
new endpoint, inspect confirmation internals, run new model calls, or claim a
biological mechanism from route-level attribution. A resolution may be
nominated only under the inherited cross-exon and control rules.

If any gate fails or the attempt stops early, the analyzer produces a
structural archive only and sets all scientific, normalization, Shapley,
interaction, resolution, nomination and combined-analysis flags false. Any
subsequent scientific attempt requires another prospective amendment.

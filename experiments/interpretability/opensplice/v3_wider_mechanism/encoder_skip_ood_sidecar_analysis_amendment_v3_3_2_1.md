# OpenSplice v3.3.2.1 controlled-stop analyzer amendment

Status: **prospective, docs-only, and CPU-only**. This document authorizes no
model, JAX, GPU, preflight, endpoint-value, activation, intervention, or
confirmation access. A separately versioned analyzer bundle must be committed
and hash-bound before its sole append-only invocation.

The only permitted result is a structural archive of the consumed v3.3.2
`compiler_graph_mismatch` stop. This amendment does not authorize a scientific
summary, a Shapley calculation, a resolution analysis, a nomination, or a new
model attempt.

## 1. Immutable v3.3.2 attempt

The model attempt is consumed and immutable at:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_2_development_ood_sidecar_one_shot`

The prospective analyzer must independently require the following exact
bindings before opening any compiler artifact:

| Binding | Exact value |
|---|---|
| Model-run Git commit | `24e2214168eeca41d4f3b60b62094b6befcadcc1` |
| Original v3.3 protocol SHA-256 | `85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0` |
| v3.3.2 amendment SHA-256 | `42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3` |
| v3.3.2 freeze SHA-256 | `baa257dfa8d0bd1afca40b11038d71c9157499eeb654523ad59df119eb2c0295` |
| Frozen v3.3.2 analyzer SHA-256 | `90f00a6d51f33ac456a0fd799f2b9caf456b58944928886dc1577731707f205e` |
| Frozen v3.3.2 analyzer-test SHA-256 | `2e7424a76840c776ff550db3961c1fb1b63dee273bd8d030800efd27bf81d11e` |
| `ATTEMPT_STARTED.json` SHA-256 | `d1bb85b3c35d892f10f67a5510b85784eb1b6b989c9ae32290b8c0fdc3c8d235` |
| `RUN_COMPLETE.json` SHA-256 | `d88c8becc96a24f1f56a1cee82da35fdd331ee3b40ba4108ae98f44ae24b4ad7` |
| `RAW_MANIFEST.json` SHA-256 | `fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd` |
| All three import-provenance SHA-256 values | `a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99` |
| `PROTOBUF_PROVENANCE.json` SHA-256 | `2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8` |
| Whole run file count / tree SHA-256 | `11` / `4ac66e45a4d7d65af2785904d11b23bf7e809e07f3f617e190772242b2e7a4ab` |
| Compiler file count / tree SHA-256 | `4` / `4378048568ff58a2bbee55ba9da750498b89fdef72c97911815cf895c8a8b7d1` |

The tree framing is the v3.3 framing: sort POSIX paths relative to the stated
root; for each regular non-symlink file append the UTF-8 path, one NUL byte,
and the 32 raw SHA-256 bytes; then SHA-256 the concatenation. Directory rows,
sizes, newlines, and hexadecimal digest text are excluded. Reject every extra,
missing, symlinked, or special file and every extra or missing directory. The
only directories are the run root, `compiler`, and `compiler/eight_row`; there
is no `raw` directory.

The exact 11-file inventory is:

| Relative path | Size (bytes) | SHA-256 |
|---|---:|---|
| `ATTEMPT_STARTED.json` | 774186 | `d1bb85b3c35d892f10f67a5510b85784eb1b6b989c9ae32290b8c0fdc3c8d235` |
| `IMPORT_PROVENANCE.json` | 41558 | `a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99` |
| `IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json` | 41558 | `a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99` |
| `IMPORT_PROVENANCE_PRE_MODEL.json` | 41558 | `a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99` |
| `PROTOBUF_PROVENANCE.json` | 3339 | `2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8` |
| `RAW_MANIFEST.json` | 145 | `fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd` |
| `RUN_COMPLETE.json` | 15145 | `d88c8becc96a24f1f56a1cee82da35fdd331ee3b40ba4108ae98f44ae24b4ad7` |
| `compiler/eight_row/COMPILER_PROVENANCE.json` | 8196 | `bd20e21a56a9ca5498d7119771bb1da9ac2e156ed3190d3ce3aa09ff2d2e312c` |
| `compiler/eight_row/graph.compiled.hlo.txt` | 16601836 | `b436435ebb14b87cf9929ee9b16fc2c74d1764460c701f8160f1dc092687b718` |
| `compiler/eight_row/graph.pre_backend.hlo.txt` | 1829833 | `675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750` |
| `compiler/eight_row/graph.stablehlo.mlir` | 3196162 | `69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd` |

The external device-preflight directory has exactly five regular files and
tree SHA-256
`797211382478ba249fe94e7ccbcc11c7192d30423d5e289ce03cf3cea37f65f5`.
Its `preflight_0000.json` is 682921 bytes with SHA-256
`309bdec9544cd4ff2cdbc83207663e81d7f5d27ece83e4ad569e1859ba412a8c`;
`.allocation.lock`, `.preflight_0000.reserved`, `preflight_0000.stderr.log`,
and `preflight_0000.stdout.log` are empty and each has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The START-bound v3.3.1 audit was already complete before this model attempt.
It records `controlled_stop_ood_tooling_failure`, no Shapley calculation, and
no nomination. Its attempt tree is two files with SHA-256
`b0e788f0df3db1678ca410da7b0c409a18ceeaa6ddcbb97c61a155188a6e719f`;
its output tree is two files with SHA-256
`f3e6eee31c3fc978356a5766c190061ae3f8fd709da6c5c0836f7ce3d47de8f0`.
The prospective analyzer must verify the complete START-bound v3.3.1 object,
not replace it with only these two digests.

## 2. Exact stop and claim boundary

`RUN_COMPLETE.json` records:

- `status=controlled_stop` and `stop_reason=compiler_graph_mismatch`;
- message `New eight-row graph/HLO differs from frozen v3.3.`;
- one eight-row compilation and zero six-row compilations;
- zero model applies of 320 planned;
- zero raw OOD records, zero invalid records, and zero unique recipient-anchor
  pairs;
- zero identity reruns, zero main-cube reruns, and zero old-OOD reuse;
- zero confirmation model calls;
- `scientific_summary_computed=false`;
- `shapley_or_nomination_computed=false`; and
- `id0_all20=false` and `id255_all20=false` because execution stopped before
  the first call, not because either biological control failed.

The raw manifest is exactly the empty manifest: count zero, empty hash map,
and empty-tree SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
There is no endpoint, movement, recovery, activation, intervention, Shapley,
interaction, resolution, or nomination value to inspect or interpret.

The compiler gate fired correctly under the frozen v3.3.2 rules. Relative to
the frozen v3.3 eight-row executable:

| Artifact | Original SHA-256 | v3.3.2 SHA-256 | Exact? |
|---|---|---|---|
| StableHLO | `69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd` | same | yes |
| Pre-backend HLO | `675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750` | same | yes |
| Compiled HLO | `0b393070983dd22cb1ea0450992896585db6a4ecf63a2530716242b62ef0e45d` | `b436435ebb14b87cf9929ee9b16fc2c74d1764460c701f8160f1dc092687b718` | no |

The original compiled HLO is 16601821 bytes; v3.3.2 is 16601836 bytes. The
original executable fingerprint is
`12283496a0987eec942bd8f9b7bbb86a9d9d676b13bee1956b30da933a4e9967`;
the new fingerprint is
`e6e482d65d82dbe27a0d78da67b475200008e31fe1d3e2db4af19f59c3ff4934`.

A read-only HLO audit found both changed source/debug metadata and changed
backend autotune choices. For example, repeated backend dot tile sizes change
from 64 to 32, Triton block-level fusion tile/stage choices change, and cuDNN
convolution algorithm/workspace configurations differ. Both artifacts contain
1,545 computations and 80,400 instruction records, and the entry module line
is exact after replacing only `fingerprint_before_lhs`; nevertheless, nested
Triton fusions increase from 115 to 124 and recorded Triton configurations
from 104 to 112. At two convolution sites, the observed choice changes from
algorithm 10 with 36,175,888 bytes of workspace to algorithm 23 with
48,760,495 bytes. Therefore the compiled HLO mismatch must not be described as
only a filename, line-number, ordering, or debug-metadata difference.
Identical StableHLO and pre-backend HLO establish the same pre-backend graph,
but do not establish byte-identical backend code.

The only claim permitted from v3.3.2 is: **the frozen infrastructure gate
stopped a graph-identical/pre-backend-identical attempt because its fresh
backend compilation was not byte-identical to v3.3, before any scientific
apply.** It provides no biological result and does not repair the original OOD
control.

## 3. Frozen analyzer failure

The frozen analyzer was not successfully consumed. Its original analysis
destination and any v3.3.2.1 attempt/output directories were absent when this
amendment was written. A read-only direct `analyze()` diagnostic on 2026-08-28
created no files and failed before compiler or raw-record validation with this
captured, non-persisted traceback:

```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2.py", line 1858, in analyze
    _validate_freeze_and_start(run_dir, bundle_root=bundle_root)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2.py", line 1244, in _validate_freeze_and_start
    raise AnalysisError('Same-process preflight PID differs from bootstrap.')
analyze_encoder_skip_ood_sidecar_v3_3_2.AnalysisError: Same-process preflight PID differs from bootstrap.
```

The message is misleading: both stored PIDs are exactly `2327369`. The frozen
analyzer combines the real PID check with demands that the same-process object
contain `jax_enable_compilation_cache` and `v3_3_runtime_environment`. Those
two names belong to the v3.3 external-preflight schema. The literal v3.3.2
runner stores the same-process observation produced by
`device_gate.collect_device_observation()` and does not add those fields. Its
exact key set is:

```text
environment, hostname, jax_default_backend, jax_gpu_devices,
jax_module_version, jaxlib_module_version, kernel, no_jit_no_array_no_model,
nvidia_smi, packages, pid, platform, python_executable, python_version,
runtime_environment
```

The external observation contains three additional derived fields:
`jax_enable_compilation_cache`, `v3_3_runtime_environment`, and
`v3_3_2_runtime_environment`. This is an analyzer schema mismatch, not a
device, PID, model-run, or compiler failure.

## 4. Sole permitted CPU-analyzer repair

Create a new analyzer, tests, freeze/manifest, and append-only wrapper under a
versioned v3.3.2.1 name. Do not edit the frozen v3.3.2 analyzer, test, freeze,
run, or preflight artifacts. Before invocation, commit and hash-bind the exact
new bundle and this amendment at a globally tracked-clean HEAD.

Use fresh append-only locations:

```text
results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1_attempt
results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1
```

The repair may change only same-process versus external-preflight schema
validation:

1. require the literal same-process key set above and reject missing, extra,
   renamed, malformed, symlink-derived, or path-escaped data;
2. require `same_process.pid == bootstrap.pid == 2327369`;
3. retain the exact RTX 3090 name, UUID, compute-capability, one-device,
   GPU-backend, `LD_LIBRARY_PATH`-absent, preallocation-disabled, no-JIT,
   package/version, runtime-environment, and START/runtime bindings emitted by
   the literal runner;
4. validate external-only fields only on the exact external artifact and its
   embedded copy; and
5. do not synthesize external-only evidence, reinterpret missing fields as
   false, mutate START on disk, or weaken any other frozen validator.

After that repair, delegate to or reproduce every remaining frozen v3.3.2
structural check unchanged. Rehash the complete original v3.3 tree, v3.3.1
audit, 75-file v3.3.2 committed bundle, v3.3.2 run tree, preflight tree,
checkpoint/reference manifests, generated protobuf bindings, import
inventories, compiler artifacts, RAW_MANIFEST, START, and RUN_COMPLETE. The
analyzer must be standard-library/CPU-only before any artifact read and must
not import JAX or model code.

The sole successful decision is
`controlled_stop_compiler_graph_mismatch`, with exactly zero raw records,
zero model applies, zero invalid records, no ID-0/ID-255 completion, no
scientific summary, no Shapley/interaction/resolution/nomination output,
`combined_analysis_permitted=false`, and confirmation model outputs,
activations, and interventions unopened. `RESULT.md` must plainly state that
the sidecar produced no biological evidence.

Write an append-only `ANALYSIS_ATTEMPT_STARTED.json` before delegation and a
terminal `ANALYSIS_COMPLETE.json` or `ANALYSIS_FAILURE.json` afterward. Bind
the attempt, result, and output tree hashes. A failure consumes v3.3.2.1; do
not delete, overwrite, resume, or retry it under the same version.

Tests must include literal captured START/preflight schemas plus failures for
each PID, key-set, runtime, package, GPU/UUID, path, hash, tree, compiler,
count, and terminal-predicate tamper. They must prove that external-only fields
are rejected when demanded from or injected into the same-process object, that
no raw/scientific value is read before all provenance checks pass, and that no
result path can contain a Shapley, interaction, resolution, nomination, or
combined-analysis payload.

## 5. Separate prerequisites for any later OOD attempt

v3.3.2 is consumed and may never be resumed or rerun in place. A later OOD
model attempt is methodologically permissible under the original
infrastructure-repair clause only because v3.3.2 executed zero scientific
applies and exposed no endpoint or activation value. It requires a different,
prospectively committed amendment and version after v3.3.2.1 has archived this
stop.

That later amendment must, before any model/preflight invocation:

- retain the exact 20 development recipients, four anchors `(0,127,128,255)`,
  order, sequences, target, intended/unrelated donor maps, four calls per
  record, 80-record/320-apply arithmetic, one fresh eight-row executable,
  zero six-row/main-cube/identity reruns, gates, thresholds, and claim limits;
- keep confirmation model outputs, activations, and interventions unopened;
- bind this amendment, the v3.3.2.1 audit, all consumed v3.3.2 hashes above,
  and a fresh output/preflight destination;
- freeze a compiler-identity policy before any output is seen; and
- forbid selecting, dropping, reordering, or retrying cases based on any
  result from v3.3, v3.3.2, or the later attempt.

Two defensible compiler policies remain, but one must be chosen and justified
prospectively. Either reproduce byte-identical backend compilation under a
separately frozen deterministic compiler diagnostic, or define StableHLO and
pre-backend-HLO byte equality plus exact program signatures/runtime/hardware
as the graph-identity gate while treating raw compiled HLO and its autotune
choices as bound provenance rather than a cross-attempt equality predicate.
The second policy must explicitly disclose that backend algorithms may differ
and rely on the single fresh executable, exact within-executable repeats, and
all route/closure controls. It must not claim byte-identical codegen.

No later attempt is currently authorized. In particular, do not silently
ignore the compiled-HLO mismatch, call it debug-only, reuse the terminated
process's executable, or relax the gate after inspecting scientific values.
If StableHLO, pre-backend HLO, source, checkpoint, runtime, target, sequence,
or signature equality fails in a later version, that version must again stop
before its first scientific apply.

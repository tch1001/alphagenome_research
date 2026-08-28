# OpenSplice v3.3.3 source-program compiler-gate amendment

Status: **prospective and docs-only**. This document chooses a principled
compiler boundary for one later development-only OOD sidecar. It authorizes no
execution by itself. The runner, tests, pre-import bootstrap, device preflight,
launcher, wrapper, offline analyzer, and machine-readable freeze must be
committed, hash-bound, and independently audited before one GPU invocation.

The scientific protocol remains
`encoder_skip_localization_protocol_v3_3.md`, SHA-256
`85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0`,
with the sole host-side OOD assertion repair in
`encoder_skip_ood_sidecar_amendment_v3_3_2.md`, SHA-256
`42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3`.
All model inputs, targets, recipients, donors, anchors, interventions, row
roles, controls, thresholds, ordering, estimands, stop rules, and claim limits
remain unchanged. This amendment changes only which compiler representation
must be equal across fresh processes.

Confirmation model outputs, activations, and interventions remain unopened.
Previously disclosed later-exon metadata/label exposure remains in scope. No
v3.3 endpoint value or v3.3.2 scientific value was used to choose this rule;
v3.3.2 produced no scientific apply or raw record.

## 1. Consumed v3.3.2 stop

v3.3.2 is consumed and immutable at:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_2_development_ood_sidecar_one_shot`

Before any v3.3.3 import, preflight allocation, append-only start, model
construction, or compilation, independently revalidate:

| Binding | Exact value |
|---|---|
| v3.3.2 model-run commit | `24e2214168eeca41d4f3b60b62094b6befcadcc1` |
| v3.3.2 freeze SHA-256 | `baa257dfa8d0bd1afca40b11038d71c9157499eeb654523ad59df119eb2c0295` |
| `ATTEMPT_STARTED.json` SHA-256 | `d1bb85b3c35d892f10f67a5510b85784eb1b6b989c9ae32290b8c0fdc3c8d235` |
| `RUN_COMPLETE.json` SHA-256 | `d88c8becc96a24f1f56a1cee82da35fdd331ee3b40ba4108ae98f44ae24b4ad7` |
| `RAW_MANIFEST.json` SHA-256 | `fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd` |
| All three import-provenance SHA-256 values | `a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99` |
| `PROTOBUF_PROVENANCE.json` SHA-256 | `2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8` |
| Compiler provenance SHA-256 | `bd20e21a56a9ca5498d7119771bb1da9ac2e156ed3190d3ce3aa09ff2d2e312c` |
| Whole-run count / tree SHA-256 | `11` / `4ac66e45a4d7d65af2785904d11b23bf7e809e07f3f617e190772242b2e7a4ab` |
| Compiler count / tree SHA-256 | `4` / `4378048568ff58a2bbee55ba9da750498b89fdef72c97911815cf895c8a8b7d1` |

The tree digest uses paths relative to the run root, including `compiler/...`,
with the exact v3.3 framing: sorted UTF-8 POSIX path, one NUL, and 32 raw
SHA-256 bytes per regular non-symlink file. Reject every missing, extra,
symlinked, or special file or directory. There is no `raw` directory.

The terminal predicates are exactly `status=controlled_stop`,
`stop_reason=compiler_graph_mismatch`, one eight-row compile, zero six-row
compiles, zero of 320 model applies, zero raw records, zero invalid records,
zero identity/main-cube reruns, zero confirmation calls, and no scientific,
Shapley, interaction, resolution, or nomination summary. The raw manifest is
the exact empty manifest with tree SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
These predicates must be recomputed, not trusted from this table.

The prospective v3.3.2.1 CPU-only controlled-stop amendment is
`encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md`, SHA-256
`81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba`
at the time this document was finalized. v3.3.3 may not run until a committed
v3.3.2.1 analyzer has successfully archived
`controlled_stop_compiler_graph_mismatch`; its final attempt/output paths,
files, sizes, SHA-256 values, and tree digests must be added to the v3.3.3
machine-readable freeze before v3.3.3 is committed. No scientific value from
that structural audit may be read because none exists.

## 2. Exact compiler diagnosis

The source program is identical through the pre-backend boundary:

| Representation | Frozen SHA-256 | Size (bytes) | v3.3 versus v3.3.2 |
|---|---|---:|---|
| StableHLO | `69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd` | 3196162 | exact |
| Pre-backend HLO | `675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750` | 1829833 | exact |

The backend compilation is not byte-identical:

| Representation | v3.3 | v3.3.2 |
|---|---|---|
| Compiled-HLO SHA-256 | `0b393070983dd22cb1ea0450992896585db6a4ecf63a2530716242b62ef0e45d` | `b436435ebb14b87cf9929ee9b16fc2c74d1764460c701f8160f1dc092687b718` |
| Size (bytes) | 16601821 | 16601836 |
| Executable fingerprint | `12283496a0987eec942bd8f9b7bbb86a9d9d676b13bee1956b30da933a4e9967` | `e6e482d65d82dbe27a0d78da67b475200008e31fe1d3e2db4af19f59c3ff4934` |

Both compiled artifacts contain 1,545 computations and 80,400 instruction
records. Their entry module line is exact after replacing only the backend-
generated `fingerprint_before_lhs` value. However, the difference is not only
debug metadata: nested Triton fusions change from 115 to 124, recorded Triton
configurations from 104 to 112, block tilings/stages differ, and two observed
convolution choices change from cuDNN algorithm 10/workspace 36,175,888 to
algorithm 23/workspace 48,760,495. A post-hoc compiled-HLO canonicalizer,
allowed-difference list, or retry-until-match rule would therefore be
misleading and is prohibited.

This is ordinary backend autotune/code-generation freedom below an identical
source-program boundary. The v3.3.2 exact-compiled-byte gate correctly stopped
under its frozen rule, but that rule is stronger than the scientific design
requires: all v3.3.3 comparisons occur within one fresh executable, and the
v3.3.2 protocol already forbids cross-executable endpoint equality as a gate.

## 3. Frozen source-program identity gate

Define:

```text
source_program_exact :=
    stablehlo_sha256 == 69dbf2...1bddd
    and stablehlo_size == 3,196,162
    and pre_backend_hlo_sha256 == 675b9c...2750
    and pre_backend_hlo_size == 1,829,833
    and program_signatures_exact
    and entry_abi_exact
    and source/runtime/device/toolchain/checkpoint/reference_exact
```

All terms are mandatory and must pass before the first model apply.

### 3.1 Program signatures and entry ABI

The exact v3.3.2 eight-row program-signature object contains the input
pytrees, treedef strings, leaf order, dtypes, and shapes for selection,
target, and eight-row interventions. Canonical JSON is UTF-8
`json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`;
its SHA-256 must be
`d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300`.
The freeze must also include the complete literal object so the hash is not the
only description.

The exact first compiled-HLO `HloModule` line binds the entry parameter/output
layout. Replace only the hexadecimal value inside
`fingerprint_before_lhs="..."` with the literal
`fingerprint_before_lhs="<backend-generated>"`, do not normalize any other
byte, omit the line-ending newline, and SHA-256 the UTF-8 result. The required
entry-ABI digest is
`ebf900771a87775a5fb657b90131fd884d68ce4725245defa164bf5066c74a80`.
This extraction is an ABI check, not a compiled-program equality claim.

The JAX `eval_shape`, lowered input/output tree, parameter order, batch size
eight, output evidence/trace tree, mixed-precision policy, and exact target
reducer must also equal the frozen v3.3.2 objects. A hash collision between
different literal trees or a reordered but shape-compatible leaf set is a
failure.

### 3.2 Exact sources and immutable inputs

Revalidate all 75 entries in
`encoder_skip_ood_sidecar_v3_3_2_freeze.json`, the original v3.3 61-file
bundle, the complete original v3.3 run tree, the complete v3.3.2 run tree, and
the completed v3.3.2.1 structural audit. Core model/factory/reducer,
interpretability instrumentation, OpenSplice pins, development manifests,
sequences, source protos/generated bindings, import inventories, checkpoint,
reference object, and mixed-precision policy
`params=float32,compute=bfloat16,output=bfloat16` must remain exact.

Checkpoint snapshot is
`a8f293a76ee73d5b57f3bf2ae146510589fcf187`; checkpoint-manifest SHA-256 is
`1ed87db4c5bd7c5418c7734ec128faa4a9ecd186df2a024437484a8bc2b6e934`;
reference-binding SHA-256 is
`da712cdca50f82113ac1d00cb2fa7171f7368f31aedf06c48ce92dbdb5897dca`.
Revalidate their full frozen contents, not only these summary hashes.

### 3.3 Exact runtime, toolchain, device, and environment

Require one visible `NVIDIA GeForce RTX 3090`, UUID
`GPU-64111645-1e42-a96d-f192-4abbec4b8090`, compute capability 8.6, driver
560.35.05, VBIOS 94.02.42.C0.05, kernel 6.8.0-136-generic, Python 3.13.5,
and the exact runtime manifest from the v3.3.2 freeze. Its critical packages
are:

```text
dm-haiku 0.0.17                 jax 0.11.0
grpcio 1.83.0                   jaxlib 0.11.0
jax-cuda12-pjrt 0.11.0          jax-cuda12-plugin 0.11.0
jmp 0.0.4                       numpy 2.5.2
nvidia-cublas-cu12 12.9.2.10    nvidia-cuda-runtime-cu12 12.9.79
nvidia-cudnn-cu12 9.24.0.43     nvidia-cusparse-cu12 12.5.10.65
orbax-checkpoint 0.12.4          protobuf 7.35.1
```

Unset `LD_LIBRARY_PATH`; require
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`JAX_ENABLE_COMPILATION_CACHE=false`; reject all compiler, persistent-cache,
kernel-cache, and autotune-cache inputs frozen as absent by v3.3/v3.3.2. Use a
fresh external and same-process RTX/UUID gate. Bind all environment variables
with the frozen JAX/XLA/CUDA/TF/compiler prefixes. No cache file, previous
executable, serialized activation, or previous compiled artifact may be used
as an execution input.

## 4. Compiled backend artifacts are provenance, not equality gates

Lower and compile exactly once in the launch process. StableHLO,
pre-backend HLO, program signatures, and entry ABI must be emitted from the
same lowered/compiled object that supplies the sole executable. Persist before
the first apply:

- raw StableHLO and pre-backend HLO with exact hashes/sizes;
- raw compiled HLO with its new SHA-256 and size;
- the new executable fingerprint;
- a deterministic descriptive summary of backend configurations, including
  computation/instruction counts, fusion kinds, Triton block-level settings,
  cuBLAS/cuDNN algorithms/workspaces, and entry ABI;
- compiler/runtime/device/environment/import/protobuf provenance; and
- diagnostic comparisons against both v3.3 and v3.3.2.

Compiled-HLO hash, size, fingerprint, computation order, fusion count, tiling,
stage/warp count, library algorithm, and workspace may differ. Such a
difference is explicitly neither a pass nor a fail. It is persisted and
reported as backend provenance. No canonicalized compiled-HLO equality,
whitelist of observed changes, minimum similarity, or search for a matching
compile is allowed.

If and only if `source_program_exact` passes, use that exact in-memory compiled
object for all 320 applies. A second compile, a cache hit, loss of provenance,
or any source-program/ABI mismatch is a controlled stop. There is no compile
retry, process retry, per-record retry, or replacement executable.

## 5. One fresh unchanged OOD sidecar

Use fresh append-only destinations whose exact paths are frozen before launch,
including a new `v3_3_3` output directory and a new preflight directory. Every
v3.3/v3.3.2/v3.3.2.1 directory remains immutable and is never resumed,
completed, copied into, or used as a raw-value input.

The run is exactly:

```text
for recipient order 0..19, in the frozen manifest order:
    for anchor ID in (0, 127, 128, 255), in that order:
        intended call
        exact intended repeat
        unrelated-donor call
        exact unrelated-donor repeat
```

This is 80 records and 320 applies from one fixed eight-row executable. It
must perform zero identity, six-row, main-cube, old-OOD, confirmation, or
extra calls. The cross-exon donor derangement, row roles, intended and
unrelated donor maps, invariant rows `[0,1,3,5,6,7]`, active recipient rows
`[2,4]`, route masks, ID-0 no-op checks, ID-255 intended/unrelated closure,
repeat equality, self controls, target algebra, finiteness, trace
fingerprints, disabled-route checks, final-seam checks, sequence hashes, and
linked-original structural bindings remain exactly v3.3.2.

Write `ATTEMPT_STARTED.json` before model construction. Persist compiler
artifacts before any apply and each raw record immediately after its four
calls and validation. Persist an exact apply count on every failure. Missing,
extra, duplicated, reordered, non-finite, invalid, or structurally inconsistent
data consumes v3.3.3 as a controlled stop. Never delete, overwrite, resume,
stitch, cherry-pick, or retry.

The GPU process performs no donor-normalized recovery, Shapley value,
interaction, resolution gate, rank, nomination, or combined scientific
analysis. The first two old v3.3 OOD records and the zero-record v3.3.2 attempt
remain failure provenance only.

## 6. Required tests and pre-run audit

CPU/synthetic tests must cover:

- exact StableHLO/pre-backend hash and size failures independently;
- literal program-signature, treedef, leaf-order, dtype, shape, and entry-ABI
  tampering;
- proof that compiled HLO/fingerprint/backend configurations may differ
  without altering `source_program_exact`;
- proof that a compiled difference never bypasses any source/runtime/input
  gate and never triggers an automatic recompile;
- one compile, one executable, 80-record/320-apply order, and zero forbidden
  family arithmetic;
- every v3.3.2 row, donor, invariant, repeat, route, closure, target, and
  provenance tamper;
- complete rehash of v3.3, v3.3.2, and v3.3.2.1 artifacts, including empty
  v3.3.2 raw/tree predicates;
- global tracked-clean, fresh-output, sanitized environment, cache absence,
  same-process PID/device, append-only start, and terminal persistence; and
- confirmation path isolation and proof that no real score is embedded in a
  fixture or read before the full provenance/source-program gates pass.

An independent read-only audit must certify exact committed bytes, test
results, dry-run arithmetic, fresh destinations, absence of model outputs,
and a tracked-clean HEAD before the sole launch.

## 7. Stop table and claim boundary

| Outcome | Required disposition |
|---|---|
| Any provenance/source/runtime/device/ABI/StableHLO/pre-backend mismatch | Controlled infrastructure stop before apply zero; no OOD or biological claim. |
| Source program exact; compiler failure or second compile | Controlled infrastructure stop; preserve compiler/apply-count artifacts; no retry or biological claim. |
| Partial/invalid 80-record prefix | Controlled OOD tooling stop; preserve prefix; no combined analysis or biological claim. |
| Exactly 80 valid records, 320 applies, all gates pass | Structurally complete OOD sidecar; still no GPU-side scientific claim. |

Even after structural completion, v3.3.3 establishes only that the prospectively
defined source program ran the unchanged OOD controls under one self-consistent
backend executable. Compiled code is not claimed byte-identical to v3.3 or
v3.3.2. A separate prospectively committed CPU analyzer must revalidate all
trees before reading any development value.

The unrelated-donor result remains an out-of-distribution raw-movement
warning, not a null distribution, rescue criterion, rejection criterion,
mechanism, or held-out validation. Shapley/resolution evidence can come only
from the unchanged frozen v3.3 six-row cube under its original gates. Every
report must disclose the earlier v3.3 OOD tooling stop, the v3.3.2 zero-apply
compiled-byte stop, this prospectively revised compiler boundary, and the lack
of confirmation validation. Confirmation remains closed until a later,
separately frozen circuit protocol authorizes it.

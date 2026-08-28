# OpenSplice v3.3.3.1 structural-analyzer amendment

Status: **prospective and docs-only**. This amendment authorizes no model,
GPU, or analysis invocation by itself. It freezes a narrow CPU-only repair for
archiving the consumed v3.3.3 source-program controlled stop. The existing
v3.3.3 runner, run artifacts, analyzer, tests, freeze, and protocol must remain
unchanged.

Confirmation model outputs, activations, and interventions remain unopened.
The v3.3.3 attempt produced zero model applies and no raw record, so this
amendment has no scientific-value input.

## 1. Immutable source and implementation bindings

The governing v3.3.3 compiler-gate protocol is
`encoder_skip_ood_sidecar_compiler_gate_amendment_v3_3_3.md`, SHA-256
`c9b00398296e683ac6e1c321fd8c4302f96b2e62bb23828e8b5ef2fe9de3f70b`,
committed at `783a7d0dfbd5f26e22152d1201dacf82f2b01d15`.

The model-run bundle was committed at
`228083b931dbc62d4a283e68df01011f5ef4bff9`. Its machine freeze is
`encoder_skip_ood_sidecar_v3_3_3_freeze.json`, SHA-256
`0e4c16a306f734e016c64509a3b7f0d76f26baf399ee0b1d41c6fb073203741b`,
with 69 top-level keys and 96 source hashes. Before reading any terminal or
compiler record, a later analyzer must rehash all 96 live files and their
exact `git show 228083b...:<path>` blobs, require a globally tracked-clean
HEAD, and bind the exact generated-protobuf exceptions already frozen by
v3.3.3.

The committed structural analyzer that must remain uninvoked is:

| File | SHA-256 |
|---|---|
| `analyze_encoder_skip_ood_sidecar_v3_3_3.py` | `5cf50aa7a9403df5d8f5555b1d2fc50ab2feb69e0508c37e64bddd5a8a1e3783` |
| `analyze_encoder_skip_ood_sidecar_v3_3_3_test.py` | `fe174d0546e97e1ac1de0ad03204df4e77cf8baa0e7edb27791d6b46ed6ed58f` |

Its append-only destinations are absent and must stay absent:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis_attempt
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis
```

The analyzer was **not invoked**. A later repair must reject either path if it
appears or becomes a symlink; it must not consume, complete, normalize, or
reuse those destinations.

## 2. Consumed v3.3.3 model-run tree

The immutable run root is:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_one_shot
```

It contains exactly 11 regular non-symlink files and three directories. Its
file-tree SHA-256 is
`bb13aa4de212c3896781401374057bc0cdfc0c7527772cc36b08b57c70451805`.
The compiler subtree contains exactly four files and two directories; using
paths relative to the run root, its tree SHA-256 is
`7ee5ad1bb94ecbd97606fcccae3abcad6b0ebec74dd9f983d81b4fc179142ef0`.

Both tree digests use sorted UTF-8 POSIX relative path, one NUL, then 32 raw
SHA-256 bytes for every regular file. Reject a missing, extra, reordered,
symlinked, or special entry, including an empty or unexpected directory.

| Relative path | Size (bytes) | SHA-256 |
|---|---:|---|
| `ATTEMPT_STARTED.json` | 871020 | `e5f7c33f2e8c82af51ed98a3884d7df83e1828e92e322df8aa8a054ec7464c65` |
| `IMPORT_PROVENANCE.json` | 41572 | `aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b` |
| `IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json` | 41572 | `aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b` |
| `IMPORT_PROVENANCE_PRE_MODEL.json` | 41572 | `aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b` |
| `PROTOBUF_PROVENANCE.json` | 3339 | `2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8` |
| `RAW_MANIFEST.json` | 145 | `fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd` |
| `RUN_COMPLETE.json` | 227159 | `43e0ff055e9f7fa4032a75120c551a2b5762e4fbd85119e80e3694f8b9f54bba` |
| `compiler/eight_row/COMPILER_PROVENANCE.json` | 102245 | `ae07b0f10784ea3c6dd26d2b87eb718c5e28d3834112ae4f0566d1c4fb7e3125` |
| `compiler/eight_row/graph.compiled.hlo.txt` | 16603075 | `f0fe2fa0b7e8326390c8f2ed38ce52ef6c64c355bc462450d85d3b2f040645f4` |
| `compiler/eight_row/graph.pre_backend.hlo.txt` | 1829833 | `675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750` |
| `compiler/eight_row/graph.stablehlo.mlir` | 3196162 | `69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd` |

There is no `raw` directory. `RAW_MANIFEST.json` is exactly:

```json
{
  "artifact_count": 0,
  "artifact_sha256": {},
  "artifact_tree_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

`RUN_COMPLETE.json` must independently recompute as
`status=controlled_stop`, `stop_reason=source_program_mismatch`, one lowering
attempt, one compile attempt, one successful eight-row compile, zero six-row
compiles, zero of 320 model applies, zero raw/invalid/unique records, zero
identity/main-cube/old-OOD/confirmation calls, and no scientific, Shapley,
interaction, resolution, nomination, or combined analysis. Its embedded
compiler, source gate, import hashes, protobuf hash, and empty manifest must
equal the current files exactly.

## 3. Immutable device-preflight and cache evidence

The sole successful external preflight root is:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_device_preflight
```

It contains exactly five regular non-symlink files and one directory. Its tree
SHA-256, under the same file-tree framing, is
`f2bae99e3b0a59a50419e0507146e26f4eea1c67f2595ddccec4e8d5aef7a0e1`.

| Relative path | Size (bytes) | SHA-256 |
|---|---:|---|
| `.allocation.lock` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.preflight_0000.reserved` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.json` | 704213 | `79e2c9937025830b309854cff4f5c93c607b7574fb44a9d51f45564b14246224` |
| `preflight_0000.stderr.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.stdout.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Require the external and same-process observations to bind the frozen RTX
3090/UUID, driver, runtime, PID, environment, and exact distinct cache roles.
The model cache had zero pre-import files. Its only terminal regular file is
`xdg/matplotlib/fontlist-v3.11.0.json`, size 163240, SHA-256
`a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125`.
The persisted historical and terminal cache bindings both report tree SHA-256
`a1dafb75097282b3d28174e4bb6c79a81d2f1c8c25cd8c64a1ea84642ba7f43a`.
Cache-output equality is descriptive and is not a scientific or execution
gate.

## 4. Exact representation-only failure diagnosis

All substantive source-program terms passed:

- StableHLO SHA/size exact;
- pre-backend HLO SHA/size exact;
- entry ABI exact;
- source, runtime, device, toolchain, checkpoint, reference, protobuf, and
  import inputs exact; and
- the same lowered object supplied the sole compiled executable.

The observed and contract program-signature canonical SHA-256 values are both
`d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300`.
The literal signature content, leaf order, treedefs, dtypes, and numeric shapes
are unchanged.

The runner nevertheless stored `program_signatures_exact=false` because
`run_superset_graph_v3_2.pytree_signature` constructs each `leaves` collection
as a tuple and every shape as `tuple(np.shape(leaf))`, while the frozen v3.3.2
compiler signature was loaded from JSON and therefore contains lists. Across
the three signature objects this is exactly three `leaves` tuples plus 29
shape tuples, or 32 tuple containers. Python direct equality treats, for
example, `(7, 8) != [7, 8]`. Canonical JSON serializes both forms as `[7,8]`;
the canonical current and prior signature payload is exactly 2877 bytes and
has SHA-256 `d8f95f...`. This is a host-language container-type mismatch, not
a model graph, dtype, shape, ABI, or biological difference.

Persisting `COMPILER_PROVENANCE.json` normalized the runtime tuples to JSON
lists. Consequently, the committed v3.3.3 analyzer now sees list-versus-list,
independently recomputes `program_signatures_exact=true`, and deterministically
raises:

```text
AnalysisError: Compiler source-program gate changed at program_signatures_exact.
```

That failure was reproduced read-only by calling only the analyzer's source-
gate validator on the immutable compiler provenance. The full analyzer was not
invoked, no append-only analyzer START was written, and no raw/scientific value
was read.

## 5. Only permitted v3.3.3.1 repair

A repair must use separate, versioned files: a CPU-only structural analyzer,
tests, machine freeze, and shell wrapper. It must be committed, hash-bound,
tracked-clean, and independently audited before exactly one invocation. It may
not edit, monkey-patch, or invoke the existing v3.3.3 analyzer or alter the
model-run artifacts. There is no dual-attempt workflow: the original v3.3.3
analyzer attempt/output paths remain absent permanently.

The new append-only destinations are frozen as:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1_attempt
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1
```

Before creating its append-only START, it must independently validate Sections
1--3, prove the original v3.3.3 analyzer attempt/output paths remain absent,
and prove its own fresh attempt/output paths are absent. After START, it must
revalidate the exact START hash and immutable run bindings before delegation.
Failure or success consumes the new attempt; no deletion, resume, overwrite,
or retry is allowed.

The representation-aware rule is deliberately narrow:

1. require the exact runner and `pytree_signature` source bytes at commit
   `228083b...`, including the frozen tuple-producing implementation;
2. require the persisted v3.3.3 and frozen v3.3.2 signature literals to be
   exactly equal after JSON serialization and to have the exact canonical hash
   `d8f95f...`;
3. require every leaf order, treedef, dtype, shape length, and integer shape
   value to match literally;
4. require the stored v3.3.3 gate to remain exactly
   `program_signatures_exact=false` and `source_program_exact=false`, with all
   other source-program flags exactly true; and
5. record the discrepancy as the consumed runner's tuple-versus-list direct-
   equality result, without rewriting it into a passed model run.

Any other signature, hash, flag, source, tree, terminal, or provenance change
must fail. In particular, this amendment does not authorize treating a true
shape, dtype, treedef, ordering, source, or ABI difference as representation-
only.

The new analyzer must implement a standalone copied/versioned structural
validator with this single declared representation-aware branch. It may import
the old module only to call unchanged, side-effect-free helper functions by
direct reference; it must never call the old `main()` or `analyze()` entry
point and must not replace any old-module binding. Recursion, temporary or
general monkey-patching, dual START records, and normalization of arbitrary
tuples/lists are forbidden.

## 6. Tests, output, and claim boundary

CPU tests must cover:

- exact reproduction of tuple-shape versus list-shape direct inequality and
  canonical-hash equality;
- rejection of every changed numeric shape, dtype, treedef, leaf order,
  canonical hash, or source-program flag;
- exact 11-file run, four-file compiler, and five-file preflight memberships,
  hashes, sizes, and tree digests;
- exact apply-zero/empty-manifest/no-science predicates and terminal linkage;
- proof that the committed v3.3.3 analyzer would deterministically reject
  before raw evidence and remains uninvoked, with both original analyzer paths
  still absent;
- CPU-only/no-JAX/no-AlphaGenome/no-model/no-confirmation import guards;
- fresh append-only START, post-START revalidation, success/failure terminal
  persistence, partial-output binding, and one-shot refusal; and
- proof that only the two frozen v3.3.3.1 destinations are used and that no
  old-module entry point or binding mutation occurs; and
- tamper tests for every bound source, run, compiler, preflight, cache, and
  prior analyzer path.

The only successful result is a **structural archive of the consumed
source-program-mismatch stop**. It must report zero applies, zero raw records,
no ID-0/ID-255 control completion, no scientific summary, no donor
normalization, no Shapley/interactions/resolution/nomination, and
`combined_analysis_permitted=false`.

This archive may state that the sole observed gate failure was caused by
tuple-versus-list host representation under otherwise exact canonical
signatures and source-program evidence. It may not claim that v3.3.3 executed
the OOD experiment, passed its frozen gate, validated a mechanism, or produced
biological evidence. It authorizes no model or GPU rerun. Any future execution
requires a separate prospective protocol and a new versioned runner.

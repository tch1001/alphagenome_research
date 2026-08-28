# OpenSplice v3.3.4 signature-representation infrastructure amendment

Status: **prospective and docs-only**. This amendment authorizes no code,
model, GPU, preflight, compiler, or analyzer invocation by itself. It permits
at most one later development-only OOD sidecar attempt after a separately
implemented runner, tests, pre-import bootstrap, device preflight, launcher,
wrapper, structural analyzer, and machine freeze have been committed,
hash-bound, and independently audited.

The scientific design remains frozen by:

- `encoder_skip_localization_protocol_v3_3.md`, SHA-256
  `85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0`;
- `encoder_skip_ood_sidecar_amendment_v3_3_2.md`, SHA-256
  `42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3`;
- `encoder_skip_ood_sidecar_compiler_gate_amendment_v3_3_3.md`, SHA-256
  `c9b00398296e683ac6e1c321fd8c4302f96b2e62bb23828e8b5ef2fe9de3f70b`,
  committed at `783a7d0dfbd5f26e22152d1201dacf82f2b01d15`; and
- `encoder_skip_ood_sidecar_analysis_amendment_v3_3_3_1.md`, SHA-256
  `4d2957d144e56e58c5b2058076bbcdb7f1495f3172e1b8829a0affa10a0ea4a9`,
  committed at `d2a013944a399ddac59a023d7d84ea5a7c23e9f4`.

All development recipients, donors, anchors, row roles, interventions,
targets, readouts, controls, ordering, thresholds, and claim limits remain
unchanged. This amendment repairs only the known Python tuple-versus-JSON-list
comparison at the program-signature gate. Confirmation model outputs,
activations, and interventions remain unopened. Previously disclosed
later-exon metadata and label exposure remains disclosed; no confirmation
model value is authorized here.

### Normative serialization and schema conventions

This document is an implementation contract, not a design sketch. Unless a
field is explicitly described as an inherited opaque binding, every JSON
object below has the exact stated key set: missing and extra keys fail. JSON is
UTF-8, ends in one newline, rejects NaN and infinity, and is written with
`sort_keys=True, indent=2`. A *canonical JSON payload* instead means exactly
`json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False).encode("utf-8")`, with no newline. A
binding is exactly `{"path": <POSIX relative path>, "sha256": <64 lowercase
hex>, "size_bytes": <non-negative integer>}`; booleans are rejected wherever
an integer is required. Every named path must be absolute in the machine
freeze, resolve inside the repository or its explicitly frozen external root,
and be a regular non-symlink file or a not-yet-created directory as specified.
A *content binding* is separately and exactly
`{"sha256":<64 lowercase hex>,"size_bytes":<non-negative integer>}` over the
canonical JSON bytes of an embedded object; it has no `path` and must never be
accepted where a file binding is required.

All timestamps are finite JSON numbers and are provenance only. All modes are
Git modes: Python, JSON, Markdown, and test files are `100644`; shell wrappers
are `100755`. A generated machine freeze must enumerate every source file,
mode, size, and SHA-256. The final freeze's own exact top-level key set and
every nested key set must be asserted literally by the bootstrap, runner,
preflight, analyzer, and tests; the freeze may not supply or redefine an
expected key set at runtime.

## 1. Immutable v3.3.3 model-run stop

The consumed v3.3.3 model run is immutable at:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_one_shot
```

Its implementation/model-run commit is
`228083b931dbc62d4a283e68df01011f5ef4bff9`. Its machine freeze is
`encoder_skip_ood_sidecar_v3_3_3_freeze.json`, SHA-256
`0e4c16a306f734e016c64509a3b7f0d76f26baf399ee0b1d41c6fb073203741b`,
with exactly 69 top-level keys and 96 source hashes. A v3.3.4 bootstrap and
analyzer must validate every live source byte and every exact
`git show 228083b931dbc62d4a283e68df01011f5ef4bff9:<path>` blob before importing a frozen helper or reading
any v3.3.3 terminal or compiler record.

The run root contains exactly 11 regular non-symlink files and the three
directories `.`, `compiler`, and `compiler/eight_row`:

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

The 11-file whole-run tree SHA-256 is
`bb13aa4de212c3896781401374057bc0cdfc0c7527772cc36b08b57c70451805`.
The four-file compiler-subtree SHA-256, with paths still relative to the run
root, is
`7ee5ad1bb94ecbd97606fcccae3abcad6b0ebec74dd9f983d81b4fc179142ef0`.
Both use sorted UTF-8 POSIX relative path, one NUL, then 32 raw SHA-256 bytes
per regular file. Reject every missing, extra, symlinked, special, or empty
directory entry.

The raw manifest is the exact empty manifest, with zero artifacts and empty
tree SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
There is no `raw` directory. Recompute, rather than trust, the exact terminal
predicates: `status=controlled_stop`,
`stop_reason=source_program_mismatch`, one lowering/compile attempt, one
successful eight-row compile, zero six-row compiles, zero of 320 model
applies, zero raw/invalid/unique records, zero identity/main-cube/old-OOD/
confirmation calls, incomplete ID-0 and ID-255 controls, and no scientific,
normalization, Shapley, interaction, resolution, nomination, or combined
analysis.

### 1.1 Exact compiler evidence

The v3.3.3 compiler record binds:

| Evidence | Exact value |
|---|---|
| StableHLO | SHA-256 `69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd`, 3196162 bytes |
| Pre-backend HLO | SHA-256 `675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750`, 1829833 bytes |
| Program signatures | canonical SHA-256 `d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300`, 2877 bytes |
| Entry ABI | normalized-line SHA-256 `ebf900771a87775a5fb657b90131fd884d68ce4725245defa164bf5066c74a80`, 18921 bytes |
| Compiled HLO | SHA-256 `f0fe2fa0b7e8326390c8f2ed38ce52ef6c64c355bc462450d85d3b2f040645f4`, 16603075 bytes |
| Executable fingerprint | `312414c1c0ca32f79d7fdf669a4877733bd744b010f91e85a2e9a9b1d32843b1` |

`stablehlo_exact`, `pre_backend_hlo_exact`, `entry_abi_exact`,
`source_runtime_device_toolchain_checkpoint_reference_exact`, and
`same_lowered_compiled_object` were true. The only false primitive was
`program_signatures_exact`, which made `source_program_exact=false`. The
compiled HLO, executable fingerprint, backend configurations, fusion choices,
tilings, library algorithms, and workspaces are descriptive provenance and
were not equality gates.

### 1.2 Exact external preflight and model cache

The immutable v3.3.3 external-preflight root is:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_device_preflight
```

It contains exactly five regular non-symlink files and no subdirectory:

| Relative path | Size (bytes) | SHA-256 |
|---|---:|---|
| `.allocation.lock` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.preflight_0000.reserved` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.json` | 704213 | `79e2c9937025830b309854cff4f5c93c607b7574fb44a9d51f45564b14246224` |
| `preflight_0000.stderr.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.stdout.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Its file-tree SHA-256 is
`f2bae99e3b0a59a50419e0507146e26f4eea1c67f2595ddccec4e8d5aef7a0e1`
under the path/NUL/raw-digest framing above.

The immutable model-cache root is:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_model_kernel_cache
```

It had zero pre-import files. Its only terminal regular file is
`xdg/matplotlib/fontlist-v3.11.0.json`, size 163240, SHA-256
`a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125`.
The exact directory set is `.`, `triton`, `xdg`, and `xdg/matplotlib`.
Under the frozen directory/file framing (`D`, `F`, NULs, relative path, and
raw file digest), its tree SHA-256 is
`a1dafb75097282b3d28174e4bb6c79a81d2f1c8c25cd8c64a1ea84642ba7f43a`.
This terminal cache is provenance only and may not be a v3.3.4 input.

## 2. Successful v3.3.3.1 structural archive

The representation-only diagnosis was archived without invoking the original
v3.3.3 analyzer entry point. Bind all three commits:

| Role | Commit |
|---|---|
| Prospective v3.3.3.1 amendment | `d2a013944a399ddac59a023d7d84ea5a7c23e9f4` |
| Committed v3.3.3.1 implementation used by START | `98c467ae16200071d110c9d73520e35e5e6d7bbf` |
| Immutable v3.3.3.1 production archive | `37bd58e88e1814f9a67bfbaaaad66d0a2b77f242` |

The implementation sources are exact:

| Bound source | SHA-256 |
|---|---|
| v3.3.3.1 amendment | `4d2957d144e56e58c5b2058076bbcdb7f1495f3172e1b8829a0affa10a0ea4a9` |
| v3.3.3.1 analyzer | `f433221f38408ee06d3bdb2c1119ae050720652ee4ec513a0b91f2d7814da063` |
| v3.3.3.1 analyzer test | `4f2e70a8f61bb1b9af7b2b98ef8f450d0937855a69d1bc83fbec9d06f21dd971` |
| v3.3.3.1 machine freeze | `96c599f3c607107b8c7ab235d7c8cef7aa1bc544189b44b15b6f3fbf1a8b3291` |
| v3.3.3.1 shell wrapper | `63a0cc95596d47ee5900fe928e1bb42115b18157f87bdae45000e5bb7ccef5c9` |

Require exact live bytes and exact blobs at the declared amendment and
implementation commits. The append-only production roots are:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1_attempt
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1
```

Their complete membership, committed at
`37bd58e88e1814f9a67bfbaaaad66d0a2b77f242`, is:

| Root | Relative path | Size (bytes) | SHA-256 |
|---|---|---:|---|
| Attempt | `ANALYSIS_ATTEMPT_STARTED.json` | 6512 | `497374d68c245c30fb0a54968859b9066d1bc16085146b978070bb092ff23bda` |
| Attempt | `ANALYSIS_COMPLETE.json` | 1179 | `e050e091743262e989693c59f5e1fcb2939190a71ee4851c5d2a345c1827c4be` |
| Output | `ANALYSIS.json` | 10060 | `f1e20b3ca4f111854b22eff1e2cd2ffdb05796d800d2831eedcc6caa1a3b7245` |
| Output | `RESULT.md` | 695 | `8ba2721c8bc350a564f4d5ffdabd65b118f60d92cbdb8ea00a8d040842012e65` |

The two-file attempt tree SHA-256 is
`cff8dd5418405dd1acef9c6de1d1e2688e63a6807b1ff4e1ef0c8b8908229307`.
The two-file output tree SHA-256 is
`4dcbaa9069b130d160efbde95b1f82b3561ea90d2a38923d259978126e889b2c`.
Use the same sorted path/NUL/raw-digest framing. Independently require exact
live bytes, exact `git show 37bd58e88e1814f9a67bfbaaaad66d0a2b77f242:<path>` blobs, memberships, sizes,
hashes, tree digests, START-to-COMPLETE linkage, and COMPLETE-to-output
linkage.

The only accepted archive state is
`status=complete_controlled_stop_structural_archive` and
`decision=controlled_stop_source_program_mismatch_representation_only`, with
zero model applies, zero raw records, incomplete ID-0/ID-255, and no
scientific summary, donor normalization, Shapley, interaction, resolution,
nomination, or combined analysis. It records exactly three `leaves` tuples,
29 `shape` tuples, 32 tuple containers, canonical payload size 2877, canonical
SHA-256 `d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300`, direct Python equality false, canonical JSON equality
true, and every other source-program term true.

The original v3.3.3 analyzer destinations remain absent and must remain absent
before, during, and after v3.3.4:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis_attempt
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_3_development_ood_sidecar_analysis
```

The v3.3.3.1 archive may be used only as immutable infrastructure provenance.
It contains no OOD result and supplies no model output, activation,
intervention, normalization, rank, or scientific-value input to v3.3.4.

## 3. Sole permitted source-program repair

The v3.3.3 runner compared an in-memory JAX signature object containing Python
tuples against the same signature loaded from JSON, where those containers
became lists. This section replaces only that raw host-container equality.
There is no general tuple/list normalizer and no permission to ignore any
semantic or structural difference.

### 3.1 Exact representation-aware signature rule

The v3.3.4 freeze must contain the complete literal v3.3.2/v3.3.3 signature
contract, not only a digest. It has exactly the three named objects
`eight_interventions`, `selection`, and `target`, with respectively 17, 9,
and 3 leaves. For the runtime object require exactly:

- three `leaves` containers, all tuples, one per named object;
- 29 `shape` containers, all tuples, one per leaf;
- no other tuple/list type discrepancy;
- exact object names and order-independent mapping keys;
- exact treedef strings;
- exact indexed leaf order and leaf count within each named object;
- exact dtype string at every leaf; and
- exact shape rank and every non-negative integer shape value, rejecting
  booleans as integers.

For the frozen JSON contract, require the corresponding three `leaves`
containers and 29 `shape` containers to be lists. After the exact type and
semantic checks below, serialize **once** the complete top-level mapping whose
only keys are `eight_interventions`, `selection`, and `target`:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

The runtime and frozen full-mapping bytes must be byte-identical, exactly 2877
bytes, and
have SHA-256
`d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300`.
The raw Python tuple-versus-list direct equality is expected to be false and
must be persisted as a diagnostic; it is neither the pass criterion nor
rewritten into the historical v3.3.3 record.

Define:

```text
program_signature_structure_exact :=
    exact three object names
    and exact treedef strings
    and exact per-object leaf counts (17, 9, 3)
    and exact indexed leaf order
    and exact dtypes
    and exact numeric shapes
    and exactly 3 runtime leaves tuples plus 29 runtime shape tuples
    and exactly 3 frozen leaves lists plus 29 frozen shape lists

program_signatures_canonical_exact :=
    program_signature_structure_exact
    and runtime_canonical_bytes == frozen_canonical_bytes
    and canonical_size == 2877
    and canonical_sha256 == d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300
```

Any different treedef, leaf count/order, dtype, rank, numeric shape, mapping,
container location, canonical byte, size, or hash is a controlled stop before
apply zero. Arbitrary recursive conversion, NumPy coercion, stringification,
set-based comparison, shape sorting, dtype promotion, or post-hoc exception is
prohibited.

#### 3.1.1 Type-tagged runtime attestation and narrow adapter

Before lowering, write
`compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json`. Its exact top-level
key set is:

```text
schema_version, script_version, attempt_id, external_freeze_authorization, object_order,
runtime_container_tags, frozen_container_tags,
runtime_semantic_mapping, frozen_semantic_mapping,
runtime_canonical, frozen_canonical, comparisons,
created_at_unix_s
```

Require `schema_version="v3.3.4-program-signature-attestation-v1"`, the exact
frozen script version and attempt ID, and
`object_order=["eight_interventions","selection","target"]`.
`runtime_container_tags` and `frozen_container_tags` are arrays of exactly 32
objects with the exact key set `path,kind`; they are sorted in this order:

1. `/eight_interventions/leaves`, followed by
   `/eight_interventions/leaves/0/shape` through
   `/eight_interventions/leaves/16/shape`;
2. `/selection/leaves`, followed by `/selection/leaves/0/shape` through
   `/selection/leaves/8/shape`; and
3. `/target/leaves`, followed by `/target/leaves/0/shape` through
   `/target/leaves/2/shape`.

Every runtime `kind` is exactly `tuple`; every corresponding frozen `kind` is
exactly `list`. No other tuple occurs anywhere in the runtime mapping and no
other runtime/frozen container-kind difference exists. The adapter is a
dedicated function that accepts only the three frozen treedefs, visits exactly
those 32 JSON-pointer paths, replaces a tuple by a list at those paths only,
and rejects an already-list runtime container, a tuple elsewhere, an absent or
extra path, or any semantic change. It is not reusable as a recursive
normalizer.

`runtime_semantic_mapping` is the adapter output and
`frozen_semantic_mapping` is the exact frozen three-object mapping. Each
`*_canonical` object has the exact keys `sha256,size_bytes`; both must be
`{"sha256":"d8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300",
"size_bytes":2877}`. `comparisons` has exactly:

```text
direct_python_equality = false
runtime_tuple_container_count = 32
runtime_leaves_tuple_count = 3
runtime_shape_tuple_count = 29
frozen_list_container_count = 32
frozen_leaves_list_count = 3
frozen_shape_list_count = 29
declared_paths_exact = true
container_kinds_exact = true
treedefs_exact = true
leaf_order_counts_dtypes_shapes_exact = true
canonical_bytes_exact = true
canonical_hash_and_size_exact = true
```

The artifact retains the complete semantic mappings, not only booleans or
digests, so the CPU analyzer can reproduce all 32 type tags and the 2877-byte
payload without importing JAX. Runtime tuple facts are captured before the
JSON write; the analyzer validates the type-tag list plus semantic mappings
and does not pretend that JSON itself preserves tuples.

If construction or validation fails before a valid attestation can be sealed,
write instead
`compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION_FAILURE.json`, with exact
keys `schema_version,script_version,attempt_id,external_freeze_authorization,status,partial_runtime_tags,
partial_frozen_tags,failure,created_at_unix_s`; require `status="failure"`,
the two partial arrays to contain only the validated prefix of the fixed
32-path order, and `failure` exactly `type,message,traceback`. It is a terminal
provenance artifact, never an adapter input and never permission to lower.

#### 3.1.2 Exact source-input and same-object attestations

Every START, compiler source gate, and terminal embeds a
`source_input_audit` object with exactly these eight keys:

```text
bootstrap_sources_and_prior_trees_exact
tracked_head_and_frozen_inventory_exact
external_device_runtime_environment_exact
same_process_device_runtime_environment_exact
checkpoint_exact
reference_object_and_sequences_exact
protobuf_binding_exact
three_import_inventories_stable_exact
```

The value domains and transitions are exact:

| Phase/object | bootstrap | tracked/freeze | external device | same-process device | checkpoint | reference | protobuf import binding | three inventories stable |
|---|---|---|---|---|---|---|---|---|
| START and post-START Gate B | bool (true on START) | bool (true on START) | bool (true on START) | bool (true on START) | null | null | null | null |
| after frozen scientific imports / PRE_MODEL / protobuf validation | true | true | true | true | null | null | independently derived bool | null |
| after model/checkpoint/reference construction attempt and POST_MODEL | true | true | true | true | derived bool if its validation ran, otherwise null | derived bool if its validation ran, otherwise null | derived bool | null |
| after terminal import capture | true | true | true | true | phase-preserved bool/null | phase-preserved bool/null | derived bool | independently derived bool |
| compiler/source record | independently rederived bool | independently rederived bool | independently rederived bool | independently rederived bool | bool | bool | bool | bool |
| dispatch/raw | true | true | true | true | true | true | true | true |

Thus START has exactly the first four values true and the last four null.
Checkpoint/reference/protobuf/import terms may not be predicted from files or
copied from the freeze before their runtime evidence exists. Before any
compiler success/failure record is sealed, capture the terminal import
inventory; a compiler record has all eight independently derived booleans.
Dispatch requires all eight true. An early controlled terminal embeds the
latest phase-appropriate bool/null object; a compiler terminal embeds the
final eight-boolean object; raw and failed-current records necessarily embed
the all-true object. RAW_MANIFEST and RUN_COMPLETE embed exactly the object
appropriate to their terminal phase, and every repeated embedding within one
phase is byte-identical. Every embedding binds the SHA-256 and size of its
canonical object; re-derivation, embedded object, and binding must agree. A
supplied `true` is never sufficient by itself, and null is never coerced to
false.

The compiler record contains `same_object_attestation` with exactly:

```text
lower_call_count
compile_call_count
stablehlo_read_from_lowered_object
pre_backend_hlo_read_from_lowered_object
compile_argument_is_lowered_object
compiled_hlo_read_from_compiled_object
signature_attestation_from_apply_arguments
apply_callable_is_compiled_object
compiler_record_is_gate_record
lowered_python_id
compiled_python_id
```

Values are phase-exact. A lowering failure has counts `(1,0)`, both IDs null,
`signature_attestation_from_apply_arguments=true` and
`compiler_record_is_gate_record=true`, with the five object-flow booleans
null. A compile failure has counts `(1,1)`, a non-negative lowered ID, null compiled ID,
`stablehlo_read_from_lowered_object`,
`pre_backend_hlo_read_from_lowered_object`, and
`compile_argument_is_lowered_object`,
`signature_attestation_from_apply_arguments`, and
`compiler_record_is_gate_record` true, with the two compiled-object booleans
null. A successful compile has counts `(1,1)`, both non-negative IDs, and all
seven booleans true. No other value pattern is legal. The applicable booleans
are evaluated by literal Python `is` checks at the point of use, not inferred
from equal HLO text. Process-local IDs are audit-only, not cross-run equality
gates. Every terminal embeds the exact phase-appropriate object and canonical
hash/size. A false applicable primitive, different embedded object, missing
evidence, or second lower or compile call is a controlled stop before
dispatch.

### 3.2 Full source-program gate remains mandatory

Define:

```text
source_program_exact :=
    stablehlo_sha256 == 69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd
    and stablehlo_size == 3,196,162
    and pre_backend_hlo_sha256 == 675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750
    and pre_backend_hlo_size == 1,829,833
    and program_signatures_canonical_exact
    and entry_abi_exact
    and source/runtime/device/toolchain/checkpoint/reference/protobuf/import_exact
    and same_lowered_compiled_object
```

The persisted `source_program_gate` object has exactly:

```text
contract, observed, stablehlo_exact, pre_backend_hlo_exact,
program_signature_structure_exact, program_signatures_canonical_exact,
entry_abi_exact,
source_runtime_device_toolchain_checkpoint_reference_exact,
source_input_audit, source_input_audit_content_binding,
same_object_attestation, same_object_attestation_content_binding,
same_lowered_compiled_object, source_program_exact
```

`contract` contains the literal frozen hashes/sizes and full signature mapping;
`observed` contains literal artifact hashes/sizes, signature-attestation
binding, ABI binding, and runtime/device/toolchain/checkpoint/reference/import/
protobuf bindings. Each primitive is independently derived and may not be
computed by copying a prior boolean. START records the frozen contract only;
compiler and terminal records persist the observed object and all primitive
results.

The entry ABI is derived from the first compiled-HLO `HloModule` line by
replacing only the hexadecimal value inside
`fingerprint_before_lhs="<one hexadecimal payload>"` with
`fingerprint_before_lhs="<backend-generated>"`, omitting the line-ending
newline, and hashing the remaining UTF-8 bytes. Require exactly one
substitution and SHA-256
`ebf900771a87775a5fb657b90131fd884d68ce4725245defa164bf5066c74a80`.

The exact JAX `eval_shape`, lowered input/output pytree, entry parameter order,
batch size eight, output evidence/trace tree, target reducer, checkpoint,
reference sequence bindings, mixed-precision policy
`params=float32,compute=bfloat16,output=bfloat16`, runtime packages, device,
driver, CUDA/JAX/XLA environment, source protos/generated bindings, and three
import inventories remain mandatory. Checkpoint snapshot
`a8f293a76ee73d5b57f3bf2ae146510589fcf187`, checkpoint-manifest SHA-256
`1ed87db4c5bd7c5418c7734ec128faa4a9ecd186df2a024437484a8bc2b6e934`,
and reference-binding SHA-256
`da712cdca50f82113ac1d00cb2fa7171f7368f31aedf06c48ce92dbdb5897dca`
must match their complete frozen contents.

### 3.3 Compiled backend remains diagnostic

Lower and compile exactly once in the GPU launch process. StableHLO,
pre-backend HLO, program signatures, entry ABI, and the executable must come
from the same lowered/compiled object. Persist compiler provenance before the
first apply. After a successful compile, the terminal import/protobuf capture,
backend-diagnostic derivation, and source-program derivation are sibling
children of `compile_succeeded`; neither is logically downstream of the other.
Derive and persist diagnostic completeness even when
`source_program_exact=false`, then evaluate the source gate. Dispatch requires
both independently true. The exact ordering is `compile -> raw graph
publication -> terminal provenance capture -> diagnostic derivation ->
source-program derivation -> compiler-record publication -> dispatch`.

The new raw compiled HLO, SHA-256, size, executable fingerprint, computation
and instruction counts, fusion kinds, Triton settings, cuBLAS/cuDNN
algorithms/workspaces, and backend configurations must be recorded and
compared descriptively against v3.3.3 and v3.3.2. They are explicitly not
equality gates. No compiled-HLO canonicalizer, observed-difference whitelist,
similarity threshold, cached executable, retry-until-match, or second compile
is allowed.

If and only if every `source_program_exact` term and
`diagnostic_provenance_complete` pass, use that exact in-memory executable for
every model apply. A compile failure, guarded second lower/compile request,
cache hit, source/ABI mismatch, applicable same-object failure,
import/protobuf instability, or loss of provenance consumes v3.3.4 as a
controlled stop with no retry. `lower_attempt_budget=compile_attempt_budget=1`
means a second-call guard runs before invoking JAX: a forbidden request is
durably recorded, but an actual second lower/compile call never occurs and the
observed call counts never exceed one.

#### 3.3.1 Exact diagnostic parser and fingerprint

`COMPILER_PROVENANCE.json` has exactly the v3.3.3 successful compiler keys
(`executable_name,compile_count,lower_attempt_count,compile_attempt_count,
successful_compile_count,compile_seconds,executable_fingerprint,artifacts,
program_signatures,program_signatures_sha256,entry_abi,source_program_gate,
backend_diagnostics,diagnostic_comparisons,kernel_cache_provenance`) plus
`program_signature_attestation`, `external_freeze_authorization`, `source_input_audit`,
`source_input_audit_content_binding`, `same_object_attestation`, and
`same_object_attestation_content_binding`, `attempt_budget_audit`, and
`diagnostic_provenance_complete`. On lower/compile/guarded-budget/same-object
failure it instead has the exact v3.3.3 failure keys plus those seven additions.
No other key is accepted. `attempt_budget_audit` has exactly
`lower_budget,compile_budget,lower_invocations,compile_invocations,
forbidden_request,forbidden_request_detected_before_invocation`; budgets are
one, invocation counts are each 0 or 1, `forbidden_request` is null, `lower`,
or `compile`, and the last boolean is true exactly for a non-null forbidden
request. No record may contain an invocation count greater than one.

The executable fingerprint formula remains literally:

```python
hashlib.sha256(bytes.fromhex(compiled_hlo_sha256)).hexdigest()
```

It is not the SHA-256 of compiled-HLO bytes. The entry ABI parser must find
exactly one first `HloModule` line and exactly one
`fingerprint_before_lhs="<hex>"` token, replace only the hex payload, and
record the original line, normalized line, sizes, hashes, and substitution
count. The backend diagnostic parser is the exact v3.3.3 parser over raw
compiled-HLO text: persist computation count, instruction-record count,
custom-call targets, fusion-kind counts, Triton fusion count and literal
backend configs and convolution algorithm/workspace summaries under the frozen
nested schema. `backend_diagnostics` has exactly
`descriptive_only_not_an_equality_gate,computation_count,
instruction_count_excluding_computation_headers,instruction_record_count,
fusion_kind_counts,triton_configuration_count,triton_configurations,
cublas_call_count,cublas_algorithms,cudnn_call_count,
cudnn_algorithms_workspaces`. A Triton row has exactly
`block_level_fusion_config,instruction_line_sha256`; a cuBLAS row has exactly
`target,gemm_backend_config,instruction_line_sha256`; a cuDNN row has exactly
`target,algorithm,workspace_size_bytes,instruction_line_sha256`. Its raw
artifact binding and derived summary
must be recomputable by the CPU analyzer.

Backend values and their differences remain diagnostic, but a parser
exception, an unrecognized mandatory record shape, or a fingerprint-formula
mismatch after all three graph artifacts are durably published is
`controlled_stop_diagnostic_provenance_failure` before dispatch. Failure to
obtain or publish raw compiled HLO cannot satisfy that exact membership and is
instead an incomplete compiler terminal with best-effort TERMINAL_FAILURE.
It is never permitted to proceed while merely marking diagnostic provenance
unknown.

For such a caught failure, do not publish a valid `COMPILER_PROVENANCE.json`.
Publish instead `COMPILER_DIAGNOSTIC_FAILURE.json`, with exact keys
`status,executable_name,lower_attempt_count,compile_attempt_count,
successful_compile_count,artifacts,program_signature_attestation_binding,
external_freeze_authorization,source_input_audit,source_input_audit_content_binding,
same_object_attestation,same_object_attestation_content_binding,
source_program_gate_without_backend_diagnostics,failure,
attempt_budget_audit,diagnostic_provenance_complete,
compiled_backend_diagnostic_only,no_dispatch,created_at_unix_s`. Require
`status="diagnostic_provenance_failure"`, counts `(1,1,1)`, the exact three
graph bindings, a successful signature boundary and independently derived
source-program object through the point of failure, `failure` exactly
`type,message,traceback`, both terminal booleans true, and
`diagnostic_provenance_complete=false`. Diagnostic failure
has precedence when diagnostics are incomplete, even if the independently
derived source gate is also false; source mismatch is used only when
diagnostics are complete. If this artifact itself cannot be published, preserve only the
incomplete prefix and best-effort TERMINAL_FAILURE; do not synthesize a
successful compiler record.

## 4. Fresh v3.3.4 state and provenance order

The v3.3.4 machine freeze must bind these exact fresh paths:

```text
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_4_device_preflight
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_4_preflight_kernel_cache
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_4_model_kernel_cache
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_one_shot
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_analysis_attempt
/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_4_development_ood_sidecar_analysis
```

At freeze generation and the independent pre-launch audit, all six paths must
be absent and non-symlinked. Thereafter absence is phase-specific rather than
global:

| Checkpoint | Paths that must exist | Paths that must remain absent |
|---|---|---|
| Before external preflight allocation | none | all six |
| After the sole external preflight | device preflight and preflight cache only | model cache, model run, analysis attempt, analysis output |
| Immediately before START | passing device preflight and sealed preflight cache; freshly allocated empty model cache | model run, analysis attempt, analysis output |
| Immediately after START | passing device preflight, both cache roots, model run | both analysis paths |
| After any model-run terminal | device preflight, both cache roots, immutable model run | both analysis paths |
| Immediately before analyzer START | device preflight, both cache roots, immutable model run | analysis attempt and analysis output |
| During/after analyzer | all prerequisite paths plus the legal append-only analysis prefix | only the not-yet-published suffix of that prefix |

Every creator must still require its own destination absent and non-symlinked
immediately before one-shot creation. No v3.3.3 or v3.3.3.1 path may be
resumed, overwritten, copied into, normalized, or used as a raw-value input.

The provenance order is a strict two-gate sequence. **Gate A** runs before
importing JAX, AlphaGenome, model code, or any frozen helper and before reading
a consumed run/compiler/terminal record. A standard-library bootstrap must
perform this order:

1. require the exact committed v3.3.4 bundle/freeze and globally tracked-clean
   HEAD;
2. rehash all 96 v3.3.3 live source rows and their exact
   `git show 228083b931dbc62d4a283e68df01011f5ef4bff9:<path>` blobs;
3. rehash the exact v3.3.3 run, compiler, preflight, and cache memberships,
   sizes, hashes, and trees from Section 1;
4. rehash all v3.3.3.1 source rows and their declared amendment/implementation
   blobs, then all four production artifacts and their exact
   `git show 37bd58e88e1814f9a67bfbaaaad66d0a2b77f242:<path>` blobs from Section 2; and
5. prove both original v3.3.3 analyzer destinations remain absent.

Only after Gate A passes may the launcher perform the sole external preflight
and the same-process no-model device/runtime check. The runner then persists
`ATTEMPT_STARTED.json` exactly once. **Gate B** immediately repeats the same
standard-library source/prior/artifact checks after START and before importing
or executing any scientific helper or constructing the model/checkpoint. Gate
B may read only START, the freeze/authorization objects, and the same inputs
already read by Gate A. A Gate-B failure writes only the exact post-START
provenance terminal from Section 6.2. This is the only legal sequence:
`Gate A -> preflight/same-process gate -> START -> Gate B -> scientific
imports/model`. There is no repeated pre-import source gate before START and
no scientific import between START and Gate B.

Use one visible `NVIDIA GeForce RTX 3090`, UUID
`GPU-64111645-1e42-a96d-f192-4abbec4b8090`, compute capability 8.6, driver
560.35.05, VBIOS 94.02.42.C0.05, kernel 6.8.0-136-generic, Python 3.13.5,
and the exact v3.3.3 frozen runtime/package manifest. Unset
`LD_LIBRARY_PATH`; require `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`JAX_ENABLE_COMPILATION_CACHE=false`, and `CUDA_CACHE_DISABLE=1`; reject all
frozen forbidden compiler, persistent-cache, kernel-cache, and autotune-cache
inputs. External-preflight and model cache roles and roots must be distinct,
fresh, and empty before import. There is exactly one external allocation
`preflight_0000`; its failure consumes this version and cannot be retried.

### 4.1 Literal machine-freeze and source inventory

The v3.3.4 machine freeze has exactly the 69 top-level key names of the
v3.3.3 freeze, with versioned values where appropriate, plus exactly these 13
new names and no others:

```text
v3_3_3_1_archive
program_signature_attestation_contract
source_input_audit_contract
same_object_attestation_contract
dispatch_journal_contract
failed_current_contract
raw_record_contract
raw_manifest_contract
terminal_contract
preflight_contract
compiled_diagnostics_contract
source_inventory_contract
external_freeze_authorization_contract
```

Thus its exact top-level key count is 82. The inherited `file_sha256` mapping
contains the exact 96 source rows from the v3.3.3 freeze, unchanged, plus
exactly these 12 prospective v3.3.4 source paths, sorted by POSIX relative
path, for exactly 108 rows:

```text
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4.py
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4.sh
experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_test.py
experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_freeze.py
experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4.py
experiments/interpretability/opensplice/run_device_preflight_v3_3_4_test.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4.py
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4.sh
experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_test.py
experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_infrastructure_amendment_v3_3_4.md
experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4.py
```

Every row has an exact lowercase SHA-256 in `file_sha256`; the parallel
`source_inventory_contract` has exact per-row `path,sha256,size_bytes,git_mode`
objects. The two shell rows have `git_mode="100755"`; the other ten are
`"100644"`. The freeze file itself is necessarily outside its self-hash
inventory. No one of the 108 inventoried sources may contain or hard-code the
final freeze digest: doing so would create an impossible hash fixed point.

The dependency is instead explicitly acyclic:

1. the 108 source bytes are finalized and their hashes are written into the
   freeze;
2. the freeze is generated, committed, and independently audited;
3. the independent coordinator authorizes one launch with three mandatory
   environment values, `V334_AUTHORIZED_GIT_HEAD`,
   `V334_AUTHORIZED_FREEZE_SHA256`, and
   `V334_AUTHORIZED_FREEZE_SIZE_BYTES`; and
4. the wrapper computes the live HEAD and freeze hash/size, compares them to
   those externally supplied audited literals, requires the freeze tracked and
   byte-identical to `git show HEAD:<freeze-path>`, and passes the validated
   authorization object in memory to the launcher/runner.

The exact `external_freeze_authorization_contract` keys are
`git_head_environment_name,freeze_sha256_environment_name,
freeze_size_environment_name,freeze_path,required_for_dry_run,
required_for_production,source_files_must_not_embed_final_freeze_digest`.
The values name the three variables above, bind the fixed freeze path, set both
required booleans and the final anti-cycle boolean true, but contain no final
HEAD/hash/size. The wrapper rejects missing, malformed, CLI-supplied, or
unapproved values. Every later preflight/START/compiler/raw/manifest/terminal
record embeds the validated runtime object with exact keys
`git_head,freeze_path,freeze_sha256,freeze_size_bytes,live_equals_git_show,
tracked_clean,authorization_source`; require both booleans true and
`authorization_source="external_post_commit_audit"`. Each process recomputes
the live freeze hash/size and compares it to that object; no process trusts a
copied digest alone. The analyzer uses the same externally audited tuple and
repeats the computation before its START and every final TOCTOU gate. Tests
carry fixture values rather than a production digest.

The old v3.3.3.1 freeze remains exactly 36 top-level keys with exactly seven
inventory entries. Its amendment is bound to commit
`d2a013944a399ddac59a023d7d84ea5a7c23e9f4`, while its implementation bytes
are bound to `98c467ae16200071d110c9d73520e35e5e6d7bbf`; those roles may not be
collapsed or relabelled.

### 4.2 Source-file inventories, loaded-module attestations, and protobuf phases

The runner writes three independently captured provenance artifacts at the
fixed names `IMPORT_PROVENANCE_PRE_MODEL.json`,
`IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json`, and
`IMPORT_PROVENANCE.json`. The historical filenames are retained, but their two
roles are now explicit and must not be conflated:

- `prospective_upstream_source_files` is a source-file inventory of exactly 26
  expected AlphaGenome files whether or not a particular file has yet appeared
  in `sys.modules`; and
- `loaded_scientific_modules` is an actual snapshot of every file-backed loaded
  module under the three allowlisted roots `alphagenome_research_checkout`,
  `upstream_alphagenome_checkout`, and `locked_opensplice_checkout`.

Each artifact has exact top-level keys
`schema_version,phase,external_freeze_authorization,prospective_upstream_source_file_count,
prospective_upstream_source_files,loaded_scientific_module_count,
loaded_scientific_modules,upstream_source_attestation,
v3_3_4_sidecar_sources,created_at_unix_s`. A prospective row has exactly
`module_name,path,declared_root,relative_path,sha256,size_bytes,source_kind,
git_mode,filesystem_mode`. Its module names, paths, hashes, and sizes are the
exact 26-row `upstream_imported_modules` mapping in the frozen v3.3.3 contract.
Exactly 22 rows have `source_kind="git_tracked"`, `git_mode="100644"`, and
clean blobs at upstream AlphaGenome commit
`95cdbfce7981411453e5e094519bcf0605720199`. Exactly four rows have
`source_kind="generated_untracked_exception"`, `git_mode=null`, are regular
non-symlink files ignored by upstream Git, and are:

```text
alphagenome.protos.dna_model_pb2
alphagenome.protos.dna_model_service_pb2
alphagenome.protos.dna_model_service_pb2_grpc
alphagenome.protos.tensor_pb2
```

All 26 prospective filesystem rows have `filesystem_mode="0664"`, including the
four generated exceptions; a different live mode is a provenance mismatch.

Their exact generated bytes, embedded headers, six source inputs, protobuf
7.35.1 and grpcio 1.83.0 runtimes, `grpcio-tools=unavailable_not_used`, and
no-regeneration claim are copied literally from
`upstream_generated_binding_exception`. Local calibration bindings remain the
two exact untracked generated outputs (2794-byte `.py` SHA-256
`4673289dd481fd8c4976f602ab36b07646304107e352e3e6d27b2abe4f9e9ebc`;
1815-byte `.pyi` SHA-256
`329dc390abeb187084fff28fbe6cb6d9868aa8867326bf53f9a52d4c83f527f9`)
both regular non-symlink mode `0664`, derived from
the frozen source-proto/dependency-proto provenance only as historical bytes;
the current standalone protoc was not used and exact regeneration is not
claimed. `PROTOBUF_PROVENANCE.json` has exactly the v3.3.3 keys
`byte_level_reproducibility,current_protoc_was_used_to_generate_frozen_outputs,
current_standalone_protoc,dependency_pb2,dependency_proto,
embedded_generated_header,generated_outputs,historical_generation_provenance,
imported_dependency_pb2,imported_pb2,protobuf_runtime_version,
regeneration_claim,source_proto,tensor_pb2,tensor_proto` plus exactly
`external_freeze_authorization`. It reproduces the literal frozen nested
schemas/values, proves the actually imported module paths and bytes, and
embeds the exact authorization object. Missing/extra keys or merely equivalent
paths fail.

Each `loaded_scientific_modules` row has exactly
`name,path,root,sha256,size_bytes,filesystem_mode`; it is sorted by
`(name,path)`. Before freeze, the implementation must deliberately import the
complete frozen scientific helper set and record its exact names, paths,
hashes, sizes, and count in
`source_inventory_contract.loaded_scientific_module_contract`. This contract
must include all 26 upstream names, all actually loaded local
`alphagenome_research` modules, every loaded locked-OpenSplice module, the
v3.3.4 runner under both legitimate `__main__`/`__mp_main__` aliases when
present, and the v3.3.4 bootstrap/helper modules. A duplicate path is legal
only for the exact `__main__`/`__mp_main__` pair with every non-name field
identical; duplicate names and all other duplicate paths are rejected.

Capture PRE_MODEL only **after all frozen scientific modules have been loaded
and attested**, but before model/checkpoint/reference construction. Capture
POST_MODEL_PRECOMPILE immediately after the construction attempt and before
signature/lowering; on a caught construction failure, capture it in the
exception handler before sealing the terminal. Capture terminal immediately
after compile/failure handling and before RAW_MANIFEST/RUN_COMPLETE. The only
allowed differences among successful captures are `phase` and timestamp;
after removing them, both the 26 source-file rows and the actual loaded-module
rows must have canonical byte equality. Lazy imports, a missing local helper,
a 21+5 or 26-tracked claim, a new generated byte, a different imported alias,
or any non-identical loaded row is a controlled stop. A failure to atomically
publish a required phase artifact follows the explicit incomplete
provenance-publication terminal in Section 6.3 and is never silently replaced
by an invented complete inventory.

### 4.3 External preflight, PID, and cache contracts

The external-preflight root has exactly one allocation and, at terminal,
exactly the five regular non-symlink files `.allocation.lock`,
`.preflight_0000.reserved`, `preflight_0000.json`,
`preflight_0000.stdout.log`, and `preflight_0000.stderr.log`, with no extra
file or directory. The JSON top-level key set is the v3.3.3 set plus the exact
v3.3.4 authorization and cache-hit-evidence additions shown here:

```text
amendment_sha256, created_at_unix_s, external_freeze_authorization, external_cache_post_observation,
external_cache_hit_evidence, failure, freeze, freeze_sha256, logs, no_jit_or_array_kernel,
no_model_or_biological_access, observation, original_protocol_sha256,
preflight_attempt_number, script_version, status, warnings
```

Require attempt number zero, `status` exactly `pass` or `fail`, the exact
freeze object, exact stdout/stderr bindings, `observation.pid` a positive
integer, one RTX 3090/UUID device, `jax_default_backend="gpu"`, and
`no_jit_no_array_no_model=true`. The preflight imports JAX/JAXLIB only; it
must not import AlphaGenome, OpenSplice, model, checkpoint, reference, or
interpretability modules and must execute no JIT/array kernel. Its cache root
is the fresh external-preflight cache, not the model cache. Main START binds
the external preflight path/hash/tree and PID. The runner records its own PID
in START and every terminal; model construction, lower, compile, and all
dispatches must have that same runner PID. The external preflight PID must be
different from the runner PID.

`observation` has exactly `atomic_publication_supported,environment,hostname,
jax_default_backend,jax_enable_compilation_cache,jax_gpu_devices,
jax_module_version,jaxlib_module_version,kernel,no_jit_no_array_no_model,
nvidia_smi,packages,pid,platform,python_executable,python_version,
runtime_environment,v3_3_4_runtime_environment`. `logs` has exactly
`stdout,stderr`, each a file binding. `failure` is null on pass and exactly
`type,message,traceback` on fail; `warnings` is an array of strings.
`external_cache_post_observation` is the exact cache binding from below and
`external_cache_hit_evidence` is the exact phase-appropriate object defined
below. These are the sole persisted locations for the external preflight's
terminal cache tree and hit decision; the same two objects are embedded
without normalization in the successful-preflight binding and START.

External preflight must prove the Section 5.2 publication primitive on the
same filesystem. It anonymously writes and no-replace-links exactly one
deterministic `atomic_publication_probe_v1.txt` into the preflight cache,
retains it sealed mode `0400`, binds its bytes in the cache terminal tree, and
records `observation.atomic_publication_supported=true`. This diagnostic
output is never an input to model compilation. Unsupported `O_TMPFILE`,
`AT_EMPTY_PATH`, no-replace link, file `fsync`, directory `fsync`, or mode seal
causes the sole preflight to fail before START.

Before START, the launcher runs a same-process, no-model gate and passes its
object by in-memory handoff to the runner; no second Python process is allowed.
That object has exactly `pid,parent_pid,external_preflight_pid,
default_backend,jax_gpu_devices,nvidia_smi,runtime_environment,
runtime_versions,freeze_sha256,external_freeze_authorization,external_preflight_binding,
external_preflight_tree_sha256,model_cache_pre_import,
current_source_inventory_exact,prior_artifacts_exact,no_model_constructed,
no_jit_or_array_kernel,created_at_unix_s`. Require runner PID equality,
external PID inequality, one exact RTX/UUID, the frozen runtime/environment,
fresh model cache, both audit booleans true, and both no-model/no-kernel
booleans true. START embeds this exact object and its content binding; terminal
rehashes all file-backed components and proves the same runner PID.

The allocation lock is mode `0600`; the reservation, JSON, stdout, and stderr
files are sealed mode `0400`. Their logs may be empty but must exist. A
different mode is a provenance failure.

Both cache roots begin with the exact directory set `.`, `triton`, `xdg` and
zero files. Their *input tree* is therefore the empty file-tree digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
the directory-aware diagnostic tree uses, in sorted order,
`b"D\0" + relpath + b"\0"` for each directory and
`b"F\0" + relpath + b"\0" + raw_sha256_bytes` for each file. The exact cache
binding object keys are `cache_role,cache_root,triton_cache_dir,xdg_cache_home,
directory_count,directory_paths,file_count,files,tree_sha256,
default_user_cache_paths_eligible,
diagnostic_outputs_only_no_cache_input`. `directory_paths` is an explicitly
stored UTF-8/POSIX-lexicographically sorted array of cache-root-relative
strings, includes `.`, and has length `directory_count`; the initial value is
exactly `[".","triton","xdg"]`. `files` is a POSIX-relative-path-sorted JSON
object mapping paths to exact `sha256,size_bytes` objects. For every
pre-import, historical, terminal, and external-preflight binding, compute the
diagnostic digest by concatenating all sorted directory frames first and all
sorted file frames second. Missing `directory_paths`, an inferred directory
set, or a count-only binding is invalid.

`cache_hit` is true if any pre-import file exists; any executable or compiler
artifact is read from a prior/default/persistent cache; JAX reports a
persistent compilation-cache hit; the sole compile is skipped; or the
pre-import routing differs from the two fresh versioned roots. A cache hit is
a stop. Files created by the sole compilation are terminal diagnostic
provenance only. Persist separate pre-import, post-compile-or-failure
historical, and terminal-live bindings; terminal bytes need not equal an
earlier snapshot and the explicit
`historical_to_terminal_equality_is_a_gate=false` remains mandatory.

The exact `cache_hit_evidence` object is
`pre_import_files_present,default_user_cache_path_eligible,
persistent_compilation_cache_hit_reported,executable_deserialized,
compile_skipped,compile_stage_not_applicable,old_cache_input_opened,
routing_exact,cache_hit`. In external preflight, the two compile-specific
values are null and `compile_stage_not_applicable=true`; in the model process
they are false and `compile_stage_not_applicable=false`. All other adverse
flags are false, `routing_exact` is true, and `cache_hit` is false under the
frozen phase-aware formula. Both preflight and main terminal persist this object. A
runtime cache-hit signal is false only when the exact JAX diagnostic source
named in the freeze was queried successfully; if that signal is unavailable,
the run stops for diagnostic-provenance failure rather than assuming false.

## 5. One full OOD sidecar, no other experiment

### 5.1 Exact development allowlist and order

The only eligible recipients are the 20 rows of
`superset_graph_v3_2_development_variants.tsv`, SHA-256
`24a0afec1c020803152c7f55a0a78ac345763173dd79a4175e889d9192db05f9`,
in this exact zero-based order:

| Order | Gene | Variant ID | Class/rank | GRCh38 allele |
|---:|---|---|---|---|
| 0 | BRAF | `BRAF_e14_A117G` | significant_effect/1 | `chr7:140754187:T>C` |
| 1 | BRAF | `BRAF_e14_T71A` | significant_effect/2 | `chr7:140754233:A>T` |
| 2 | BRAF | `BRAF_e14_T71G` | significant_effect/3 | `chr7:140754233:A>C` |
| 3 | BRAF | `BRAF_e14_A117C` | significant_effect/4 | `chr7:140754187:T>G` |
| 4 | BRAF | `BRAF_e14_A89G` | significant_effect/5 | `chr7:140754215:T>C` |
| 5 | BRAF | `BRAF_e14_A77C` | significant_effect/6 | `chr7:140754227:T>G` |
| 6 | BRAF | `BRAF_e14_T121C` | neutral_control/1 | `chr7:140754183:A>G` |
| 7 | BRAF | `BRAF_e14_A69G` | neutral_control/2 | `chr7:140754235:T>C` |
| 8 | BRAF | `BRAF_e14_C68T` | neutral_control/3 | `chr7:140754236:G>A` |
| 9 | BRAF | `BRAF_e14_G118A` | neutral_control/4 | `chr7:140754186:C>T` |
| 10 | SLC25A48 | `SLC25A48_e8_G70A` | significant_effect/1 | `chr5:135880772:G>A` |
| 11 | SLC25A48 | `SLC25A48_e8_A69C` | significant_effect/2 | `chr5:135880771:A>C` |
| 12 | SLC25A48 | `SLC25A48_e8_A69T` | significant_effect/3 | `chr5:135880771:A>T` |
| 13 | SLC25A48 | `SLC25A48_e8_T68G` | significant_effect/4 | `chr5:135880770:T>G` |
| 14 | SLC25A48 | `SLC25A48_e8_G70C` | significant_effect/5 | `chr5:135880772:G>C` |
| 15 | SLC25A48 | `SLC25A48_e8_G71T` | significant_effect/6 | `chr5:135880773:G>T` |
| 16 | SLC25A48 | `SLC25A48_e8_C67T` | neutral_control/1 | `chr5:135880769:C>T` |
| 17 | SLC25A48 | `SLC25A48_e8_C67G` | neutral_control/2 | `chr5:135880769:C>G` |
| 18 | SLC25A48 | `SLC25A48_e8_T68C` | neutral_control/3 | `chr5:135880770:T>C` |
| 19 | SLC25A48 | `SLC25A48_e8_C6A` | neutral_control/4 | `chr5:135880708:C>A` |

The exact unrelated-donor derangement is
`0..5 -> 10..15`, `10..15 -> 0..5`, `6..9 -> 16..19`, and
`16..19 -> 6..9`, preserving the within-class rank. The freeze stores the
literal 20 recipient objects and 20-entry donor map. No runtime TSV sort,
filter, score selection, substitution, or confirmation row is permitted.

The eight row roles are exactly, in order,
`reference_baseline, alternate_baseline, reference_into_alternate,
alternate_into_alternate_self_control, alternate_into_reference,
reference_into_reference_self_control, unrelated_reference_donor,
unrelated_alternate_donor`. Natural identity rows are
`[0,1,1,1,0,0,6,7]`; intended donor rows are `[0,1,0,1,1,0,6,7]`;
unrelated donor rows are `[0,1,6,1,7,0,6,7]`; invariant rows are
`[0,1,3,5,6,7]`; and active recipient rows are `[2,4]`. These literal arrays
are repeated in freeze, START, every raw record, and terminal and must agree.

### 5.2 Exact execution and append-only dispatch journal

After a durable `ATTEMPT_STARTED.json`, exact source-program pass, and exactly
one successful eight-row compile, execute only:

```text
for recipient order 0..19, in the frozen development-manifest order:
    for anchor ID in (0, 127, 128, 255), in that order:
        intended call
        exact intended repeat
        unrelated-donor call
        exact unrelated-donor repeat
```

This is exactly 80 recipient-anchor records and 320 model applies from one
fixed eight-row executable. The execution index is `4*recipient_order +
anchor_minor_index`, where anchor-minor order is `(0,127,128,255)`. Each
record has four call indices in exact order: `0=intended`,
`1=intended_repeat`, `2=unrelated`, `3=unrelated_repeat`. Global dispatch
index is `4*execution_index + call_index`, hence `0..319`.

Durability is an explicit append-only ledger, not a mutable counter. Before
every model call, exclusively create, flush, `fsync` the file, and `fsync` its
parent directory:

```text
dispatch_journal/started/{global_dispatch_index:03d}.json
```

After a successful return and before inspecting the returned arrays, do the
same for:

```text
dispatch_journal/completed/{global_dispatch_index:03d}.json
```

A started-event object has exactly:

```text
schema_version, event, attempt_id, script_version, execution_index,
recipient_order, recipient_variant_id, anchor_id,
call_index_within_record, call_role, global_dispatch_index,
runner_pid, source_input_audit_sha256, same_object_attestation_sha256,
started_at_unix_s
```

Require `event="dispatch_started"`. A completed-event object has exactly:

```text
schema_version, event, attempt_id, script_version, execution_index,
recipient_order, recipient_variant_id, anchor_id,
call_index_within_record, call_role, global_dispatch_index,
runner_pid, source_input_audit_sha256, same_object_attestation_sha256,
started_event_sha256, returned, completed_at_unix_s
```

Require `event="dispatch_completed"` and `returned=true`; it omits
`started_at_unix_s`. Every event identity must match the frozen order and
runner PID. Both directories contain a strict zero-based prefix, no gaps,
extras, symlinks, special files, or empty directories. `started_count` is the
number of attempted calls; `completed_count` is the number that returned.
There is never a completion without its exact started-event binding.
Every event binding used in a raw/failed-current record is an object with
exact keys `path,sha256,size_bytes`; `path` is the exact model-run-root-relative
ledger path above. Raw-record binding arrays contain four such objects in call
order. Failed-current arrays contain the strict started/completed prefix for
that record. Manifest/terminal binding maps instead use those relative paths
as JSON object keys and exact `sha256,size_bytes` values. All ledger-tree
digests use the same run-root-relative UTF-8 path + NUL + raw-digest framing as
Section 6.3.

Every sealed artifact uses one publication primitive: serialize and validate
the complete bytes in memory; open an anonymous same-filesystem inode with
Linux `O_TMPFILE` at mode `0600`; write all bytes; flush and `fsync` the inode;
set mode `0400`; then publish with `linkat(..., AT_EMPTY_PATH)` to the exact
previously absent final name using no-replace semantics, and `fsync` the parent
directory. The final name is never opened for writing. There is no named
temporary file, no `os.replace`, and no interval in which a partial final file
is visible. A failure before link leaves no named artifact. A failure after
link preserves the sealed final artifact but is an incomplete publication
failure: best-effort `TERMINAL_FAILURE.json` may be written, but a standard
RAW_MANIFEST/RUN_COMPLETE must not claim it. There is no unlink, cleanup,
second link, or publication retry. Tests inject failure at every step and
assert exact visible membership.

After all four completion events, validate and exclusively persist the raw
record, then `fsync` it and its parent before advancing. A mid-dispatch or
validation failure writes exactly one separate `failed_current` artifact and
never writes an invalid record into the valid raw namespace. Resume,
replacement, stitching, or deletion is prohibited.

### 5.3 Raw and failed-current schemas

A valid raw path is exactly
`raw/ood_anchors/{recipient_order:03d}_{slug(variant_id)}/{anchor_id:03d}.json`.
Its object has exactly the v3.3.3 raw-record keys listed below, with
`family="v3_3_4_unrelated_donor_sidecar_anchor"` and versioned provenance,
plus `dispatch_started_bindings`, `dispatch_completed_bindings`,
`source_input_audit`, `source_input_audit_content_binding`,
`same_object_attestation`, and `same_object_attestation_content_binding`:

```text
status, family, script_version, amendment_sha256, amendment_commit,
original_protocol_sha256, freeze_sha256, external_freeze_authorization, execution_index,
sidecar_execution_index, execution_order, eight_row_executable_fingerprint,
same_eight_row_compiled_executable, six_row_executable_used, recipient_case,
donor_case, coalition, batch_roles, natural_identity_rows,
intended_donor_rows, unrelated_donor_rows, invariant_rows_between_calls,
active_recipient_rows, active_recipient_cross_call_equality_gate,
active_recipient_cross_call_inequality_gate, original_artifact_bindings,
original_ood_records_used_as_data, recipient_sequence_sha256,
donor_sequence_sha256, runtime_interventions, intended_target_readout,
intended_repeat_target_readout, unrelated_target_readout,
unrelated_repeat_target_readout, intended_trace_fingerprint,
intended_repeat_trace_fingerprint, unrelated_trace_fingerprint,
unrelated_repeat_trace_fingerprint, rowwise_trace_fingerprints, raw_movement,
model_apply_count_through_record, checks, failure, seconds,
dispatch_started_bindings, dispatch_completed_bindings,
source_input_audit, source_input_audit_content_binding,
same_object_attestation, same_object_attestation_content_binding,
confirmation_scope_disclosure,
created_at_unix_s
```

For a valid record, `status="complete"`, `failure=null`, both dispatch arrays
contain exactly four bindings in call order, and
`model_apply_count_through_record=4*(execution_index+1)`. Values, trace trees,
rowwise fingerprints, checks, and intervention records retain these literal
v3.3.3 nested schemas:

- each case has exactly `order,selection_version,selection_class,
  observed_effect_sign,gene,exon_id,ensembl_exon_id,chromosome,strand,
  exon_start_1based,exon_end_1based,variant_id,position_1based,
  reference_bases,alternate_bases,region,mut_type,delta_psi,delta_logit`;
- coalition has exactly `coalition_id,t,e_mask,e_bits,e_bits_binary,
  enabled_players,coalition_bit_order,shapley_player_order`;
- each target readout has exactly `endpoint_axis,selected_logit_axis,
  selected_logits,endpoint_margins,means,totals,num_values`, with axes
  `acceptor,donor` and `relevant_class,padding_class` and shapes `[8,2,2]`,
  `[8,2]`, `[8]`, `[8]`, scalar 2;
- each intervention record has exactly `transformer_output,encoder_skips,
  final_embedding,phase_r_residuals`; whole transfers have exactly
  `donor_batch_indices,natural_identity_batch_indices,transfer_mask`, residual
  transfers exactly `donor_batch_indices,transfer_mask`, and phase-R residuals
  exactly the three frozen residual route names;
- rowwise trace fingerprints have exactly `natural_final_embeddings,
  effective_final_embeddings,transformer_output_natural_fingerprint,
  encoder_skips_natural_fingerprints`; a compact rowwise object has exactly
  `full_shape,dtype,row_count,rows,collision_semantics`, and each of eight row
  objects exactly `row,shape,dtype,size_bytes,sha256`; each natural-fingerprint
  value object has exactly `shape,dtype,values`;
- raw movement has exactly `intended,unrelated`, each with exactly
  `reference_into_alternate,alternate_into_reference`; seconds has exactly the
  four call-role keys; original artifact bindings have exactly
  `recipient_identity,donor_identity,recipient_six_row_coalition`; and
- checks have exactly `passed,corrected_host_assertion_version,
  upstream_transformer_natural_tensors_all8_exact_between_calls,
  upstream_T_E_natural_fingerprints_all8_exact_between_calls,
  natural_final_invariant_rows_exact_between_calls,natural_final_invariant_rows,
  active_rows_cross_call_equality_not_required,
  active_rows_forced_difference_not_required,
  full_within_call_natural_effective_final_exact,
  endpoint_invariant_rows_exact_between_calls,self_rows_exact_within_each_call,
  id0_all8_natural_final_exact_between_calls,
  id0_within_call_natural_final_recipient_noop_exact,
  id0_all8_endpoint_exact_between_calls,id0_recipient_noop_exact,
  id255_intended_endpoint_closure_exact,
  id255_unrelated_endpoint_closure_exact,intended_route_tensor_donor_exact,
  unrelated_route_tensor_donor_exact,enabled_disabled_T_E_exact,
  runtime_route_masks_and_maps_exact,intended_target_repeat_exact,
  intended_trace_repeat_exact,unrelated_target_repeat_exact,
  unrelated_trace_repeat_exact,transformer_internal_seams_disabled_exact,
  final_embedding_disabled_exact,normalization_computed`.

Missing/extra nested keys, changed axes/shapes/types, or permissive arbitrary
mappings fail. The machine freeze embeds these recursive schemas literally and
tests assert every branch.

There is at most one
`raw/failed_current/{execution_index:03d}_{slug(variant_id)}/{anchor_id:03d}.json`.
Its exact keys are:

```text
schema_version, status, attempt_id, script_version, external_freeze_authorization, execution_index,
recipient_order, recipient_variant_id, anchor_id, failed_or_next_call_role,
d_completed, started_count, completed_count, started_event_bindings,
completed_event_bindings, partial_call_outputs, failure_phase, failure,
source_input_audit_content_binding, same_object_attestation_content_binding,
confirmation_scope_disclosure,
created_at_unix_s
```

Require `status="failed_current"`; `failure` has exactly
`type,message,traceback`; `partial_call_outputs` has exactly the four call-role
keys and uses null for calls that did not return. A returned call is encoded
losslessly, not coerced to JSON floats: its value has exact keys
`status,treedef,leaf_count,leaves`; `status="returned"`; every leaf has exact
`path,dtype_name,byte_order,shape,encoding,data_base64,sha256,size_bytes`.
`treedef` is a recursive JSON AST with exact node keys `kind,metadata,children`:
`kind` is one of `dict,list,tuple,leaf`; dict metadata is the ordered array of
string keys, list/tuple metadata is the non-negative child count, leaf metadata
is null, and `children` is the ordered child-node array (empty only for a
leaf). A leaf `path` is an array of exact tokens, each either
`{"kind":"dict_key","key":<string>}` or
`{"kind":"sequence_index","index":<non-negative integer>}`, and it must be
the unique path produced by that AST.

`dtype_name` is the exact frozen semantic dtype spelling for that leaf (for
example `bfloat16`, `float32`, `int32`, or `bool`), never a platform alias;
`byte_order` is `little` for every multi-byte numeric dtype and
`not_applicable` for one-byte values. Shape is an array of non-negative JSON
integers. Require `encoding="base64_c_order_raw_bytes"`, C-contiguous
little-endian bytes, and RFC 4648 standard-alphabet base64 with mandatory `=`
padding to a length divisible by four. Decoding must yield exactly
`size_bytes=product(shape)*itemsize`, and its SHA-256 must match. The freeze
contains the exact allowed dtype/shape at each leaf path; a generic dtype or
container normalizer is forbidden. This preserves NaN/Inf payload bits without
emitting non-finite JSON numbers. It is structural failure evidence, not a
valid raw or scientific record.

Let `k` be the number of
valid prior raw records and `d` the number of successfully returned calls for
the current record, `d in {0,1,2,3,4}`. Then
`completed_count=4*k+d`. For `d<4`, the next call was attempted and failed, so
`started_count=4*k+d+1`; for `d=4`, all calls returned but record validation or
pre-publication serialization failed, so
`started_count=completed_count=4*k+4`. A setup failure
before dispatch has `d=0`, `started_count=completed_count=4*k`, and
`failed_or_next_call_role="intended"`. A first-call model failure also has
`d=0` but is
distinguished by `started_count=4*k+1` and a started-event binding for
`intended`. No other `d=0` pattern is legal.
`failed_current` is preserved but excluded from the valid raw manifest and all
scientific use.

`failure_phase` is exactly one of `record_setup`, `model_dispatch`,
`record_validation`, or `record_serialization` and is never null.
`failed_or_next_call_role` names the next call for setup, the failed call for
dispatch, and is null for `d=4`. No other nullable/role combination is legal.

The run performs:

- one eight-row lowering/compile attempt and at most one successful compile;
- zero six-row compiles or calls;
- zero separate identity-family reruns (anchor ID 0 remains one of the four
  required OOD anchors, not an identity rerun);
- zero main-cube reruns;
- zero old-OOD record reuse;
- zero confirmation calls; and
- zero extra, warm-up, diagnostic model calls.

The cross-exon donor derangement, row roles, intended and unrelated donor
maps, invariant rows `[0,1,3,5,6,7]`, active recipient rows `[2,4]`, route
masks, sequence hashes, exact repeats, rowwise and whole-trace fingerprints,
target algebra, finiteness, ID-0 no-op checks, ID-255 intended/unrelated
closure, disabled-route checks, and final-seam checks remain byte-for-byte and
semantically identical to the frozen v3.3.3 design. A missing, extra,
duplicated, reordered, non-finite, drifted, or structurally invalid call or
record is a controlled OOD-tooling stop.

The GPU process must not compute donor-normalized recovery, effect size,
Shapley value, interaction, resolution gate, rank, nomination, combined
analysis, or biological interpretation. Every GPU terminal, including a
complete 80/320 terminal, must retain:

```text
scientific_summary_computed = false
donor_normalization_computed = false
shapley_or_nomination_computed = false
interaction_or_resolution_computed = false
nomination_performed = false
combined_analysis_permitted = false
confirmation_model_calls = 0
```

Raw development outputs are append-only evidence for a later CPU structural
audit, not a GPU-side scientific result.

## 6. Append-only terminal and CPU structural audit

There is at most one v3.3.4 model-run attempt. Preflight failure, source-gate
failure, compile failure, partial prefix, or full completion consumes it. No
file or directory may be deleted, replaced, resumed, retried, cherry-picked,
or stitched. Persist compiler artifacts before apply zero, each raw record
before advancing, an exact raw manifest, all import/protobuf/cache provenance,
and a terminal record on every caught failure. Preserve uncaught START-only
states if terminal persistence itself fails.

### 6.1 Launch and phase-state lifecycle

The GPU wrapper accepts no production options; `--dry-run` is its only option
and cannot allocate a production path. It rejects caller-supplied preflight,
checkpoint, output, cache, bounds, resume, overwrite, force, or retry flags.
One non-dry invocation is the sole authorized v3.3.4 launch. The only required
caller inputs are the three post-commit authorization environment values from
Section 4.1; the wrapper rejects a CLI equivalent and rejects any extra
`V334_AUTHORIZED_*` variable.

| Phase | Required durable state | Model applies | Disposition |
|---|---|---:|---|
| Gate A fails before allocation | All six fresh paths remain absent; nonzero stderr is external coordinator evidence only | 0 | Version blocked; no same-version rerun. |
| External preflight allocated, then fails | Exact five-file preflight root and its cache; `preflight_0000.json status=fail`; model cache/run/analysis paths absent | 0 | Version consumed; no rerun. |
| Preflight passes but same-process pre-START validation fails | Passing preflight and preflight cache exist; model run and both analysis paths absent; an allocated model-cache prefix is preserved if allocation preceded failure | 0 | Version consumed; preserve stderr; no rerun. |
| START persists, then Gate B fails | START plus one stdlib-only `POST_START_PROVENANCE_FAILURE.json`, or START only if that publication fails | 0 | Version consumed; no helper/model import and no rerun. |
| Main START exists; later caught failure | Exact append-only run prefix plus RAW_MANIFEST and RUN_COMPLETE | As journaled | Version consumed; CPU structural archive may inspect structure only. |
| Failure prevents RUN_COMPLETE persistence | Preserve exact START/compiler/journal/raw prefix and `TERMINAL_FAILURE.json` if that write succeeds | As journaled | Incomplete terminal; no retry or structural completeness claim. |
| Full pass | Exact 80 valid raws, 320 started and completed calls, manifest, RUN_COMPLETE | 320 | Structurally complete, still no scientific claim. |

No scientific helper, model, checkpoint, or reference import may occur in the
first four rows. START is created only after Gate A, external preflight, and
the same-process no-model validation pass. Gate B then repeats the source/prior
checks after START and before any such import. This paragraph and the sequence
in Section 4 are authoritative; no sentence may be read as moving Gate B
before START. Every caught exception after START writes the applicable
terminal; an uncaught or unwritable terminal is preserved as an incomplete
append-only attempt, never repaired in place.

### 6.2 Exact START and compiler memberships

`ATTEMPT_STARTED.json` has exactly:

```text
status, attempt_id, script_version, amendment_sha256, amendment_commit,
original_protocol_sha256, freeze_path, freeze_sha256, git_head,
external_freeze_authorization,
runner_pid, parent_pid, started_at_unix_s, successful_preflight,
same_process_preflight, same_process_preflight_content_binding,
fresh_paths, budgets, execution_contract, source_inventory_attestation,
prior_v3_3_3_binding, prior_v3_3_3_1_archive_binding,
source_input_audit, source_input_audit_content_binding,
program_signature_contract, cache_isolation_contract,
confirmation_scope_disclosure, confirmation_model_calls,
scientific_summary_computed, donor_normalization_computed,
shapley_or_nomination_computed, interaction_or_resolution_computed,
nomination_performed, combined_analysis_permitted
```

Require `status="attempt_started"`, the committed launch HEAD, exact absolute
fresh paths, `budgets={"max_wall_time_seconds":7200,
"max_output_bytes":1073741824,"expected_records":80,
"expected_model_applies":320,"lower_attempt_budget":1,
"compile_attempt_budget":1,"run_complete_size_cap_bytes":16777216}`, and the
frozen later-exon disclosure. All six
science/nomination/combined booleans are false and confirmation calls are
zero. `successful_preflight` is the exact binding plus tree/PID audit of
Section 4.3. `execution_contract` literally contains the 20 recipients, donor
map, anchor/call order, eight roles, donor/identity maps, invariant/active
rows, one-eight/zero-six restrictions, and exact raw/ledger paths.
`external_freeze_authorization` is the validated runtime object from Section
4.1; `freeze_sha256` and `git_head` must equal its independently recomputed
values.

The disclosure string is exactly: `Later-exon metadata/labels were exposed
after protocol freeze; no later-exon model outputs, activations, or
interventions are used.` It is byte-identical in START, every raw/failed-current
artifact, RUN_COMPLETE, and ANALYSIS.

The START nested key sets are exact. `successful_preflight` has
`artifact_binding,root_file_count,root_file_tree_sha256,external_pid,status,
external_freeze_authorization,external_cache_post_observation,
external_cache_hit_evidence`.
`fresh_paths` has exactly `device_preflight,preflight_kernel_cache,
model_kernel_cache,model_run,analysis_attempt,analysis_output`.
`source_inventory_attestation` has `row_count,rows,tree_sha256,
git_head,tracked_clean,live_equals_head`; its 108 rows use Section 4.1's
schema/order. `execution_contract` has exactly `recipient_cases,donor_order,
recipient_orders,anchor_ids,call_roles,execution_order,record_count,
applies_per_record,expected_model_apply_count,eight_row_roles,
natural_identity_rows,intended_donor_rows,unrelated_donor_rows,
invariant_rows,active_recipient_rows,eight_row_compile_count,
six_row_compile_count,identity_rerun_count,main_cube_rerun_count,
old_ood_records_reused,confirmation_model_calls,raw_path_template,
started_event_path_template,completed_event_path_template,
failed_current_path_template`. The two prior bindings are the exact full
Section 1 and Section 2 immutable objects, not status booleans or abbreviated
hashes. `cache_isolation_contract` is the exact literal machine-freeze object
and `program_signature_contract` is the full three-object mapping plus all
hash/size/type-tag rules.

`POST_START_PROVENANCE_FAILURE.json` has exactly `status,stop_reason,message,
failure,attempt_id,script_version,amendment_sha256,freeze_sha256,git_head,
external_freeze_authorization,runner_pid,source_inventory_failure,model_constructed,model_apply_count,
source_input_audit,source_input_audit_content_binding,
confirmation_model_calls,scientific_summary_computed,
combined_analysis_permitted,failed_at_unix_s`. Require
`status="controlled_stop_post_start_provenance_failure"`,
`stop_reason="post_start_provenance_failure"`, model/apply/confirmation counts
zero, and both booleans false. `failure` is exactly `type,message,traceback`.
The embedded Gate-B audit has the first four primitives independently
rederived as booleans with at least one false, the last four exactly null, and
an exact canonical hash/size binding; copying START's all-true first four is
forbidden.
This two-file state is terminal infrastructure evidence, not a RUN_COMPLETE;
no import/protobuf/compiler/raw/manifest directory or file may exist. A
START-only state is incomplete, never eligible for analyzer completion, and
requires a new prospective version rather than repair.

After Gate B passes, a normal caught terminal whose required provenance files
were all published contains these seven base files at the root: START, all
three import-provenance artifacts, PROTOBUF_PROVENANCE, RAW_MANIFEST, and
RUN_COMPLETE. A required provenance publication failure is the sole exception
and uses the phase-specific replacement contract below. The compiler directory
contains
`PROGRAM_SIGNATURE_ATTESTATION.json` once runtime signatures exist and
`COMPILER_PROVENANCE.json` once lowering is attempted, except that a caught
diagnostic-provenance failure uses the separately named failure artifact and a
terminal-import publication failure has no compiler record under its explicit
four-artifact compiler prefix.
Exact additional
compiler membership is:

- model/checkpoint/reference/setup failure before signatures: no compiler
  directory;
- signature-attestation failure before lowering: the signature-attestation
  failure artifact only;
- lowering failure: no graph text file;
- compile failure after lowering: StableHLO and pre-backend HLO only;
- successful compile, source mismatch, diagnostic-provenance stop, partial
  dispatch, or full completion: StableHLO, pre-backend HLO, and compiled HLO;
  diagnostic-provenance stop has `COMPILER_DIAGNOSTIC_FAILURE.json` instead
  of `COMPILER_PROVENANCE.json`; and
- terminal-import publication failure after compile: signature attestation and
  all three graph texts, with no compiler record.

Thus the strict normal-provenance no-dispatch file counts for the first five
states are respectively 7, 8, 9, 11, and 12; the last special state is the
11-file publication prefix defined below.
Unexpected graphs, a compiled HLO after a compile failure, or a missing
expected graph is fatal. All sealed JSON/graph/ledger/raw files are regular
non-symlink mode `0400` on disk; temporary files must never be visible. Any
allocation lock is `0600`. Committed archive copies, if later authorized, are
`100644`.

A model-cache hit detected after START but before any scientific import has an
exact early special prefix. Publish `MODEL_CACHE_PRE_IMPORT_HIT.json` with
exact keys `status,attempt_id,script_version,external_freeze_authorization,
model_cache_pre_import,cache_hit_evidence,source_input_audit,
source_input_audit_content_binding,failure,model_constructed,
model_apply_count,created_at_unix_s`; require
`status="model_cache_pre_import_hit"`, the exact nonempty/adverse cache binding
and evidence, the START-phase source audit (first four true, last four null),
`failure` exactly `type,message,traceback`, `model_constructed=false`, and zero
applies. Then publish empty RAW_MANIFEST and RUN_COMPLETE with
`status="controlled_stop_cache_hit"`,
`stop_reason="model_cache_pre_import_hit"`. The model-run root contains exactly
four regular files: START, the hit artifact, manifest, and RUN_COMPLETE; no
import/protobuf/compiler/raw/ledger directory or artifact exists,
`import_provenance_phases` is all null, and
`protobuf_provenance_sha256=null`. A cache hit detected before START is instead
a consumed pre-START/preflight state under Section 6.1 and creates no run
terminal.

If atomic publication of a required import/protobuf artifact fails, do not
invent the missing file. If failure occurs before the intended final name is
linked, publish exactly one root `PROVENANCE_PUBLICATION_FAILURE.json` with
keys `status,attempt_id,script_version,external_freeze_authorization,artifact_role,intended_path,
successfully_published_phase_bindings,failure,model_constructed,
lower_attempt_count,compile_attempt_count,successful_compile_count,
model_apply_count,created_at_unix_s`; require
`status="provenance_publication_failure"`, `failure` exactly
`type,message,traceback`, and counts matching the actual prefix. Then publish
an empty RAW_MANIFEST and RUN_COMPLETE with
`status="controlled_stop_provenance_publication_failure"` and
`stop_reason` exactly one of `pre_model_import_publication_failure`,
`protobuf_publication_failure`, `post_model_import_publication_failure`, or
`terminal_import_publication_failure`. Before compiler files, the root file
counts are respectively 4, 5, 6, and 7: START; the successfully published
required phase files in order PRE_MODEL, PROTOBUF, POST_MODEL; the one failure
artifact; RAW_MANIFEST; RUN_COMPLETE. Existing compiler files are added only
for a terminal-import publication failure after compile: exactly the successful
signature attestation and the three graph texts, with no compiler provenance
record because its required terminal-source input does not exist. That state
has 11 regular files total, `compiler_binding=null`, exact four-entry
`compiler_artifact_bindings`, one lower/compile/success, source gate null,
diagnostics null, and no dispatch. No other publication-failure/compiler
combination is legal.

A failure after the intended final name is linked preserves that file but
cannot prove whether the publication protocol completed; it writes
best-effort TERMINAL_FAILURE only and has no RUN_COMPLETE. Failure to publish
the failure artifact, manifest, or RUN_COMPLETE is likewise an incomplete
TERMINAL_FAILURE prefix, never a complete controlled stop.

A successfully published but invalid import/protobuf artifact stops at once;
the runner must not execute the next scientific phase merely to fill later
filenames. Publish exactly one root `PROVENANCE_VALIDATION_FAILURE.json` with
keys `status,attempt_id,script_version,external_freeze_authorization,
artifact_role,artifact_binding,validation_predicates,failure,
model_constructed,lower_attempt_count,compile_attempt_count,
successful_compile_count,model_apply_count,created_at_unix_s`; require
`status="provenance_validation_failure"`, the exact current artifact binding,
literal primitive predicates (not a copied aggregate), and `failure` exactly
`type,message,traceback`. PRE_MODEL import mismatch therefore has exactly five
root files (START, PRE_MODEL, validation failure, empty manifest,
RUN_COMPLETE); protobuf mismatch has six (also PROTOBUF); POST_MODEL mismatch
has seven (START, PRE_MODEL, PROTOBUF, POST_MODEL, validation failure,
manifest, RUN_COMPLETE). A terminal-import mismatch after successful compile
uses the three published import artifacts and protobuf artifact, no separate
validation-failure file, the normal seven root files plus the five successful
compiler files, for 12 total. Missing later provenance phases are null under
the exact prefix rule; no placeholder artifact is allowed.

### 6.3 Exact raw manifest and terminal schema

`RAW_MANIFEST.json` has exactly:

```text
schema_version, status, attempt_id, external_freeze_authorization, valid_artifact_count,
artifact_bindings, artifact_tree_sha256, valid_recipient_anchor_pairs,
failed_current_binding, dispatch_started_count, dispatch_completed_count,
dispatch_started_bindings, dispatch_started_tree_sha256,
dispatch_completed_bindings, dispatch_completed_tree_sha256,
source_input_audit_content_binding, same_object_attestation_content_binding,
created_at_unix_s
```

Every path in these objects is relative to the v3.3.4 model-run root; absolute
paths, `..`, `.`, doubled separators, and backslashes are forbidden.
`artifact_bindings` is a JSON object whose keys are exactly
`raw/ood_anchors/{recipient_order:03d}_{slug}/{anchor_id:03d}.json` for the
valid prefix and whose values have exactly `sha256,size_bytes`.
`dispatch_started_bindings` and `dispatch_completed_bindings` are JSON objects
keyed exactly by `dispatch_journal/started/{global_index:03d}.json` and
`dispatch_journal/completed/{global_index:03d}.json`, respectively, with the
same exact value schema. JSON object insertion order is not trusted: digest
inputs are sorted by their numeric frozen order and must also be
POSIX-lexicographically increasing for these zero-padded paths.
`valid_recipient_anchor_pairs` is an array in execution-index order; each item
has exactly `execution_index,recipient_order,anchor_id`, all JSON integers, and
must satisfy the formulas in Section 5.2. `failed_current_binding` is null or
has exactly `path,sha256,size_bytes`, with the path in the run-root-relative
failed-current grammar from Section 5.3. Each tree uses the UTF-8 bytes of its
run-root-relative POSIX path + NUL + 32 raw SHA-256 bytes; the empty tree is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Only complete valid OOD records are in
`artifact_bindings`; `failed_current_binding` is null or the one separate
artifact. The manifest's `status` is exactly `complete80`,
`controlled_prefix`, or `empty_controlled_stop`.
`same_object_attestation_content_binding` is null for every terminal before a
lowering attempt (import/protobuf/publication/pre-import-cache/model-setup/
signature stops) and otherwise binds the phase-appropriate compiler object.

`RUN_COMPLETE.json` has one common exact key set for every caught terminal:

```text
status, stop_reason, message, failure, attempt_id, script_version,
amendment_sha256, amendment_commit, original_protocol_sha256,
freeze_sha256, git_head, external_freeze_authorization, runner_pid, started_at_unix_s,
completed_at_unix_s, phase_state, terminal_detail, budgets,
source_input_audit, source_input_audit_content_binding, same_object_attestation,
same_object_attestation_content_binding, program_signature_attestation_binding,
source_program_gate, compiler_binding, compiler_artifact_bindings,
attempt_budget_audit, diagnostic_provenance_complete,
compiled_backend_diagnostic_only, backend_diagnostics,
diagnostic_comparisons, dispatch_journal, raw_manifest,
preterminal_tree_binding,
valid_record_count, failed_current_binding, model_apply_attempt_count,
model_apply_success_count, expected_model_apply_count,
eight_row_lower_attempt_count, eight_row_compile_attempt_count,
eight_row_successful_compile_count, six_row_compile_count,
identity_rerun_count, main_cube_rerun_count, old_ood_records_reused,
confirmation_model_calls, all_80_recipient_anchors_complete,
id0_all20, id255_all20, import_provenance_phases,
protobuf_provenance_sha256, model_kernel_cache_final,
prior_v3_3_3_binding, prior_v3_3_3_1_archive_binding,
confirmation_scope_disclosure, scientific_summary_computed,
donor_normalization_computed, shapley_or_nomination_computed,
interaction_or_resolution_computed, nomination_performed,
combined_analysis_permitted, no_retry
```

The common status/phase requirements are:

| Status | `stop_reason` | Compiler state | `k,d` and call accounting |
|---|---|---|---|
| `controlled_stop_import_provenance_failure` | exactly `pre_model_import_inventory_mismatch`, `post_model_import_inventory_mismatch`, or `terminal_import_inventory_mismatch` | no compiler for the first two; terminal mismatch has one successful compiler | `k=0,d=0`, started/completed 0; exactly 5, 7, or 12 files respectively |
| `controlled_stop_protobuf_provenance_failure` | `protobuf_binding_mismatch` | no lower/compile | `k=0,d=0`, started/completed 0; exactly 6 files |
| `controlled_stop_provenance_publication_failure` | one of the four publication reasons above | exact published provenance/compiler prefix from Section 6.2 | `k=0,d=0`, started/completed 0; phase-specific count |
| `controlled_stop_cache_hit` | exactly `model_cache_pre_import_hit` or `model_cache_post_compile_hit` | no scientific import/compiler for pre-import; one successful compiler for post-compile | `k=0,d=0`, started/completed 0; exactly 4 or 12 files |
| `controlled_stop_model_setup_failure` | `model_setup_failure` | no lower or compile; no compiler record | `k=0,d=0`, started/completed 0; 7 files |
| `controlled_stop_signature_attestation_failure` | `signature_attestation_failure` | no lower or compile; attestation preserves failure | `k=0,d=0`, started/completed 0; 8 files |
| `controlled_stop_lower_failure` | `lower_failure` | lower attempts 1; compile attempts/successes 0 | `k=0,d=0`, started/completed 0; 9 files |
| `controlled_stop_compile_failure` | `compile_failure` | lower 1; compile attempts 1; successes 0 | `k=0,d=0`, started/completed 0; 11 files |
| `controlled_stop_attempt_budget_violation` | exactly `second_lower_attempt_forbidden` or `second_compile_attempt_forbidden` | guard fires before forbidden call; observed calls remain at most one; prior lower/full-compile membership | `k=0,d=0`, started/completed 0; 11 or 12 files |
| `controlled_stop_same_object_provenance_failure` | exactly `lowered_object_identity_lost`, `compile_argument_identity_lost`, `compiled_object_identity_lost`, or `apply_callable_identity_lost` | phase-appropriate applicable primitive false; no dispatch | `k=0,d=0`, started/completed 0; exactly 9, 11, or 12 files for lowered-object, compile-argument, or post-compile identity loss |
| `controlled_stop_source_program_mismatch` | `source_program_mismatch` | one successful compile, diagnostics complete, source gate false | `k=0,d=0`, started/completed 0; 12 files |
| `controlled_stop_diagnostic_provenance_failure` | exactly `diagnostic_parser_failure`, `diagnostic_persistence_failure`, `cache_signal_unavailable`, or `fingerprint_formula_mismatch` | one successful compile; diagnostics incomplete; source gate independently recorded | `k=0,d=0`, started/completed 0; 12 files |
| `controlled_stop_partial_dispatch` | `record_setup_failure` or `model_dispatch_failure` | source gate true | `0<=k<80`, `0<=d<4`; ledger arithmetic from 5.3 |
| `controlled_stop_four_call_invalid` | `record_validation_or_serialization_failure` | source gate true | `0<=k<80,d=4`; started=completed=`4k+4` |
| `complete_structural_sidecar` | null | source gate true | `k=80`, no failed-current, started=completed=320 |

The strict whole-run regular-file counts are therefore phase-derived, not
guessed: 7 for a root-only complete-provenance stop, 8 for signature-attestation
failure, 9 for lower failure, 11 for a lowered/precompiled stop, 12 for a
successfully compiled stop, and 732 for full completion (`12 base + 640 ledger
+ 80 raw`). The publication-failure counts are the explicit exception in
Section 6.2; provenance-validation failures have the exact 5/6/7/12 counts
above, and the early model-cache-hit prefix has exactly four. For a
record-setup failure at valid-prefix length `k`, the count is
`13+9k`.
For a failed model call with `d in {0,1,2,3}` successfully completed current
calls it is `14+9k+2d`. For four returned calls followed by invalid validation
or pre-publication serialization it is `21+9k`.
`TERMINAL_FAILURE.json`, if present, adds one and
means none of these complete-terminal memberships applies.

The full-completion directory set is exactly `.`, `compiler`,
`compiler/eight_row`, `dispatch_journal`, `dispatch_journal/started`,
`dispatch_journal/completed`, `raw`, `raw/ood_anchors`, plus the 20 frozen
recipient directories beneath `raw/ood_anchors`, exactly 28 directories.
Controlled prefixes include
only directories required by their exact files; a failed-current state adds
`raw/failed_current` and its one recipient directory. All directories are
mode `0700`; unexpected empty, extra, symlinked, or special directories fail.

`phase_state` has exactly the boolean keys
`preflight_passed,start_persisted,post_start_source_gate_passed,
protobuf_persisted,pre_model_import_inventory_persisted,
model_construction_attempted,model_constructed,
reference_cases_loaded,signatures_captured,
signature_attestation_persisted,post_model_import_inventory_persisted,
lower_attempted,lower_succeeded,compile_attempted,compile_succeeded,
terminal_import_inventory_persisted,source_program_gate_passed,
diagnostic_provenance_passed,dispatch_begun`. Values must follow the status
table and this exact DAG: preflight -> START -> Gate B -> scientific imports ->
PRE_MODEL/protobuf -> model-construction attempt -> POST_MODEL; only a
successful construction continues to signature -> lower -> compile;
terminal-import persistence, diagnostic derivation, and source-program
derivation are then independently evaluated at the applicable phase;
`dispatch_begun` requires both diagnostic and source-program pass. In
particular, `post_model_import_inventory_persisted=true` with
`model_constructed=false` is legal only for a caught construction failure
because it depends on `model_construction_attempted`, not construction success.
`diagnostic_provenance_passed=true` with
`source_program_gate_passed=false` is required for source mismatch and is not a
phase-order violation. No child may be true when its named parent is false.
For the early post-START model-cache hit, exactly
`preflight_passed,start_persisted,post_start_source_gate_passed` are true and
every later phase-state key is false.
`terminal_detail` has exactly `k_valid_records,d_completed,
failed_execution_index,failed_call_role,failure_phase,forbidden_operation,
provenance_artifact_role`; nullable fields must be null for non-applicable
states. `failure_phase` is one of `imports,protobuf,model_setup,signatures,
cache_pre_import,cache_post_compile,lower,compile,post_compile_diagnostics,source_program,record_setup,
model_dispatch,record_validation,record_serialization,terminal_publication`.
`forbidden_operation` is null except the two attempt-budget stops, where it is
`lower` or `compile`; `provenance_artifact_role` is null except provenance
validation/publication failure. `failure` is null only for full completion and otherwise
exactly `type,message,traceback`.
For `controlled_stop_cache_hit`, the mapping is literal:
`model_cache_pre_import_hit -> failure_phase="cache_pre_import"` and
`model_cache_post_compile_hit -> failure_phase="cache_post_compile"`; neither
cache phase is legal for any other stop reason.

The compiler/source nullability is exact:

| Terminal phase | Signature binding | Same-object object/binding | Source gate | Compiler binding | Diagnostics/comparisons |
|---|---|---|---|---|---|
| import/protobuf/pre-import-cache/model setup or precompiler publication failure | null | null | null | null | null/null |
| terminal-import publication failure after compile | success-attestation binding | full in-memory object/binding | null | null | null/null |
| signature attestation failure | failure-artifact binding | null | null | null | null/null |
| lower failure | success-attestation binding | lowering-failure object/binding | null | failure compiler binding | null/null |
| compile failure or second-lower guard | success-attestation binding | phase object/binding | null | failure compiler binding | null/null |
| second-compile guard or applicable same-object failure | success-attestation binding | phase object/binding with the applicable false primitive | null | failure compiler binding | null/null |
| successfully compiled cache/import/provenance stop | success-attestation binding | full object/binding | independently derived object | successful compiler binding | present/present unless the reason itself is diagnostic failure |
| source mismatch | success-attestation binding | full object/binding | present and false | successful compiler binding | present/present and complete |
| diagnostic failure | success-attestation binding | full object/binding | independently derived object | diagnostic-failure binding | null/null in RUN_COMPLETE; failure artifact carries the partial diagnostic audit |
| dispatch prefix or full completion | success-attestation binding | full object/binding | present and true | successful compiler binding | present/present and complete |

`compiler_artifact_bindings` is always a POSIX-run-root-relative-path-sorted
JSON object mapping each allowed compiler file to exact `sha256,size_bytes`;
it is empty where the table says no compiler. `source_input_audit` remains the
phase-appropriate independently derived object and may contain false/null only
as prescribed. No null is converted to false and no unavailable field is
synthesized.
`attempt_budget_audit` is null before any lower request and otherwise exact.
`diagnostic_provenance_complete` is null before successful compile, false only
for diagnostic failure, and true for source mismatch, compiled controlled
prefixes, and full completion. It is null for terminal-import publication
failure because diagnostic derivation was not reached.

`preterminal_tree_binding` is computed immediately before RUN_COMPLETE and has
exact keys `file_count,directory_count,file_bindings,file_tree_sha256,
directory_paths,directory_tree_sha256`. It covers every then-existing run-root
file and directory but necessarily excludes RUN_COMPLETE itself. `file_bindings`
is a JSON object mapping exact run-root-relative POSIX path strings to
`{"sha256":<64 lowercase hex>,"size_bytes":<non-negative integer>}`; keys are
sorted before framing. File-tree framing is UTF-8 path + NUL + raw digest.
Directory paths include `.` and are
sorted; directory-tree framing is `b"D\0" + path + b"\0"`. The offline analyzer
also reports the full immutable tree including RUN_COMPLETE. Both walkers use
`lstat`, reject symlinks/specials and undeclared empty directories, and require
the exact mode and membership for the terminal phase.

The remaining terminal nested schemas are exact. `budgets` has
`max_wall_time_seconds,elapsed_wall_time_seconds,wall_time_within_budget,
max_output_bytes,preterminal_output_bytes,run_complete_size_cap_bytes,
preterminal_plus_terminal_cap_within_budget`; freeze the terminal-record cap at
16,777,216 bytes and require preterminal bytes plus that cap remain within the
1-GiB output budget before serialization. After publication the analyzer also
requires the actual immutable whole-run size within 1 GiB.
`dispatch_journal` has `started_count,completed_count,started_bindings,
completed_bindings,started_tree_sha256,completed_tree_sha256,
started_prefix_exact,completed_prefix_exact`. `raw_manifest` is byte-for-byte
the object in RAW_MANIFEST. `import_provenance_phases` has exactly
`pre_model,post_model_precompile,terminal`. Each value is a file binding for a
normal terminal. For provenance publication or validation failure, all phases
after the failed/invalid artifact are null; the invalid current phase itself
is bound for validation failure and null for publication failure; earlier
values are exact bindings, and `provenance_artifact_role` names the current
phase. No other null pattern is legal. Each import artifact separately contains both the prospective 26-file
inventory and actual loaded-module attestation from Section 4.2.
For the exact early pre-import cache-hit terminal, all three values are null;
this is the only all-null RUN_COMPLETE pattern.
`protobuf_provenance_sha256` is null before successful protobuf artifact
publication (including the early cache hit and PRE_MODEL/protobuf publication
failure) and otherwise is the exact live artifact digest; no empty-string
sentinel is legal.
`model_kernel_cache_final` has `pre_import,historical_stage,
historical_binding,terminal,cache_hit_evidence,
historical_to_terminal_tree_exact,
historical_to_terminal_equality_is_a_gate,
historical_snapshot_not_reauthenticated_as_live_files,
default_user_cache_paths_eligible,cache_outputs_are_diagnostic_only`.
`prior_v3_3_3_binding` and `prior_v3_3_3_1_archive_binding` are the same full
objects independently rederived for START, compiler, and terminal; their
content bindings must match but the derivations remain distinct.

For every controlled outcome, `all_80_recipient_anchors_complete`,
`id0_all20`, and `id255_all20` are false. For full completion they are
recomputed from raw controls and true. Every terminal independently embeds and
hash-binds the exact prerequisite source terms; copied START booleans do not
suffice. All science/normalization/Shapley/interaction/resolution/nomination/
combined fields remain false, `confirmation_model_calls=0`, and
`no_retry=true`.

If terminal writing itself fails, `TERMINAL_FAILURE.json` is the only
additional allowed file and has exactly `status,type,message,traceback,
external_freeze_authorization,valid_record_count,model_apply_attempt_count,model_apply_success_count,
eight_row_compile_count,six_row_compile_count,identity_rerun_count,
main_cube_rerun_count,confirmation_model_calls,created_at_unix_s`. It does not
replace RAW_MANIFEST or RUN_COMPLETE and confers no completion claim.
Require `status="incomplete_terminal_persistence_failure"`; if RUN_COMPLETE
already exists, TERMINAL_FAILURE is forbidden.

The v3.3.4 CPU structural analyzer must be separately versioned, committed in
the audited pre-run bundle, hash-bound in the v3.3.4 freeze, and use the fresh
analysis paths in Section 4. It has at most one append-only invocation after
the model-run terminal is immutable. Before reading any raw development value,
it must independently revalidate the current clean source bundle, Sections
1--2, the complete v3.3.4 run/compiler/preflight/cache trees, START/terminal/
manifest linkage, and confirmation isolation.

The CPU analyzer may validate only structural protocol predicates: exact
prefix/order, apply accounting, repetitions, invariant/self/closure checks,
trace and sequence bindings, finiteness, and complete/controlled-stop state.
It must not compute normalized recovery, Shapley values, interactions,
resolution results, ranks, nominations, or a combined scientific result.
Even after an exact 80/320 pass, emit
`combined_analysis_permitted=false`; a later combined analysis requires a
separate prospective protocol and versioned implementation.

### 6.4 Structural analyzer boundary and append-only schema

The analyzer shell accepts exactly one literal acknowledgement token,
`--acknowledge-structural-only-v3-3-4`, and no path, repair, normalization,
resume, overwrite, force, score, or confirmation argument. Without the token
it exits before creating a path. It exclusively creates the frozen analysis
attempt directory, then writes `ANALYSIS_ATTEMPT_STARTED.json` with exact keys
`status,analysis_version,attempt_id,acknowledgement,git_head,freeze_sha256,
external_freeze_authorization,analyzer_binding,test_binding,run_root,run_terminal_binding,
fresh_output_dir,old_analyzer_destinations_absent,started_at_unix_s`.
Require `status="analysis_attempt_started"` and the exact external freeze
authorization object from Section 4.1.

On success it publishes `RESULT.md` then `ANALYSIS.json` in the fresh output
directory and then `ANALYSIS_COMPLETE.json` in the attempt directory. COMPLETE
has exactly `status,attempt_id,analysis_attempt_start_binding,analysis_binding,
result_binding,output_tree_sha256,run_terminal_binding,completed_at_unix_s`.
Require `status="analysis_complete"`. On failure it creates only
`ANALYSIS_FAILURE.json` beside START, with exact keys
`status,attempt_id,analysis_attempt_start_binding,type,message,traceback,raw_values_read,
scientific_analysis_performed,output_dir_state,failed_at_unix_s`. Paths are
append-only singletons; the analyzer refuses a pre-existing attempt or output.
Attempt/output directories are mode `0700`; every published JSON/Markdown
artifact is mode `0400` and uses the same anonymous-inode publication primitive
as the run.

`output_dir_state` has exactly `state,file_bindings,file_tree_sha256`.
`state` is `absent`, `result_only`, or `result_and_analysis`, matching the only
legal publication prefix if a final TOCTOU or publication check fails. A
failure never deletes a published RESULT/ANALYSIS, never writes COMPLETE, and
binds the exact surviving prefix. No other partial output membership is legal.
Require failure `status="analysis_failure"`,
`scientific_analysis_performed=false`, and an exact
`analysis_attempt_start_binding`.

The public CLI is the only entry point that may create the analyzer START.
After publishing it, the CLI computes the exact
`{"path":<fixed absolute path>,"sha256":<digest>,"size_bytes":<integer>}`
binding and calls the internal raw-reading entry with a mandatory
`active_attempt` object having exactly `attempt_id,start_binding,
acknowledgement`. The internal entry rejects a missing/extra object, a direct
call without the literal acknowledgement token, a different attempt ID, a
non-singleton attempt directory, or a current START hash/size that differs
from `start_binding`, before opening any run/raw value. It revalidates the
active START binding immediately before each raw-record open and at every
final TOCTOU gate. ANALYSIS, COMPLETE, and FAILURE all bind that exact START;
`attempt_id` alone is never sufficient linkage.

Before any raw JSON value is opened, a standard-library phase must rehash the
current tracked-clean analyzer bundle/freeze, all 108 source rows, all prior
Section 1--2 trees, the external preflight/cache, and every v3.3.4 non-raw
run/compiler/terminal/ledger/manifest binding. It first validates the raw path
set and hashes from the manifest without parsing raw JSON. Only then may it
parse development raw records. Confirmation-like paths are rejected by exact
allowlisted path fields and root containment; disclosure text is allowed and
never treated as a path.

`ANALYSIS.json` is structural only. Its exact top-level keys are
`status,decision,analysis_version,analysis_attempt_start_binding,
run_binding,preflight_binding,
model_cache_binding,source_and_prior_audit,compiler_and_signature_audit,
dispatch_journal_audit,raw_prefix_audit,control_audit,terminal_audit,
confirmation_boundary,claim_boundary,scientific_summary_computed,
donor_normalization_computed,shapley_or_nomination_computed,
interaction_or_resolution_computed,nomination_performed,
combined_analysis_permitted,completed_at_unix_s`. No score aggregates,
effect sizes, normalized recoveries, Shapley values, interactions, ranks, or
nominees are legal keys. `RESULT.md` renders only provenance, exact counts,
control pass/failure, structural decision, and the no-science claim boundary.
Even a full pass says `combined_analysis_permitted=false`; only a later
prospective protocol may combine this sidecar with the frozen main cube.

The nested schemas are also literal:

- `run_binding`: `path,file_count,directory_count,file_bindings,
  file_tree_sha256,directory_paths,directory_tree_sha256,terminal_kind,terminal_binding,
  start_binding,strict_membership_exact`;
- `preflight_binding`: `path,file_count,directory_count,directory_paths,
  file_bindings,file_tree_sha256,directory_tree_sha256,
  status,external_pid,runner_pid,pids_distinct,device_exact,cache_role_exact`;
- `model_cache_binding`: `path,pre_import_binding,historical_binding,
  terminal_live_binding,directory_paths_exact,cache_hit,cache_hit_evidence,
  historical_to_terminal_equality_is_a_gate`;
- `source_and_prior_audit`: `current_108_source_rows_exact,
  historical_96_source_rows_exact,git_head_exact,tracked_clean,
  external_freeze_authorization_exact,
  prior_v3_3_3_exact,prior_v3_3_3_1_exact,old_analyzer_paths_absent,
  pre_start_exact,post_start_exact,final_exact`;
- `compiler_and_signature_audit`: `compiler_state,artifact_membership_exact,
  signature_attestation_state,type_tag_paths_exact,semantic_mapping_exact,
  canonical_sha256,canonical_size_bytes,source_input_audit_exact,
  same_object_attestation_exact,stablehlo_exact,pre_backend_exact,
  entry_abi_exact,source_program_exact,compiled_backend_diagnostic_only,
  diagnostic_provenance_complete,compile_counts_exact`;
- `dispatch_journal_audit`: `started_count,completed_count,started_prefix_exact,
  completed_prefix_exact,event_schemas_exact,event_hash_links_exact,
  call_order_exact,pid_exact,publication_membership_exact`;
- `raw_prefix_audit`: `valid_record_count,failed_current_count,
  valid_pairs,expected_next_pair,manifest_exact,raw_paths_exact,
  raw_schemas_exact,failed_current_schema_exact,k,d,
  started_completed_arithmetic_exact,lossless_partial_encoding_exact`;
- `control_audit`: `all_80_complete,id0_all20,id255_all20,
  invariant_rows_exact,repeat_fingerprints_exact,donor_maps_exact,
  sequence_bindings_exact,finiteness_exact,control_state_eligible`;
- `terminal_audit`: `status,stop_reason,phase_state_exact,membership_exact,
  count_arithmetic_exact,budgets_exact,disclosure_exact,no_retry,
  no_forbidden_calls,terminal_linkage_exact`;
- `confirmation_boundary`: `confirmation_paths_opened,
  confirmation_model_calls,later_exon_metadata_label_exposure_disclosed,
  model_outputs_activations_interventions_blind`; and
- `claim_boundary`: `structural_only,no_biological_claim,
  no_scientific_summary,no_normalization,no_shapley,no_interaction,
  no_resolution,no_nomination,combined_analysis_permitted,
  future_protocol_required`.

The output enums are fixed by the immutable model terminal:

| Model terminal | ANALYSIS `status` | ANALYSIS `decision` | `compiler_state` | `terminal_kind` | `control_state_eligible` |
|---|---|---|---|---|---|
| `POST_START_PROVENANCE_FAILURE.json` | `complete_controlled_stop_structural_archive` | `controlled_stop_post_start_provenance_failure` | `not_reached` | `post_start_provenance_failure` | false |
| precompile import/protobuf/provenance-publication/pre-import-cache/model-setup RUN_COMPLETE | `complete_controlled_stop_structural_archive` | exact RUN_COMPLETE `status` | `not_reached` | `run_complete` | false |
| terminal-import mismatch or post-compile cache-hit RUN_COMPLETE | `complete_controlled_stop_structural_archive` | exact RUN_COMPLETE `status` | `compiled_ready_controlled_stop` | `run_complete` | false |
| terminal-import publication-failure RUN_COMPLETE | `complete_controlled_stop_structural_archive` | `controlled_stop_provenance_publication_failure` | `compiled_artifacts_no_gate_record` | `run_complete` | false |
| signature-attestation stop | `complete_controlled_stop_structural_archive` | `controlled_stop_signature_attestation_failure` | `signature_attestation_failed` | `run_complete` | false |
| lower stop | `complete_controlled_stop_structural_archive` | `controlled_stop_lower_failure` | `lower_failed` | `run_complete` | false |
| compile stop | `complete_controlled_stop_structural_archive` | `controlled_stop_compile_failure` | `compile_failed` | `run_complete` | false |
| guarded budget stop | `complete_controlled_stop_structural_archive` | `controlled_stop_attempt_budget_violation` | exactly `second_lower_guarded` or `second_compile_guarded` from reason | `run_complete` | false |
| same-object stop | `complete_controlled_stop_structural_archive` | `controlled_stop_same_object_provenance_failure` | exactly `lowered_identity_failed` or `compiled_identity_failed` from failure phase | `run_complete` | false |
| source mismatch | `complete_controlled_stop_structural_archive` | `controlled_stop_source_program_mismatch` | `compiled_source_mismatch` | `run_complete` | false |
| diagnostic stop | `complete_controlled_stop_structural_archive` | `controlled_stop_diagnostic_provenance_failure` | `compiled_diagnostic_failure` | `run_complete` | false |
| partial/four-call stop | `complete_controlled_stop_structural_archive` | exact RUN_COMPLETE `status` | `compiled_ready_controlled_prefix` | `run_complete` | false |
| full 80/320 terminal | `complete_structural_sidecar_audit` | `structurally_complete_no_scientific_analysis` | `compiled_ready_complete` | `run_complete` | true only if every frozen structural control passes |

For rows whose decision is the exact RUN_COMPLETE status, the only legal
values are `controlled_stop_import_provenance_failure`,
`controlled_stop_protobuf_provenance_failure`,
`controlled_stop_provenance_publication_failure`,
`controlled_stop_cache_hit`, `controlled_stop_model_setup_failure`,
`controlled_stop_partial_dispatch`, or
`controlled_stop_four_call_invalid`. No free-form decision, warning-only
status, or biological pass label is permitted.

Values are strict booleans/integers/nulls according to terminal phase, not
free-form status text. For post-START provenance failure, `run_binding`
accepts only the exact two-file state and `terminal_kind` names that artifact;
all compiler/dispatch/raw counts are zero and control eligibility is false.
For START-only or a malformed terminal, the analyzer writes FAILURE only and
no ANALYSIS/RESULT.

The analyzer is standalone standard-library code: it does not import JAX,
JAXLIB, AlphaGenome, OpenSplice model code, or an older analyzer and does not
monkeypatch any frozen module. Before analysis START it performs all source and
fresh-path gates. Immediately after START it repeats both current and
historical source gates before reading the run. Immediately before publishing
ANALYSIS, and again before publishing COMPLETE, it repeats global clean/HEAD,
all 108 source bytes, run/preflight/cache immutable trees, old-destination
absence, and singleton attempt/output state before RESULT, again before
ANALYSIS, and again before COMPLETE. Any TOCTOU change yields the one
append-only FAILURE and no COMPLETE; it cannot delete a partially published
output, so ANALYSIS publication follows RESULT and COMPLETE follows only after its
exact binding is revalidated.

## 7. Required tests and independent audit

Before any invocation, CPU/synthetic tests must cover:

- literal 82-key freeze schema, exact inherited 96 plus new 12 source rows,
  every SHA/size/mode row, acyclic external freeze authorization,
  freeze-tracked/HEAD-byte/global-clean gates, and
  source-gate ordering before any consumed artifact read or helper import;

- exact v3.3.3 96-row live/historical source gate before every consumed-record
  read and helper import, pre- and post-START;
- exact v3.3.3 run/compiler/preflight/cache memberships, directories, modes,
  sizes, hashes, tree framing, linkage, terminal predicates, and tamper of
  every bound row;
- exact v3.3.3.1 amendment/implementation/archive commits, every source and
  artifact row, both tree digests, linkage, no-science predicates, and
  permanent absence of both original v3.3.3 analyzer destinations;
- exactly three runtime `leaves` tuples and 29 runtime `shape` tuples versus
  the corresponding frozen JSON lists, with direct inequality and canonical
  byte equality;
- exact 32-path type-tag artifact, one canonical serialization of the complete
  three-object mapping, exact 2877-byte payload, full semantic mapping, narrow
  adapter-only locations, and no reliance on post-JSON tuple recovery;
- every one of the eight `source_input_audit` primitives, canonical same-object
  embedding/hash, literal `is`-based source-flow primitives, false/missing/
  copied-object failures, exact START first-four-true/last-four-null and every
  later phase-null transition, exact Gate-B terminal first-four-bool with at
  least one false/last-four-null binding, and all
  START/compiler/raw/manifest/terminal copies;
- independent rejection of every object-name, treedef, per-object leaf count,
  leaf-order, dtype, rank, numeric-shape, container-location, canonical-size,
  canonical-hash, StableHLO, pre-backend, and entry-ABI mutation;
- exact failed-current treedef AST/path-token grammar, semantic dtype and
  little-endian spelling, C-order byte length/hash, RFC 4648 padding, and
  lossless NaN/Inf payload handling without non-finite JSON numbers;
- proof that an arbitrary tuple/list normalization outside the 32 declared
  locations fails;
- proof that changed compiled HLO/fingerprint/backend configurations remain
  diagnostic, while no compiled difference can bypass another source gate or
  trigger a second compile;
- exactly one executable, 20 recipients, anchor order `(0,127,128,255)`, 80
  records, 320 applies, and zero six-row/identity/main-cube/old-OOD/
  confirmation/extra calls;
- exact 20-row development allowlist/alleles/classes/order, exact 20-entry
  cross-exon donor map, and rejection of any reorder, alternate TSV, score-led
  selection, duplicate, omission, or confirmation path;
- every donor-map, row-role, invariant, repeat, trace, route, target, ID-0,
  ID-255, sequence, finiteness, and closure tamper;
- apply-counter persistence before dispatch, every valid and invalid prefix,
  append-only START/terminal behavior, output caps, one-shot refusal, and
  absence of deletion/resume/retry paths;
- runner-shaped lifecycle fixtures for every import/protobuf/publication/cache/
  model/signature/lower/compile/budget/same-object/source/diagnostic status and
  reason, their phase-specific compiler/null/file/count mappings, full 80/320
  completion, and for
  every failed-current `d=0,1,2,3,4`; assert exact started/completed event
  prefixes, `4k+d` arithmetic, setup-vs-dispatch distinction, raw exclusion,
  strict path/membership/mode, and the exact common terminal schema;
- the exact four-file early model-cache-hit prefix, first-four-true/last-four-
  null source audit, all-null import phases, null protobuf binding, and proof
  that no scientific import/helper executes in that fixture;
- exhaustive tamper loops over every START/compiler/signature-attestation/raw/
  failed-current/manifest/RUN_COMPLETE/freeze/preflight/cache field and every
  bound file row; exact raw path/record order, manifest framing, terminal file
  membership for each compiler phase, wall-time/output budgets, disclosures,
  and status/type/null semantics;
- exact compiled-diagnostic parser and executable-fingerprint formula,
  entry-line one-substitution rule, diagnostic-only backend differences, and
  the named diagnostic-provenance stop for absent/unparseable provenance;
- exact external/model cache objects, empty-input and D/F tree framing, all
  cache-hit definitions, non-gating terminal outputs, PID separation and
  same-process main PID, five-file preflight tree, all three prospective
  26-row 22-tracked+4-generated source-file inventories, the independently
  frozen actual loaded-module rows (including local helpers), and protobuf
  phases;
- exact sorted cache `directory_paths`, D/F digests, separately persisted
  external cache-hit evidence, every pre-import/post-compile hit stop, and the
  exact reason-to-`cache_pre_import`/`cache_post_compile` phase matrix;
- exact fresh preflight/output/cache/analysis paths, cache/env sanitation,
  same-process PID and RTX/UUID gates, tracked-clean HEAD, source TOCTOU, and
  symlink/special/extra-path rejection; and
- static and runtime proof that the GPU process computes no scientific
  summary and accesses no confirmation model output, activation, or
  intervention.

The analyzer suite must exercise the exact acknowledgement CLI, fresh
attempt/output one-shot gates, START/COMPLETE/FAILURE schemas, active START
hash/size/token revalidation before every raw open, direct-call rejection,
every terminal-specific status/decision/compiler-state/terminal-kind enum,
non-raw
provenance checks before the first raw-value read, path-like-field confirmation
rejection with legitimate disclosure acceptance, complete and every
controlled prefix, and absence of every forbidden scientific key. The combined
analysis flag is false in every fixture; no test may imply that a structurally
complete sidecar is already eligible for combination under this amendment.

Fixtures may contain only synthetic values or frozen structural provenance.
No real development score may be embedded in a test fixture or read before
all source/provenance gates pass. No confirmation path may be opened.

An independent read-only audit must verify the exact committed bytes and file
modes, machine-freeze inventory, all test suites, shell/static checks, dry-run
80/320 arithmetic, fresh-path absence, cache isolation, tracked-clean HEAD,
and absence of model/GPU output before authorizing the sole launch.

## 8. Stop table and claim boundary

| Outcome | Required disposition |
|---|---|
| Any prior-source/archive/run/preflight/cache provenance mismatch | Stop before import or apply; no retry and no OOD/biological claim. |
| Any signature structure/canonical bytes, StableHLO, pre-backend, ABI, runtime, device, checkpoint, reference, import, or protobuf mismatch | Controlled infrastructure stop before apply zero; no retry and no OOD/biological claim. |
| Import/protobuf drift or required-provenance publication failure | Exact phase-specific controlled/incomplete terminal; no dispatch, retry, or scientific claim. |
| Compile failure, cache hit, loss of same-object provenance, or guarded second lower/compile request | Exact controlled infrastructure stop; preserve phase/compiler/apply-count evidence; actual lower/compile counts never exceed one; no retry. |
| Partial or invalid 80-record prefix | Controlled OOD-tooling stop; preserve exact prefix; no combined or biological claim. |
| Exactly 80 valid records and 320 applies, with every structural gate passing | Structurally complete development-only OOD sidecar; still no GPU-side scientific or biological claim. |
| CPU structural audit passes | Provenance/control execution is eligible for a separately prospective combined analysis; `combined_analysis_permitted` remains false here. |

v3.3.4 can establish only that the already-frozen OOD controls executed under
one prospectively defined source program and one self-consistent eight-row
backend executable. It cannot establish a biological mechanism, validate the
encoder-skip hypothesis, produce a null distribution, rescue or reject a
candidate, nominate a resolution, or validate confirmation data. Compiled
code is not claimed byte-identical to v3.3.3 or v3.3.2.

The unrelated donor remains an out-of-distribution raw-movement warning, not
a biological null, rescue criterion, rejection criterion, or mechanistic
intervention. Shapley and resolution evidence remain only in the frozen v3.3
main cube under its original gates. Every later report must disclose the
v3.3.3 apply-zero tuple/list stop, its v3.3.3.1 no-science structural archive,
the narrow prospective v3.3.4 repair, the absence of confirmation validation,
and any partial/invalid v3.3.4 state. Any further repair or analysis requires
a new prospective version; no post-hoc v3.3.4 exception is permitted.

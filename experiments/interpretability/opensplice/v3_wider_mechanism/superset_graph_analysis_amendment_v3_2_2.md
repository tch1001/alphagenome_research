# OpenSplice v3.2.2 analyzer-only amendment

Status: prospective. This amendment must be committed and hash-bound by a
separately versioned v3.2.2 analyzer before another corrected analysis is
started.

Scientific protocol: unchanged from `superset_graph_protocol_v3_2.md`
(SHA-256
`1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea`).

Preceding analyzer-only amendment: `superset_graph_analysis_amendment_v3_2_1.md`
(SHA-256
`69207fbee072bfacca3af7d21361b6257839b6a0bab3791643f1c4d36bf75ed3`).

## 1. Immutable model run and consumed v3.2.1 attempt

The model run remains complete and byte-for-byte unchanged at:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_2_development_superset_graph_one_shot`

| Artifact | Frozen value |
|---|---|
| Original model-run Git commit | `e3dbc09ee15859299d735279c9e46da6945c4f5e` |
| Original freeze SHA-256 | `526b40899736e1a0442f51bf5b5dae3a2cea89ab4aa680b7c0c6189ce0d8dc4f` |
| Model-run `ATTEMPT_STARTED.json` SHA-256 | `2fb661d35c431ce03f2f62199a32a2ef5a4827294c91a980029c233616e060c8` |
| Model-run `RUN_COMPLETE.json` SHA-256 | `4d59b74584528d6ed8149a36f80e506c37d6ee939cd75b4a22da5bb230fd2425` |
| Model-run `RAW_MANIFEST.json` SHA-256 | `2d63d7dfeaa69e2c1ad8cde731e656e134e37e639023f0745daadb564f17a665` |
| Raw artifact count / tree SHA-256 | `2660` / `4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8` |
| Identity / Phase-R / Stage-A counts | `20 / 2592 / 48` |
| Confirmation model calls | `0` |

The committed v3.2.1 analyzer bundle was created at Git commit
`34c5737855c7c9ebc1c6765f3677e2e53d5fa6f7`. Its analyzer SHA-256 was
`250236f4e1fd4c7712f7862da06e4bf933d731343dd306ae967a03a19e274313`
and its test SHA-256 was
`deb96ebdf4d2d8c961bea87249565d721e046a85b23defefe9932160275f128c`.

The v3.2.1 corrected-analysis attempt is consumed and must never be deleted or
reused:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_2_development_superset_graph_analysis_v3_2_1_attempt`

| Attempt artifact | SHA-256 |
|---|---|
| `ANALYSIS_ATTEMPT_STARTED.json` | `2a77efe48eff892fd3d61b227b486c6820107236461de80a888414bed88ba38b` |
| `ANALYSIS_FAILURE.json` | `174e4bdc56126cab1c9fddd358e06ecaa796d770bbbf526676ac2262dc26311e` |

`ANALYSIS_FAILURE.json` has status `failed_consumed_no_retry`, binds the start
artifact above, and records that neither `ANALYSIS.json` nor `RESULT.md`
exists. The frozen scientific analysis destination remains absent.

## 2. Exact failure boundary

The following traceback is copied verbatim from the append-only
`ANALYSIS_FAILURE.json` artifact:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2_1.py", line 637, in main
    result = analyze(
        args.run_dir, ignored_paths=ignored,
        amendment_binding=amendment_binding,
        attempt_started_sha256=started_sha,
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2_1.py", line 570, in analyze
    result = _v3.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 1921, in analyze
    bootstrap_audit = _validate_bootstrap_attestation(
        start, freeze, freeze_sha, bundle_root=bundle_root
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 1248, in _validate_bootstrap_attestation
    raise ValueError(
        f'Bootstrap artifact {name} differs from frozen protobuf binding.'
    )
ValueError: Bootstrap artifact generated_pyi differs from frozen protobuf binding.
```

The failure occurred in bootstrap-provenance validation before preflight,
raw-manifest validation, or any individual identity, Phase-R, or Stage-A
artifact was read. No target, recovery, `B`, `q`, `Q`, Shapley, ranking, or
decision value was inspected. Only the committed analyzer code, top-level
provenance/failure artifacts, freeze schema, generated-file hashes/sizes, and
filesystem paths were inspected to prepare this amendment.

## 3. Cause

The generated `.pyi` bytes are not missing or different. The live file still
has the exact frozen binding:

| Generated artifact | Size | SHA-256 |
|---|---:|---|
| `src/alphagenome_research/protos/calibration_scores_pb2.py` | `2794` | `4673289dd481fd8c4976f602ab36b07646304107e352e3e6d27b2abe4f9e9ebc` |
| `src/alphagenome_research/protos/calibration_scores_pb2.pyi` | `1815` | `329dc390abeb187084fff28fbe6cb6d9868aa8867326bf53f9a52d4c83f527f9` |

The original bootstrap validator recursively constructs `frozen_paths` only
from mapping nodes that contain an explicit `path` field alongside `sha256`.
The frozen `protobuf_binding.generated_outputs` node instead uses the two
absolute generated-output paths as dictionary keys; each value contains only
`sha256` and `size_bytes`. The `.py` is redundantly present in a separate
`imported_pb2` node with an explicit `path`, so it is collected. The `.pyi`
exists only in `generated_outputs`, so it is omitted from `frozen_paths` and is
then falsely reported as different even though its independently recomputed
size and SHA-256 match exactly.

This is a schema-normalization defect in the offline validator. It is not a
model-run failure, generated-byte mismatch, or scientific-gate failure.

## 4. Sole permitted schema-normalization repair

The original protocol, freeze, v3.2 analyzer, v3.2.1 analyzer, both consumed
attempts, raw tree, thresholds, estimands, and scientific reducers remain
unchanged. A new CPU-only v3.2.2 analyzer may change only bootstrap-provenance
schema normalization as follows:

1. Retain the original recursive collection of explicit
   `{path, sha256, size_bytes?}` nodes.
2. Additionally inspect only the exact node
   `freeze['protobuf_binding']['generated_outputs']`. Require it to be a
   mapping with exactly the two already-frozen absolute keys for
   `calibration_scores_pb2.py` and `calibration_scores_pb2.pyi`.
3. Require each value to have exactly `{sha256, size_bytes}`, with a lowercase
   SHA-256 and non-negative integer size. Normalize each dictionary key into
   the corresponding path binding. Do not interpret keys anywhere else in the
   freeze as paths.
4. Require both normalized paths to remain inside the AlphaGenome Research
   repository, to equal the paths recorded in the historical same-process
   bootstrap artifact set, and to match those artifact entries' exact size and
   SHA-256.
5. Independently hash and stat the two live generated outputs and require the
   same exact size and SHA-256. Preserve the historical disclosures that the
   generator/argv are unknown, the outputs are intentionally untracked exact
   build artifacts, and no byte-regeneration claim is made.
6. If a normalized path collides with an explicit-path record, require its
   size and hash to be identical; any ambiguity, extra output, absent output,
   path escape, malformed schema, or mismatch fails closed.

The repair must not broadly reinterpret arbitrary mapping keys as filesystem
paths and must not modify or regenerate either protobuf output.

Synthetic tests must cover at least: the exact keyed `generated_outputs`
schema, the `.pyi`-only keyed binding, an extra/missing output, malformed key or
value schema, key/path escape, live-byte or size tampering, collision agreement,
collision disagreement, and preservation of the original symlink and
scientific analyzer suites.

## 5. v3.2.2 execution and stop rules

Before any individual scientific raw artifact is read, the v3.2.2 analyzer
must:

- verify all model-run hashes/counts in section 1 and the original raw tree;
- verify the exact v3.2.1 start/failure hashes, their internal linkage, the
  `failed_consumed_no_retry` status, and absence of v3.2.1 scientific outputs;
- verify the original protocol/freeze and the unchanged v3.2/v3.2.1 analyzer
  bundles, amendments, checkpoint/reference bindings, and 12-link snapshot
  audit;
- bind this amendment's SHA-256 plus its own analyzer/test hashes at a globally
  tracked-clean Git HEAD;
- verify the frozen scientific analysis destination and a new, separately
  named v3.2.2 attempt directory are absent; and
- verify that JAX and AlphaGenome model modules are not imported.

After those checks, create exactly one append-only v3.2.2 analysis-attempt
record before delegating to the unchanged scientific analyzer. Persist either
completion or failure. Any v3.2.2 failure consumes that attempt and permits no
retry without another prospective amendment.

The v3.2.2 process is CPU-only. It must not invoke JAX, the AlphaGenome model,
GPU/device preflight, the model launcher, sequence fetching, protobuf
generation, or confirmation data. It must not alter the raw run or either
earlier attempt. The final result, if validation succeeds, must disclose both
analyzer-only failures and identify v3.2.2 as a provenance-schema repair only.
All scientific claims remain limited by the original v3.2 protocol, including
Phase-R layers 0--5 and the disclosed later-exon metadata/label exposure.

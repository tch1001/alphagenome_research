# OpenSplice v3.3.1 analyzer-only amendment

Status: prospective. This amendment, the v3.3.1 analyzer and its tests must be
committed and hash-bound before the corrected CPU-only analysis is started.

Scientific protocol: unchanged from
`encoder_skip_localization_protocol_v3_3.md` (SHA-256
`85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0`).

## 1. Immutable model run and controlled-stop boundary

The one-shot v3.3 GPU run is consumed and must remain byte-for-byte unchanged
at:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_development_encoder_skip_factorial_one_shot`

The original frozen bundle and run bindings observed before this amendment
are:

| Artifact | Frozen value |
|---|---|
| Original model-run Git commit | `9e8324434a3c0b2d5c53d209ad04c2d88b99f4bc` |
| Original protocol SHA-256 | `85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0` |
| Original freeze SHA-256 | `98860ed4e60c427a76ac05879d800f36b65c10a310f4b2b981819fa48af767b3` |
| Original analyzer SHA-256 | `0a65a27a5c424bb9dddacb5475e02d28a0999fc7cd593d4dd63ba4be06c39a46` |
| Original analyzer-test SHA-256 | `d027f73fb07682e8cb54d46653a5bd9aa900aaeac433b1748dbdf4886c6d5034` |
| `ATTEMPT_STARTED.json` SHA-256 | `b74081fd0cbd1c8d6ec5445b3b71661f40ac4d47dd77fd2d9bd3675b4cf9c3c3` |
| `RUN_COMPLETE.json` SHA-256 | `ddc8350361ae9091ac47878a2c2d043897c46ef1d7722a401869d8d69e4be463` |
| `RAW_MANIFEST.json` SHA-256 | `6c50c86153fbce5136ed99205ca4726f87a00ef56216f1205dba5c25d3d27cd7` |
| Raw artifact count / tree SHA-256 | `5142` / `e7376062ce31090b349e88b91bd41700caf4e690511c15993e50f2bd0d47f770` |
| Whole immutable run file count / tree SHA-256 | `5158` / `2d8125fe6d13773ba9621e527870361b6a195c516c5b4f044c7dad64c9310aaa` |
| Compiler file count / tree SHA-256 | `8` / `9a03dcbc9d439cb9bf197941af3bbdb3e6bda067cf661b90de6d7eab1f4d87eb` |
| All three import-provenance SHA-256 values | `64a5538499e5b06e29cb506a2b08585bb002b3766bd1be210d1a568b9ec5110e` |
| `PROTOBUF_PROVENANCE.json` SHA-256 | `2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8` |
| `TARGET_ELIGIBILITY.json` SHA-256 | `b216692d8028faab09b5f6590e3e68d9c8805d3c715ddedd99e8956019cedcf0` |
| Device preflight SHA-256 | `b983c7f4910ef4fc5f68bc72486552063f4497f90bba64497eb29a09d3d1809d` |
| Preflight stdout / stderr SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` / same |

`RUN_COMPLETE.json` is not a successful full-run record. It records the
protocol-defined controlled stop `ood_tooling_failure` with these exact
counts:

| Field | Frozen value |
|---|---:|
| Identities / invalid identities | `20 / 0` |
| Eligible effects | `12` |
| Coalitions / invalid coalitions | `5120 / 0` |
| OOD anchors / invalid OOD anchors | `2 / 1` |
| Scientific records / model applies | `5142 / 10288` |
| Compiled executables | `2` |
| Confirmation model calls | `0` |

All effects were target-eligible, all neutrals were retained, and the ID0 and
ID255 endpoint gates passed for all 20 development variants. The run then
stopped at the first invalid OOD record, as the frozen protocol required. It
did not execute the remaining 78 OOD records and must not be resumed, retried
or described as a complete 5,220-record run. The complete 256-coalition grids
exist for all 20 development variants, but the controlled stop forbids a
Shapley summary, resolution nomination, spatial trigger or mechanistic claim.

Before this amendment, an independent byte-only audit recomputed all 5,142
manifest hashes and the raw tree above, confirmed exact manifest membership
and count/apply arithmetic, and found no `TERMINAL_FAILURE.json`. This is
structural/provenance evidence only. Scientific record values were not opened
or interpreted, and the controlled stop still requires a corrected analyzer
audit.

## 2. Original analyzer failure

The original analyzer created neither the frozen analysis directory

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_development_encoder_skip_factorial_analysis`

nor a scientific `ANALYSIS.json` or `RESULT.md`.

The following stderr was captured by the coordinator from session `90015`.
The failed command did not persist it to disk:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_localization_v3_3.py", line 3836, in <module>
    main()
    ~~~~^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_localization_v3_3.py", line 3820, in main
    result = analyze(args.run_dir)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_localization_v3_3.py", line 3482, in analyze
    freeze, freeze_sha, start_audit = _validate_start(
                                      ~~~~~~~~~~~~~~~^
        run_dir, bundle_root=bundle_root, cases=cases
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_localization_v3_3.py", line 3174, in _validate_start
    raise AnalysisError('Pre-import generated-output path set changed.')
AnalysisError: Pre-import generated-output path set changed.
```

The failure occurred inside `_validate_start`, before `RUN_COMPLETE.json`,
`RAW_MANIFEST.json` or any individual identity, coalition or OOD record was
read. The analyzer had loaded only frozen development metadata and top-level
provenance. No endpoint logit, movement, recovery, Shapley value, interaction,
ranking or nomination was inspected before this amendment. No model, JAX
computation, GPU rerun or confirmation artifact was used.

## 3. Exact cause: seven provenance roles versus two generated outputs

The same-process pre-import bootstrap correctly records exactly seven artifact
roles:

```text
dependency_pb2
dependency_proto
generated_pb2
generated_pyi
source_proto
tensor_pb2
tensor_proto
```

The frozen `protobuf_binding.generated_outputs` mapping correctly contains
only the two artifacts generated for this repository:

```text
/home/degen2/alphafold-stuff/alphagenome_research/src/alphagenome_research/protos/calibration_scores_pb2.py
/home/degen2/alphafold-stuff/alphagenome_research/src/alphagenome_research/protos/calibration_scores_pb2.pyi
```

Their exact frozen bindings are:

| Generated output | Bytes | SHA-256 |
|---|---:|---|
| `calibration_scores_pb2.py` | `2794` | `4673289dd481fd8c4976f602ab36b07646304107e352e3e6d27b2abe4f9e9ebc` |
| `calibration_scores_pb2.pyi` | `1815` | `329dc390abeb187084fff28fbe6cb6d9868aa8867326bf53f9a52d4c83f527f9` |

The original analyzer incorrectly formed the path set from all seven
bootstrap roles and required it to equal the two-path `generated_outputs`
set. The additional five rows are expected source/dependency provenance, not
unexpected generated outputs. The original analyzer's separate protobuf
validator already recursively binds those source and dependency bytes. This
is an offline provenance-set classification defect, not changed protobuf
bytes, a model-run failure or a reason to relax a scientific gate.

## 4. Sole permitted v3.3.1 repair

The original protocol, freeze, v3.3 analyzer/test, GPU run, raw tree,
thresholds, estimands, reducers and controlled-stop decision remain unchanged.
Do not edit the original analyzer because its exact bytes are bound by the
original freeze. A separately versioned, CPU-only v3.3.1 analyzer may change
only the pre-import generated-binding validation as follows:

1. Require the bootstrap `artifacts` mapping to contain exactly the seven
   named roles above. Every row must have exactly `path`, `sha256` and
   `size_bytes`; paths must be canonical, contained in the already frozen
   repository roots, regular non-symlink files, and byte-exact when independently
   hashed and statted.
2. Require `protobuf_binding.generated_outputs` to have exactly the two
   absolute paths and bindings in section 3. Compare only `generated_pb2` and
   `generated_pyi` to those two rows, including role-to-filename, path, size
   and SHA-256. Reject a missing, extra, swapped, duplicated or escaping row.
3. Cross-bind each of the other five roles to its exact named frozen node:
   `source_proto` to `protobuf_binding.source_proto`, `dependency_pb2` to
   `protobuf_binding.dependency_pb2`, `dependency_proto` to
   `protobuf_binding.dependency_proto`, `tensor_pb2` to
   `protobuf_binding.tensor_pb2`, and `tensor_proto` to
   `protobuf_binding.tensor_proto`. Where the frozen named node omits a byte
   size, require the bootstrap size to equal the independently stated live
   regular-file size while path and SHA-256 still equal the frozen node.
4. Preserve and verify the exact two-row
   `generated_artifact_exception`, embedded header, protobuf runtime,
   historical-generator/argv-unknown disclosure and false regeneration claim.
   Do not generate, rewrite or normalize either protobuf output.
5. Do not interpret arbitrary mapping keys as paths, accept an arbitrary
   seven-to-two subset, weaken any existing recursive protobuf validation, or
   modify scientific validation after `_validate_start`.

Tests must cover at least: the exact seven-role/two-output schema; missing,
extra, swapped and duplicated roles; generated-output path escape; source or
dependency role misclassification; path/hash/size tampering for every role
class; malformed value schema; exception/header/runtime/regeneration-claim
tampering; and proof that the unchanged original analyzer suites and
controlled-stop no-Shapley/no-nomination behavior still pass.

## 5. CPU-only execution, append-only attempt and claim limit

Use this fresh attempt directory exactly once:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_development_encoder_skip_factorial_analysis_v3_3_1_attempt`

Before reading an individual scientific raw record, the v3.3.1 wrapper must:

- verify every binding, count and tree in section 1, including all 5,142 raw
  hashes, the eight compiler files, top-level provenance and preflight;
- verify the original protocol/freeze/analyzer/test and the original
  61-file bundle at commit `9e8324434a3c0b2d5c53d209ad04c2d88b99f4bc`;
- verify that the original analyzer failure wrote no analysis output or prior
  repair-attempt artifact;
- bind this amendment's SHA-256 and the new analyzer/test hashes at a globally
  tracked-clean Git HEAD;
- verify that the raw run directory is unchanged, the attempt directory and
  frozen analysis destination are absent, and confirmation-named paths are
  rejected; and
- verify that JAX and AlphaGenome model modules are not imported.

After those prechecks, write one append-only
`ANALYSIS_ATTEMPT_STARTED.json` before delegating to the unchanged scientific
validation logic. The process may then read the immutable development records
solely to audit the frozen controlled-stop prefix. It must write either a
durable `ANALYSIS_COMPLETE.json` or `ANALYSIS_FAILURE.json` in the attempt
directory. Failure consumes the attempt and permits no deletion, overwrite or
retry without another prospective amendment.

The corrected final files, if validation succeeds, remain the original frozen
paths `v3_3_development_encoder_skip_factorial_analysis/ANALYSIS.json` and
`RESULT.md`. The result must be `controlled_stop_ood_tooling_failure`, with no
Shapley decomposition, resolution nomination, spatial trigger or mechanistic
claim. It must disclose that the GPU run used v3.3 and offline provenance
validation used the prospective v3.3.1 seven-role/two-output repair.

The v3.3.1 process must not import JAX/model code, invoke a GPU or device
preflight, fetch sequence, regenerate protobufs, resume or rerun the model,
open confirmation model outputs/activations/interventions, or use later-exon
information. The maximum claim is that the development v3.3 run reached a
provenance-audited OOD tooling controlled stop after completing the frozen
coalition grids. It is not a resolution-localization result, molecular
mechanism, pathway, experimental replication or held-out validation.

# OpenSplice v3.2.3 analyzer-only amendment

Status: prospective. This amendment must be committed and hash-bound by a
separately versioned v3.2.3 analyzer before another corrected analysis is
started.

Scientific protocol: unchanged from `superset_graph_protocol_v3_2.md`
(SHA-256
`1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea`).

Earlier analyzer amendments remain unchanged:

- v3.2.1 SHA-256
  `69207fbee072bfacca3af7d21361b6257839b6a0bab3791643f1c4d36bf75ed3`;
- v3.2.2 SHA-256
  `a25d08c8a609703532a749ac0e5d0246614446627b84b4196f28be91ffdecb4f`.

## 1. Unchanged model run and consumed v3.2.2 attempt

The completed model run remains byte-for-byte unchanged at:

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

The committed v3.2.2 analyzer bundle was created at Git commit
`a3d2d9bdede0eabd47977d8c223e78b0487dbfc9`. Its analyzer SHA-256 was
`874d66b8e9f9e39e8a9e56329f2d04cb219eb7d62be38a0bc038cc8de010e2b4`
and its test SHA-256 was
`15477a6e171a42d24d75fd549960ba831804427209238125745e0158174099b9`.

The v3.2.2 corrected-analysis attempt is consumed and must never be deleted or
reused:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_2_development_superset_graph_analysis_v3_2_2_attempt`

| Attempt artifact | SHA-256 |
|---|---|
| `ANALYSIS_ATTEMPT_STARTED.json` | `11bf87f577d6aafbaa37139b5b98dc2c96c4dec317552564b4ca2bf6db88f117` |
| `ANALYSIS_FAILURE.json` | `f1904bf9593d6f43acad6bd3394e6010404b8e9e2e4b8c43f8b90344554d92a9` |

`ANALYSIS_FAILURE.json` has status `failed_consumed_no_retry`, binds the start
artifact above, and records that neither `ANALYSIS.json` nor `RESULT.md`
exists. The frozen scientific analysis destination remains absent.

## 2. Exact failure boundary

The following traceback is copied verbatim from the append-only v3.2.2
`ANALYSIS_FAILURE.json` artifact:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2_2.py", line 504, in main
    result = analyze(
        run_dir, ignored_paths=ignored, amendment_binding=binding,
        attempt_started_sha256=started_sha,
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2_2.py", line 339, in analyze
    result = _base.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 1924, in analyze
    preflight_audit = _validate_preflight(start, freeze, freeze_sha)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 705, in _validate_preflight
    raise ValueError('Embedded external preflight differs from its artifact.')
ValueError: Embedded external preflight differs from its artifact.
```

The v3.2.2 repairs passed. The failure occurred at the original external-
preflight equality check, before raw-manifest validation or any individual
identity, Phase-R, or Stage-A artifact was read. No target, recovery, `B`, `q`,
`Q`, Shapley, ranking, or decision value was inspected. Inspection for this
amendment was limited to committed analyzer/runner code, top-level provenance
and failure artifacts, the embedded START preflight binding, the saved
preflight JSON and its two log files.

## 3. Exact mismatch

The externally persisted preflight artifact is:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_2_device_preflight/preflight_0000.json`

Its SHA-256 is
`06e0d79f751dc2beb63355a87964f8c3f88d6ae8f843bc1132af5ea6d7ea2b35`.
Both bound log files are empty and have SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

A recursive provenance-only comparison found exactly one schema difference:

- every key and value shared by the saved preflight JSON and embedded
  `START.external_preflight` is identical;
- the embedded record adds `path` and `sha256`, which identify and bind the
  saved artifact as the frozen validator expects; and
- the embedded record additionally adds one runner-derived key,
  `validated_logs`, which is absent from the saved artifact. Its value is
  exactly equal to the saved artifact's existing `logs` mapping.

The model runner deliberately created this augmentation in
`validate_external_preflight`: after verifying both log paths and hashes it
returned `{path, sha256, **record, validated_logs}`. The offline analyzer
removed only `path` and `sha256` before requiring exact equality with the saved
artifact, but did not remove or validate the runner-derived `validated_logs`.
The result is a false inequality despite identical underlying provenance.

This is an offline START-embedding schema-normalization defect. It is not a
device-preflight failure, file/hash mismatch, model-run failure, or scientific-
gate failure.

## 4. Sole permitted preflight-schema repair

The protocol, freeze, model run, saved preflight/log artifacts, v3.2--v3.2.2
analyzers, earlier attempts, raw tree, estimands, and thresholds remain
unchanged. A new CPU-only v3.2.3 analyzer may change only preflight-provenance
schema normalization as follows:

1. Read and hash the exact path named by `START.external_preflight.path` and
   require the embedded `sha256` to equal both the frozen value above and the
   independently recomputed artifact SHA-256.
2. Require the embedded external-preflight key set to equal exactly the saved
   artifact's key set plus `{path, sha256, validated_logs}`. Require every
   saved-artifact key/value to be identical in the embedded record. Reject any
   other missing, extra, or changed field.
3. Require the saved artifact's `logs` mapping to contain exactly `stdout` and
   `stderr`. For each, require exactly `{path, sha256}`, require its resolved
   path to stay inside the frozen preflight directory, reject symlinks and
   non-files, and independently recompute the exact log SHA-256.
4. Reconstruct `validated_logs` only from those newly verified `logs` entries,
   using their resolved paths and exact hashes. Require the reconstructed
   mapping to equal both embedded `validated_logs` and the saved artifact's
   `logs` mapping. Do not trust or interpret `validated_logs` independently.
5. On a deep copy of START, remove only the now-verified runner-derived
   `validated_logs` key. Pass that normalized copy, the original unchanged
   freeze and original freeze hash to the frozen v3.2 preflight validator.
6. Preserve every original device, environment, UUID, compute-capability,
   no-model/no-biological-access, no-JIT, cache, log, path, hash and freeze
   check. Any ambiguity or mismatch fails closed.

The repair must not modify the saved preflight JSON, its logs, START, the
freeze, or any raw/model artifact. It must not delete arbitrary unknown keys or
generalize normalization beyond this exact `validated_logs` augmentation.

Synthetic tests must cover at least: the exact historical augmented schema;
missing or extra embedded fields; a changed shared artifact value; missing,
extra or malformed `validated_logs`; inequality between `validated_logs` and
`logs`; a log path escape/symlink/missing file; current log-byte/hash tampering;
and successful delegation through the frozen preflight validator. All existing
v3.2, v3.2.1 and v3.2.2 analyzer suites remain required.

## 5. v3.2.3 execution and stop rules

Before any individual scientific raw artifact is read, v3.2.3 must:

- verify all unchanged model-run/raw hashes and counts in section 1;
- verify the exact consumed v3.2.1 and v3.2.2 start/failure trees, hashes,
  internal linkage, failure messages, `failed_consumed_no_retry` status and
  absence of scientific outputs;
- verify all earlier amendments/analyzer bundles, checkpoint/reference,
  protobuf, preflight/log and import-provenance bindings;
- bind this amendment's SHA-256 plus its own analyzer/test hashes at a globally
  tracked-clean Git HEAD;
- verify the frozen scientific analysis destination and a new separately named
  v3.2.3 attempt directory are absent; and
- verify that JAX and AlphaGenome model modules are not imported.

After those checks, create exactly one append-only v3.2.3 attempt record before
delegating to the unchanged scientific analyzer. Post-start revalidation must
require the exact singleton start artifact, its SHA-256 and embedded amendment
binding before scientific delegation. Persist either completion or failure;
any failure consumes the attempt and permits no retry without another
prospective amendment.

The v3.2.3 analyzer is CPU-only. It must not invoke JAX, the AlphaGenome model,
GPU/device preflight, model launcher, sequence fetching, protobuf generation,
or confirmation data. It must not alter the raw run, preflight artifacts or any
earlier attempt. A successful result must disclose all analyzer-only failures
and identify v3.2.3 as a START/preflight schema repair only. Scientific claims
remain limited by the original v3.2 protocol, including Phase-R layers 0--5 and
the disclosed later-exon metadata/label exposure.

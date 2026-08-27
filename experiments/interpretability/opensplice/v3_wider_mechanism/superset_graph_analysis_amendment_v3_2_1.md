# OpenSplice v3.2.1 analyzer-only amendment

Status: prospective; this amendment must be committed and hash-bound by the
v3.2.1 analyzer before any corrected analysis is run.

Scientific protocol: unchanged from
`superset_graph_protocol_v3_2.md` (SHA-256
`1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea`).

## 1. Completed raw run and failure boundary

The one-shot v3.2 model run completed before the analyzer was invoked. Its
immutable directory is:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_2_development_superset_graph_one_shot`

The completed-run bindings observed before this amendment are:

| Artifact | Frozen value |
|---|---|
| Source Git commit | `e3dbc09ee15859299d735279c9e46da6945c4f5e` |
| `ATTEMPT_STARTED.json` SHA-256 | `2fb661d35c431ce03f2f62199a32a2ef5a4827294c91a980029c233616e060c8` |
| `RUN_COMPLETE.json` SHA-256 | `4d59b74584528d6ed8149a36f80e506c37d6ee939cd75b4a22da5bb230fd2425` |
| `RAW_MANIFEST.json` SHA-256 | `2d63d7dfeaa69e2c1ad8cde731e656e134e37e639023f0745daadb564f17a665` |
| Raw artifact count | `2660` |
| Raw artifact-tree SHA-256 | `4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8` |
| Identity / Phase-R / Stage-A counts | `20 / 2592 / 48` |
| Invalid Phase-R / Stage-A counts | `0 / 0` |
| Eligible development effects | `12` |
| Confirmation model calls | `0` |
| Original freeze SHA-256 | `526b40899736e1a0442f51bf5b5dae3a2cea89ab4aa680b7c0c6189ce0d8dc4f` |

`RUN_COMPLETE.json` has status `complete`. The failed analyzer created no
`v3_2_development_superset_graph_analysis` directory and wrote no scientific
result.

The following traceback was copied from coordinator-captured stderr; the
failed command did not persist stderr automatically. Repeated abbreviated path
prefixes in the captured copy have been expanded to the same absolute analyzer
path, and the elided multiline call at line 1906 is shown at its source line:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 2607, in <module>
    main()
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 2587, in main
    result = analyze(args.run_dir, ignored_paths=ignored)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 1906, in analyze
    immutable_inputs = _validate_checkpoint_and_reference_inputs(
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 826, in _validate_checkpoint_and_reference_inputs
    raise ValueError(f'Checkpoint file is missing or symlinked: {relative}.')
ValueError: Checkpoint file is missing or symlinked: .gitattributes.
```

No per-variant target, recovery, `B`, `q`, `Q`, Shapley, or ranking value was
inspected before freezing this amendment. Inspection was limited to provenance,
completion/count fields, filesystem shape, hashes, and the validation
traceback.

## 2. Cause and permitted repair

The pinned Hugging Face snapshot uses its normal content-addressed layout: the
12 lexical snapshot files are symbolic links into the model repository's
`blobs/` directory. The original runner followed those links, checked the
dereferenced size and SHA-256 against the exact 12-row manifest, and recorded
the same binding in `ATTEMPT_STARTED.json`. The v3.2.0 analyzer instead rejected
every symlink before performing the already-frozen content check. It stopped at
the first entry, `.gitattributes`.

This is an analyzer input-validation defect. It does not authorize another
model execution, a changed scientific estimand, a changed raw artifact, a
changed threshold, or selection after viewing results.

The only permitted repair is a separately versioned CPU-only v3.2.1 analyzer.
The original v3.2 analyzer, protocol, freeze, run directory, raw manifest, and
all 2,660 raw artifacts remain byte-for-byte unchanged. The v3.2.1 analyzer and
its tests must be committed before execution, must bind this amendment's
SHA-256 in code and output, and must verify a tracked, HEAD-clean repository
before reading any individual scientific raw artifact. Because the original
freeze binds the v3.2.0 analyzer's bytes, the repair must not silently replace
that file and pretend it still satisfies the original freeze.

## 3. Fail-closed checkpoint rule

The manifest path, snapshot name/path, 12 lexical relative paths, sizes,
SHA-256 digests, and `ATTEMPT_STARTED.json` checkpoint binding stay exact. For
each manifest entry the corrected analyzer must apply all of the following:

1. Reject absolute paths, `..`, duplicate entries, malformed sizes/digests,
   missing entries, non-file targets, and lexical paths outside the snapshot.
2. If the lexical entry is not a symlink, require its resolved path to remain
   inside the pinned snapshot.
3. If it is a symlink, allow it only for the exact layout
   `models--google--alphagenome-all-folds/snapshots/<pinned snapshot>`. The
   link must contain exactly `len(relative.parts) + 1` leading `..`
   components, followed by `blobs/<blob-id>`, with no other component. Thus a
   top-level entry uses `../../blobs/<blob-id>`, `d/<entry>` uses
   `../../../blobs/<blob-id>`, and `ocdbt.process_0/d/<entry>` uses
   `../../../../blobs/<blob-id>`. Require the normalized target to be a direct
   child of that model repository's `blobs/` directory, require the blob to be
   a regular non-symlink file, and reject dangling, chained, absolute,
   cross-repository, nested-target, or escaping links. `<blob-id>` must be
   lowercase hexadecimal with the content-addressed length used by the pinned
   cache (`40` or `64`).
4. In both cases, follow the validated entry and independently recompute its
   regular-file size and SHA-256. Both must exactly equal the frozen manifest
   row and the recorded attempt binding.
5. Require exactly the 12 frozen lexical file entries and no extra file,
   symlink, socket, device, or other unrecognized tree entry. Ordinary
   directories may exist only as parents of the 12 frozen entries.

Tests must cover at least: a valid real-file entry, a valid one-hop link into
the exact model `blobs/` directory, an absolute/escaping or cross-repository
link, a chained link, a dangling link, a blob-byte/size mutation, and an extra
tree entry. The existing checkpoint/reference, raw-tree, confirmation-path,
and scientific-gate tests remain required.

## 4. Analyzer-only execution and stop rules

Before corrected analysis, the v3.2.1 analyzer must fail closed unless:

- the four raw-run hashes and artifact-tree count/hash in section 1 match;
- `RUN_COMPLETE.json` is still `complete` with the exact recorded counts and
  zero confirmation calls;
- the original protocol, freeze, original 57-file bundle, checkpoint manifest,
  reference binding, preflight, compiler provenance, import inventories, and
  generated-binding provenance pass their existing checks;
- this amendment and the new analyzer/tests are tracked at a clean Git HEAD,
  and their paths and SHA-256 digests are persisted in the analysis result;
- the frozen analysis destination is still absent; and
- neither JAX nor AlphaGenome model code is imported by the analyzer process.

Exactly one CPU-only corrected analysis may then read the completed development
raw tree and write the previously empty append-only analysis destination. It
must not invoke the model, GPU launcher, device preflight, sequence fetch, or
confirmation data. A repaired analysis failure is persisted outside the raw
run and does not permit deleting, overwriting, retrying, or cherry-picking a
partial scientific result without another prospective amendment.

The corrected result may claim only what the unchanged v3.2 protocol permits.
It must disclose that the model run used v3.2.0 and that offline interpretation
used the prospective v3.2.1 symlink-validation repair. Phase-R remains limited
to layers 0--5; later-exon metadata/labels were exposed, but confirmation model
outputs, activations, and interventions remain unopened.

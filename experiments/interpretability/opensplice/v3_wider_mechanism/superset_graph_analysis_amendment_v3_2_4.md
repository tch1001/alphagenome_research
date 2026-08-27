# OpenSplice v3.2.4 analyzer-only amendment

Status: prospective. This amendment must be committed and hash-bound by a
separately versioned v3.2.4 analyzer before another corrected analysis is
started.

Scientific protocol: unchanged from `superset_graph_protocol_v3_2.md`
(SHA-256
`1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea`).

Earlier analyzer amendments v3.2.1--v3.2.3 remain unchanged. Their SHA-256
digests are, respectively:

- `69207fbee072bfacca3af7d21361b6257839b6a0bab3791643f1c4d36bf75ed3`;
- `a25d08c8a609703532a749ac0e5d0246614446627b84b4196f28be91ffdecb4f`;
- `e0c18b53dfdce93be443c84c178766bbb744ce7b017463b83ca449833f91e95e`.

## 1. Unchanged model run and consumed v3.2.3 attempt

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

The committed v3.2.3 analyzer bundle was created at Git commit
`5ccbe55c9c121fa997eae0d7ce5111c650abe4d1`. Its analyzer SHA-256 was
`9f84b094aa064b3dc789e72f53e6d028e346eecfe6dcfc8a0c7febc7fcc1397c`
and its test SHA-256 was
`9c8a937791f9ae966dfd45be5384989917912a1714fb1b147f8e7a5268cf1bd8`.

The v3.2.3 corrected-analysis attempt is consumed and must never be deleted or
reused:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_2_development_superset_graph_analysis_v3_2_3_attempt`

| Attempt artifact | SHA-256 |
|---|---|
| `ANALYSIS_ATTEMPT_STARTED.json` | `00dbe3749a1a810e1bf7de02d1eeecf02a9a363ee196709d8c7c76283b778325` |
| `ANALYSIS_FAILURE.json` | `70b4521dfbcfbc0702bd3e8c3039c45f1a6d8f66081d2c51e5181d4fa2320871` |

`ANALYSIS_FAILURE.json` has status `failed_consumed_no_retry`, binds the start
artifact above, and records that neither `ANALYSIS.json` nor `RESULT.md`
exists. The frozen scientific analysis destination remains absent.

## 2. Exact failure boundary

The following traceback is copied verbatim from the append-only v3.2.3
`ANALYSIS_FAILURE.json` artifact:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2_3.py", line 464, in main
    result = analyze(
        args.run_dir, ignored_paths=ignored, amendment_binding=binding,
        attempt_started_sha256=started_sha,
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2_3.py", line 306, in analyze
    result = _base.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 1962, in analyze
    import_audit = _validate_import_phases(
        run_dir, complete, bundle_root=bundle_root
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 1001, in _validate_import_phases
    phases[name] = _validate_import_provenance(
                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        run_dir / filename, bindings[name], bundle_root=bundle_root
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py", line 958, in _validate_import_provenance
    raise ValueError('IMPORT_PROVENANCE contains duplicate module/path rows.')
ValueError: IMPORT_PROVENANCE contains duplicate module/path rows.
```

The v3.2.1--v3.2.3 repairs and the original checkpoint, protobuf, preflight,
raw-manifest hash tree, RUN_COMPLETE and compiler-provenance checks passed. The
raw files were read only as bytes for the frozen manifest hash-tree audit. The
failure occurred before any identity, Phase-R or Stage-A JSON was parsed and
before any target, recovery, `B`, `q`, `Q`, Shapley, ranking or decision value
was inspected.

Inspection for this amendment was limited to committed analyzer/runner/launcher
code, top-level provenance/failure artifacts and the three bound import-
provenance JSON files. No model was imported or run by this audit.

## 3. Exact alias pattern

The three files are:

- `IMPORT_PROVENANCE_PRE_MODEL.json`;
- `IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json`;
- `IMPORT_PROVENANCE.json`.

All three are byte-identical and `24700` bytes, each has SHA-256
`b542e76b7db0cbe74a5322af9c6e647a0dfd5bea931657af746c41560949dd7d`,
and `RUN_COMPLETE.json` binds that digest for all three phases. Each contains
exactly 74 rows, sorted by unique module name, with the exact row schema
`{name, path, root, sha256, size_bytes}`.

There are no duplicate module names and no exact duplicate rows. There is
exactly one duplicated resolved path, represented by exactly two rows:

| Name | Path | Root | Size | SHA-256 |
|---|---|---|---:|---|
| `__main__` | `experiments/interpretability/opensplice/run_superset_graph_v3_2.py` | `alphagenome_research_checkout` | `68642` | `d3d3335ee47fcd477f25fd8dbf13f515a6ec909b7054e35c7209acac45f2f1eb` |
| `__mp_main__` | same | same | `68642` | same |

The absolute path of both rows is
`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/run_superset_graph_v3_2.py`.
The other 72 rows have unique resolved paths.

The launcher executes that runner with
`runpy.run_path(..., run_name='__main__')`. The runner inventories loaded
modules by iterating `sys.modules` names without path deduplication. Python's
multiprocessing main-module alias therefore records the same executed runner
under both `__main__` and `__mp_main__`. Both rows bind the identical live file,
root, size and hash; they are two names for one source file, not conflicting
module bytes.

The original analyzer correctly rejected arbitrary duplicate names or paths,
but did not distinguish this exact byte-identical standard alias pair. This is
an import-provenance alias-normalization defect, not an import mutation, code-
byte mismatch, model-run failure or scientific-gate failure.

## 4. Sole permitted import-alias repair

The protocol, freeze, model run, import-provenance files, earlier analyzers and
attempts, raw tree, estimands and thresholds remain unchanged. A CPU-only
v3.2.4 analyzer may change only import-provenance path-uniqueness handling:

1. Require all three exact filenames and the exact SHA-256 above, require their
   `RUN_COMPLETE` phase bindings, and require the three JSON objects to be
   byte-identical.
2. Require exactly `{module_count, modules}`, `module_count == 74`, exactly 74
   rows sorted by name, and exactly
   `{name, path, root, sha256, size_bytes}` for every row.
3. Preserve the original name, declared-root containment, live file SHA-256,
   live size, required-module, runner-presence and cross-phase consistency
   checks for every row. Reject symlinked or non-regular live module files.
4. Require all 74 names to be unique. Group rows by resolved path. Require
   exactly 73 path groups: 72 singleton groups and one two-row group. The sole
   two-row group must have names exactly `{__main__, __mp_main__}` and the exact
   runner path, root, size and SHA-256 in section 3. The two rows must differ
   only in `name`.
5. Treat that exact pair as one path identity solely for the path-uniqueness
   check while preserving both original rows and names in the audit output.
   Reject any additional alias, third row, duplicate name, alternate path,
   altered root/hash/size, or non-identical collision.
6. Preserve the original phase comparison by module name. Because the three
   inventories are exact and byte-identical, require no missing/changed module
   and no lazy addition between phases.

The repair must not delete or rewrite either alias row, alter any provenance
artifact, or generalize acceptance to other names or byte-identical paths.

Synthetic tests must cover at least: the exact two-name runner alias; duplicate
name; unrelated duplicate path; third alias; wrong alias name/path/root/hash/
size; differing live bytes or size; symlink/non-file; missing/extra/unsorted
row; changed phase artifact or binding; and successful validation of all three
real provenance phases without parsing scientific raw records. All existing
v3.2--v3.2.3 analyzer suites remain required.

## 5. v3.2.4 execution and stop rules

Before any individual scientific raw artifact is parsed, v3.2.4 must:

- verify the unchanged model-run/raw hashes and counts in section 1;
- verify the exact consumed v3.2.1--v3.2.3 start/failure trees, hashes,
  internal linkage, failure messages, `failed_consumed_no_retry` status and
  absence of scientific outputs;
- verify every earlier amendment/analyzer bundle and the checkpoint,
  reference, protobuf, preflight/log, compiler and import-provenance bindings;
- bind this amendment's SHA-256 plus its own analyzer/test hashes at a globally
  tracked-clean Git HEAD;
- verify the frozen scientific analysis destination and a new separately named
  v3.2.4 attempt directory are absent; and
- verify that JAX and AlphaGenome model modules are not imported.

After those checks, create exactly one append-only v3.2.4 attempt record before
delegating to the unchanged scientific analyzer. Post-start revalidation must
require the exact singleton start artifact, its SHA-256 and embedded amendment
binding. Persist either completion or failure; any failure consumes the attempt
and permits no retry without another prospective amendment.

The v3.2.4 analyzer is CPU-only. It must not invoke JAX, the AlphaGenome model,
GPU/device preflight, model launcher, sequence fetching, protobuf generation or
confirmation data. It must not alter the raw run, provenance artifacts or any
earlier attempt. A successful result must disclose all analyzer-only failures
and identify v3.2.4 as an exact import-alias validation repair only. Scientific
claims remain limited by the original v3.2 protocol, including Phase-R layers
0--5 and the disclosed later-exon metadata/label exposure.

# OpenSplice v3.3.2.2 saved-validator analyzer amendment

Status: **prospective, docs-only, and CPU-only**. This amendment authorizes no
model, JAX, GPU, preflight, scientific-score, activation, intervention, or
confirmation access. A new analyzer, tests, freeze, and append-only wrapper
must be committed and hash-bound before one CPU-only invocation.

The only permitted result is the structural archive already specified by
`encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md`, SHA-256
`81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba`:
`controlled_stop_compiler_graph_mismatch`, zero model applies, zero raw
records, and no scientific summary, Shapley value, interaction, resolution
analysis, nomination, or combined analysis. This document changes only how
the frozen original validator function is called while its module attribute is
temporarily patched.

## 1. Consumed v3.3.2.1 analyzer attempt

The v3.3.2.1 CPU-analyzer attempt is consumed and immutable at:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1_attempt`

It contains exactly two regular non-symlink files and no directories below
the attempt root:

| Relative path | Size (bytes) | SHA-256 |
|---|---:|---|
| `ANALYSIS_ATTEMPT_STARTED.json` | 8616 | `a87c4e15ed67a363d07c434ca232540687950d145e67492b9ed9c17d9adebf1d` |
| `ANALYSIS_FAILURE.json` | 2163 | `1cd933623ecdfb328d5db458b16df909e632a560361dbb547f83c22cf13ab7c7` |

The two-file tree SHA-256 is
`5e97b191e781c5141d2f308deefacfa8f6a196449fd7e11f36c22828a13f036a`.
Use the v3.3 framing: sort UTF-8 POSIX paths relative to the attempt root and,
for each file, append the path, one NUL, and 32 raw SHA-256 bytes. Reject any
extra, missing, symlinked, special, or empty-directory entry.

The bound v3.3.2.1 implementation was committed at
`b43051aa4a893e24a38e932900d349278c9ead88` with these exact bytes:

| Artifact | SHA-256 |
|---|---|
| v3.3.2.1 analyzer | `35db9ca198cb5d7f03621ccf322ea116f98cea3bfdc711006dfd20bc809e8048` |
| v3.3.2.1 analyzer test | `a8733f3ffb35920dda2f6a856076cbe82a9e90aa2fe483169150dbad4421a1b8` |
| v3.3.2.1 freeze | `3871ab41b16105a94673e89381d32d7253b014c64eda5c6789eaecf16477c061` |
| v3.3.2.1 shell wrapper | `ea5cce6ae631ba3fa2bf0082d691d0896ef0fed7b20f0d908034e17775060caa` |
| v3.3.2.1 amendment | `81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba` |

`ANALYSIS_ATTEMPT_STARTED.json` binds that commit and freeze, the exact
v3.3.2 11-file run tree
`4ac66e45a4d7d65af2785904d11b23bf7e809e07f3f617e190772242b2e7a4ab`,
the four-file compiler tree
`4378048568ff58a2bbee55ba9da750498b89fdef72c97911815cf895c8a8b7d1`,
the five-file preflight tree
`797211382478ba249fe94e7ccbcc11c7192d30423d5e289ce03cf3cea37f65f5`,
all 75 frozen model-run source bytes, zero model applies, no model rerun, no
scientific-value read, and zero confirmation calls. Revalidate the complete
literal START object; these summary digests do not replace it.

The following destinations are absent and must remain absent:

```text
results/v3_3_2_development_ood_sidecar_analysis
results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1
```

The v3.3.2.1 attempt may not be deleted, resumed, overwritten, retried, or
given a later terminal/output file.

## 2. Exact persisted failure

`ANALYSIS_FAILURE.json` records:

- `status=failed_consumed_no_retry`;
- failure type `RecursionError`;
- exact message `maximum recursion depth exceeded`;
- the exact START SHA-256 above;
- `analysis_dir_exists=false`;
- `model_apply_count=0`;
- `scientific_summary_computed=false`;
- `shapley_or_nomination_computed=false`; and
- `combined_analysis_permitted=false`.

The persisted traceback is:

```text
Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1.py", line 899, in main
    result = analyze(
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1.py", line 769, in analyze
    result = _v332.analyze(run_dir, bundle_root=bundle_root)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2.py", line 1858, in analyze
    _validate_freeze_and_start(run_dir, bundle_root=bundle_root)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1.py", line 536, in _validate_freeze_and_start_repaired
    _v332._validate_freeze_and_start(  # pylint: disable=protected-access
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1.py", line 536, in _validate_freeze_and_start_repaired
    _v332._validate_freeze_and_start(  # pylint: disable=protected-access
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1.py", line 536, in _validate_freeze_and_start_repaired
    _v332._validate_freeze_and_start(  # pylint: disable=protected-access
  [Previous line repeated 993 more times]
RecursionError: maximum recursion depth exceeded
```

## 3. Failure boundary and no-science audit

The v3.3.2.1 `main()` completed its standard-library preconditions, persisted
START, and entered `analyze()`. The second precondition pass again rehashed
the committed analyzer bundle, frozen v3.3.2 source/run/compiler/preflight
trees, and append-only START. It then installed the scoped repaired validator
on the frozen analyzer module and called the frozen analyzer's `analyze()`.

The frozen analyzer immediately resolved its validator through the patched
module global. The repaired validator then tried to call that same patched
module global and recursed. No invocation of the saved frozen validator body
occurred after START. Therefore the process did not reach RUN_COMPLETE parsing,
compiler semantic validation, raw-manifest parsing, record iteration, result
construction, or output writing.

The preconditions hashed immutable files as opaque bytes and read structural
provenance JSON. The consumed v3.3.2 model run has no `raw` directory and its
raw manifest is the exact empty manifest. No endpoint, movement, recovery,
activation, intervention, Shapley, interaction, resolution, rank, or
nomination value was parsed or computed. The analyzer's CPU-only import gate
remained active; JAX, JAXlib, AlphaGenome, and AlphaGenome model modules were
not imported. No model/GPU/preflight process was run and confirmation model
outputs, activations, and interventions remained unopened.

This is a Python function-reference/control-flow defect only. It does not
change or question any v3.3.2 artifact, compiler stop, device provenance, or
scientific claim boundary.

## 4. Sole permitted v3.3.2.2 repair

Create a new versioned analyzer, tests, freeze, and shell wrapper. Do not edit
the frozen v3.3.2 or v3.3.2.1 analyzer bundles or either consumed attempt.
Before its sole invocation, commit and independently hash-audit every new
file at a globally tracked-clean HEAD. The new freeze must itself be tracked
and byte-exact to `git show HEAD:<freeze>`.

Use fresh append-only destinations:

```text
results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_2_attempt
results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_2
```

The only code-path repair is to save the frozen original validator function
before installing the temporary patch, and to call that saved object inside
the repaired validator. In schematic form:

```python
saved_frozen_validator = frozen_module._validate_freeze_and_start

def repaired_validator(run_dir, *, bundle_root):
    # Under the already frozen historical-HEAD shim, invoke the saved object,
    # never the mutable module attribute.
    try:
        saved_frozen_validator(run_dir, bundle_root=bundle_root)
    except frozen_module.AnalysisError as error:
        if str(error) != exact_known_schema_error:
            raise
    # Validate the literal same-process and external schemas and reconstruct
    # the frozen return object exactly as prospectively specified in v3.3.2.1.

@contextmanager
def scoped_repair():
    require frozen_module._validate_freeze_and_start is saved_frozen_validator
    frozen_module._validate_freeze_and_start = repaired_validator
    try:
        yield
    finally:
        frozen_module._validate_freeze_and_start = saved_frozen_validator
```

The saved reference must be captured once, immediately after exact-byte import
of the frozen v3.3.2 analyzer and before any monkeypatch. Fail closed if the
module attribute is not that exact saved object at context entry or if it is
not restored at exit. The repaired validator must never call
`frozen_module._validate_freeze_and_start` while the patch is active. Do not
raise the recursion limit, catch `RecursionError` as success, normalize START,
synthesize external fields, skip the known frozen failure, or copy a result
without completing every remaining frozen structural validator.

All other v3.3.2.1 behavior remains exact:

- current tracked-clean HEAD plus exact current/HEAD bytes for the new bundle;
- all 75 live v3.3.2 source bytes equal their model-run-commit bytes before
  replaying only the historical `rev-parse HEAD` response into the frozen
  model-run attestation;
- exact literal 15-key same-process versus 18-key external-preflight schemas,
  PID 2327369, RTX 3090/UUID/runtime/package/environment checks;
- complete v3.3, v3.3.1, v3.3.2, preflight, protobuf, import, checkpoint,
  reference, compiler, START, RUN_COMPLETE, and empty-raw-tree validation;
- standard-library/CPU-only execution and no JAX/model imports;
- no scientific raw parsing before every provenance and terminal predicate
  passes; and
- exact zero-apply controlled-stop/no-science result enforcement.

The v3.3.2.2 START must bind the complete two-file v3.3.2.1 failure attempt,
its exact traceback and output absence, the v3.3.2.1 committed bundle, this
amendment, and every previously frozen artifact. Persist either a terminal
complete artifact or a terminal failure artifact. Any failure consumes
v3.3.2.2; never delete, overwrite, resume, or retry it.

## 5. Required tests

In addition to all frozen v3.3.2 and v3.3.2.1 tests, CPU-only tests must prove:

1. the saved validator reference is captured before patching and differs from
   the repaired function;
2. inside the scoped context, the module attribute is the repaired function
   while the repaired function calls the saved original exactly once;
3. the known same-process schema error alone is repaired and every other
   frozen error propagates unchanged;
4. context exit restores the exact saved object after success and every
   exception;
5. a deliberately low recursion limit is not reached and no recursive call
   to the repaired function occurs;
6. direct or concurrent pre-patching of the module attribute fails closed;
7. the exact v3.3.2.1 START/FAILURE hashes, two-file tree, traceback, zero
   counts, and absent output paths are mandatory;
8. extra, missing, symlinked, special, mutated, or later-added v3.3.2.1
   attempt/output artifacts fail;
9. the v3.3.2.2 freeze is tracked and exact to its HEAD blob; and
10. the only accepted delegated result is
    `controlled_stop_compiler_graph_mismatch` with zero records/applies,
    no scientific payload, and `combined_analysis_permitted=false`.

No fixture may contain a real model score.

## 6. Claim and downstream boundary

A successful v3.3.2.2 run only archives the already known zero-apply compiler
stop. It is not evidence for an encoder-skip mechanism, donor specificity,
splicing biology, or model validity. It does not make the incomplete v3.3.2
OOD sidecar scientifically usable.

The prospective v3.3.3 compiler-gate amendment may not be implemented or run
until v3.3.2.2 successfully completes, its attempt/output files and tree
digests are frozen, and the v3.3.3 freeze is updated prospectively to bind
v3.3.2.2 rather than expecting v3.3.2.1 completion. That later change cannot
alter the v3.3.3 cohort, order, graph/ABI gate, one-compile/80-record/320-apply
design, controls, or confirmation boundary.

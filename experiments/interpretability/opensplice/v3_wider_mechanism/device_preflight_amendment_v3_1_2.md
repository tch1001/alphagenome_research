# Device-preflight amendment v3.1.2

**Design date:** 2026-08-28

**Status:** prospective infrastructure-only amendment; commit this document,
the v3.1.2 runner/wrapper/tests and their freeze manifest before any v3.1.2
device preflight or model access

**Scope:** exact-source Phase-R identity diagnostic on the same 20 development
rows; no active intervention and no confirmation access

## 1. Frozen disposition of v3.1.1

The v3.1.1 bundle was committed as
`c95c7c284487227fc6bcc2d1ae05a00088e37b17` before its process started. The
append-only attempt directory contains only `ATTEMPT_STARTED.json`:

| Artifact | SHA-256 |
|---|---|
| `ATTEMPT_STARTED.json` | `1c1d24219e49e089806a02956bf8d7a44bdbd103e28adad6217eb3f94b424587` |
| one-file artifact tree | `a310ecddcace66bc9362e18249b08506c8e84ed7217a594a17e464ffb130d510` |

The start record captured:

- `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/lib:`;
- an available RTX 3090 in `nvidia-smi`, but only one CPU device in JAX;
- JAX/JAX CUDA plugin 0.11.0 with bundled CUDA 12.9 packages; and
- the exact v3.1.1 source, protocol, protobuf and import provenance.

The inherited library path shadowed the CUDA libraries selected by the JAX
environment. The console reported a cuSPARSE initialization failure followed
by CPU fallback. The process was interrupted during metadata loading. It wrote
no post-model import record, compiler artifact, identity artifact, target,
activation, comparison, active intervention or confirmation result.

v3.1.1 is therefore a **partial infrastructure failure**. It is neither a
Gate-0 pass nor a numerical/biological negative. Its directory must not be
deleted, overwritten, completed, resumed or retried. A later run must not use
the v3.1.1 attempt ID or output path.

## 2. Why v3.1.2 is allowed

A separately versioned v3.1.2 attempt is methodologically allowed because the
only observed information was device/runtime health recorded before model
construction completed. No AlphaGenome output or internal value was available
to change a threshold, variant, target, direction, compiler choice or
mechanistic hypothesis. Sanitizing the shared-library environment and adding a
fail-closed device gate repairs an infrastructure omission already handled by
the successful OpenSplice and TAL1 wrappers; it does not adapt to model
performance.

v3.1.2 retains without change:

- pristine exact source commit
  `fd4dc6913335a6966420d60ef04bc4643b751a27`;
- checkpoint, manifest, exon, sequence, endpoint and position-set bindings;
- the standalone-protoc build artifact and complete import inventory;
- 16,384-bp dense attention, six-row mapping and mixed-precision policy;
- all 20 development identities in frozen order, two calls per identity;
- the inclusive `2^-8` identity threshold and all exact/direction gates;
- zero active Phase-R groups, zero Stage-A calls and zero confirmation access;
  and
- append-only per-case persistence, continuation after ordinary numerical
  failures and unconditional stop after the identity cohort.

No v3.1.1 CPU value may be used as a baseline, and v3.1.2 cannot retroactively
turn v3.1.1 into a pass.

## 3. Frozen environment correction

Before **any** Python/JAX process is launched, the v3.1.2 shell wrapper must:

```text
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

`LD_LIBRARY_PATH` must be absent from `os.environ`, not present with an empty
value. `XLA_PYTHON_CLIENT_PREALLOCATE` must equal the literal lowercase string
`false`. Both conditions are checked again by the external preflight and the
main runner. A direct invocation that bypasses the wrapper fails before model
construction and before creation of the v3.1.2 attempt directory.

Do not add a deterministic-operation, precision, autotune, CUDA-version or
compilation-cache setting as part of this repair. Record all existing
`XLA_FLAGS`, `JAX_*`, `CUDA_*` and `NVIDIA_*` values exactly. The correction is
limited to removing the known shadowing path and disabling whole-device JAX
preallocation, consistent with the successful local wrappers. It does not
guarantee deterministic model arithmetic.

## 4. External JAX-only device preflight

The committed v3.1.2 wrapper first launches a fresh, sanitized subprocess that
imports JAX but imports no AlphaGenome Research, AlphaGenome model, OpenSplice
runner, checkpoint, FASTA, manifest or variant data. It performs no `jit`,
model call, array kernel, compilation or numerical target computation.

The preflight passes only if all of the following are true:

1. `LD_LIBRARY_PATH` is absent and
   `XLA_PYTHON_CLIENT_PREALLOCATE == "false"`;
2. JAX/JAXLIB and CUDA-plugin versions are recorded;
3. `jax.default_backend() == "gpu"`;
4. `jax.devices("gpu")` succeeds and returns exactly one visible device;
5. that device has `platform == "gpu"` and identifies as an NVIDIA GeForce RTX
   3090;
6. `nvidia-smi` succeeds and records exactly the expected visible physical GPU,
   UUID `GPU-64111645-1e42-a96d-f192-4abbec4b8090`, compute capability 8.6 and
   its current driver/VBIOS; and
7. no CUDA, cuSPARSE or backend initialization warning/error was suppressed.

The expected device-kind spelling is prospectively frozen as the literal
`NVIDIA GeForce RTX 3090` and compared exactly in the final freeze manifest.
The preflight must not accept `nvidia-smi` alone: the v3.1.1 failure
demonstrated that a visible physical GPU can coexist with a CPU-only JAX
backend.

### 4.1 Durable preflight records

Every external preflight writes an atomic, append-only record outside the
scientific one-shot directory, for example under
`results/v3_1_2_device_preflight/`. It records:

- monotonically increasing attempt number, timestamp and pass/fail status;
- committed preflight/wrapper/protocol/freeze hashes;
- Python executable and version;
- sanitized environment fields;
- JAX/JAXLIB/plugin and CUDA-library package versions;
- full `jax.default_backend`, JAX device and `nvidia-smi` observations; and
- complete exception/warning text on failure.

A failed device preflight does not consume the scientific v3.1.2 attempt
because it accesses no model or biological data and the one-shot directory
does not yet exist. Infrastructure may be repaired and the preflight repeated,
but every failed record is retained; preflights are never deleted or selected
based on AlphaGenome output. A pass is usable only while its committed code and
sanitized environment contract are unchanged.

The main runner is passed the successful record path and SHA-256. It refuses a
missing, failed, edited or wrong-version record.

## 5. Same-process fail-closed GPU gate

After exact source/protobuf/import/freeze validation but before model creation
and before the append-only v3.1.2 start record, the actual long-lived runner
repeats the Section 4 environment, default-backend, one-GPU, platform, device-
kind and `nvidia-smi` checks in its own process. This detects a time-of-check or
process-environment difference. It still performs no compiled probe.

Any mismatch exits before creating the v3.1.2 scientific output directory.
The runner must not merely serialize a CPU device and continue, and it must not
request a CPU fallback. The successful external-preflight hash and same-process
observations are embedded in `ATTEMPT_STARTED.json`.

Only after both preflights pass may the runner atomically create the new
scientific attempt directory and load AlphaGenome metadata/checkpoint state.

## 6. One-shot boundary and failure policy

The v3.1.2 attempt uses a new script version, attempt ID and output directory.
Before any preflight, commit or otherwise immutably timestamp:

- this amendment;
- the v3.1.2 preflight, runner, wrapper and tests;
- a freeze manifest binding their hashes and the unchanged v3.1.1/exact-source
  dependencies; and
- a v3.1.1 partial-failure record binding the one-file tree above.

The new attempt directory must be absent when the main runner starts. Once its
`ATTEMPT_STARTED.json` is created, **any** later failure—including metadata,
checkpoint, compile, device loss, persistence or identity failure—consumes
v3.1.2 under the existing one-shot/no-resume rule. It may not be deleted or
rerun. Ordinary per-case numerical/validation failures are still persisted
and collection continues; unrecoverable runtime/device failures stop the
cohort and remain partial.

## 7. Interpretation

- A preflight failure supports only “the recorded JAX GPU environment was not
  ready”; it says nothing about AlphaGenome.
- A v3.1.2 cohort pass supports only the exact-source reproduction claim
  already defined for v3.1.1, explicitly in the corrected recorded GPU
  environment.
- A v3.1.2 identity miss is a failed exact-source reproduction under the
  unchanged numerical rules; it cannot be excused by the v3.1.1 CPU fallback.
- A post-start runtime failure is an incomplete tooling diagnostic.

Under every outcome, v3.1.1 remains a disclosed partial infrastructure
failure, v3.0.2/v3.1 remain failed, the locked Phase-R result remains separate,
no active mechanism work is authorized and confirmation remains blind.

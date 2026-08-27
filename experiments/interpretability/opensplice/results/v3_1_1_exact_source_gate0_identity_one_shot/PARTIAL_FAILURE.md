# Exact-source v3.1.1 partial tooling failure

**Date:** 2026-08-28

**Frozen implementation commit:**
`c95c7c284487227fc6bcc2d1ae05a00088e37b17`

**Attempt status:** partial and failed; never resume, delete, or rerun as
v3.1.1

## Provenance and ordering

The five-file diagnostic bundle was committed and pushed before the attempt:
Git records commit `c95c7c2` at `05:26:37 +08:00`; the append-only start record
was created at `05:26:55 +08:00`. Its SHA-256 is
`1c1d24219e49e089806a02956bf8d7a44bdbd103e28adad6217eb3f94b424587`.

The start record binds the frozen launcher, protocol, source worktree,
generated protobuf artifacts, transitive import tree and mixed-precision
policy. It reports only a CPU JAX device and records the inherited environment
variable:

```text
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/lib:
```

## Failure

During JAX initialization, the CUDA plugin could not load cuSPARSE and emitted
an error ending with:

```text
RuntimeError: Unable to load cuSPARSE. Is it installed?
```

JAX then fell back to CPU. The launcher lacked a fail-closed assertion that a
CUDA device was active before publishing the one-shot start record. The
process was interrupted during remote metadata loading rather than allowed to
perform hours of unintended CPU work.

Only `ATTEMPT_STARTED.json` exists. There are:

- zero compiled model executables;
- zero identity calls or target values;
- zero active interventions;
- zero confirmation calls.

The protocol explicitly defines a terminally interrupted attempt as partial
and failed with no resume. Therefore this attempt is closed permanently. It
does not answer the exact-source numerical-reproduction question and provides
no biological or mechanistic evidence.

Any continuation requires a newly versioned prospective protocol. At minimum,
the wrapper must establish the same working CUDA-library environment used by
the validated RTX 3090 runs and a separate device preflight must assert a CUDA
backend before the new append-only scientific attempt can be created.

# OpenSplice v3.3.4.6 controlled-stop archive

Date: 2026-09-02 (Asia/Singapore)

Status: `complete_controlled_stop_structural_archive`

Decision: `controlled_stop_device_preflight_environment_drift`

Scientific result: none. This archive contains no model outputs, no dispatches,
no interventions and no biological evidence.

## Authorized attempt

The sole v3.3.4.6 invocation was authorized against:

- amendment commit `686b67bd772a7771a6c4540c5b188b6497dbdec7`;
- source commit `a00c7561add8d227a5b3047b8425b0ad7c53de22`;
- freeze commit `cfbc673a198229c771f8c3c409d121eeebd41ca3`;
- freeze SHA-256
  `8bdaba150ae8e4a5a9ecf1b41f2f7fa230d9a7560935ba3bce97d8e4adb1d9db`;
- freeze size 251355 bytes; and
- immutable v3.3.4.5 controlled-stop archive content binding
  `9112a72de718a1e25f374798f4d76118c5d7ee8b93b38227b4517adda4af6e3d`
  over 19072 canonical bytes.

The postcommit dry run passed Gate A and reported exactly 20 recipients,
anchors `[0,127,128,255]`, 80 records, 320 planned applies, one planned
eight-row compile, zero six-row compiles, zero confirmation calls and
`production_paths_created=false`.

The production wrapper was invoked exactly once. It exited 2 after publishing
the sole external preflight record. The attempt is consumed and must not be
retried, resumed, renamed, deleted or used as a cache input.

## Exact stop

`preflight_0000.json` has status `fail`, a null observation, no warnings and
the exact failure:

```text
ValueError: Frozen v3.3 platform/GPU version manifest changed.
```

The record asserts `no_jit_or_array_kernel=true` and
`no_model_or_biological_access=true`. The atomic no-replace publication probe
passed; its deliberate collision preserved one named temporary diagnostic.
Cache-hit evidence is false, no old cache input was opened and the two files
under the external cache root are diagnostic outputs only.

A read-only postmortem compared the inherited frozen manifest with the live
host. Python 3.13.5, platform family, GPU UUID, RTX 3090 device name, compute
capability 8.6, VBIOS 94.02.42.C0.05 and driver 560.35.05 still match. The
host kernel is now `6.8.0-138-generic`; the inherited v3.3 freeze requires
`6.8.0-136-generic`. That kernel drift is sufficient for the fail-closed
preflight result. It is not evidence about AlphaGenome internals.

## Immutable attempt bytes

External preflight-cache root:

| Relative path | Mode | Bytes | SHA-256 |
|---|---:|---:|---|
| `.v3345.tmp.443727.000001.7f9560523ee0ed4c682bfadae2cc4884` | 0400 | 39 | `d9eaea234b10d919daaabddf3d560d34c946ba3259bb1e16cfd9a005a05d85b2` |
| `atomic_publication_probe_v3_3_4_6.txt` | 0400 | 49 | `db249bd851ab5028d95d71a10d6edad91f15b18e023722a3086383f7aab04a65` |

The cache root and its `triton` and `xdg` children are mode 0700. Its frozen
post-observation file-tree digest is
`d3f60f76a2f3fb1253e1236c283bae4c404960f2a8c37fe4dd63b67fa6949431`.

External preflight-record root:

| Relative path | Mode | Bytes | SHA-256 |
|---|---:|---:|---|
| `.allocation.lock` | 0600 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.preflight_0000.reserved` | 0400 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `preflight_0000.json` | 0400 | 46668 | `15d6c1cae48c0df34b3cbcfb4cdad802b386a65cb76c0e723bb27f4b8564f92a` |
| `preflight_0000.stderr.log` | 0400 | 767 | `fdfc92d41ed6c18ce3b77787f1be7bd318742b003776145b0f6684e708960e4d` |
| `preflight_0000.stdout.log` | 0400 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The preflight root is mode 0700.

## Zero-work and path boundary

The model-kernel cache, model-run directory, analysis-attempt directory and
analysis-output directory all remain absent. Consequently the exact counts
are:

- model loads: 0;
- model compiles: 0;
- model applies: 0;
- dispatch-journal events: 0;
- OOD records: 0 of 80;
- raw manifests: 0;
- main-cube reruns: 0;
- confirmation calls or reads: 0; and
- scientific/Shapley/interaction/nomination outputs: 0.

The two present v3.3.4.6 roots and the four absent roots are now immutable
archive state.

## Analyzer disposition

The frozen v3.3.4.6 analyzer was not invoked. Static inspection after the
controlled stop found that its parser creates
`args.acknowledge_structural_only_v3_3_4_5`, while `main()` reads
`args.acknowledge_structural_only_v3_3_4_6`. Invoking it would therefore fail
before allocating either analysis root. Modifying that frozen source after the
attempt would violate the source/freeze authority chain, so both analysis
roots were deliberately left absent.

Any future model attempt must be a new prospectively documented attempt. It
must explicitly decide whether kernel patch-level equality is scientifically
necessary, freeze the live runtime it actually authorizes, and separately fix
and test the analyzer CLI destination. Nothing in this archive licenses a
retry of v3.3.4.6 or a biological claim.

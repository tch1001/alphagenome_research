# Independent audit of the v3.1.2 exact-source diagnostic

**Audit date:** 2026-08-28

**Scope:** development identities only. This audit read the 20 raw identity
JSON files, the append-only provenance files and the summary in this directory.
It did not run a model, inspect an activation, run an intervention, or open a
confirmation artifact.

## Result

The v3.1.2 process completed all 20 frozen development identities without a
runtime or validation failure, but it did **not** reproduce the old Phase-R
saved targets at the unchanged inclusive tolerance of `2^-8 = 0.00390625`.
Only 4/20 identities passed that historical comparison; 16/20 were numerical
failures.

| Recomputed quantity | Result |
|---|---:|
| Raw identity files | 20 |
| BRAF pass / total | 4 / 10 |
| SLC25A48 pass / total | 0 / 10 |
| Historical-lock pass / failure | 4 / 16 |
| Validation / runtime failures | 0 / 0 |
| Median per-identity maximum absolute lock difference | 0.0078125 |
| Largest per-identity maximum absolute lock difference | 0.03125 |
| Significant effects with current direction and `abs(DeltaL) >= 0.01` | 12 / 12 |
| Significant effects with locked direction and `abs(DeltaL) >= 0.01` | 12 / 12 |

The four passing rows were `BRAF_e14_T71A`, `BRAF_e14_T71G`,
`BRAF_e14_T121C`, and `BRAF_e14_C68T`. The first two are significant effects
and the latter two are experimental neutral controls. A historical-lock pass is
not a biological accuracy label.

The 20 per-row maximum absolute differences had the following exact counts:

| Maximum absolute difference | Rows |
|---:|---:|
| 0.00390625 | 4 |
| 0.005859375 | 1 |
| 0.0078125 | 8 |
| 0.009765625 | 2 |
| 0.01171875 | 2 |
| 0.013671875 | 1 |
| 0.015625 | 1 |
| 0.03125 | 1 |

## Controls recomputed from raw JSON

All 20 identities independently satisfied every within-executable control
serialized by the runner:

- exact REF and ALT duplicate target rows;
- exact target repeat;
- exact compact-trace duplicate rows and repeat;
- exact equality of the two-target total and twice the reported mean;
- exact case, interval, endpoint, position-set and sequence linkage; and
- finite current targets with no skipped assertion.

All 12 significant-effect rows retained the experimental sign and exceeded
the frozen `0.01` predicted-effect floor in both the current and historical
values. Thus v3.1.2 is not evidence that the target reducer, allele mapping or
six-row identities failed. It is evidence that saved BF16 targets from a
different compiled execution are not a stable numerical identity oracle at
`2^-8`.

The external preflight passed on exactly one JAX GPU identified as an NVIDIA
GeForce RTX 3090 with UUID
`GPU-64111645-1e42-a96d-f192-4abbec4b8090`. `LD_LIBRARY_PATH` was absent,
preallocation was disabled, both captured logs were empty, and no warning was
recorded. The same-process observation also recorded the RTX 3090 before the
attempt started.

## Integrity recomputation

The summary's 27-entry artifact hash map matches every non-summary file. The
independently recomputed artifact-tree SHA-256 is
`88e756dbbc1370a4e1b129d47ee15d90472100588df525b1e213993276b42dbd`,
identical to the summary.

| Artifact | SHA-256 |
|---|---|
| External preflight | `e9cb47e7a15cc44b43e713d567aff9b4d2297f3c512fe5a9dca296d0df2bc499` |
| `ATTEMPT_STARTED.json` | `c35fc8fd3df8bbd9dfba6b976fd3bced57029917b2a930d68b1b6ec8625f7ab6` |
| `SUMMARY.json` | `16772391bde09f3a768224cc5bb7bf02da1021d5c386fff4a35f66be53d6b08d` |
| `COMPLETION_PROVENANCE.json` | `5d93f2bbea55eec04f2c78f586e4f192c00f04de0ad584a7eb23e47a70e4cc3b` |

The attempt took about 255.18 seconds from its start record to summary. The
single compilation accounted for 161.49 seconds; the 40 identity executions
accounted for about 4.65 seconds in total. Timing is a tooling observation, not
a benchmark because initialization and metadata costs are included only in the
wall time.

## Claim boundary and disposition

The maximum defensible claim is:

> In the corrected recorded RTX 3090 environment, the exact-source Phase-R
> identity program was internally repeatable and preserved the expected
> direction for all 12 development effects, but 16/20 current target vectors
> did not reproduce an older compiled target lock within `2^-8`.

This does not validate or invalidate a causal mechanism. v3.1.2 ran zero active
patches and zero confirmation calls. The failed historical comparison must not
be relaxed, rounded, offset-corrected, retried or cherry-picked. It should be
retired as a gate for future intervention work. Future baselines, self controls,
donors and patched targets must instead be generated inside one prospectively
frozen superset graph and compared only within that graph.

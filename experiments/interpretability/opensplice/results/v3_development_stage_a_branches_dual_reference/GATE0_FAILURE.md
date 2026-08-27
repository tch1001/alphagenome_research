# Stage-A v3.1 Gate-0 failure

**Date:** 2026-08-28

**Git commit:** `5860172ee3a0b856b925bc360baac330cb0d6bdc`

**Runner:** `opensplice-stage-a-branches-v3.1.0`
**Status:** failed prospectively; do not resume or retry as v3.1

## What ran

The frozen all-or-nothing development command was:

```bash
./experiments/interpretability/opensplice/run_stage_a_branches_v3.sh
```

The runner first attempted all 20 fresh current-Phase-R (`R_current`)
references. It was required to reproduce the immutable Phase-R identities
(`R_lock`) within the unchanged inclusive absolute tolerance
`2^-8 = 0.00390625` before any Stage-A graph was called.

## Exact outcome

The first two references completed and were written under
`phase_r_reference/`:

- `BRAF_e14_A117G`: maximum absolute difference `0.00390625`;
- `BRAF_e14_T71A`: maximum absolute difference `0.00390625`.

The third case, `BRAF_e14_T71G`, failed before its artifact was written. Its
six-row absolute-difference vector was:

```text
[0.00390625,
 0.014404296875,
 0.014404296875,
 0.014404296875,
 0.00390625,
 0.00390625]
```

The locked REF mean was `2.546875` and the locked ALT mean was
`3.876708984375`. Because the runner raised before serialization, the sign and
raw current target for the failing case were not retained and must not be
inferred from the absolute differences.

The first reference's compile-and-run time was about 160.23 seconds; later
warm calls were about 0.114 seconds. Thus all three cases used the same current
compiled executable. Within that executable, exact duplicate rows, self rows
and the immediate repeat passed. Case, interval, sequence hashes, canonical
target, all selected position sets, checkpoint snapshot, helper hashes and
the amendment hash were linked before execution.

## What did not run

- zero Stage-A identities;
- zero final-A/D closures;
- zero joint T+E closures;
- zero isolated T or E branch interventions;
- zero Shapley partitions;
- zero confirmation-exon calls.

No mechanistic conclusion follows from this attempt. It is a numerical
reproducibility failure of the historical-reference gate.

## Interpretation and next-step boundary

The original Phase-R lock was produced at commit `fd4dc69`, whereas the
current checkout includes additive Stage-A instrumentation in core source
files. The Phase-R function bodies and linked inputs are unchanged, but the
historical optimized HLO, GPU autotune choices and executable were not saved.
Different BF16/XLA compilation decisions are therefore a plausible cause;
the failure does not prove a specific kernel or a single-ULP mechanism.

It is not valid to delete the two artifacts, retry until all 20 happen to
pass, enlarge/round the threshold, or choose compiler settings by closeness to
the lock. The next permitted action is a separately frozen, identity-only,
one-shot diagnostic from the immutable original source commit with fuller
compiler provenance. If that fails, v3.1 is permanently closed and any
substantive continuation must be a newly versioned within-executable study.

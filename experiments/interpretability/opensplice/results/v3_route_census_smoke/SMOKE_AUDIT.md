# OpenSplice v3 route-census smoke audit

- **Audit date:** 2026-08-28
- **Scope:** one development variant and one route component; no model
  inference was rerun for this audit
- **Verdict:** the artifacts pass as a tooling smoke test. They do not support
  a biological mechanism, route ranking, or “encoder stage 0 explains 55.83%”
  claim.

## Provenance checked

The run is pinned to commit
`0985c0d2238ea643fe6b8506b9903632f76f3f52`, checkpoint snapshot
`a8f293a76ee73d5b57f3bf2ae146510589fcf187`, dense attention and 16,384-bp
context. The five recorded source hashes still match the files at that commit,
and the recorded tracked-diff hash is the SHA-256 of an empty diff. The frozen
input hashes also match:

- `selected_variants_v2.tsv`:
  `09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6`
- `frozen_exons_v2.tsv`:
  `b95f8fc540f19222546322bebfb817a6c0f2147dd41325091086833584a09a75`

Recomputing the canonical-JSON configuration fingerprints produced the exact
stored identity fingerprint
`d89bed04df4ef6d3ff627844eafb2f22b3e3bae0e9ab0191f9355d0ecc53547c`
and component fingerprint
`d081bff0562f50813f7f8def28ba769c1a3f2804bcd3db42616050cfcadd6439`.
The three smoke files have SHA-256 values:

- `summary.json`:
  `ef7daa6723337f6c3d359150f70eb10930bb8b9ee11f0d1e0e22e2c5aa5635a7`
- identity JSON:
  `2e93fad68ad25ed03594794f776b94e44e6a52df355cb57506b33d9434ec7152`
- encoder-stage-0 component JSON:
  `63ebdf4128550570eb04cda5828927eb473a5e60c29ff6fc1ebe283a20cc6dd9`

## Target and sign audit

The case is the negative-strand `BRAF_e14_A117G` variant. The interval starts
at GRCh38 position 140,746,017 (0-based). Recomputing output indices gives:

- acceptor: `140754233 - 1 - 140746017 = 8215`, class track 3;
- donor: `140754187 - 1 - 140746017 = 8169`, class track 2; and
- background/padding: track 4.

These tracks match the pinned metadata order: donor+, acceptor+, donor-,
acceptor-, Padding. The hashed factory reads only
`splice_sites_classification/logits`; its reducer computes, in FP32,

```text
T = mean(logit_acceptor- - logit_padding at 8215,
         logit_donor-    - logit_padding at 8169).
```

The identity artifact reports `num_values = 2` and that `total = 2 * mean` was
checked before serialization. The stored means independently give

```text
T_REF = 2.537109375
T_ALT = 3.089843750
DeltaT = T_ALT - T_REF = +0.552734375
```

The sign agrees with the positive experimental OpenSplice effect
(`delta_logit = +4.36370722301283`) and exceeds the frozen `0.01` eligibility
threshold. The magnitudes are not on a common calibrated scale: one is an
AlphaGenome class-evidence margin in native sequence and the other is an
experimental minigene delta-logit(PSI).

The JSON does not retain the four endpoint logits or the two endpoint margins.
Therefore the target formula, coordinates, track mapping and aggregate
arithmetic are independently auditable, but the endpoint-level numerical
reduction is represented by the run's passed check rather than reconstructible
from the JSON alone.

## Six-row direction and recovery audit

The component artifact records the following rows. A donor label matters only
for the four patched/self-control rows; baseline transfer masks are false.

| Row | Recipient DNA | Donor | Target |
|---|---:|---:|---:|
| REF baseline | REF | -- | 2.537109375 |
| ALT baseline | ALT | -- | 3.089843750 |
| REF into ALT | ALT | REF row 0 | 2.781250000 |
| ALT into ALT self | ALT | ALT row 1 | 3.089843750 |
| ALT into REF | REF | ALT row 1 | 2.992187500 |
| REF into REF self | REF | REF row 0 | 2.537109375 |

Both transfers move in the donor-consistent direction: REF into ALT decreases
the target by `-0.30859375`, while ALT into REF increases it by `+0.455078125`.
Baseline drift and both self-control drifts are exactly zero. Recomputing the
stored recovery definition gives:

```text
R_REF->ALT = (2.78125 - 3.08984375)
             / (2.537109375 - 3.08984375)
           = 0.558303886925795

R_ALT->REF = (2.9921875 - 2.537109375)
             / (3.08984375 - 2.537109375)
           = 0.823321554770318

B = min(R_REF->ALT, R_ALT->REF) = 0.558303886925795
```

Thus `55.83%` is the minimum of two normalized directional movements for this
one intervention, not the fraction of a biological mechanism explained.

## Trace-seam audit

At encoder stage 0 the runner selects
`S = unique(variant, acceptor, donor)`. Here the variant is the donor, leaving
two active 1-bp slots: indices 8169 and 8215. The intervention replaces the
entire 768-channel `DnaEmbedder` output vector at both slots. Core code takes
all donor vectors simultaneously from the natural seam input, applies them to
the recipient, gathers the effective vectors for the audit, then uses the
modified tensor both as the 1-bp skip state and as input to pooling and every
downstream stage.

The run checked exact natural-donor/effective-recipient vector equality for
REF->ALT, ALT->ALT, ALT->REF and REF->REF at the active slots. It also ran an
all-false identity call twice and checked bit-exact targets and traces across
repeats and duplicate REF/ALT rows. Four focused CPU synthetic tests were
rerun during this audit and passed: six-row donor semantics, encoder no-op/tree
preservation, logit-margin common-shift invariance, and standard-checkpoint
factory compatibility.

Raw trace vectors are not serialized. Accordingly, the exact-vector result is
a code-pinned, fail-closed assertion made during the real run, not an equality
that a reader can recompute from these JSON files alone.

## Exact claim boundary

The defensible claim is:

> In a real, development-only smoke run for `BRAF_e14_A117G`, the v3 JAX
> plumbing performed a code-pinned, six-row bidirectional transfer of the two
> selected full encoder-stage-0 vectors with exact baseline/self controls. The
> splice-classification logit-margin moved toward the donor in both directions;
> normalized directional recoveries were 0.5583 and 0.8233 (minimum 0.5583).

Do not yet claim that encoder stage 0 is an AlphaGenome “bottleneck,” that it
explains 55.83% of the variant's biological effect, or that a splicing
mechanism was localized. This was one positively selected BRAF variant and the
first of 51 route components, with no cross-variant, cross-exon, neutral or
positional-control comparison. Stage 0 is also a common ancestor of the skip
and pooled trunk paths, the intervention replaces all 768 channels at both
canonical sites, the recovery metric is non-additive, and the scalar average
does not reveal whether acceptor or donor evidence moved.

The observation becomes a route-census result only after the preregistered
development run covers the locked cases/components and is summarized across
variants and both exons without choosing examples or routes post hoc. Future
artifacts should additionally store the two endpoint margins (or four selected
logits) and compact hashes of the natural/effective selected vectors so those
two assertions can be independently reconstructed without retaining dense
traces.

# Spatial encoder-skip model-behavior result

The mask-110 route is spatially localized in the development examples.
Variant (`V`), acceptor (`A`), and acceptor/donor-union (`S`) supports
pass the frozen two-gene rule; donor-only (`D`) fails in SLC25A48.
All 160 equal-shape shifted controls have exactly zero recovery.

## Primary effect-variant result

| Support | BRAF median B | SLC25A48 median B | BRAF median q | SLC25A48 median q | Pass both genes |
|---|---:|---:|---:|---:|:---:|
| V | 0.41409 | 0.39649 | 0.41409 | 0.39649 | yes |
| A | 0.38700 | 0.65977 | 0.38700 | 0.65977 | yes |
| D | 0.27905 | 0.00893 | 0.27905 | 0.00893 | no |
| S | 0.51613 | 0.66509 | 0.51613 | 0.66509 | yes |

Here `q = B_intended - max(B_upstream, B_downstream)` per variant.
Every effect variant has positive `q` for every support, but `D` misses
the preregistered `B >= 0.25` threshold in SLC25A48.

## What the intervention establishes

- Whole-skip recovery is not diffuse: equal-shape patches shifted at
  least 512 bp away recover exactly zero in all 160 controls.
- A compact V-local patch recovers median `B=0.41409` in BRAF and
  `B=0.39649` in SLC25A48.
- The larger A/D union is stronger (`0.51613`, `0.66509`), showing that
  useful skip information is distributed across the splice-local
  region.
- The donor region contributes in BRAF but is nearly irrelevant in
  SLC25A48 (`B=0.00893`), consistent with exon-specific routing.

## Important boundaries

V versus A/D is not a clean biological contrast in this benchmark.
Many effect variants are at or very near a canonical splice site, so
guarded supports overlap at several resolutions. The result localizes
a V/splice-site neighborhood; it does not prove distinct variant,
acceptor or donor modules.

The BRAF specificity problem also remains. For every passing support,
median absolute movement is larger in the four experimental BRAF
neutrals than in the six effects. These neutrals are therefore not
AlphaGenome-null, and the result should not be promoted to a general
biological mechanism.

## Next scientific step

Use the V-local support as the compact seed and rank channels inside
`E32+E16+E8+E2+E1` by causal loss of bidirectional recovery. Require
cross-gene consistency and the same shifted-position controls before
connecting high-impact channels back to sequence patterns or known
splicing motifs.

Confirmation examples were not accessed.

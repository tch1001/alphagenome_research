# V-local channel-group model-behavior result

The completed development screen identifies reproducible channel
subspaces inside the spatially localized five-skip route. The strongest
shared candidate is `E1` channels 160-191. SLC25A48 additionally shows
a striking 160-191 signal at every tested resolution, while BRAF ranks
`E16` channels 512-543 first within that gene.

## Screen and controls

- 20 development variants, 172 nonoverlapping 32-channel blocks and
  3,520 model applies completed.
- All selected-channel donor, withheld-channel natural-value,
  same-allele, baseline and non-route no-op controls passed.
- Identity and full-route repeats were bit-exact. Group conditions were
  intentionally single-shot after the prior spatial repeat cube was
  exact.
- Confirmation examples were not accessed.

## Leading cross-gene necessity blocks

| Rank | Block | BRAF median loss | SLC25A48 median loss | Maximin |
|---:|---|---:|---:|---:|
| 1 | `E1_c0160_0191` | 0.02007 | 0.04814 | 0.02007 |
| 2 | `E32_c0000_0031` | 0.01750 | 0.01567 | 0.01567 |
| 3 | `E16_c0000_0031` | 0.01779 | 0.00963 | 0.00963 |
| 4 | `E2_c0096_0127` | 0.00874 | 0.00771 | 0.00771 |
| 5 | `E16_c0320_0351` | 0.00750 | 0.01532 | 0.00750 |
| 6 | `E32_c0928_0959` | 0.01147 | 0.00674 | 0.00674 |
| 7 | `E16_c1088_1119` | 0.00674 | 0.00852 | 0.00674 |
| 8 | `E32_c0960_0991` | 0.02260 | 0.00632 | 0.00632 |
| 9 | `E8_c1024_1055` | 0.00822 | 0.00591 | 0.00591 |
| 10 | `E1_c0480_0511` | 0.00852 | 0.00545 | 0.00545 |

Loss is `B_full V - B_without block`; positive values mean that
withholding the block reduced reciprocal recovery. The first three
blocks also have a larger median loss for effects than experimental
neutrals in both genes, so they are the locked shared refinement set:

- `E1_c0160_0191`
- `E32_c0000_0031`
- `E16_c0000_0031`

## Architecture-linked clue

For SLC25A48, the same channel-number band 160-191 is highly ranked at
E32, E16, E8, E2 and E1; it is the top SLC25A48 block at E2 and the
strongest shared block at E1. `DownResBlock` preserves the existing
channel prefix through a zero-padded residual connection while adding
128 new channels at each downsampling step. The causal recurrence is
therefore consistent with a persistent multiscale feature family.

This is an inference, not yet a biological label: learned convolutions
mix channels, so equal indices do not prove an invariant feature.

## Boundaries and next experiment

The screen is a causal search result, not a motif or molecular-mechanism
claim. Effects are nonlinear and nonadditive, only six effect variants
per gene were used, and no one 32-channel block accounts for most of the
full-route recovery.

Next, split the three shared parents and the two top gene-ranked parents
into contiguous 8-channel children. Surviving children then advance to
individual-channel necessity, only-group sufficiency, and V-versus-
shifted localization. Sequence optimization or motif attribution should
start only after those causal checks.

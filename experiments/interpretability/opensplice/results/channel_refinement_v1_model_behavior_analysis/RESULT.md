# Eight-channel refinement model-behavior result

The refinement separates the coarse cross-gene candidates into mostly
adjacent, gene-specific subspaces. In all three 32-channel parents
selected as shared candidates, BRAF and SLC25A48 have different
dominant 8-channel children.

## Dominant child within each shared parent

| Parent | BRAF top child (median loss) | SLC25A48 top child (median loss) | Same child? |
|---|---:|---:|:---:|
| `E1_c0160_0191` | `E1_c0160_0167` (0.01280) | `E1_c0168_0175` (0.04306) | no |
| `E32_c0000_0031` | `E32_c0000_0007` (0.01900) | `E32_c0016_0023` (0.01243) | no |
| `E16_c0000_0031` | `E16_c0000_0007` (0.01731) | `E16_c0016_0023` (0.00854) | no |

The repeated offset is informative. BRAF favors channels 0-7 at both
E32 and E16. SLC25A48 favors 16-23 in those same parents, and its two
strongest children overall are channels 168-175 at E2 and E1. Thus the
persistent 160-191 SLC25A48 band from the coarse screen narrows to the
same eight-channel slice at two resolutions.

## Locked individual-channel candidates

Using the top two positive effect-over-neutral children per gene:

- BRAF: `E32_c0000_0007`, `E16_c0000_0007`
- SLC25A48: `E2_c0168_0175`, `E1_c0168_0175`

## Boundaries

All 480 planned applies completed and every causal runtime control
passed. Identity and full-route repeats were bit-exact; child calls
were intentionally single-shot. Confirmation data remained sealed.

These are model subspaces, not biological factors. The maximin shared
losses at 8-channel resolution are small, losses are nonadditive, and
there are six effect variants per gene. The result argues for parallel
exon-specific programs rather than a universal splice channel.

Next, test all 32 individual channels within the four locked children,
together with only-child sufficiency and shifted-position controls.
Only channels surviving those tests should be mapped to activating
sequences, motifs or candidate splicing factors.

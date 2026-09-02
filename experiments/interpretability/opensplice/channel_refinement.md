# Eight-channel refinement of V-local skip features

The 32-channel necessity screen selected three shared candidates and one top
within-gene candidate for each exon:

- shared: `E1:160-191`, `E32:0-31`, and `E16:0-31`;
- BRAF-ranked: `E16:512-543`; and
- SLC25A48-ranked: `E2:160-191`.

The shared candidates have positive median necessity in both genes and larger
median losses for effects than experimental neutrals in both genes. The two
additional parents preserve the strongest exon-specific signals rather than
forcing a shared circuit prematurely.

Each 32-channel parent is split into four contiguous, nonoverlapping
8-channel children. For every child, the intervention transfers all other
channels in the V-local `E32+E16+E8+E2+E1` route and leaves that child at its
natural recipient value. The score remains

```text
loss(child) = B_full_V - B_without_child.
```

All 20 development variants are retained. Identity and full-route conditions
are repeated, and every child call verifies exact donor values in selected
channels, exact natural values in withheld channels, same-allele controls and
no-op behavior outside the route. The full design is 480 model applies in one
fixed-shape graph. Confirmation examples remain sealed, and operating-system
metadata is not a scientific gate.

This experiment narrows causal subspaces; it does not label them. A surviving
8-channel child must still pass individual-channel necessity, only-child
sufficiency and shifted-position controls before sequence preferences or known
splicing motifs are assigned.

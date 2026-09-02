# Single-channel spatial sufficiency

The individual necessity screen advances three AlphaGenome coordinates: BRAF
E16 channel 3 and SLC25A48 channel 175 at E1 and E2. This development-only
experiment asks whether each coordinate can transfer reciprocal splice effect
by itself.

For each coordinate, transfer only that channel between alleles at the three
guarded V tokens. Repeat the same one-channel intervention at equal-shape
upstream and downstream positions. The primary statistic is bidirectional
bottleneck recovery, with per-variant spatial contrast

```text
B_intended - max(B_upstream, B_downstream).
```

A coordinate passes if, in its selected gene, median intended recovery and
median spatial contrast are positive, at least four of six effects have a
positive contrast, and effect median intended recovery exceeds the
gene-matched neutral median.

The complete design contains 9 conditions across 20 development variants,
plus repeated identity and full-route calls, for 260 model applies. Every call
checks exact donor values in the selected channel, natural values everywhere
else, same-allele controls and no-op behavior outside the skip route.
Confirmation examples remain sealed; OS metadata is not a scientific gate.

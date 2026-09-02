# Individual-channel necessity and subspace sufficiency validation

The 8-channel refinement favors two subspaces per gene:

- BRAF: `E32:0-7` and `E16:0-7`;
- SLC25A48: `E2:168-175` and `E1:168-175`.

This frozen development experiment combines two tests. First, it withholds
each of the 32 constituent channels individually while transferring every
other channel in the V-local five-resolution route. This measures

```text
necessity(channel) = B_full_V - B_without_channel.
```

Second, it transfers each selected 8-channel subspace by itself at the
intended V neighborhood and at equal-shape upstream and downstream positions.
This tests whether a necessary subspace is also independently sufficient and
spatially localized rather than merely participating in a distributed route.

The experiment retains all 20 development variants and has 44 conditions per
variant: 32 individual-channel necessity conditions plus 12 child-only
sufficiency/location conditions. Together with repeated identity and full-V
calls, the complete run is 960 model applies in one fixed-shape graph.

Every call checks exact selected donor channels, exact natural values in
withheld channels, same-allele controls and no-op behavior outside the skip
route. Confirmation examples remain sealed. The operating-system kernel is
not a scientific gate.

Channels advance only if their effect-variant median necessity is positive,
exceeds the gene-matched neutral median, and is positive in at least four of
six effects. Even an advancing channel remains a model coordinate, not a
biological factor. Sequence optimization, motif matching and controlled edits
are required to turn it into a biological hypothesis.

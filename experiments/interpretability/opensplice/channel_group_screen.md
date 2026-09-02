# V-local channel-group necessity screen

## Question

The spatial experiment localized reciprocal splice-effect recovery to a compact
variant/splice-site neighborhood inside `E32+E16+E8+E2+E1`. This screen asks
which channel subspaces within those five skip tensors are causally necessary
for that recovery.

This remains a development-only model-behavior experiment. It does not access
confirmation examples or use operating-system metadata as a scientific gate.

## Intervention

The existing live decoder-skip transfer accepts an optional channel mask. At
the three guarded V-local tokens in each candidate stage, the six-row protocol
can therefore transfer selected donor channels while leaving every withheld
channel at its natural recipient value. The ordinary whole-vector behavior is
unchanged when no channel mask is supplied.

The candidate widths are:

| Stage | Channels | 32-channel groups |
|---|---:|---:|
| E32 | 1,408 | 44 |
| E16 | 1,280 | 40 |
| E8 | 1,152 | 36 |
| E2 | 896 | 28 |
| E1 | 768 | 24 |
| Total | 5,504 | 172 |

For each condition, transfer every candidate channel except one contiguous,
non-overlapping group. The per-variant necessity score is

```text
loss(group) = B_full_V - B_without_group.
```

A positive value means withholding that group reduced bidirectional recovery.
Because the network is nonlinear and groups may be redundant, losses are not
assumed to add to the full-route result.

## Scope and controls

The screen includes all 20 development variants so the 12 effects and 8
experimental neutrals can be compared. Identity and full-V conditions are
repeated exactly. Each of the 172 group conditions is called once; repeating
all of them would add little after the prior spatial experiment produced 240
bit-exact repeats.

Every group call checks:

- exact live donor values in every selected channel;
- exact natural recipient values in every withheld channel;
- exact no-op behavior at invalid positions and non-V routes;
- unchanged REF/ALT baseline rows and same-allele self controls; and
- finite reciprocal splice-logit-margin recovery.

The complete screen is 3,520 model applies in one fixed-shape compiled graph.
The new graph records its difference from the previous spatial run, but exact
cross-executable equality is not required: compilation can change BF16-scale
rounding even in untouched baseline rows. All causal comparisons used for
ranking are within the same executable.

## Analysis and follow-up

Rank blocks jointly by the smaller of their BRAF and SLC25A48 median necessity
losses, and separately within each gene. A shared high-ranked block is evidence
for a cross-exon causal subspace; divergent rankings support exon-specific
feature programs.

This group screen is a search stage, not a feature interpretation. The leading
blocks must be recursively split toward individual channels, then tested for
only-group sufficiency and V-versus-shifted spatial localization before their
sequence preferences are interpreted as motifs.

# SLC25A48 channel-175 saturation-mutagenesis result

The controlled edit map closes a sequence-to-feature-to-output mechanism at
this exon. The reference sequence is `TAGG` at -3..0, exactly the
core preferred by the channel-175 checkpoint kernel.

All 6/6 substitutions of the invariant acceptor `AG` at -2/-1 reduce the
acceptor logit margin (median `-10.672`). Across
all 12 substitutions in the `TAGG` -3..0 core, the median acceptor change is
`-6.775` and median absolute change is
`6.775`, versus
`0.453` outside the core.

Across all 123 SNVs, E2 channel-175 change at the acceptor predicts acceptor
logit change with Pearson `r=0.935` (tie-aware Spearman
`rho=0.633`). E1 is similarly predictive
(`r=0.933`). The nonlinear E1 update (`r=0.933`)
is much more predictive than the direct kernel alone
(`r=0.622`), confirming that channel 175 is a composite learned
detector rather than a single PWM filter.

The six strongest acceptor-decreasing edits are:

- `-1_G>T`: acceptor -11.719, E2:175 -16.000
- `-2_A>C`: acceptor -11.438, E2:175 -14.000
- `-2_A>T`: acceptor -11.094, E2:175 -15.406
- `-2_A>G`: acceptor -10.250, E2:175 -13.875
- `-1_G>A`: acceptor -10.219, E2:175 -14.000
- `-1_G>C`: acceptor -9.312, E2:175 -17.094

Together with the prior necessity/sufficiency interventions, these results
support the following model mechanism: AlphaGenome recognizes the local
splice-acceptor sequence in E1 channel 175, nonlinearly sharpens disruption of
the `TAGG`/invariant-`AG` neighborhood, carries and amplifies that feature at
E2, and uses it causally in the splice prediction. This is a model mechanism at
SLC25A48; it is not yet a claim about a named splicing factor or genome-wide
universality.

All 50 applies completed, reference outputs were exact across all 25 batches,
and confirmation data remained sealed.

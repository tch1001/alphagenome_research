# Encoder feature-decomposition result

SLC25A48 channel 175 is a learned splice-acceptor detector. The direct
15-bp DNA kernel prefers `TAGG` at
offsets -3..0 relative to its output base, matching the core of the tested
acceptor neighborhood. Across all ten SLC25A48 variants, direct-kernel weight
differences predict the measured direct-convolution allele differences to a
maximum absolute error of
`0.000976562`.

The causal E1 feature is not merely that short linear filter. In the six
effects, the direct branch has median allele-difference L2
`0.407`, while the learned E1
residual branch has median L2
`12.877` and produces a
negative acceptor activation change in all 6/6 effects. The final E1 effect
vectors are already strongly aligned (median pairwise cosine
`0.997`), versus an effect median L2 of
`13.112` and neutral median L2 of
`1.359`.

E2 mostly inherits that composite E1 detector through the explicit
zero-padded residual path, then selectively strengthens it. The E2 output
norm exceeds its carried-input norm in 6/6 effects, with median amplification
`1.129x`; the neutral median is
`0.981x`.
Effect median L2 rises to `14.008`
while neutral median L2 is `1.341`.

The decomposition revises the earlier robustness hypothesis: E2 does not make
the six natural effect vectors more mutually aligned (median cosine changes
from `0.997` to
`0.986`). Instead, its learned updates
amplify an already coherent E1 acceptor-disruption signal. The greater causal
portability of E2:175 therefore likely also reflects how the decoder consumes
the coarser skip, not just a cleaner encoder feature.

BRAF E16:3 remains a weaker, distributed inherited feature and is not assigned
a biological motif here. All 40 natural-allele applies passed exact-repeat,
finite-value and padded-position controls; confirmation data remained sealed.

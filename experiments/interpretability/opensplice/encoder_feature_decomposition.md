# Encoder feature decomposition

This development-only experiment follows the causal channel tests by asking
what the frozen AlphaGenome encoder actually computes at those coordinates.
It does not inspect confirmation data.

For every development allele pair, it traces channels 3 and 175 at the same
variant/acceptor/donor tokens used by the causal experiments. Each encoder
output is separated into the exact residual terms present in the model code:

- at E1, the direct 15-bp DNA convolution and its learned residual update;
- at E2 through E64, the pooled inherited channel, first learned convolution,
  and second learned convolution.

The primary comparison is channel 175 across the six SLC25A48 effects. If E2
really makes this feature more portable than E1, the E2 allele-difference
vectors should be more consistently aligned, and the decomposition should
show whether that consistency comes from max-pooled inheritance or a learned
update. The direct DNA kernels and standardized E2 kernels are also recorded
from the frozen checkpoint, but no raw kernel is treated as a named motif.

The generated plan is
[`encoder_feature_decomposition_plan_v1.json`](encoder_feature_decomposition_plan_v1.json).

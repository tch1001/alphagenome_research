# Interpretation of the one-variant encoder-stage-0 smoke

**Date:** 2026-08-28

**Scope:** development-only tooling smoke; not a ranked v3 result

**Variant:** `BRAF_e14_A117G`
**Component artifact SHA-256:**
`63ebdf4128550570eb04cda5828927eb473a5e60c29ff6fc1ebe283a20cc6dd9`

## Bottom line

The smoke is encouraging evidence that the new live-transfer seam works and
that an allele-dependent signal exists very early in AlphaGenome. It is not
evidence for a U-Net bypass. Encoder stage 0 is immediately after
`DnaEmbedder`, before both storage of the 1-bp skip and pooling into every
coarser encoder/transformer state. Patching it therefore changes the skip route
**and** the continuing trunk route. It is a common upstream cause, not a
branch-isolating intervention.

This example is also unusually easy: the SNV is the negative-strand canonical
donor itself, so `V` and `D` coincide. The selected stage-0 set contains the
donor/SNV base and the acceptor base. Restoring an input-adjacent learned
representation at the scored canonical endpoint can recover classification
evidence without identifying a distributed splice circuit.

## What the numbers establish

The paired logit-margin identities were

```text
REF = 2.537109375
ALT = 3.089843750
DeltaL = 0.552734375

ALT recipient <- REF = 2.781250000
REF recipient <- ALT = 2.992187500
```

Thus the audited recoveries are

```text
REF -> ALT = (2.78125 - 3.08984375) / (2.537109375 - 3.08984375)
           = 0.558303887

ALT -> REF = (2.9921875 - 2.537109375) / (3.08984375 - 2.537109375)
           = 0.823321555

B = 0.558303887
```

The all-false duplicate/repeat audit, baseline identity, self targets, and live
donor-vector checks passed exactly. The reciprocal asymmetry is compatible
with a nonlinear recipient computation; it is not by itself a failure. These
facts validate this one intervention execution. They do not establish a
population effect, spatial specificity, branch exclusivity, or generalization.

## Required evidence before using “skip bypass”

1. **Finish Phase R first.** Run the frozen target audit and unchanged
   72-member v2 residual grid under the logit-margin reducer. The present smoke
   may be retained as a labelled engineering artifact, but it must not rank a
   route. If Phase R passes, stop the wider search and call the result
   readout-dependent. If it fails, rerun the complete predeclared route census;
   do not promote this already-seen component selectively.
2. **Pass target closure.** Patching the complete donor post-GELU 1-bp output
   embedding at both canonical endpoints must reproduce the donor logit-margin
   target bit-for-bit in both directions for every development effect.
3. **Pass route closure.** Jointly transferring the complete transformer input
   to the decoder and all seven live skip tensors at decoder consumption must
   reproduce the donor target. Use complete sequences or prospectively computed
   exact ancestral supports, not only the bins containing `V`, `A`, and `D`.
   Failure means the census is missing an active path.
4. **Intervene after the branch.** Compare `T` (transformer output), `E` (all
   seven skip tensors at decoder consumption), and `T+E`. The 1-bp skip alone
   is `decoder_skip_states` stage 6; `encoder_outputs` stage 0 cannot isolate
   it. Report raw movements, both recoveries, the two-route Shapley accounting,
   and the nonlinear interaction. Leave-one-skip-out and cumulative scale tests
   are useful necessity/localization checks.
5. **Use matched spatial controls.** At each resolution/support, run the frozen
   upstream, downstream, and 32 same-shape random translations. Separate
   `V/D` from `A` where possible and report endpoint-specific margins. This
   example cannot distinguish `V` from `D` because they are the same base.
6. **Apply the full development gate.** Require exact self controls, median
   bidirectional `B >= 0.25` and median `q > 0` in both BRAF and SLC25A48,
   performance above the matched random null, reciprocal directionality, and
   intended-output/effect-neutral specificity on the original 12 effects and
   eight neutral controls. One canonical-site variant is not a gate.

Only robust recovery by `E` after branch isolation, with `T`, `T+E`, closure,
interaction, and matched controls reported, supports the narrow statement that
the tested AlphaGenome logit effect is causally transmitted through U-Net skip
states. Even then, “bypass” means a computational route around the transformer;
it does not identify a spliceosomal, RBP, or biochemical pathway, and it should
not imply that the transformer route is unnecessary when the branch interaction
is material.

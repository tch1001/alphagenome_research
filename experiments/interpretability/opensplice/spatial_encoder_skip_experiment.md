# Spatial encoder-skip experiment

## Question

The completed development cube shows that the broad AlphaGenome splice effect
can be transferred through `E32+E16+E8+E2+E1`, but whole-tensor replacement
changes every position and channel in five encoder skips. This experiment asks
whether that route localizes near the variant, canonical acceptor or canonical
donor.

This is a model-behavior experiment. It does not test or record the operating
system kernel, and it does not access confirmation examples.

## Existing model hook

No new model intervention primitive is required. AlphaGenome's instrumented
route-census graph already accepts a dynamic `SequenceResidualBatchTransfer`
at the decoder skip-consumption seam. It can transfer selected live skip
vectors for several resolutions and positions while keeping tensor shapes and
the compiled graph fixed.

## Design

Use the same 20 development variants, 16,384-bp contexts, checkpoint, splice
classification logit-margin target and six-row reciprocal batch as the prior
experiments. Keep transformer output in its natural recipient state.

Enable only the exploratory mask-110 players:

```text
E32, E16, E8, E2, E1
```

At every enabled resolution, construct four biological supports:

- `V`: the token containing the variant;
- `A`: the token containing the canonical acceptor;
- `D`: the token containing the canonical donor; and
- `S`: the union of acceptor and donor tokens.

Add one token of guard on each side of every contiguous support component.
For every support, include an equal-shape upstream and downstream translation
at least 512 bp away and disjoint from all V/A/D supports. Coordinates are
determined from sequence/exon metadata before inference and stored in
`spatial_encoder_skip_plan_v1.json`.

There are 12 conditions per variant: four supports times intended, upstream
and downstream locations. Each identity and condition is executed twice for
exact repeat checks. The complete design is 520 model applies and one fixed-
shape compilation:

```text
20 variants * 2 identity applies
+ 20 variants * 12 conditions * 2 applies
= 520 applies
```

## Controls

- reciprocal REF-to-ALT and ALT-to-REF transfer;
- ALT-to-ALT and REF-to-REF self controls;
- exact repeated calls;
- exact donor-vector equality at enabled skip positions;
- exact natural/no-op behavior at disabled resolutions and positions; and
- equal-shape, within-variant upstream/downstream spatial controls.

Experimental neutral variants remain secondary specificity controls. They are
not assumed to be AlphaGenome-null because the completed cube shows large
model effects for several BRAF neutrals.

## Analysis

For each intended support and gene, compute median bidirectional bottleneck
recovery `B` and the spatial contrast

```text
q = B_intended - max(B_upstream, B_downstream).
```

A support is spatially promising only if both genes have median intended
`B >= 0.25` and median `q > 0`. Report effect-versus-neutral absolute movement
per gene as a secondary warning, not a rescue criterion.

If no support passes, the whole-skip signal is diffuse or nonspecific at these
supports. If one passes, the next step is causal channel ranking only inside
that spatially localized route. Motif interpretation remains downstream of
channel-level causal evidence.

## Claim boundary

A passing support would localize a development computational route. It would
not by itself identify a motif, RBP, spliceosome step, endogenous necessity or
general biological mechanism.

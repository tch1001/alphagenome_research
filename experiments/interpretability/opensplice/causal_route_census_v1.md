# AlphaGenome sequence causal-route census v1

This experimental path extends the six-row live paired-batch intervention from
the transformer residual stream to every sequence route that can affect the
OpenSplice splice-site classification target. It is opt-in: normal prediction,
checkpoint initialization, and the existing OpenSplice v2 artifacts are
unchanged. No confirmation-exon internals are part of this implementation.

## Frozen batch semantics

Every active census component uses one forward execution with these rows:

| Row | DNA | Role | Live donor |
|---:|---|---|---:|
| 0 | REF | REF baseline | unchanged |
| 1 | ALT | ALT baseline | unchanged |
| 2 | ALT | REF into ALT | 0 |
| 3 | ALT | ALT self-control | 1 |
| 4 | REF | ALT into REF | 1 |
| 5 | REF | REF self-control | 0 |

Donors are gathered from the natural tensor before any recipient is updated.
The dynamic transfer arrays have shape `[stage, 6, selected_position]`.
`paired_six_row_batch_transfer(component_mask)` expands a dynamic
`[stage, selected_position]` mask into this exact mapping. An all-false mask
therefore retains the identical pytree and executable shape at every seam.

## Census seams

Selections are fixed padded arrays. Each trace contains only selected vectors,
not the full activation tensor. Natural and effective trace elements have shape
`[6, selected_position, channel]`; channel width can vary between stages, so
stage traces are returned as fixed-length tuples.

`interpretability.CAUSAL_ROUTE_FAMILIES` exposes the ordered family name,
resolution, channel width, and seam description for all 51 components, so a
separate target reducer can pair a scalar result with an unambiguous route key.

### Encoder outputs

`ENCODER_ROUTE_RESOLUTIONS = (1, 2, 4, 8, 16, 32, 64, 128)` and selection
arrays have shape `[8, R_encoder]`.

| Stage | Resolution | Width | Exact seam |
|---:|---:|---:|---|
| 0 | 1 bp | 768 | after `DnaEmbedder`, before storing the 1-bp skip and pooling |
| 1 | 2 bp | 896 | after down block 0, before storing the skip and pooling |
| 2 | 4 bp | 1,024 | after down block 1, before storing the skip and pooling |
| 3 | 8 bp | 1,152 | after down block 2, before storing the skip and pooling |
| 4 | 16 bp | 1,280 | after down block 3, before storing the skip and pooling |
| 5 | 32 bp | 1,408 | after down block 4, before storing the skip and pooling |
| 6 | 64 bp | 1,536 | after down block 5, before storing the skip and pooling |
| 7 | 128 bp | 1,536 | after final pooling, before organism embedding and transformer |

An encoder intervention at stages 0–6 intentionally affects both the continuing
encoder path and the stored U-Net skip. The separate skip seams below isolate
the skip route at decoder consumption.

### Transformer

All nine blocks, including layers 6–8, retain the existing three residual
seams on `[6, sequence_bp / 128, 1536]`:

1. `pre_attention`
2. `post_attention`
3. `post_mlp`

The route census nests the existing compact pair-bias and per-head traces, but
the complete-route intervention grid uses the live sequence residual seams.

### U-Net skip consumption and decoder outputs

`DECODER_ROUTE_RESOLUTIONS = (64, 32, 16, 8, 4, 2, 1)`. Skip and decoder
selection arrays independently have shapes `[7, R_skip]` and
`[7, R_decoder]`.

| Stage | Resolution | Width | Skip seam | Decoder seam |
|---:|---:|---:|---|---|
| 0 | 64 bp | 1,536 | after skip lookup, before `UpResBlock` | after block output |
| 1 | 32 bp | 1,408 | same | same |
| 2 | 16 bp | 1,280 | same | same |
| 3 | 8 bp | 1,152 | same | same |
| 4 | 4 bp | 1,024 | same | same |
| 5 | 2 bp | 896 | same | same |
| 6 | 1 bp | 768 | same | same |

### Final sequence embeddings

`FINAL_EMBEDDING_ROUTE_RESOLUTIONS = (128, 1)` and selection arrays have shape
`[2, R_final]`.

| Stage | Resolution | Width | Exact seam |
|---:|---:|---:|---|
| 0 | 128 bp | 3,072 | after the 128-bp `OutputEmbedder`, before it is supplied as the 1-bp embedder skip |
| 1 | 1 bp | 1,536 | after the 1-bp `OutputEmbedder`, before prediction heads |

The rank-4 pair embedding is emitted through the original unchanged path. It is
not a sequence-position route and does not feed the splice-site classification
target; a pair-edge census would require a separate paired-coordinate protocol.

## Code entry points

- `interpretability.CausalRouteTraceSelection`: positions and valid masks.
- `interpretability.CausalRouteInterventions`: fixed transfer pytrees.
- `interpretability.no_causal_route_interventions`: all-false identity setup.
- `interpretability.paired_six_row_batch_transfer`: frozen donor semantics.
- `AlphaGenome.forward_trunk_with_route_census`: opt-in trunk instrumentation.
- `dna_model.create_paired_targeted_route_census_apply`: fused paired target
  reducer and compact route trace using the normal checkpoint tree.

The normal `create_model` return tuple and public prediction methods are not
modified. Dense and Pallas-tiled attention both use the existing transformer
implementation; no attention kernel is changed.

## CPU validation

Run instrumentation tests with the platform pinned explicitly so convolution
tests cannot allocate GPU autotuning workspaces:

```bash
JAX_PLATFORMS=cpu \
  ../agvenv/bin/python \
  src/alphagenome_research/model/interpretability_test.py
```

Tests cover parameter/state-tree identity, exact all-false outputs, BF16 live
self transfers at encoder, transformer layers 6–8, decoder/skip and final
embedding seams, frozen six-row donor mapping, and dense/Pallas route-factory
checkpoint compatibility. No GPU census or confirmation inference is run by
these tests.

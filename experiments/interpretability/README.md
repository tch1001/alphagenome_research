# Causal interpretability experiments for AlphaGenome

This directory asks a narrower and more testable question than “can we explain
AlphaGenome?”:

> When a known regulatory variant changes a prediction, which internal
> representations are causally necessary for that change, and do those
> representations localize to the experimentally established regulatory
> element?

The distinction matters. An attribution map can show that AlphaGenome is
sensitive to a sequence position, but it does not show how the effect is
carried through the network. Conversely, patching an activation can establish
a causal statement about the model's computation, but it does not by itself
prove that the model has recovered the literal biochemical pathway. We report
these as separate claims.

## What was added

The experimental API traces and intervenes at four compact internal seams:

1. the 128-bp sequence residual stream before attention, after attention and
   after the MLP in each of the nine transformer layers;
2. each attention head's weighted-value output before the output projection;
3. the learned compact pair-bias edge for a selected 2,048-bp by 2,048-bp
   region; and
4. a fused scalar target reducer for a selected output track and genomic
   interval.

Only selected positions and edges are returned. This is important for
megabase-scale AlphaGenome: exporting all intermediate activations or a full
attention matrix would defeat the memory savings of the tiled-attention
backend.

The normal inference API and checkpoint parameter tree are unchanged. The
interpretability path is opt-in and uses the same restored parameters as
normal AlphaGenome inference.

## First control: the TAL1 neo-enhancer

The initial tooling control is the known T-ALL insertion
`chr1:47239296:C>ACG` (GRCh38). It creates a MYB motif in a neo-enhancer about
7.5 kb from `TAL1`; AlphaGenome's paper already reports the expected increase
in enhancer-associated tracks and `TAL1` expression. That makes it a useful
positive control for the machinery, but not an independent biological
discovery.

The experiment compares REF and ALT sequences in the same 131,072-bp crop and
uses exact activation patching in both directions:

- **denoising:** put the REF activation into the ALT run and measure recovery
  toward REF;
- **noising:** put the ALT activation into the REF run and measure movement
  toward ALT;
- **self controls:** put ALT back into ALT and REF back into REF to measure the
  numerical/intervention-graph floor;
- **negative controls:** patch matched upstream/downstream regions, reversed
  and self pair edges, and unrelated heads; and
- **specificity check:** compare the effect on `TAL1` with the nearby
  `PDZK1IP1` output. This check currently fails to establish gene specificity:
  relative to its own predicted allelic effect, `PDZK1IP1` is also recovered.

The crop is shifted by 959 bp so the three-base insertion sits inside both a
128-bp sequence token and a 2,048-bp pair bin. An earlier centred crop put the
insertion exactly on token/bin boundaries; those earlier JSON files are kept
only as provenance and must not be used for the main claim.

## Current result

The valid shifted run gives a clear model-level localization result:

- ALT increases the selected CD34+ RNA output over the `TAL1` target interval
  by 58.02%;
- replacing the ALT enhancer residual with the REF residual before transformer
  layer 0 gives 101.31% self-control-corrected recovery toward REF; the
  reciprocal ALT-into-REF patch gives 102.16% recovery toward ALT;
- the largest corrected distance-control residual effect anywhere in the
  tower is 3.05%, about 33 times smaller than the early enhancer effect;
- patching one proposed promoter-to-enhancer pair-bias cell is near-null, even
  when combined across heads and layers;
- individual local head-value patches recover at most 2.27% in this scan; and
- broad whole-head ablations can have large effects, but are too global and
  distribution-shifting to identify a local variant circuit on their own.

The strongest supported statement is therefore: **the variant effect is
encoded in a localized early enhancer representation and causally drives the
model's downstream output changes, including its `TAL1` prediction.** This is
an instrumentation positive control, not yet a discovered internal circuit:
replacing the pre-layer-0 representation is close to reverting the variant at
the model-input boundary, and the effect is not `TAL1`-specific. We have not
shown that a particular MYB feature, attention head or pair-bias edge transports
that effect, and we do not claim to have extracted a literal molecular pathway.

The preferred residual/head artifact is
`results/tal1_residual_head_self_controls_v2_131kb_shift959.json`; the matching
pair-edge artifact is `results/tal1_pair_self_controls_131kb_shift959.json`.
See the adjacent audit Markdown files for independent numerical reviews and
the remaining caveats.

## Files to read

Start with these files in order:

1. `src/alphagenome_research/model/interpretability.py` — immutable selections,
   interventions, traces and target reduction.
2. `src/alphagenome_research/model/attention.py` — compact pair-bias and local
   head-value hooks.
3. `src/alphagenome_research/model/model.py` — the nine-layer residual tracing
   and patching path.
4. `src/alphagenome_research/model/dna_model.py` — opt-in apply factories that
   reuse normal checkpoint parameters.
5. `run_tal1_pair_trace.py` — the complete clean/corrupt experiment and its
   controls.
6. `docs/mechanistic_interpretability_landscape.md` — literature, novelty and
   claim boundaries.
7. `docs/interpretability_positive_controls.md` — the frozen validation ladder.

## Reproduce the TAL1 experiment

From the repository root, with the AlphaGenome all-folds checkpoint already in
the Hugging Face cache:

```bash
experiments/interpretability/run_tal1_pair_trace.sh \
  --max-pair-interventions 0 \
  --max-head-ablations 0 \
  --output experiments/interpretability/results/tal1_residual_head_self_controls_v2_131kb_shift959.json
```

The bidirectional, self-controlled pair-bias scan is:

```bash
experiments/interpretability/run_tal1_pair_trace.sh \
  --max-pair-interventions 432 \
  --max-head-ablations 0 \
  --max-residual-interventions 0 \
  --max-local-head-patches 0 \
  --output experiments/interpretability/results/tal1_pair_self_controls_131kb_shift959.json
```

The wrapper selects the workspace environment, unsets a conflicting CUDA
library path, disables JAX preallocation and uses the opt-in Pallas tiled-
attention backend. Use `--dry-run` to validate coordinates and configuration
without loading the model.

Run the focused tests with:

```bash
../agvenv/bin/python -m pytest \
  src/alphagenome_research/model/interpretability_test.py \
  experiments/interpretability/opensplice/select_holdout_test.py
```

## Next validation: OpenSplice

TAL1 is an official AlphaGenome example, so it proves the tooling rather than
novel generalization. The next stage freezes 50 experimentally measured splice
variants across five OpenSplice exons without consulting AlphaGenome scores.
OpenSplice postdates the original AlphaGenome work, although its authors have
since benchmarked AlphaGenome, so it is a temporal holdout rather than a fully
model-blind dataset.

The source-pinned data plan and deterministic selector are under
`opensplice/`. Large barcode/read artifacts and existing predictor-result
tables are explicitly excluded. The benchmark will first require the correct
output direction, then test whether exact internal patches recover known
splice-regulatory effects better than matched controls.

## Claim boundary

These experiments can support statements about causal components of
AlphaGenome's computation. Biological claims require independent experimental
measurements, held-out loci and matched controls. In particular:

- parallel output heads are not a biochemical chain;
- a learned pair bias is not an attention probability or a chromatin contact;
- a decoded or correlated feature is not necessarily model-causal; and
- one famous locus is a positive control, not evidence of generality.

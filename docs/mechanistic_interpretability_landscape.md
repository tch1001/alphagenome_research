# Mechanistic interpretability for AlphaGenome: landscape and research plan

Last reviewed: 27 August 2026

## Executive conclusion

The most defensible research direction is **causal tracing of an experimentally
validated cis-regulatory effect through AlphaGenome's internal computation**.
The distinctive target is the path from sequence representations, through the
learned pair representation and pair-derived attention bias, to a target-gene
prediction.

The project should not be framed as extracting a complete molecular or chemical
pathway. AlphaGenome is a sequence-to-function model: it predicts many molecular
measurements from DNA, but most output heads read the same shared embeddings in
parallel. A predicted change in H3K27ac therefore does not computationally cause
a predicted change in RNA abundance. The model can contain a computational
circuit that agrees with a known cis-regulatory mechanism without containing the
biological pathway itself.

The proposed proof of concept is:

1. choose mechanisms already established by perturbation experiments;
2. verify that AlphaGenome predicts the direction of each perturbation;
3. use internal interventions to recover the sequence positions, layers,
   attention components, and pair edges that mediate the model's prediction;
4. freeze the discovery procedure and evaluate it on held-out mechanisms; and
5. only then apply it to unresolved loci to produce prospective experimental
   hypotheses.

The publishable contribution is not “we plotted attention” or “we trained a
sparse autoencoder.” A stronger description is:

> Causal, memory-efficient circuit tracing in a megabase-scale supervised
> sequence-to-function model, including interventions on AlphaGenome's learned
> pair-biased enhancer–promoter communication, benchmarked against
> experimentally established cis-regulatory mechanisms.

## What AlphaGenome can and cannot represent

AlphaGenome maps DNA sequence and organism identity to functional-genomics
tracks. Its scope includes transcription, accessibility, TF binding, histone
marks, contact maps, and splicing, but not dynamic protein concentrations,
receptor activation, kinase cascades, or metabolite flux. Suitable biological
objects are therefore cis-regulatory and splicing mechanisms such as:

```text
motif -> regulatory element -> target promoter -> predicted transcription

sequence element -> donor/acceptor choice -> predicted splice junction
```

Complete signalling or metabolic pathways such as the following are outside the
model's causal graph:

```text
ligand -> receptor -> kinase cascade -> TF activation -> metabolism
```

### The relevant AlphaGenome computational graph

The official implementation has a multi-resolution U-Net-style encoder and
decoder around nine Transformer blocks. At 1 Mb context, the Transformer works
on 8,192 sequence positions at 128-bp resolution. On even-numbered blocks, the
sequence state updates a 512 x 512 x 128 pair state at 2,048-bp resolution. The
pair state is projected to eight attention-head biases and each coarse cell is
expanded over its corresponding 16 x 16 sequence-token block.

In simplified form:

```text
DNA
  |
  v
multi-resolution sequence encoder and skip states
  |
  v
128-bp sequence residual x <--------------------+
  |                                               |
  +-> sequence-to-pair -> pair update -> pair_x --+
                              |                   pair-derived bias
                              +-> contact head          |
                                                       v
                         MHA(Q, shared K/V, pair bias) -> MLP
                                      |
                                      v
                              sequence decoder
                                      |
                                      v
                         parallel task-specific heads
```

This creates a genuine internal causal route from sequence to pair state to
attention bias and back to the sequence state. It is the most AlphaGenome-
specific object for mechanistic investigation.

There are several important architectural qualifications:

- The sequence attention has eight query heads but shared key and value
  projections. “Head-specific key” and “head-specific value” claims would be
  architecturally incorrect.
- The 1-bp decoder receives encoder skip states. Local output effects can bypass
  the Transformer tower, so the skip route must be included in causal tracing.
- Standard output heads independently transform shared embeddings. Predicted
  accessibility, histone, contact, and expression tracks are not sequential
  causal stages inside the model.
- A pair-state feature is not automatically a chromatin contact. It becomes
  evidence for enhancer–promoter communication only if an intervention on the
  relevant pair edge changes the predefined target prediction and survives
  matched controls.

The architecture claims above can be checked in the official
[`model.py`](https://github.com/google-deepmind/alphagenome_research/blob/main/src/alphagenome_research/model/model.py)
and
[`attention.py`](https://github.com/google-deepmind/alphagenome_research/blob/main/src/alphagenome_research/model/attention.py).

## Three claim levels that must remain separate

| Claim level | Example evidence | What it establishes |
| --- | --- | --- |
| Input attribution | ISM, gradients, motif deletion, CREME necessity test | A sequence change is associated with a changed model prediction. |
| Model mechanism | Activation replacement, pair-edge intervention, component ablation and rescue | A particular internal component causally contributes to the prediction under the tested intervention. |
| Biological mechanism | Base editing, CRISPRi, MPRA, minigene, allele-specific or other orthogonal perturbation data | The biological system responds consistently with the model-derived hypothesis. |

None of these levels implies the next. A linearly decodable motif need not be
used by the model. A model-causal circuit may be a shortcut or training-data
artefact. Agreement with an assay used as a training target is weaker evidence
than agreement with an independent perturbation experiment.

The central reporting distinction should be:

> Activation patching identifies how AlphaGenome computed a prediction. Only
> independent experimental agreement supports the claim that the computation
> captured a biological mechanism.

## Prior work and novelty boundary

### AlphaGenome output interpretation and ISM

The AlphaGenome paper already uses multimodal output differences and in silico
mutagenesis to interpret pathogenic variants. Its TAL1 example shows that the
chr1:47239296 C>ACG neo-enhancer mutation creates a MYB motif and increases
predicted TAL1 expression about 7.5 kb away, while altering several regulatory
tracks. This is an excellent infrastructure smoke test because the expected
input feature and output are known. Reproducing the output-level result is not
novel; tracing the internal route would be the new experiment.

Primary source: [Avsec et al., *Nature* 2026](https://www.nature.com/articles/s41586-025-10014-0).

### CREME

CREME treats a sequence-to-function model as an in silico assay and performs
context, context-swap, necessity, sufficiency, distance, multiplicity, and
higher-order interaction tests. Applied to Enformer, it identified model-
predicted enhancers, silencers, and interacting regulatory elements. These are
valuable input-response experiments, but they do not identify or intervene on
internal neural circuits. A CREME port to AlphaGenome would be useful baseline
work rather than the primary novelty.

Primary source: [Toneyan and Koo, *Nature Genetics* 2024](https://www.nature.com/articles/s41588-024-01923-3).
Code: [p-koo/creme-nn](https://github.com/p-koo/creme-nn).

### AlphaGenome motif ablation

Liang et al. nominate C/EBPbeta motifs, delete individual motif instances in
AlphaGenome inputs, compare local predicted H3K27ac changes with length-matched
sham deletions, and compare direction with CEBPB-knockout CUT&RUN in a related
multiple-myeloma cell line. This establishes that calibrated AlphaGenome input
ablation and experimental comparison have already been done. It does not
inspect internal activations.

The study also demonstrates why validation must be cautious: effects were
context-dependent, at least one motif reversed direction, and global TF knockout
is not equivalent to editing a single motif. Internal patching should therefore
use alignment-preserving substitutions and, where possible, validation against
motif-specific editing rather than a global TF knockout.

Primary source: [Liang et al., *Briefings in Bioinformatics* 2026](https://academic.oup.com/bib/article/27/4/bbag422/8753614).

### Sparse autoencoders on Borzoi

Korsakova and Kelley trained TopK sparse autoencoders on Borzoi's first four
convolutional layers. The discovered features align with TF/RBP motifs,
transposable elements, and other annotations. This is strong precedent for
sparse feature discovery in supervised regulatory-genomics models.

It does not establish long-range internal circuits. The studied receptive fields
were only 16–31 bp, and replacing activations by SAE reconstructions preserved
output tracks with correlations of roughly 0.59–0.72 rather than exactly. A
straight AlphaGenome SAE port with MEME/TomTom annotation would therefore be
incremental. AlphaGenome work needs a target-output intervention, deeper or pair
representations, and causal controls.

Primary source: [Korsakova and Kelley, NeurIPS FM4LS workshop 2025](https://openreview.net/pdf?id=AlLZnZX01x).
Code: [calico/sae-borzoi](https://github.com/calico/sae-borzoi).

### Input-space surrogate and motif-grammar methods

Global Importance Analysis quantifies population-level effects of inserted
sequence patterns. SQUID fits interpretable additive or pairwise surrogate
models to model-generated mutational libraries and can recover motif effects and
epistasis while accounting for global nonlinearities. Both are strong baselines
for regulatory grammar, but neither reverse-engineers an internal AlphaGenome
circuit.

Primary sources:

- [Koo et al., *PLOS Computational Biology* 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8118286/)
- [Seitz et al., *Nature Machine Intelligence* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11823438/)

### Genomic activation patching and causal dictionary learning

Activation patching is no longer new in genomic models generally. The public
MINTS project combines probes, QK/OV analysis, attention enrichment, activation
patching, and SAEs for DNABERT-2 and Nucleotide Transformer. Its strict CTCF
motif-detector result is negative, and its strongest promoter-TATA patch
over-restores the target, illustrating why attention or probe evidence alone can
produce overconfident stories.

A recent preprint also trains sparse dictionaries on DNABERT-2 and Nucleotide
Transformer, controls motif associations for sequence composition, and ablates
dictionary directions to test their effect on masked-token predictions. Thus,
“causal SAE validation in a genomic language model” is not a safe novelty claim.

Relevant sources:

- [MINTS repository and manuscript](https://github.com/ArjunCodess/MINTS) — public workshop-style work, not treated here as a peer-reviewed result
- [Ali, arXiv:2607.19618, 2026](https://arxiv.org/abs/2607.19618) — preprint

### Defensible novelty

The following would be incremental on their own:

- plotting AlphaGenome attention;
- running gradients, DeepSHAP-like attribution, or ISM;
- applying CREME tests without inspecting internal computation;
- deleting motifs and comparing output tracks;
- training an SAE and matching its features to motif databases; or
- claiming that a probe's label decodability identifies a biological circuit.

The strongest novelty boundary is instead the combination of:

- an experimentally supervised, megabase-scale sequence-to-function model;
- exact interventions on multi-resolution sequence and pair states;
- enhancer-to-promoter pair-bias interventions without materializing dense
  attention;
- bidirectional necessity and rescue tests of a proposed internal path; and
- systematic recovery of held-out experimentally established regulatory
  mechanisms rather than a single case study.

Absence-of-prior-work claims should remain qualified. A literature search can
support “we found no prior public AlphaGenome pair-edge causal tracing study,”
not an unqualified “first mechanistic interpretability study in genomics.”

## Recommended causal-patching workflow

### 1. Define the biological contrast and output metric before inspecting internals

For every example, define:

- a clean and corrupted sequence differing in one interpretable feature;
- the relevant AlphaGenome output track and biosample;
- a fixed genomic aggregation window; and
- the expected experimental direction.

Use length-preserving motif substitutions or carefully matched point mutations.
Deleting a motif shifts all downstream coordinates and makes token-wise
activation replacement ambiguous. For an insertion such as the TAL1 neo-
enhancer, compare the oncogenic inserted sequence with a same-length,
motif-destroyed inserted control, or begin with a known SNV. The official indel
stitching utilities may help output alignment, but they do not remove the causal
ambiguity of patching misaligned internal tokens.

The primary output-space metric should be biologically meaningful, such as a
predefined RNA or CAGE track sum at the target gene. A pre-softplus target-head
logit should also be captured as a robustness metric because it is closer to
linear in the shared embedding. Do not select the window after viewing patching
results.

Only attempt circuit recovery where AlphaGenome predicts the known direction
with a sufficiently large clean–corrupt difference. A wrong or insensitive
prediction has no successful internal mechanism to recover and should be
reported as a model failure.

### 2. Establish input-level baselines

Run input-gradient, gradient-times-input, ISM, and calibrated sham perturbations.
For distal regulation, add CREME-style necessity and sufficiency tests. These
baselines answer which sequence matters and provide a comparison for whether
internal circuit measures improve target-gene recovery.

### 3. Perform coarse activation patching

Let `m_clean` and `m_corrupt` be the predefined scalar model scores. In a
denoising intervention, run the corrupted sequence while replacing a selected
activation with its clean value:

```text
denoising recovery =
    (m_patched_into_corrupt - m_corrupt)
    / (m_clean - m_corrupt)
```

In the reverse noising intervention, run the clean sequence while replacing the
activation with its corrupted value:

```text
noising destruction =
    (m_clean - m_patched_into_clean)
    / (m_clean - m_corrupt)
```

Sweep coarse regions before individual features:

- every encoder scale and decoder skip state;
- the 128-bp residual before and after MHA and MLP in every Transformer block;
- enhancer, promoter, target gene, and matched control positions; and
- the pair state after each of its five update stages.

Report raw score changes as well as normalized recovery. Exclude or flag
near-zero denominators using a threshold fixed in advance. Recovery above 1 is
over-restoration, not automatically stronger evidence; it may reflect nonlinear
interactions or an off-distribution patched state. Run both denoising and noising
because they test different counterfactuals and can disagree.

### 4. Decompose AlphaGenome-specific components

After coarse localization, test:

- MHA output for each query head before the output projection;
- query activations separately from the shared key and value activations;
- MLP outputs;
- final projected pair bias for each layer, head, 2-kb query region, and 2-kb
  key region; and
- individual or grouped pair-state features before bias projection.

For enhancer-to-promoter communication, the primary directed edge is a promoter
query attending to an enhancer key. Test the reverse edge as a control rather
than assuming pair symmetry. Compare:

1. the full attention logit;
2. the sequence-derived QK contribution;
3. the pair-derived bias contribution; and
4. the resulting head output written to the promoter residual.

Intervening on the coarse projected bias is preferable to building its dense
expanded form. At full context, a bfloat16 512 x 512 x 8 projected bias is about
4 MiB, whereas its 8 x 8,192 x 8,192 expansion is about 1 GiB before other
attention intermediates. The tiled backend can accept sparse override masks and
values for selected coarse edges.

### 5. Test an explicit path, not only isolated nodes

A candidate long-range path might be:

```text
motif-containing enhancer token
  -> sequence-to-pair update
  -> enhancer/promoter pair state
  -> pair-derived promoter-to-enhancer attention bias
  -> promoter-query head output
  -> promoter residual
  -> target CAGE/RNA head
```

Test each edge and the joint path. A useful circuit should show:

- **necessity:** corrupted activations inserted into the clean run reduce the
  target effect;
- **sufficiency or rescue:** clean activations inserted into the corrupted run
  restore the effect;
- **specificity:** the intervention affects the intended target more than
  matched nearby genes or unrelated tracks;
- **minimality:** removing candidate components from the joint set lowers
  recovery; and
- **replication:** the result survives multiple corruptions, loci, and model
  checkpoints.

### 6. Scale localization, then verify exactly

Exhaustively patching every position and component is expensive. Gradient times
clean–corrupt activation difference, or Attribution Patching, can rank candidate
components with one or a small number of backward passes. Exact activation
patching must verify the leading candidates because linear approximations can
miss effects through cancellation and saturated attention softmaxes.

AtP* provides diagnostics and mitigations for these failures:
[Kramár et al., arXiv:2403.00745](https://arxiv.org/abs/2403.00745).
Practical patching choices and metric sensitivity are discussed by
[Heimersheim and Nanda, arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
and
[Zhang and Nanda, arXiv:2309.16042](https://arxiv.org/abs/2309.16042).

If the tiled Pallas path has no gradient or custom VJP, begin with vectorized
exact patches over coarse regions and layers. Gradient screening should not
silently fall back to a numerically different implementation.

### 7. Add sparse features only when needed

If causal information remains distributed across raw channels, train a sparse
autoencoder on selected residual or pair states. Require:

- train/validation separation by genomic fold;
- activation and target-output reconstruction fidelity;
- live-feature and sparsity diagnostics;
- motif annotation with GC-, repeat-, and activity-matched controls;
- stability across seeds and widths; and
- direct feature ablation/activation effects on the predefined target output.

MEME or TomTom similarity names a candidate feature; it does not show that the
model uses the feature. An SAE is a lossy learned model layered on top of
AlphaGenome and must not be treated as privileged ground truth.

### 8. Freeze the method and evaluate held-out recovery

A case study is useful for debugging but not for demonstrating generality.
Separate examples into development and held-out sets, and fix thresholds and
ranking rules before evaluating the held-out mechanisms. Useful endpoints are:

- true regulatory-element rank among distance- and activity-matched elements;
- true target-gene rank among nearby genes;
- recovery of the known TF-family motif;
- activating/repressive direction accuracy;
- correct biosample or tissue ranking;
- patch recovery and destruction fractions;
- specificity relative to matched negative edges; and
- replication across fold-specific checkpoints.

Use fold-specific AlphaGenome models where the evaluated locus was held out from
training when possible, and replicate the final result in the distilled or
all-fold model. This reduces, but does not eliminate, memorization concerns.

## Required controls

| Risk | Required control |
| --- | --- |
| Coordinate shift | Prefer equal-length substitutions; explicitly align every tensor position for indels. |
| Generic sequence disruption | Multiple same-length motif-destroying substitutions, dinucleotide-preserving shuffles, and matched sham edits. |
| GC/repeat confounding | GC-, repeat-, accessibility-, and motif-score-matched negatives. |
| Attention reflects distance | Same-distance pair edges, reverse-direction edges, and edges to matched non-target promoters. |
| Arbitrary component sensitivity | Random layers, heads, positions, pair features, and MLP components. |
| Off-distribution ablation | Compare clean/corrupt replacement with matched-activation resampling; do not rely on zero ablation alone. |
| Donor-specific patch artefact | Patch activations from several matched donor loci, not only one convenient example. |
| Small denominator | Predefine a minimum clean–corrupt effect; report raw changes and denominator failures. |
| Corruption-specific result | Repeat with several biologically equivalent corruptions and both patching directions. |
| Strand artefact | Reverse-complement consistency tests with correctly transformed coordinates and output strands. |
| Checkpoint instability | Replicate across appropriate fold-specific models and the distilled/all-fold checkpoint. |
| Output cherry-picking | Predefine track, biosample, strand, resolution, and genomic aggregation window. |
| Training-label circularity | Prefer CRISPRi, base editing, MPRA, allele-specific, minigene, or later perturbation datasets over simple agreement with a track used for training. |
| Global versus local biology | Prefer motif/base editing for motif-specific claims; treat whole-TF knockout as a different perturbation with secondary effects. |
| Multiple testing | Hold out mechanisms, calibrate against matched nulls, correct feature-level tests, and bootstrap over loci rather than individual bases. |

## Recommended proof-of-concept sequence

### Tooling smoke test: TAL1

The TAL1 MYB neo-enhancer is the best first engineering target because
AlphaGenome's output response and the known motif have already been documented.
Success means the tooling can recover an expected motif-to-target effect and
localize a plausible internal route. It is not independent biological discovery
and should not be the headline result.

For alignment-safe patching, use the oncogenic inserted allele versus one or
more same-length inserted sequences that destroy the MYB motif, or select one of
the known TAL1 SNVs. Confirm that the model still predicts the expected target
expression contrast before inspecting internals.

### Method validation: local promoter or splice effects

Length-preserving promoter-motif and splice-site variants provide sharply
defined positive controls. They can validate capture, replacement, metrics, and
negative controls before long-range pair-edge experiments. They also test
whether the workflow correctly finds a predominantly local route rather than
forcing every example into a pair-attention story.

### Primary study: held-out enhancer-to-gene benchmark

Assemble experimentally tested enhancer–gene pairs from CRISPRi or base-editing
data, with positives and distance/activity-matched negatives in biosamples that
AlphaGenome supports. Use a development subset to define the procedure and a
held-out subset to measure recovery. The most persuasive result would show that
pair-edge or circuit scores identify the correct enhancer and target gene better
than distance, ISM, input attribution, raw attention, and output-only baselines.

### Prospective phase

Only after held-out recovery should the frozen procedure be applied to unresolved
variants or candidate enhancers. A useful prospective prediction should state:

```text
edit this motif
in this regulatory element and biosample,
measure this gene or junction,
and expect this direction of change.
```

That is the point at which mechanistic interpretation can plausibly reduce the
experimental search space.

## Interpretation of possible outcomes

| Model circuit | Experiment | Interpretation |
| --- | --- | --- |
| Coherent and causal | Concordant | AlphaGenome captured a computation aligned with the tested biological mechanism. |
| Coherent and causal | Discordant | AlphaGenome learned a model-specific shortcut, training artefact, or incomplete mechanism. |
| No stable circuit | Correct output | The effect may be distributed, redundant, reached through an unexpected route, or exposed by an inadequate intervention. |
| Incorrect output | Discordant | Ordinary model failure; mechanistic interpretation cannot repair it. |

Negative results remain scientifically useful. Finding that pair bias mostly
encodes distance, that raw attention is not causally important, or that distal
effects use a different route would constrain how AlphaGenome predictions
should be interpreted. Such conclusions are stronger than an attractive but
uncontrolled attention visualization.

## Source index

- [AlphaGenome paper](https://www.nature.com/articles/s41586-025-10014-0)
- [Official AlphaGenome research code](https://github.com/google-deepmind/alphagenome_research)
- [CREME paper](https://www.nature.com/articles/s41588-024-01923-3)
- [CREME code](https://github.com/p-koo/creme-nn)
- [AlphaGenome motif-ablation paper](https://academic.oup.com/bib/article/27/4/bbag422/8753614)
- [SAE-Borzoi paper](https://openreview.net/pdf?id=AlLZnZX01x)
- [SAE-Borzoi code](https://github.com/calico/sae-borzoi)
- [SQUID paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11823438/)
- [Global Importance Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8118286/)
- [MINTS public repository](https://github.com/ArjunCodess/MINTS)
- [Causal dictionary learning for genomic language models](https://arxiv.org/abs/2607.19618)
- [AtP*](https://arxiv.org/abs/2403.00745)
- [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
- [Activation-patching metrics and methods](https://arxiv.org/abs/2309.16042)

# SLC25A48 channel-175 saturation mutagenesis

This frozen development-only experiment tests the weight-derived acceptor
hypothesis with controlled sequence edits. Every possible single-nucleotide
substitution from -20 through +20 bp around the SLC25A48 exon-8 acceptor is
evaluated against the same GRCh38 sequence context.

For each edit the experiment measures E1 and E2 channel 175 across a compact
local window, as well as AlphaGenome's separate acceptor and donor
class-minus-padding splice logits. A reference row is repeated in every
six-sequence batch, and unused rows in the final batch are exact reference
padding controls. Confirmation examples remain sealed.

The plan is [`slc25a48_channel175_ism_plan_v1.json`](slc25a48_channel175_ism_plan_v1.json).

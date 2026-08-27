# Canonical TAL1 results

Two result artifacts make up the current TAL1 tooling-control result. They use
the same shifted 131,072-bp crop, sequence hashes, all-folds checkpoint,
CD34+ RNA track and scalar `TAL1` interval target.

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `tal1_residual_head_self_controls_v2_131kb_shift959.json` | Bidirectional residual and local-head patches with REF→REF and ALT→ALT controls | `0c34029b28d2d7587b2cf4dc366e351d479e6b0774251a5395fdf9489b745282` |
| `tal1_pair_self_controls_131kb_shift959.json` | All 432 directional compact pair-edge cells with bidirectional same-allele controls | `2a64e204b19195d60020f737bb4294266698fbb4a6e65fa3bfeb936bdc5f22a4` |

The independently recomputed reports are
`tal1_residual_head_self_controls_v2_audit.md` and
`tal1_pair_self_controls_audit.md`. The artifact-integrity test independently
recomputes every stored corrected recovery in the canonical JSON files.

## Result in one paragraph

The selected ALT sequence raises the predicted CD34+ RNA total over the TAL1
target by about 58%. Swapping the five enhancer-token residuals before layer 0
gives roughly 101% corrected recovery in both directions, while every matched
distance-control effect is at most about 3%. This establishes a localized
model-causal dependence. It does not identify a selective long-range circuit:
local single-head patches explain at most about 2.3%; the nominated compact
pair-bias edge explains less than 0.5% in either direction; and the same early
residual transplant also swaps the nearby PDZK1IP1 prediction. The appropriate
label is **validated intervention tooling and localized early residual
dependence; unresolved feature, route and gene specificity**.

## Reproduce

Commands are recorded in the parent `README.md`. Within-process REF and ALT
identity repeats are exact in both artifacts. Cross-process totals can move at
the sub-percent level under the experimental BF16/Pallas execution path, so
all recovery values use matched baselines and self controls from the same
process.

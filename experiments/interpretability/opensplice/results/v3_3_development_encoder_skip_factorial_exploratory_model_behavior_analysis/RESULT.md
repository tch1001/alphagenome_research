# Exploratory AlphaGenome encoder-skip model-behavior analysis

This CPU-only analysis independently reconstructed the complete 20-by-256
development coalition cube from raw relevant/padding logits. It made no
model call and read neither confirmation nor incomplete OOD-anchor artifacts.

## Coarse route anchors

| Coalition | BRAF median B | SLC25A48 median B |
|---|---:|---:|
| empty | -0.00000 | 0.00000 |
| all encoder skips | 0.71908 | 0.93746 |
| T only | 0.22746 | 0.05552 |
| T + all skips | 1.00000 | 1.00000 |

## Resolution-level Shapley profile

Median normalized contributions conditional on natural T:

| Skip | BRAF | SLC25A48 |
|---|---:|---:|
| E64 | 0.07876 | 0.03153 |
| E32 | 0.21743 | 0.10381 |
| E16 | 0.18651 | 0.04182 |
| E8 | 0.07276 | 0.09452 |
| E4 | 0.02640 | 0.08200 |
| E2 | 0.04356 | 0.14949 |
| E1 | 0.03106 | 0.42005 |

## Exploratory resolution candidate

The deterministic historical rule selects **E32 + E16 + E8 + E2 + E1** (mask `110`).

| Gene | Median B | Retention versus all skips |
|---|---:|---:|
| BRAF | 0.62056 | 0.86299 |
| SLC25A48 | 0.77170 | 0.82318 |

## Biological-specificity warning

| Gene | Effect median absolute movement | Neutral median | Effect > neutral |
|---|---:|---:|:---:|
| BRAF | 0.57324 | 2.67822 | false |
| SLC25A48 | 4.38184 | 0.06445 | true |

The candidate fails the effect-versus-neutral comparison in BRAF. It is
therefore evidence about a broad computational route, not a biologically
specific splice mechanism.

## Interpretation

- BRAF is weighted toward E32 and E16.
- SLC25A48 is weighted toward E1 and E2, with E1 the largest contributor.
- No single resolution explains both exons; the shared route is multiscale.
- E64 and E4 can be omitted together while retaining most all-skip recovery.
- The next experiment should spatially localize the five retained skips and
  add better behavior-matched controls before any channel or motif claim.

These are development-only causal-computation observations. They do not
identify an RBP, motif, spliceosome step, endogenous necessity or mechanism
that generalizes beyond the two analyzed exons.

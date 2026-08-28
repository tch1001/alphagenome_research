# OpenSplice v3.3.2 OOD-sidecar amendment

Status: **prospective and docs-only**. This amendment, a separate sidecar
runner, its tests and a machine-readable freeze must be committed and
hash-bound before any v3.3.2 model call. No implementation or execution is
authorized by this document alone.

The scientific protocol remains
`encoder_skip_localization_protocol_v3_3.md`, SHA-256
`85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0`,
except for the single host-side OOD trace assertion defined below. All
cohorts, anchors, donor mappings, model inputs, targets, thresholds, estimands,
gates and claim limits remain frozen.

Confirmation status: model outputs, activations and interventions remain
unopened. The previously disclosed later-exon metadata/label exposure remains
in scope. This amendment was written from source semantics, artifact hashes,
status fields and exactness booleans only; no v3.3 endpoint logit, movement,
recovery, Shapley value, interaction, ranking or nomination was inspected.

## 1. Consumed v3.3 attempt and exact failure boundary

The v3.3 model attempt is consumed and immutable at:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_development_encoder_skip_factorial_one_shot`

The v3.3.2 bootstrap and offline analyzer must independently verify these
bindings before accepting any old or new raw record:

| Artifact | Exact binding |
|---|---|
| Original model-run Git commit | `9e8324434a3c0b2d5c53d209ad04c2d88b99f4bc` |
| Original protocol SHA-256 | `85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0` |
| Original freeze SHA-256 | `98860ed4e60c427a76ac05879d800f36b65c10a310f4b2b981819fa48af767b3` |
| Original runner SHA-256 | `56eef2cc5b87f3ff9ad5837d19b891b98bbb4a7e126e20713ea9bc8b21c409c5` |
| Original analyzer SHA-256 | `0a65a27a5c424bb9dddacb5475e02d28a0999fc7cd593d4dd63ba4be06c39a46` |
| Original analyzer-test SHA-256 | `d027f73fb07682e8cb54d46653a5bd9aa900aaeac433b1748dbdf4886c6d5034` |
| Original `model.py` SHA-256 | `7aee357d776f1f10f9ef04b1602103496ad543d89f49d5e59af459afca217ea1` |
| Original `interpretability.py` SHA-256 | `d00a4dd8a4e62c2d8a7d583a74cbf5632121f98892e901c7f8927539ee156500` |
| `ATTEMPT_STARTED.json` SHA-256 | `b74081fd0cbd1c8d6ec5445b3b71661f40ac4d47dd77fd2d9bd3675b4cf9c3c3` |
| `RUN_COMPLETE.json` SHA-256 | `ddc8350361ae9091ac47878a2c2d043897c46ef1d7722a401869d8d69e4be463` |
| `RAW_MANIFEST.json` SHA-256 | `6c50c86153fbce5136ed99205ca4726f87a00ef56216f1205dba5c25d3d27cd7` |
| Raw artifact count / tree SHA-256 | `5142` / `e7376062ce31090b349e88b91bd41700caf4e690511c15993e50f2bd0d47f770` |
| Whole immutable run file count / tree SHA-256 | `5158` / `2d8125fe6d13773ba9621e527870361b6a195c516c5b4f044c7dad64c9310aaa` |
| Compiler file count / tree SHA-256 | `8` / `9a03dcbc9d439cb9bf197941af3bbdb3e6bda067cf661b90de6d7eab1f4d87eb` |
| All three import-provenance SHA-256 values | `64a5538499e5b06e29cb506a2b08585bb002b3766bd1be210d1a568b9ec5110e` |
| `PROTOBUF_PROVENANCE.json` SHA-256 | `2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8` |
| `TARGET_ELIGIBILITY.json` SHA-256 | `b216692d8028faab09b5f6590e3e68d9c8805d3c715ddedd99e8956019cedcf0` |
| Device-preflight SHA-256 | `b983c7f4910ef4fc5f68bc72486552063f4497f90bba64497eb29a09d3d1809d` |
| Preflight stdout / stderr SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` / same |

The tree digest framing is fixed: sort the exact file paths; for each file,
append its UTF-8 POSIX path relative to the run root, one NUL byte, and the 32
raw bytes represented by that file's SHA-256; then SHA-256 the concatenation.
There are no directory, size, newline or hexadecimal-text rows. The whole-run
tree contains exactly 5,142 raw files, eight top-level files and eight compiler
files. The compiler tree contains those exact eight files expressed relative
to the run root as `compiler/...`.

The original 61-file committed bundle must also be revalidated from every
`file_sha256` entry in `encoder_skip_factorial_v3_3_freeze.json`; the table
above is not a replacement for that per-file validation. The checkpoint,
reference object, development manifests, OpenSplice source pins, runtime
versions, mixed-precision policy, 22+4 upstream-module split, protobuf
bindings and compiler environment remain exactly those frozen by that file.

`RUN_COMPLETE.json` records `controlled_stop` with stop reason
`ood_tooling_failure` and the following exact status-only counts:

| Family | Records | Invalid |
|---|---:|---:|
| Identities | `20` | `0` |
| Six-row coalitions | `5120` | `0` |
| OOD anchors | `2` | `1` |
| All scientific records | `5142` | — |

It also records 12 eligible effects, all effects target-eligible, all neutrals
retained, ID 0 no-op and ID 255 closure true for all 20 variants, 10,288 model
applies, two compiled executables and zero confirmation model calls. These are
structural predicates, not permission to inspect the frozen scientific values.

The only two original OOD records are bound as follows:

| Record | SHA-256 | Frozen predicate |
|---|---|---|
| `raw/ood_anchors/000_BRAF_e14_A117G/000.json` | `97917119318b21e679bb0c2d11f40937f1e0d8b2ec41c20275dc9f9305d0e680` | `status=complete`, `execution_index=5140`, `checks.passed=true`; every applicable exactness boolean is true, `normalization_computed=false`, and the two ID-255 flags are false only because this is ID 0 |
| `raw/ood_anchors/000_BRAF_e14_A117G/127.json` | `4245778e3c5edca8075b8e0a703cea470d6567e8083c369370b735c390397998` | `status=invalid`, `execution_index=5141`, `checks=null`, failure type `ValueError`, exact message `Eight-row natural route differs between calls: natural_final_embeddings.` |

Both records bind the original eight-row executable fingerprint
`12283496a0987eec942bd8f9b7bbb86a9d9d676b13bee1956b30da933a4e9967`.
For the failed recipient, the original ID-0 and ID-127 linked six-row
coalition hashes are respectively
`1a0a512b01ba2c8153c78f6f77b7dd4d1400e5c8bf66c1886bf895a714cc4882`
and
`d2d15fa7d1ee19738b2652f90440da32002e3f30d3d945f8732935fcb1517498`.
The original run stopped immediately after persisting the invalid ID-127
record; the remaining 78 OOD anchors do not exist. The original directory may
not be resumed, completed in place, deleted, normalized or rewritten.

## 2. v3.3.1 analyzer amendment status

The committed v3.3.1 controlled-stop analyzer amendment is
`encoder_skip_analysis_amendment_v3_3_1.md`, commit
`186c25f`, SHA-256
`37e23b251f53ab87bae99b63024a381c367ce33bbc950a2227b3267fbc9668d1`.
At the prospective freeze point for this document it is **unconsumed**: its
attempt directory and the original frozen v3.3 analysis destination are
absent. It permits only a CPU-side seven-provenance-role versus two-generated-
output classification repair and requires a final
`controlled_stop_ood_tooling_failure` result with no Shapley calculation or
nomination.

The OOD sidecar neither supersedes that amendment nor depends on a scientific
value from it. Its freeze must record whether v3.3.1 is still unconsumed or has
subsequently produced a terminal artifact, with exact path and SHA-256. It may
inspect only status, provenance, hashes and structural booleans from that
attempt. A changed v3.3.1 status is not permission to modify this sidecar's
cohort or order.

## 3. Exact infrastructure defect

The eight rows have these frozen roles and natural identities:

```text
row 0  recipient REF baseline       natural row 0
row 1  recipient ALT baseline       natural row 1
row 2  recipient ALT <- donor       natural row 1
row 3  recipient ALT self           natural row 1
row 4  recipient REF <- donor       natural row 0
row 5  recipient REF self           natural row 0
row 6  unrelated REF donor source   natural row 6
row 7  unrelated ALT donor source   natural row 7

intended donor rows  [0,1,0,1,1,0,6,7]
unrelated donor rows [0,1,6,1,7,0,6,7]
```

The two calls therefore have identical intervention semantics on exactly rows
`[0,1,3,5,6,7]`. Rows 2 and 4 are active recipients whose donor changes.

In the frozen `model.py`, `natural_final_embeddings` is gathered from the
one-base embeddings **after** the effective transformer-output transfer and
the effective encoder-skip transfers have passed through the decoder, and
immediately **before** the disabled final-embedding transfer. Consequently:

- within one call, full-array
  `natural_final_embeddings == effective_final_embeddings` remains mandatory
  because the final seam is disabled;
- between intended and unrelated calls, the tensor must be equal on invariant
  rows `[0,1,3,5,6,7]`;
- between those calls, rows 2 and 4 may legitimately differ whenever T or an
  encoder skip is enabled; and
- equality or inequality of rows 2 and 4 must not itself be required, ranked
  or interpreted.

The original host validator instead compared the complete eight-row
`natural_final_embeddings` arrays. ID 0 passed because every route mask is
false. ID 127 enabled all seven skip routes and exposed the invalid assertion.
This is a host-side trace-invariance defect, not a changed model graph,
scientific threshold, donor definition, target or closure rule.

Full intended-versus-unrelated equality remains mandatory for genuinely
upstream natural transformer-seam tensors,
`transformer_output_natural_fingerprint`, and
`encoder_skips_natural_fingerprints`. This amendment does not weaken those
checks.

## 4. Sole permitted v3.3.2 repair

The sidecar may change only the intended-versus-unrelated comparison of
`natural_final_embeddings` from all eight rows to exact rows
`[0,1,3,5,6,7]`. It must additionally retain the stronger all-eight-row
comparison for ID 0. No core model, target reducer, graph factory, route mask,
donor map, precision, tolerance or scientific calculation may change.

The original runner and artifacts remain byte-exact. Implement the repair in
a new versioned runner/test/freeze rather than editing or importing altered
semantics into the bound v3.3 runner. The new bundle must show that the model,
factory and reducer sources remain byte-exact to the v3.3 freeze and that the
only scientific-path source difference is the corrected host assertion plus
the machinery needed for a standalone sidecar.

## 5. Fixed sidecar execution

Use a fresh append-only output directory:

`/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/results/v3_3_2_development_ood_sidecar_one_shot`

The successful sidecar consists of exactly 80 OOD raw records and 320 model
applies:

```text
for recipient order 0..19, in frozen manifest order:
    for anchor ID in (0, 127, 128, 255), in this order:
        intended call
        exact intended repeat
        unrelated-donor call
        exact unrelated-donor repeat
```

The execution index is `4 * recipient_order + anchor_index`, from 0 through
79. The cross-exon donor derangement is unchanged: effect orders 0--5 pair
with 10--15 by class rank and neutral orders 6--9 pair with 16--19 by class
rank, in both directions. The eight-row roles, natural identities and two
donor maps are exactly those in section 3.

The sidecar must compile exactly one new fixed-shape eight-row executable and
use that same executable for all 320 applies. It must not compile a six-row
executable; rerun an identity, six-row coalition or main-cube record; import a
host-serialized activation; or reuse the prior executable as if it were the
new attempt. Persist and bind the new StableHLO, pre-backend HLO, compiled HLO,
executable fingerprint, pytree/input-output signatures and compiler/runtime
provenance. The exact graph and source inputs must match the frozen v3.3
eight-row graph; any graph/HLO discrepancy is a tooling failure requiring a
new prospective amendment. Cross-executable endpoint equality with v3.3 is
not a gate and must not be asserted.

Every new OOD record must bind and rehash its original recipient identity,
mapped donor identity and original six-row coalition at the same anchor ID.
The sidecar may reconstruct the exact frozen sequences and target selection
from the original manifests/reference, but it may not read an old endpoint
value to choose, skip or alter a call.

Before any model import or append-only start, require:

1. a globally tracked-clean committed HEAD and exact hashes for this
   amendment, runner, tests, launcher, wrapper, preflight and freeze;
2. revalidation of every original binding and all 5,158 immutable files from
   section 1, including the 5,142-entry manifest and both tree digests;
3. exact checkpoint, reference, manifest, environment, mixed-precision,
   protobuf, import-root and generated-module containment checks from v3.3;
4. a fresh external and same-process RTX 3090/UUID preflight after sanitizing
   `LD_LIBRARY_PATH`, JAX cache and autotune state exactly as in the successful
   v3.3 launcher;
5. absence of the sidecar output directory before the one allowed attempt;
6. zero confirmation paths, model calls, outputs, activations or
   interventions; and
7. no inspection of original endpoint logits, movements, recoveries,
   Shapley/interaction values, ranks or nominations.

Write `ATTEMPT_STARTED.json` before model construction and persist every raw
record immediately after its four applies and validation. Persist import,
protobuf, compiler and completion provenance plus a manifest and framed tree
over the new raw family. A crash or validation failure must preserve the
partial directory and an exact apply count. There is no deletion, overwrite,
resume, per-case retry, replacement or completion from a later process.

## 6. Per-record exactness gates

Every intended and unrelated call and repeat must retain all original gates:

- endpoint logit-margin shape, class-minus-padding algebra, denominator,
  finiteness and strand/position selection;
- exact evidence and compact-trace repeats;
- exact runtime route masks, donor indices, identity indices and fixed pytree;
- transformer-internal Phase-R seams disabled;
- natural T and all seven natural E tensors matching their same-sequence
  identities;
- rows 0, 1, 6 and 7 unchanged at every active T/E route;
- each disabled T/E route equal to natural and each enabled route equal to its
  requested live donor, including same-allele self rows;
- full-array natural-versus-effective final-embedding equality within each
  call because the final seam remains disabled;
- full eight-row intended-versus-unrelated equality for upstream natural
  transformer tensors and natural T/E fingerprints;
- intended-versus-unrelated exact equality for
  `natural_final_embeddings` on rows `[0,1,3,5,6,7]` only;
- intended-versus-unrelated endpoint equality on rows `[0,1,3,5,6,7]`;
- row 3 equal to row 1 and row 5 equal to row 0 within each call; and
- exact donor/recipient identities, sequence hashes and linked-original
  bindings.

Anchor-specific gates are:

- **ID 0:** all route masks false; rows 2/4 equal their natural recipient
  state; intended and unrelated endpoint readouts and
  `natural_final_embeddings` equal on all eight rows.
- **ID 127:** all seven E routes enabled and T natural; no donor-endpoint
  closure requirement.
- **ID 128:** T enabled and all seven E routes natural; no donor-endpoint
  closure requirement.
- **ID 255:** intended row 2 closes exactly to row 0 and row 4 to row 1;
  unrelated row 2 closes exactly to row 6 and row 4 to row 7.

Rows 2/4 at IDs 127, 128 and 255 have no cross-call equality or inequality
predicate beyond the frozen donor, repeat and closure rules. Record the raw
intended and unrelated movements and all endpoint logits, but never compute a
donor-normalized recovery or B for this unmatched OOD control.

## 7. Completion, stop rules and tests

Success requires exactly 80 unique records in the frozen order, zero invalid
records, exactly 320 applies, one eight-row compilation, all four anchors for
every development variant and every gate in section 6. Missing, extra,
duplicated, reordered, non-finite or invalid records, a second compilation,
an original-tree mismatch or any confirmation access consumes the sidecar as
a controlled tooling stop. Partial OOD results may be reported only as an
append-only failure audit; they cannot be combined with the old first two OOD
records or used for a scientific summary.

Synthetic and CPU-only tests must cover at least:

- exact 20-by-4 order, 80-record/320-apply arithmetic and one-executable cap;
- refusal to run only the 78 missing records or to rerun any main-cube family;
- all row maps and active/invariant-row derivation;
- a passing case in which rows 2/4 differ but `[0,1,3,5,6,7]` remain exact;
- failure for drift in each invariant row and every upstream natural tensor;
- ID-0 all-row equality and ID-255 intended/unrelated closure;
- rejection of forced-difference logic on rows 2/4;
- repeat, donor, disabled-route, final-seam, target-algebra and finiteness
  tampering;
- original raw/tree/compiler/provenance and linked-record tampering;
- one-shot freshness, append-only failure persistence and global-clean gates;
  and
- confirmation-path, model-call and metadata-scope isolation.

No synthetic test may contain or select a real model score.

## 8. Combined-analysis prerequisites and claim limits

The sidecar GPU process performs no Shapley, interaction, ranking, resolution
gate or nomination. A later CPU-only combined analyzer is permitted only if:

1. the v3.3.2 sidecar completed all gates in section 7;
2. the original v3.3 tree and the new sidecar tree are independently
   revalidated in full and are both immutable;
3. the v3.3.1 controlled-stop audit has itself completed successfully and its
   attempt/completion and output hashes are bound as structural evidence, not
   as a source of scientific values;
4. a separately versioned analyzer, tests, freeze and wrapper are prospectively
   committed at a globally tracked-clean HEAD before any endpoint value is
   read;
5. that analyzer reconstructs the original frozen scientific estimands solely
   from the unchanged 20 identities and 5,120 six-row coalitions, and uses all
   80 new OOD records only for the original raw-movement warning; and
6. all original completeness, Shapley-efficiency, effect-versus-neutral,
   resolution nomination and claim gates remain unchanged.

If v3.3.1 is not successfully completed, or either raw family fails
validation, no combined scientific analysis is allowed. The first two v3.3
OOD records are retained for failure provenance but are never stitched into,
substituted for or averaged with the fresh 80-record sidecar.

Even after a successful combined analysis, every report must state that the
original v3.3 attempt controlled-stopped, and that v3.3.2 prospectively
repaired only an overbroad OOD host assertion. The unrelated-donor result
remains an out-of-distribution raw-movement warning. It is not a null
distribution, formal nomination gate, rescue criterion, rejection criterion,
biological mechanism or held-out validation. Any resolution or Shapley claim
comes only from the unchanged complete six-row v3.3 cube under the original
protocol's outcome table. Confirmation remains closed until all later spatial
prerequisites and a separate circuit lock are satisfied.

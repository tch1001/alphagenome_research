#!/usr/bin/env python3
"""Deterministically selects the SNV-only OpenSplice v2 circuit benchmark.

This program intentionally has no AlphaGenome dependency. It consumes only an
explicit allowlist of experimental columns and joins exact alleles from the
released per-exon VCF files. Existing predictor columns in the released master
table are never read by the selection logic.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import io
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence, TextIO
import zipfile


SELECTION_VERSION = "opensplice-circuit-v2-snv-only"
EFFECTS_PER_EXON = 6
NEUTRAL_PER_EXON = 4

EXPERIMENTAL_COLUMNS = frozenset({
    "gene",
    "exon_id",
    "ensembl_exon_id",
    "variant_id",
    "start",
    "end",
    "length",
    "region",
    "mut_type",
    "psi_r1",
    "psi_r2",
    "psi_r3",
    "wt_psi",
    "psi",
    "delta_psi",
    "logit_psi_wt",
    "logit_psi",
    "delta_logit",
    "se_d",
    "padj",
    "significant",
    "measured",
})

OUTPUT_COLUMNS = (
    "selection_version",
    "exon_order",
    "gene",
    "exon_id",
    "ensembl_exon_id",
    "selection_class",
    "observed_effect_sign",
    "class_rank",
    "matched_effect_variant_id",
    "neutral_match_tier",
    "chromosome",
    "position_1based",
    "reference_bases",
    "alternate_bases",
    "vcf_id",
    "variant_id",
    "region",
    "mut_type",
    "minigene_start",
    "minigene_end",
    "mutation_length_bp",
    "psi_r1",
    "psi_r2",
    "psi_r3",
    "wt_psi",
    "psi",
    "delta_psi",
    "delta_logit",
    "se_d",
    "padj",
    "significant",
    "measured",
)


@dataclasses.dataclass(frozen=True)
class FrozenExon:
  order: int
  gene: str
  exon_id: str
  ensembl_exon_id: str


@dataclasses.dataclass(frozen=True)
class VcfRecord:
  chromosome: str
  position: int
  identifier: str
  reference_bases: str
  alternate_bases: str


@dataclasses.dataclass(frozen=True)
class Candidate:
  exon: FrozenExon
  row: Mapping[str, str]
  vcf: VcfRecord
  delta_logit: float
  padj: float
  significant: bool

  @property
  def variant_id(self) -> str:
    return self.row["variant_id"]

  @property
  def region(self) -> str:
    return self.row["region"]

  @property
  def mut_type(self) -> str:
    return self.row["mut_type"]


@dataclasses.dataclass(frozen=True)
class Selection:
  candidate: Candidate
  selection_class: str
  class_rank: int
  matched_effect_variant_id: str = ""
  neutral_match_tier: str = ""


def _open_text(path: Path) -> TextIO:
  if path.suffix == ".gz":
    return gzip.open(path, "rt", encoding="utf-8", newline="")
  return path.open("r", encoding="utf-8", newline="")


def _require_columns(fieldnames: Sequence[str] | None, required: Iterable[str]):
  if fieldnames is None:
    raise ValueError("Input has no header")
  missing = sorted(set(required) - set(fieldnames))
  if missing:
    raise ValueError(f"Missing required columns: {', '.join(missing)}")


def read_frozen_exons(path: Path) -> tuple[FrozenExon, ...]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    _require_columns(
        reader.fieldnames,
        ("selection_order", "gene", "exon_id", "ensembl_exon_id"),
    )
    exons = tuple(
        FrozenExon(
            order=int(row["selection_order"]),
            gene=row["gene"],
            exon_id=row["exon_id"],
            ensembl_exon_id=row["ensembl_exon_id"],
        )
        for row in reader
    )
  if not exons:
    raise ValueError("Frozen exon file is empty")
  if len({exon.order for exon in exons}) != len(exons):
    raise ValueError("Duplicate selection_order in frozen exon file")
  if len({exon.ensembl_exon_id for exon in exons}) != len(exons):
    raise ValueError("Duplicate Ensembl exon ID in frozen exon file")
  return tuple(sorted(exons, key=lambda exon: exon.order))


def _normalise_chromosome(value: str) -> str:
  value = value.strip()
  return value if value.startswith("chr") else f"chr{value}"


def _parse_vcf(handle: TextIO) -> dict[str, VcfRecord]:
  header = None
  records = {}
  for line in handle:
    if line.startswith("##"):
      continue
    if line.startswith("#CHROM"):
      header = line.lstrip("#").rstrip("\n").split("\t")
      continue
    if line.startswith("#") or not line.strip():
      continue
    if header is None:
      raise ValueError("VCF record encountered before #CHROM header")
    values = line.rstrip("\n").split("\t")
    row = dict(zip(header, values, strict=True))
    identifier = row["ID"]
    if not identifier or identifier == ".":
      raise ValueError("OpenSplice VCF record has no stable ID")
    record = VcfRecord(
        chromosome=_normalise_chromosome(row["CHROM"]),
        position=int(row["POS"]),
        identifier=identifier,
        reference_bases=row["REF"].upper(),
        alternate_bases=row["ALT"].upper(),
    )
    if identifier in records:
      raise ValueError(f"Duplicate VCF ID: {identifier}")
    records[identifier] = record
  if header is None:
    raise ValueError("VCF has no #CHROM header")
  return records


def read_vcf_records(
    source: Path, exons: Sequence[FrozenExon]
) -> dict[str, dict[str, VcfRecord]]:
  result = {}
  if source.is_dir():
    for exon in exons:
      matches = sorted(source.glob(f"{exon.ensembl_exon_id}_*.vcf"))
      if len(matches) != 1:
        raise ValueError(
            f"Expected one VCF for {exon.ensembl_exon_id}, found {len(matches)}"
        )
      with matches[0].open("r", encoding="utf-8", newline="") as handle:
        result[exon.ensembl_exon_id] = _parse_vcf(handle)
    return result

  if not zipfile.is_zipfile(source):
    raise ValueError("--vcf-source must be a directory or ZIP archive")
  with zipfile.ZipFile(source) as archive:
    names = archive.namelist()
    for exon in exons:
      matches = [
          name
          for name in names
          if name.endswith(".vcf")
          and Path(name).name.startswith(f"{exon.ensembl_exon_id}_")
      ]
      if len(matches) != 1:
        raise ValueError(
            f"Expected one VCF for {exon.ensembl_exon_id}, found {len(matches)}"
        )
      with archive.open(matches[0]) as binary_handle:
        with io.TextIOWrapper(binary_handle, encoding="utf-8", newline="") as handle:
          result[exon.ensembl_exon_id] = _parse_vcf(handle)
  return result


def _finite_float(value: str, field: str, variant_id: str) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{variant_id}: non-numeric {field}={value!r}") from error
  if not math.isfinite(parsed):
    raise ValueError(f"{variant_id}: non-finite {field}={value!r}")
  return parsed


def _is_true(value: str) -> bool:
  return value.strip().lower() in {"1", "true", "yes"}


def _vcf_identifier(exon: FrozenExon, variant_id: str) -> str:
  prefix = f"{exon.exon_id}_"
  if not variant_id.startswith(prefix):
    raise ValueError(
        f"{variant_id}: expected prefix {prefix!r} for exact VCF join"
    )
  return f"{exon.ensembl_exon_id}_{variant_id[len(prefix):]}"


def read_candidates(
    experimental_path: Path,
    exons: Sequence[FrozenExon],
    vcf_records: Mapping[str, Mapping[str, VcfRecord]],
) -> dict[str, tuple[Candidate, ...]]:
  by_ensembl = {exon.ensembl_exon_id: exon for exon in exons}
  candidates = {exon.ensembl_exon_id: [] for exon in exons}
  seen = set()
  with _open_text(experimental_path) as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    _require_columns(reader.fieldnames, EXPERIMENTAL_COLUMNS)
    for full_row in reader:
      ensembl_id = full_row["ensembl_exon_id"]
      if ensembl_id not in by_ensembl:
        continue
      # This is the leakage boundary: no non-allowlisted value survives.
      row = {column: full_row[column] for column in EXPERIMENTAL_COLUMNS}
      exon = by_ensembl[ensembl_id]
      variant_id = row["variant_id"]
      if row["exon_id"] != exon.exon_id or row["gene"] != exon.gene:
        raise ValueError(f"{variant_id}: frozen exon metadata mismatch")
      if not _is_true(row["measured"]) or row["mut_type"].lower() == "wt":
        continue
      numeric_fields = (
          "psi_r1",
          "psi_r2",
          "psi_r3",
          "wt_psi",
          "psi",
          "delta_psi",
          "logit_psi_wt",
          "logit_psi",
          "delta_logit",
          "se_d",
          "padj",
      )
      parsed = {
          field: _finite_float(row[field], field, variant_id)
          for field in numeric_fields
      }
      if parsed["se_d"] < 0 or not 0 <= parsed["padj"] <= 1:
        raise ValueError(f"{variant_id}: invalid uncertainty or adjusted P value")
      vcf_id = _vcf_identifier(exon, variant_id)
      try:
        vcf = vcf_records[ensembl_id][vcf_id]
      except KeyError as error:
        raise ValueError(
            f"{variant_id}: exact VCF record {vcf_id} is missing"
        ) from error
      key = (ensembl_id, variant_id)
      if key in seen:
        raise ValueError(f"Duplicate measured variant: {variant_id}")
      seen.add(key)
      candidates[ensembl_id].append(
          Candidate(
              exon=exon,
              row=row,
              vcf=vcf,
              delta_logit=parsed["delta_logit"],
              padj=parsed["padj"],
              significant=_is_true(row["significant"]),
          )
      )
  return {
      key: tuple(sorted(values, key=lambda candidate: candidate.variant_id))
      for key, values in candidates.items()
  }


def _neutral_match_tier(neutral: Candidate, effect: Candidate) -> tuple[int, str]:
  same_region = neutral.region == effect.region
  same_type = neutral.mut_type == effect.mut_type
  if same_region and same_type:
    return 0, "same_region_and_mut_type"
  if same_region:
    return 1, "same_region"
  if same_type:
    return 2, "same_mut_type"
  return 3, "unmatched"


def select_for_exon(candidates: Sequence[Candidate]) -> tuple[Selection, ...]:
  if not candidates:
    raise ValueError("Frozen exon has no high-quality measured candidates")
  exon_id = candidates[0].exon.exon_id
  snvs = [
      candidate
      for candidate in candidates
      if candidate.mut_type == "sub"
      and len(candidate.vcf.reference_bases) == 1
      and len(candidate.vcf.alternate_bases) == 1
      and "," not in candidate.vcf.alternate_bases
  ]
  effects = sorted(
      (
          candidate
          for candidate in snvs
          if candidate.significant and candidate.padj < 0.1
      ),
      key=lambda candidate: (-abs(candidate.delta_logit), candidate.variant_id),
  )[:EFFECTS_PER_EXON]
  if len(effects) != EFFECTS_PER_EXON:
    raise ValueError(f"{exon_id}: fewer than six significant SNVs")

  neutral_pool = [
      candidate
      for candidate in snvs
      if not candidate.significant and candidate.padj >= 0.1
  ]
  if len(neutral_pool) < NEUTRAL_PER_EXON:
    raise ValueError(f"{exon_id}: fewer than four nonsignificant neutral SNVs")

  selections = [
      Selection(candidate, "significant_effect", rank)
      for rank, candidate in enumerate(effects, start=1)
  ]

  match_targets = effects[:NEUTRAL_PER_EXON]
  unused_neutrals = list(neutral_pool)
  for rank, effect in enumerate(match_targets, start=1):
    def match_key(neutral: Candidate):
      tier, _ = _neutral_match_tier(neutral, effect)
      return tier, abs(neutral.delta_logit), neutral.variant_id

    neutral = min(unused_neutrals, key=match_key)
    unused_neutrals.remove(neutral)
    _, tier_name = _neutral_match_tier(neutral, effect)
    selections.append(
        Selection(
            neutral,
            "neutral_control",
            rank,
            matched_effect_variant_id=effect.variant_id,
            neutral_match_tier=tier_name,
        )
    )
  return tuple(selections)


def run_selection(
    experimental_path: Path,
    frozen_exons_path: Path,
    vcf_source: Path,
) -> tuple[Selection, ...]:
  exons = read_frozen_exons(frozen_exons_path)
  vcf_records = read_vcf_records(vcf_source, exons)
  candidates = read_candidates(experimental_path, exons, vcf_records)
  selections = []
  for exon in exons:
    exon_selections = select_for_exon(candidates[exon.ensembl_exon_id])
    selections.extend(exon_selections)
  return tuple(selections)


def _format_float(value: str) -> str:
  return format(float(value), ".17g")


def selection_rows(selections: Sequence[Selection]) -> tuple[dict[str, str], ...]:
  rows = []
  for selection in selections:
    candidate = selection.candidate
    source = candidate.row
    rows.append({
        "selection_version": SELECTION_VERSION,
        "exon_order": str(candidate.exon.order),
        "gene": candidate.exon.gene,
        "exon_id": candidate.exon.exon_id,
        "ensembl_exon_id": candidate.exon.ensembl_exon_id,
        "selection_class": selection.selection_class,
        "observed_effect_sign": (
            "positive"
            if selection.selection_class == "significant_effect"
            and candidate.delta_logit > 0
            else "negative"
            if selection.selection_class == "significant_effect"
            else "neutral_control"
        ),
        "class_rank": str(selection.class_rank),
        "matched_effect_variant_id": selection.matched_effect_variant_id,
        "neutral_match_tier": selection.neutral_match_tier,
        "chromosome": candidate.vcf.chromosome,
        "position_1based": str(candidate.vcf.position),
        "reference_bases": candidate.vcf.reference_bases,
        "alternate_bases": candidate.vcf.alternate_bases,
        "vcf_id": candidate.vcf.identifier,
        "variant_id": candidate.variant_id,
        "region": candidate.region,
        "mut_type": candidate.mut_type,
        "minigene_start": source["start"],
        "minigene_end": source["end"],
        "mutation_length_bp": source["length"],
        "psi_r1": _format_float(source["psi_r1"]),
        "psi_r2": _format_float(source["psi_r2"]),
        "psi_r3": _format_float(source["psi_r3"]),
        "wt_psi": _format_float(source["wt_psi"]),
        "psi": _format_float(source["psi"]),
        "delta_psi": _format_float(source["delta_psi"]),
        "delta_logit": _format_float(source["delta_logit"]),
        "se_d": _format_float(source["se_d"]),
        "padj": _format_float(source["padj"]),
        "significant": "yes" if candidate.significant else "no",
        "measured": "true",
    })
  return tuple(rows)


def render_tsv(selections: Sequence[Selection]) -> str:
  output = io.StringIO(newline="")
  writer = csv.DictWriter(
      output, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n"
  )
  writer.writeheader()
  writer.writerows(selection_rows(selections))
  return output.getvalue()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--frozen-exons", type=Path, required=True)
  parser.add_argument("--vcf-source", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print canonical selection to stdout; do not create an output file.",
  )
  args = parser.parse_args(argv)
  if args.dry_run and args.output is not None:
    parser.error("--dry-run and --output are mutually exclusive")
  if not args.dry_run and args.output is None:
    parser.error("--output is required unless --dry-run is set")
  return args


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv if argv is not None else sys.argv[1:])
  selections = run_selection(args.input, args.frozen_exons, args.vcf_source)
  rendered = render_tsv(selections)
  digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
  if args.dry_run:
    sys.stdout.write(rendered)
  else:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
  print(
      f"selected={len(selections)} sha256={digest}",
      file=sys.stderr,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

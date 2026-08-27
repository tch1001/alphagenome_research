"""Tests for the deterministic SNV-only OpenSplice v2 selector."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).with_name("select_holdout_v2.py")
_SPEC = importlib.util.spec_from_file_location("select_holdout_v2", _MODULE_PATH)
select_holdout = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = select_holdout
_SPEC.loader.exec_module(select_holdout)


class SelectHoldoutTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.TemporaryDirectory()
    self.root = Path(self.tempdir.name)
    self.exons = self.root / "frozen.tsv"
    self.exons.write_text(
        "selection_order\tgene\texon_id\tensembl_exon_id\n"
        "1\tGENE\tGENE_e1\tENSETEST001\n",
        encoding="utf-8",
    )
    self.vcf_dir = self.root / "vcfs"
    self.vcf_dir.mkdir()

  def tearDown(self):
    self.tempdir.cleanup()
    super().tearDown()

  def _rows(self):
    specifications = (
        ("T1A", "Exon", "sub", 6.0, 0.001, "yes"),
        ("T2A", "Intron up", "sub", -5.0, 0.001, "yes"),
        ("T3A", "Exon", "sub", 4.0, 0.001, "yes"),
        ("T4A", "Intron down", "sub", -3.0, 0.001, "yes"),
        ("T5A", "5'SS", "sub", 2.0, 0.001, "yes"),
        ("T6A", "3'SS", "sub", 1.0, 0.001, "yes"),
        ("del7to27", "Exon", "del21", -100.0, 0.001, "yes"),
        ("T8A", "Exon", "sub", 0.01, 0.9, "no"),
        ("T9A", "Intron up", "sub", -0.02, 0.8, "no"),
        ("T10A", "Intron down", "sub", 0.03, 0.7, "no"),
        ("T11A", "5'SS", "sub", -0.04, 0.6, "no"),
        ("T12A", "3'SS", "sub", 0.001, 0.5, "no"),
        ("del13to13", "Exon", "del1", 0.0, 0.9, "no"),
    )
    rows = []
    for index, (token, region, mut_type, delta, padj, significant) in enumerate(
        specifications, start=1
    ):
      rows.append({
          "gene": "GENE",
          "exon_id": "GENE_e1",
          "ensembl_exon_id": "ENSETEST001",
          "variant_id": f"GENE_e1_{token}",
          "start": str(index),
          "end": str(index),
          "length": "1",
          "region": region,
          "mut_type": mut_type,
          "psi_r1": "50",
          "psi_r2": "51",
          "psi_r3": "49",
          "wt_psi": "50",
          "psi": str(50 + delta),
          "delta_psi": str(delta),
          "logit_psi_wt": "0",
          "logit_psi": str(delta),
          "delta_logit": str(delta),
          "se_d": "0.1",
          "padj": str(padj),
          "significant": significant,
          "measured": "True",
          "alphagenome_genome__mean_delta_splice": str(index * 1000),
      })
    return rows

  def _write_inputs(self, rows, path, *, write_vcf=True):
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)

    if not write_vcf:
      return
    lines = [
        "##fileformat=VCFv4.1",
        "##reference=GRCh38",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    for index, row in enumerate(rows, start=100):
      token = row["variant_id"].removeprefix("GENE_e1_")
      ref, alt = ("AT", "A") if token.startswith("del") else ("T", "A")
      lines.append(
          f"chr1\t{index}\tENSETEST001_{token}\t{ref}\t{alt}\t.\t.\t."
      )
    (self.vcf_dir / "ENSETEST001_variants.vcf").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

  def test_selection_is_order_and_predictor_score_independent(self):
    rows = self._rows()
    first = self.root / "first.tsv"
    self._write_inputs(rows, first)
    selection_one = select_holdout.run_selection(
        first, self.exons, self.vcf_dir
    )
    rendered_one = select_holdout.render_tsv(selection_one)

    reversed_rows = list(reversed(rows))
    for row in reversed_rows:
      row["alphagenome_genome__mean_delta_splice"] = "-999999"
    second = self.root / "second.tsv"
    self._write_inputs(reversed_rows, second, write_vcf=False)
    selection_two = select_holdout.run_selection(
        second, self.exons, self.vcf_dir
    )
    rendered_two = select_holdout.render_tsv(selection_two)

    self.assertEqual(rendered_one, rendered_two)
    self.assertEqual(len(selection_one), 10)
    self.assertEqual(
        [item.candidate.variant_id for item in selection_one[:6]],
        [
            "GENE_e1_T1A",
            "GENE_e1_T2A",
            "GENE_e1_T3A",
            "GENE_e1_T4A",
            "GENE_e1_T5A",
            "GENE_e1_T6A",
        ],
    )
    self.assertTrue(
        all(item.candidate.mut_type == "sub" for item in selection_one)
    )
    self.assertTrue(
        all(
            len(item.candidate.vcf.reference_bases) == 1
            and len(item.candidate.vcf.alternate_bases) == 1
            for item in selection_one
        )
    )
    self.assertNotIn(
        "GENE_e1_del7to27",
        [item.candidate.variant_id for item in selection_one],
    )
    self.assertNotIn(
        "GENE_e1_del13to13",
        [item.candidate.variant_id for item in selection_one],
    )
    output_rows = select_holdout.selection_rows(selection_one)
    self.assertEqual(
        [row["observed_effect_sign"] for row in output_rows[:6]],
        ["positive", "negative", "positive", "negative", "positive", "positive"],
    )
    self.assertTrue(
        all(item.candidate.vcf.position >= 100 for item in selection_one)
    )

  def test_missing_exact_vcf_mapping_fails_closed(self):
    rows = self._rows()
    path = self.root / "input.tsv"
    self._write_inputs(rows, path)
    vcf_path = self.vcf_dir / "ENSETEST001_variants.vcf"
    text = vcf_path.read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if "ENSETEST001_T1A" not in line
    ) + "\n"
    vcf_path.write_text(text, encoding="utf-8")

    with self.assertRaisesRegex(ValueError, "exact VCF record.*is missing"):
      select_holdout.run_selection(path, self.exons, self.vcf_dir)


if __name__ == "__main__":
  unittest.main()

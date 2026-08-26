import math
import shutil
import tempfile
import unittest
from pathlib import Path

from fedrbtvis.artifacts import ArtifactCorruptError, sha256_file
from fedrbtvis.legacy import (
    LegacyFormatError,
    LegacyHashError,
    LegacyRepository,
    parse_predict_file,
)


FIXTURE = Path(__file__).parent / "fixtures" / "legacy_valid.txt"
EVIDENCE_DIR = Path(__file__).resolve().parents[3] / "evidence" / "legacy"
_TEMP_DIRECTORIES: list[tempfile.TemporaryDirectory[str]] = []


def write_temp(content: str) -> Path:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "predict.txt"
    path.write_text(content, encoding="utf-8")
    _TEMP_DIRECTORIES.append(directory)
    return path


class LegacyParserTest(unittest.TestCase):
    def tearDown(self) -> None:
        while _TEMP_DIRECTORIES:
            directory = _TEMP_DIRECTORIES.pop()
            directory.cleanup()

    def test_fixture_assigns_block_and_inferred_fields(self) -> None:
        result = parse_predict_file(
            FIXTURE,
            expected_sha256=sha256_file(FIXTURE),
            expected_clients=(100, 101),
        )
        self.assertEqual(len(result.observations), 4)
        self.assertEqual(result.blocks, 2)
        last = result.observations[-1]
        self.assertEqual(last.block_index, 1)
        self.assertEqual(last.k_index, 2)
        self.assertEqual(last.round_index, 1)
        self.assertAlmostEqual(last.inferred_target_noise, 0.1)
        self.assertAlmostEqual(last.inferred_emd_base, 0.05)
        self.assertAlmostEqual(result.total_runtime_seconds, 121.0)

    def test_wrong_hash_is_rejected_before_parse(self) -> None:
        with self.assertRaises(LegacyHashError):
            parse_predict_file(
                FIXTURE,
                expected_sha256="0" * 64,
                expected_clients=(100, 101),
            )

    def test_incomplete_block_is_rejected(self) -> None:
        path = write_temp("i=100 0 0 2 200\n")
        with self.assertRaises(LegacyFormatError):
            parse_predict_file(
                path,
                expected_sha256=sha256_file(path),
                expected_clients=(100, 101),
            )

    def test_non_finite_value_is_rejected(self) -> None:
        path = write_temp("i=100 nan 0 2 200\n")
        with self.assertRaises(LegacyFormatError):
            parse_predict_file(
                path,
                expected_sha256=sha256_file(path),
                expected_clients=(100,),
            )


class LegacyRepositoryTest(unittest.TestCase):
    def test_repository_loads_generated_evidence(self) -> None:
        repository = LegacyRepository.from_directory(EVIDENCE_DIR)
        summary = repository.manifest_summary()
        rows = repository.observations()

        self.assertEqual(summary["rows"], 4500)
        self.assertEqual(summary["blocks"], 180)
        self.assertEqual(summary["clients_per_block"], 25)
        self.assertEqual(len(rows), 4500)
        self.assertTrue(all(row["source"] == "legacy" for row in rows))
        self.assertTrue(all(row["sample_count"] == 200 for row in rows))
        self.assertTrue(all(math.isfinite(row["lid"]) for row in rows))

    def test_repository_rejects_corrupt_csv_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copy2(EVIDENCE_DIR / "manifest.json", target / "manifest.json")
            shutil.copy2(
                EVIDENCE_DIR / "observations.csv",
                target / "observations.csv",
            )
            with (target / "observations.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(ArtifactCorruptError):
                LegacyRepository.from_directory(target)

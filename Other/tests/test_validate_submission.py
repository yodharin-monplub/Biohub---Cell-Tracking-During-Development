from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.validate_submission import COLUMNS, SubmissionError, validate


class ValidateSubmissionTests(unittest.TestCase):
    def write_rows(self, rows: list[list[object]]) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "submission.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        return path

    def test_valid_lineage_with_division(self) -> None:
        path = self.write_rows(
            [
                [0, "movie", "node", 1, 0, 2, 3, 4, -1, -1],
                [1, "movie", "node", 2, 1, 2, 4, 4, -1, -1],
                [2, "movie", "node", 3, 1, 2, 2, 4, -1, -1],
                [3, "movie", "edge", -1, -1, -1, -1, -1, 1, 2],
                [4, "movie", "edge", -1, -1, -1, -1, -1, 1, 3],
            ]
        )
        result = validate(path)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["totals"]["divisions"], 1)
        self.assertEqual(result["totals"]["tracks"], 1)

    def test_rejects_nonconsecutive_edge(self) -> None:
        path = self.write_rows(
            [
                [0, "movie", "node", 1, 0, 2, 3, 4, -1, -1],
                [1, "movie", "node", 2, 2, 2, 4, 4, -1, -1],
                [2, "movie", "edge", -1, -1, -1, -1, -1, 1, 2],
            ]
        )
        with self.assertRaisesRegex(SubmissionError, "only t -> t\\+1 is scoreable"):
            validate(path)

    def test_rejects_duplicate_node(self) -> None:
        path = self.write_rows(
            [
                [0, "movie", "node", 1, 0, 2, 3, 4, -1, -1],
                [1, "movie", "node", 1, 1, 2, 4, 4, -1, -1],
            ]
        )
        with self.assertRaisesRegex(SubmissionError, "duplicate node_id"):
            validate(path)

    def test_rejects_noncontiguous_dataset_blocks(self) -> None:
        path = self.write_rows(
            [
                [0, "a", "node", 1, 0, 2, 3, 4, -1, -1],
                [1, "b", "node", 1, 0, 2, 3, 4, -1, -1],
                [2, "a", "node", 2, 1, 2, 4, 4, -1, -1],
            ]
        )
        with self.assertRaisesRegex(SubmissionError, "not in one contiguous block"):
            validate(path)


if __name__ == "__main__":
    unittest.main()


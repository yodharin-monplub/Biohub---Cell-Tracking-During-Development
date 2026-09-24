"""CPU-only checks for the unexecuted clean-fold training plan."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model156.train_clean import DEFAULT_REPO, build_plan


class CleanPlanTests(unittest.TestCase):
    def test_two_complete_embryo_disjoint_folds(self) -> None:
        plans = [build_plan(index, DEFAULT_REPO) for index in (0, 1)]
        for plan in plans:
            train = set(plan["train"])
            outer = set(plan["outer_eval"])
            self.assertFalse(train & outer)
            self.assertEqual(len(train | outer), 199)
            self.assertTrue(set(plan["trainer_internal_test"]) <= train)
            self.assertFalse(
                {name.split("_")[0] for name in train}
                & {name.split("_")[0] for name in outer}
            )
        self.assertEqual(set(plans[0]["train"]), set(plans[1]["outer_eval"]))
        self.assertEqual(set(plans[1]["train"]), set(plans[0]["outer_eval"]))

    def test_rejects_overlapping_embryo_even_with_distinct_movies(self) -> None:
        bad = [{"held_out_embryo": "44b6", "train": ["44b6_a"],
                "test": ["44b6_b"]}]
        with patch("model156.train_clean.folds", return_value=bad):
            with self.assertRaisesRegex(RuntimeError, "Outer embryo"):
                build_plan(0, DEFAULT_REPO)

    def test_rejects_unreviewed_trainer_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scripts = Path(directory) / "scripts"
            scripts.mkdir()
            (scripts / "train_unet_transformer.py").write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "source hash changed"):
                build_plan(0, Path(directory))


if __name__ == "__main__":
    unittest.main()

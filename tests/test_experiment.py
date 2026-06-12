from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragile_jscc.experiment import ExperimentRun


class ExperimentRunTests(unittest.TestCase):
    def test_run_creates_manifest_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "configs" / "EXP-S0-999_test.json"
            config_path.parent.mkdir()
            config_path.write_text("{}", encoding="utf-8")
            config = {"experiment_id": "EXP-S0-999", "seed": 7}

            with ExperimentRun(root, config_path, config) as run:
                run.append_metrics({"value": 1.0})

            self.assertTrue(
                (root / "outputs" / "EXP-S0-999" / "run_manifest.json").exists()
            )
            self.assertTrue(
                (root / "checkpoints" / "EXP-S0-999").is_dir()
            )
            with self.assertRaises(FileExistsError):
                with ExperimentRun(root, config_path, config):
                    pass

    def test_invalid_experiment_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "bad.json"
            config_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperimentRun(
                    root,
                    config_path,
                    {"experiment_id": "experiment-one"},
                )


if __name__ == "__main__":
    unittest.main()


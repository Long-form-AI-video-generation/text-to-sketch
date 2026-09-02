from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class TrainScriptSmokeTest(unittest.TestCase):
    def test_train_script_dry_run_composes_smoke_experiment(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/sketchformer/train.py",
                "--experiment",
                "smoke_test",
                "--dry-run",
            ],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("experiment=smoke_test", result.stdout)
        self.assertIn("model=sketchformer_tok_dict", result.stdout)
        self.assertIn("checkpoint_dir=", result.stdout)


if __name__ == "__main__":
    unittest.main()

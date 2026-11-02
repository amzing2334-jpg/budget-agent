import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "runs/iteration-20261102-14182-controls/run_controls.sh"


class ControlInterpreterTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "controls run on a Linux grading host")
    def test_relative_venv_symlink_survives_change_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory)
            venv = caller / "grader venv"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--symlinks", str(venv)], check=True)
            python = venv / "bin/python"
            self.assertTrue(python.is_symlink())
            site = subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
            # Record the actual interpreter at startup, then let the real preflight
            # halt on missing inputs before it can invoke SWE-bench or Docker.
            (Path(site) / "control-probe.pth").write_text(
                "import sys, pathlib; pathlib.Path('observed-prefix').write_text(sys.prefix)\n",
                encoding="utf-8")
            (caller / "inputs").mkdir()
            result = subprocess.run(["bash", str(SCRIPT), str(python.relative_to(caller)), "inputs", "work"],
                                    cwd=caller, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONNOUSERSITE": "1"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pinned-dataset.json", result.stderr)
            marker = caller / "work/observed-prefix"
            self.assertTrue(marker.is_file(), "the grading interpreter did not load its venv")
            self.assertEqual(marker.read_text(), str(venv))

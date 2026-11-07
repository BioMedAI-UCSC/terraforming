import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import os
import subprocess
import tempfile

HERE = Path(__file__).parent
import sys
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location("run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
sys.modules["run"] = run


class RuntimeTests(unittest.TestCase):
    def test_retry_preserves_failed_attempt_and_skips_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source = work / "source"
            source.mkdir()
            (source / ".paper-source.json").write_text('{}')
            (source / ".paper-staged").write_text(run.SOURCE_ID)
            prior_cwd = Path.cwd()
            calls = []

            def execute(args, **kwargs):
                if "experiment.py" in args:
                    calls.append(args)
                    if len(calls) == 1:
                        raise subprocess.CalledProcessError(10, args)

            try:
                os.chdir(source)
                with patch.object(sys, "argv", ["run.py", "all", "--work", str(work)]), patch.object(run, "setup", return_value=(source, Path("python"))), patch.object(run, "experiments", return_value={"fit": ["experiment.py --output {output}"]}), patch.object(run, "command", side_effect=execute):
                    with self.assertRaisesRegex(RuntimeError, "Failed experiments"):
                        run.main()
                    run.main()
                    run.main()
                self.assertEqual(len(calls), 2)
                self.assertNotEqual(calls[0][-1], calls[1][-1])
                self.assertEqual(len(list((work / "outputs" / run.SOURCE_ID / "paper-v1" / "fit").glob("*/run.log"))), 2)
            finally:
                os.chdir(prior_cwd)

    def test_protocols_and_safe_outputs(self):
        tasks = run.experiments()
        self.assertEqual(tasks["radiation-0"], tasks["radiation-repeat"])
        for name, commands in tasks.items():
            for command in commands:
                rendered = command.format(output="/work/new-attempt/results", climate="/work/prior/results")
                self.assertNotIn("{", rendered)
                self.assertIn("/work/new-attempt/results", rendered)
        for seed in range(3):
            self.assertIn(f"calibration-{seed}", tasks)
            self.assertIn(f"neural-{seed}", tasks)
        self.assertIn("--sols 2006", tasks["climate"][0])
        self.assertIn("--dt 300", tasks["climate"][0])
        self.assertTrue(all(len(x) == 40 for x in run.REVISIONS))

    def test_cluster_gate_rejects_wrong_input_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source=work/'source';source.mkdir()
            marker=source/'.paper-staged'
            marker.write_text("wrong-source")
            prior=Path.cwd()
            try:
                os.chdir(source)
                with patch.object(sys,'argv',['run.py','stage','--work',str(work)]), patch.object(run,'setup',return_value=(source,Path('python'))):
                    with self.assertRaisesRegex(RuntimeError,'Input source identity mismatch'):
                        run.main()
            finally:
                os.chdir(prior)


if __name__ == "__main__":
    unittest.main()

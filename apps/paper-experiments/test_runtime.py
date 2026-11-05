import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import os
import subprocess
import tempfile

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
import sys
sys.modules["run"] = run
spec = importlib.util.spec_from_file_location("jobs", HERE / "jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)


class RuntimeTests(unittest.TestCase):
    def test_retry_preserves_failed_attempt_and_skips_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source = work / "source"
            source.mkdir()
            (source / ".paper-source.json").write_text('{}')
            (source / ".paper-staged").touch()
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
                self.assertEqual(len(list((work / "runs" / run.SOURCE_ID / "paper-v1" / "fit").glob("*/run.log"))), 2)
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

    def test_gpu_and_storage_contract(self):
        stage = jobs.manifests("stage", "samkash/test@sha256:abc", "test")
        gpu = jobs.manifests("all", "samkash/test@sha256:abc", "test")
        for manifest in (stage, gpu):
            self.assertTrue(all(x["metadata"]["namespace"] == "razvanlab" for x in manifest["items"]))
            self.assertEqual(manifest["items"][0]["spec"]["accessModes"], ["ReadWriteOnce"])
        cpu_resources = stage["items"][1]["spec"]["template"]["spec"]["containers"][0]["resources"]
        gpu_resources = gpu["items"][1]["spec"]["template"]["spec"]["containers"][0]["resources"]
        self.assertNotIn("nvidia.com/a100", cpu_resources["requests"])
        self.assertEqual(gpu_resources["requests"]["nvidia.com/a100"], gpu_resources["limits"]["nvidia.com/a100"])
        init = gpu["items"][1]["spec"]["template"]["spec"]["initContainers"][0]
        self.assertTrue(init["volumeMounts"][0]["readOnly"])
        self.assertIn(run.SOURCE_ID, init["command"][-1])
        self.assertNotIn("initContainers", stage["items"][1]["spec"]["template"]["spec"])

    def test_cluster_gate_rejects_wrong_input_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            marker = work / "source" / run.SOURCE_ID / ".paper-staged"
            marker.parent.mkdir(parents=True)
            (work / ".paper.lock").touch()
            script = jobs.WAIT_FOR_INPUTS.replace("/work", temporary)
            marker.write_text("wrong-source")
            result = subprocess.run([sys.executable, "-c", script], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            marker.write_text(run.SOURCE_ID)
            result = subprocess.run([sys.executable, "-c", script], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

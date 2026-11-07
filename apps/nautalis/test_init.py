import tempfile
import unittest
from pathlib import Path

from init import create_experiment


class InitTests(unittest.TestCase):
    def test_creates_contract_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = create_experiment('climate-study', Path(tmp) / 'experiments')
            self.assertTrue((target / 'run.py').is_file())
            compile((target / 'run.py').read_text(), str(target / 'run.py'), 'exec')
            readme = (target / 'README.md').read_text()
            self.assertIn('EXPERIMENT_OUTPUT_DIR', readme)
            self.assertIn('--experiments ', readme)
            self.assertIn('climate-study', readme)

    def test_rejects_path_traversal_and_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            with self.assertRaises(ValueError):
                create_experiment('../outside', parent)
            create_experiment('existing', parent)
            with self.assertRaises(FileExistsError):
                create_experiment('existing', parent)


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from runtime_loader import load_runtime

class SourceVersionTests(unittest.TestCase):
    def test_deployed_source_change_loads_new_engine_without_reusing_cached_classes(self):
        with TemporaryDirectory() as t:
            root=Path(t);(root/'__init__.py').write_text('')
            (root/'storage.py').write_text('class DriveStore: pass\n')
            (root/'runner.py').write_text('class Jobs:\n    version = "old"\n')
            first=load_runtime(root)
            self.assertEqual(first.Jobs.version,'old')
            self.assertIs(load_runtime(root).Jobs,first.Jobs)
            (root/'runner.py').write_text('class Jobs:\n    version = "corrected-editor"\n')
            second=load_runtime(root)
            self.assertNotEqual(first.revision,second.revision)
            self.assertIsNot(first.Jobs,second.Jobs)
            self.assertEqual(second.Jobs.version,'corrected-editor')

if __name__=='__main__':unittest.main()

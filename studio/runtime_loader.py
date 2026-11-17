"""Load a source-versioned engine so a page refresh cannot retain an older editor."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib
import importlib.util
import sys

def load_runtime(package_dir=None):
    folder=Path(package_dir) if package_dir else Path(__file__).with_name('pipeline')
    digest=hashlib.sha256()
    for path in sorted(folder.rglob('*.py')):
        digest.update(path.relative_to(folder).as_posix().encode());digest.update(b'\0');digest.update(path.read_bytes())
    revision=digest.hexdigest()[:16];namespace='_studio_engine_'+revision
    if namespace not in sys.modules:
        spec=importlib.util.spec_from_file_location(namespace,folder/'__init__.py',submodule_search_locations=[str(folder)])
        if spec is None or spec.loader is None:raise RuntimeError('The editing engine could not be loaded')
        module=importlib.util.module_from_spec(spec);sys.modules[namespace]=module
        try:spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(namespace,None);raise
    runner=importlib.import_module(namespace+'.runner');storage=importlib.import_module(namespace+'.storage')
    return SimpleNamespace(Jobs=runner.Jobs,DriveStore=storage.DriveStore,revision=revision)

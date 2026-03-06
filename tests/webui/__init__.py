import os as _os
import sys as _sys

# When pytest adds tests/ to sys.path, 'webui' resolves to this directory
# (tests/webui/) instead of the real webui/ package. Ensure the project root
# is on sys.path before the project root entry so the real webui package wins.
_project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

# Re-import the real webui package to shadow this stub
import importlib.util as _util

_spec = _util.spec_from_file_location(
    "webui",
    _os.path.join(_project_root, "webui", "__init__.py"),
    submodule_search_locations=[_os.path.join(_project_root, "webui")],
)
if _spec is not None:
    _real_webui = _util.module_from_spec(_spec)
    _sys.modules["webui"] = _real_webui
    _spec.loader.exec_module(_real_webui)

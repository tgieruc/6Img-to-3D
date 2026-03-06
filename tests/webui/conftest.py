import os
import sys

# Ensure the project root is on sys.path so that `webui.backend` resolves
# to the actual package, not the tests/webui/ directory.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

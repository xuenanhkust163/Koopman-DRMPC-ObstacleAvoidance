"""Add the project source and repository roots for direct script execution."""

from pathlib import Path
import sys


def bootstrap():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    source_root = project_root / "src"

    for path in (project_root, source_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

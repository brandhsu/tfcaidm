import os
from pathlib import Path
import sys


def set_path(root=os.getcwd(), path="tfcaidm"):
    root = Path(root).parent.parent
    fpath = str(root)
    if fpath not in sys.path:
        sys.path.append(fpath)

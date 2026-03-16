from pathlib import Path
from typing import Dict, Union

def prepare_output_dirs(output_root: Union[str, Path]) -> Dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    names = ["logs", "fits", "preds", "metrics", "figures"]
    out = {}

    for name in names:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        out[name] = d
    return out


def prepare_output_dir(output_root: Union[str, Path]) -> Dict[str, Path]:
    """Backward-compatible alias."""
    return prepare_output_dirs(output_root)

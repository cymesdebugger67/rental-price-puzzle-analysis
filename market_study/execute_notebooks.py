from pathlib import Path
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = (
    "01_market_context.ipynb",
    "02_model_comparison.ipynb",
    "03_geographic_generalization.ipynb",
)


def execute_all(root):
    kernel_dir = Path(sys.prefix) / "share" / "jupyter" / "kernels" / "rental_puzzle_local"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Rental Price Puzzle (local)",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel, indent=2), encoding="utf-8")
    jupyter_dir = str(Path(sys.prefix) / "share" / "jupyter")
    runtime = Path(sys.prefix) / "jupyter_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    old = {key: os.environ.get(key) for key in ("JUPYTER_PATH", "JUPYTER_RUNTIME_DIR")}
    os.environ["JUPYTER_PATH"] = jupyter_dir + (os.pathsep + old["JUPYTER_PATH"] if old["JUPYTER_PATH"] else "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    executed = []
    try:
        for name in NOTEBOOKS:
            source = root / "notebooks" / name
            nb = nbformat.read(source, as_version=4)
            nbformat.validate(nb)
            NotebookClient(
                nb,
                timeout=180,
                kernel_name="rental_puzzle_local",
                resources={"metadata": {"path": str(root)}},
            ).execute()
            executed.append(name)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return executed


def main():
    for name in execute_all(Path.cwd()):
        print("PASS - Notebook executed without saving outputs:", name)


if __name__ == "__main__":
    main()

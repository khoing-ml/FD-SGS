import json
from pathlib import Path

import pytest


def test_colab_notebook_is_valid_python_and_covers_the_run():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/fd_sgs_colab.ipynb"
    if not notebook_path.exists():
        pytest.skip("Colab notebook is local and excluded from Git")
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for flag in ("--prompt-dataset", "--eval-rewards", "--wandb-mode", "--offload", "--dry-run"):
        assert flag in source
    assert 'SOURCE_MODE = "upload"' in source and 'SOURCE_MODE == "github"' in source
    assert 'WANDB_API_KEY = "PASTE_YOUR_WANDB_API_KEY_HERE"' in source
    assert "userdata.get(" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "fd_sgs_colab.ipynb", "exec")

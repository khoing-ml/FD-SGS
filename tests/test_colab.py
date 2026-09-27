import json
from pathlib import Path

import pytest


def test_colab_notebook_is_valid_python_and_covers_the_run():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/fd_sgs_colab.ipynb"
    if not notebook_path.exists():
        pytest.skip("Colab notebook is local and excluded from Git")
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    # Keep locally hardcoded tokens out of pytest's assertion output.
    source = "\n".join(
        "".join((line.split("=", 1)[0].strip() + " = <redacted>\n")
                if line.lstrip().startswith(("WANDB_API_KEY = ", "HF_TOKEN = ")) else line
                for line in cell["source"])
        for cell in notebook["cells"]
    )
    for flag in ("--prompt-dataset", "--eval-rewards", "--wandb-mode", "--offload", "--dry-run"):
        assert flag in source
    assert 'SOURCE_MODE = "upload"' in source and 'SOURCE_MODE == "github"' in source
    assert "WANDB_API_KEY = <redacted>" in source
    assert 'BACKBONE = "hyper-sdxl"' in source
    assert 'TWIN_SAMPLER = "auto"' in source
    assert 'EFFECTIVE_TWIN_SAMPLER = ' in source
    assert 'STEPS = ' in source
    assert '"--steps", str(STEPS)' in source
    assert '"hyper-sd3": "flow-grpo"' in source
    assert '"--lora-scale", str(LORA_SCALE)' in source
    assert 'HF_TOKEN = <redacted>' in source
    assert 'hyper_sd3.py' in source
    assert '#@title Probe sampler settings' in source
    assert 'EXPLORATION = ' in source
    assert 'DDIM_ETA = ' in source
    assert '"--ddim-eta", str(DDIM_ETA)' in source
    assert '"--unet-batch-size", str(UNET_BATCH_SIZE)' in source
    assert '"--backbone", BACKBONE' in source
    assert 'GUIDANCE_STEPS = [' in source
    assert 'GUIDANCE_EVAL = "final-rollout" if BACKBONE.startswith("hyper-")' in source
    assert '"--guidance-eval", GUIDANCE_EVAL' in source
    assert 'RHO_SCHEDULE = ' in source
    assert '"--rho-start-multiplier", str(RHO_START_MULTIPLIER)' in source
    assert '"--guidance-steps", *[str(step) for step in active_guidance_steps]' in source
    assert '"--exploration", str(EXPLORATION)' in source
    assert "userdata.get(" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            # Colab accepts notebook shell lines that are not Python syntax.
            python_source = "".join(line for line in cell["source"] if not line.lstrip().startswith("!"))
            compile(python_source, "fd_sgs_colab.ipynb", "exec")

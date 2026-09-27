import hashlib
import json
from pathlib import Path

import pytest

from fd_sgs.prompts import DATASETS, load_prompts


def test_bundled_prompt_snapshots_have_expected_provenance():
    root = Path(__file__).resolve().parents[1] / "fd_sgs/data/prompts"
    source = json.loads((root / "source.json").read_text())
    assert source["revision"] == "f410ab526c996bb9e7828ba92f02b68e35473a2f"
    for dataset, filename in DATASETS.items():
        records, metadata = load_prompts(dataset=dataset)
        assert len(records) == source["files"][filename]["prompts"]
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == source["files"][filename]["sha256"]
        assert metadata["sha256"] == source["files"][filename]["sha256"]
    ir, _ = load_prompts(dataset="image-reward", start=5, limit=2)
    assert len(ir) == 2 and ir[0].index == 5 and ir[0].id == "005937-0170"
    hps, _ = load_prompts(dataset="hpsv2", start=1, limit=1)
    assert hps[0].text == "Three people are preparing a meal in a small kitchen."


def test_custom_prompt_json_and_text_files(tmp_path):
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps([{"id": "a", "prompt": "  cat  "}, "dog"]))
    records, info = load_prompts(path=source, start=1)
    assert records[0].id == "1" and records[0].text == "dog"
    assert info["total_prompts"] == 2 and info["selected_prompts"] == 1
    lines = tmp_path / "prompts.txt"
    lines.write_text(" first \n\n second \n")
    records, _ = load_prompts(path=lines)
    assert [record.text for record in records] == ["first", "second"]


@pytest.mark.parametrize("kwargs", [dict(), dict(prompt="a", dataset="hpsv2"),
                                     dict(prompt="a", start=2), dict(prompt="a", limit=0)])
def test_prompt_selection_errors(kwargs):
    with pytest.raises(ValueError):
        load_prompts(**kwargs)

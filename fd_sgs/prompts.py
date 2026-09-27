"""Read the bundled SGS prompt snapshots or user-supplied TXT/JSON files."""

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path


DATASETS = {"image-reward": "benchmark_ir.json", "hpsv2": "hps_v2_all_eval.txt"}


@dataclass(frozen=True)
class Prompt:
    index: int
    id: str
    text: str


def load_prompts(*, prompt=None, dataset=None, path=None, start=0, limit=None):
    if sum(value is not None for value in (prompt, dataset, path)) != 1:
        raise ValueError("Choose exactly one of prompt, dataset or path")
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("start must be nonnegative and limit must be positive")
    source = {"type": "inline"}
    if prompt is not None:
        records = [prompt]
    else:
        root = files("fd_sgs").joinpath("data/prompts")
        resource = root.joinpath(DATASETS[dataset]) if dataset is not None else Path(path)
        text = resource.read_text(encoding="utf-8")
        if resource.name.endswith(".json"):
            records = json.loads(text)
            if not isinstance(records, list):
                raise ValueError("Prompt JSON must be a list of strings or {id, prompt} records")
        elif resource.name.endswith(".txt"):
            records = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            raise ValueError("Prompt files must have .txt or .json extensions")
        import hashlib
        source = {"type": "dataset" if dataset else "file", "name": dataset or str(path),
                  "sha256": hashlib.sha256(resource.read_bytes()).hexdigest()}
        if dataset is not None:
            source["upstream"] = json.loads(root.joinpath("source.json").read_text())
    prompts = []
    for index, record in enumerate(records):
        value = record.get("prompt") if isinstance(record, dict) else record
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid prompt at index {index}")
        identity = str(record.get("id", index)) if isinstance(record, dict) else str(index)
        prompts.append(Prompt(index, identity, value.strip()))
    selected = prompts[start:None if limit is None else start + limit]
    if not selected:
        raise ValueError("No prompts selected; check the file and --start-index/--max-prompts")
    source.update(total_prompts=len(prompts), start_index=start, selected_prompts=len(selected))
    return selected, source

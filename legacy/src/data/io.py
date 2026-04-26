from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from legacy.src.pcer_types import SerializedGraphPair


EVENT_FILE_PATTERN = re.compile(r"^(input|target)_(\d+)\.npy$")


@dataclass(frozen=True)
class DatasetSplitFiles:
    split_dir: Path
    pairs: tuple[tuple[Path, Path], ...]

    def __len__(self) -> int:
        return len(self.pairs)


def _event_index_from_name(path: Path) -> int:
    match = EVENT_FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected dataset file name: {path}")
    return int(match.group(2))


def discover_split_files(split_dir: str | Path) -> DatasetSplitFiles:
    root = Path(split_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset split directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset split path is not a directory: {root}")

    input_files = { _event_index_from_name(path): path for path in sorted(root.glob("input_*.npy")) }
    target_files = { _event_index_from_name(path): path for path in sorted(root.glob("target_*.npy")) }
    event_ids = sorted(set(input_files) & set(target_files))
    if not event_ids:
        raise FileNotFoundError(f"No paired input_/target_ files found in {root}")

    pairs = tuple((input_files[event_id], target_files[event_id]) for event_id in event_ids)
    return DatasetSplitFiles(split_dir=root, pairs=pairs)


def load_serialized_graph_pair(input_path: str | Path, target_path: str | Path) -> SerializedGraphPair:
    input_file = Path(input_path).resolve()
    target_file = Path(target_path).resolve()
    event_id = _event_index_from_name(input_file)
    if _event_index_from_name(target_file) != event_id:
        raise ValueError(f"Mismatched event ids for {input_file.name} and {target_file.name}")

    input_graph = np.load(input_file, allow_pickle=True).item()
    target_graph = np.load(target_file, allow_pickle=True).item()
    if not isinstance(input_graph, dict) or not isinstance(target_graph, dict):
        raise TypeError("Serialized dataset entries must contain Python dictionaries.")

    return SerializedGraphPair(
        event_id=event_id,
        input_graph=input_graph,
        target_graph=target_graph,
        input_path=input_file,
        target_path=target_file,
    )


def iter_serialized_graph_pairs(split_dir: str | Path, max_events: int | None = None):
    split_files = discover_split_files(split_dir)
    pairs = split_files.pairs if max_events is None else split_files.pairs[:max_events]
    for input_path, target_path in pairs:
        yield load_serialized_graph_pair(input_path, target_path)


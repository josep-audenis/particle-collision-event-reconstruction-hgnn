from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legacy.src.data.graph_builder import build_hetero_graph_record
from legacy.src.data.io import DatasetSplitFiles, discover_split_files, iter_serialized_graph_pairs
from legacy.src.pcer_types import HeteroGraphRecord


@dataclass
class HeteroEventDataset:
    split_dir: Path
    max_events: int | None = None

    def files(self) -> DatasetSplitFiles:
        return discover_split_files(self.split_dir)

    def __len__(self) -> int:
        file_count = len(self.files())
        return file_count if self.max_events is None else min(file_count, self.max_events)

    def iter_records(self):
        for pair in iter_serialized_graph_pairs(self.split_dir, max_events=self.max_events):
            yield build_hetero_graph_record(pair)

    def materialize(self) -> list[HeteroGraphRecord]:
        return list(self.iter_records())


def summarize_split(split_dir: str | Path, max_events: int | None = None) -> dict[str, int | str]:
    dataset = HeteroEventDataset(Path(split_dir), max_events=max_events)
    files = dataset.files()
    counts = {"events": 0, "tracks": 0, "track_edges": 0, "pvs": 0, "track_to_pv_edges": 0}
    for record in dataset.iter_records():
        counts["events"] += 1
        counts["tracks"] += int(record.track_features.shape[0])
        counts["track_edges"] += int(record.track_to_track_edges.shape[1])
        counts["pvs"] += int(record.pv_features.shape[0])
        counts["track_to_pv_edges"] += int(record.track_to_pv_edges.shape[1])

    if counts["events"] == 0:
        mean_tracks = 0.0
        mean_track_edges = 0.0
        mean_pvs = 0.0
    else:
        mean_tracks = counts["tracks"] / counts["events"]
        mean_track_edges = counts["track_edges"] / counts["events"]
        mean_pvs = counts["pvs"] / counts["events"]

    return {
        "split_dir": str(files.split_dir),
        "num_pairs": len(files),
        "loaded_events": counts["events"],
        "mean_tracks": round(mean_tracks, 2),
        "mean_track_edges": round(mean_track_edges, 2),
        "mean_pvs": round(mean_pvs, 2),
        "total_track_to_pv_edges": counts["track_to_pv_edges"],
    }

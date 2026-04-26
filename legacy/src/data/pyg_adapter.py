from __future__ import annotations

from pathlib import Path

from legacy.src.data.dataset import HeteroEventDataset
from legacy.src.pcer_types import HeteroGraphRecord


TRACKS = "tracks"
PVS = "pvs"
TO = "to"
REV_TO = "rev_to"


def _require_pyg():
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch and PyTorch Geometric are required for the PyG backend."
        ) from exc
    return torch, HeteroData


def to_pyg_heterodata(record: HeteroGraphRecord):
    torch, HeteroData = _require_pyg()

    data = HeteroData()
    data[TRACKS].x = torch.as_tensor(record.track_features, dtype=torch.float32)
    data[PVS].x = torch.as_tensor(record.pv_features, dtype=torch.float32)

    track_track = (TRACKS, TO, TRACKS)
    track_pv = (TRACKS, TO, PVS)
    pv_track = (PVS, REV_TO, TRACKS)

    data[track_track].edge_index = torch.as_tensor(record.track_to_track_edges, dtype=torch.long)
    data[track_track].edge_attr = torch.as_tensor(record.track_to_track_features, dtype=torch.float32)
    data[track_track].y_lcag = torch.as_tensor(record.lcag_edge_labels.argmax(axis=1), dtype=torch.long)
    data[track_track].y_same_b = torch.as_tensor(record.same_b_hadron_labels, dtype=torch.float32).unsqueeze(-1)

    data[track_pv].edge_index = torch.as_tensor(record.track_to_pv_edges, dtype=torch.long)
    data[track_pv].edge_attr = torch.as_tensor(record.track_to_pv_features, dtype=torch.float32)
    data[track_pv].y = torch.as_tensor(record.pv_assignment_labels, dtype=torch.float32).unsqueeze(-1)

    reverse_edge_index = record.track_to_pv_edges[[1, 0], :]
    data[pv_track].edge_index = torch.as_tensor(reverse_edge_index, dtype=torch.long)
    data[pv_track].edge_attr = torch.as_tensor(record.track_to_pv_features, dtype=torch.float32)

    data[TRACKS].y = torch.as_tensor(record.track_origin_labels, dtype=torch.float32).unsqueeze(-1)
    data[TRACKS].event_id = torch.as_tensor([record.event_id], dtype=torch.long)
    data["globals"].x = torch.as_tensor(
        [[record.track_features.shape[0], record.pv_features.shape[0], record.track_to_track_edges.shape[1]]],
        dtype=torch.float32,
    )
    return data


class PygHeteroEventDataset:
    def __init__(self, split_dir: str | Path, max_events: int | None = None) -> None:
        self._dataset = HeteroEventDataset(Path(split_dir), max_events=max_events)

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for record in self._dataset.iter_records():
            yield to_pyg_heterodata(record)

    def materialize(self):
        return [to_pyg_heterodata(record) for record in self._dataset.iter_records()]

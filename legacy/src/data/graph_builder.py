from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legacy.src.pcer_types import HeteroGraphRecord, SerializedGraphPair


TRACK_COORDINATE_SLICE = slice(0, 3)
TRACK_MOMENTUM_SLICE = slice(3, 6)
TRACK_MINIP_PV_SLICE = slice(6, 9)
TRACK_CHARGE_INDEX = 9
TRACK_TRUE_PV_SLICE = slice(10, 13)
PV_COORD_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PVDeduplicationResult:
    pv_features: np.ndarray
    pv_index_by_track: np.ndarray


def _require_key(mapping: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in mapping:
        raise KeyError(f"Expected key '{key}' in serialized graph entry.")
    return np.asarray(mapping[key])


def _as_edge_index(senders: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    if senders.shape != receivers.shape:
        raise ValueError("Senders and receivers must have the same shape.")
    return np.stack((senders.astype(np.int64), receivers.astype(np.int64)), axis=0)


def _infer_track_origin_labels(lcag_edge_labels: np.ndarray, edge_index: np.ndarray, num_tracks: int) -> np.ndarray:
    if lcag_edge_labels.shape[0] != edge_index.shape[1]:
        raise ValueError("LCAG labels and edge index must agree on the number of edges.")
    if lcag_edge_labels.ndim != 2 or lcag_edge_labels.shape[1] == 0:
        raise ValueError("LCAG edge labels must be a 2D one-hot or score matrix.")

    lcag_class = np.argmax(lcag_edge_labels, axis=1)
    positive_edge_mask = lcag_class > 0
    labels = np.zeros((num_tracks,), dtype=np.int64)
    if positive_edge_mask.any():
        positive_edges = edge_index[:, positive_edge_mask]
        labels[np.unique(positive_edges.reshape(-1))] = 1
    return labels


def _deduplicate_pvs(track_nodes: np.ndarray) -> PVDeduplicationResult:
    if track_nodes.ndim != 2 or track_nodes.shape[1] < TRACK_TRUE_PV_SLICE.stop:
        raise ValueError("Track node matrix does not contain the expected true PV coordinates.")

    true_pv_coords = track_nodes[:, TRACK_TRUE_PV_SLICE].astype(np.float32)
    if true_pv_coords.shape[0] == 0:
        return PVDeduplicationResult(
            pv_features=np.zeros((0, 4), dtype=np.float32),
            pv_index_by_track=np.zeros((0,), dtype=np.int64),
        )

    rounded = np.round(true_pv_coords / PV_COORD_TOLERANCE).astype(np.int64)
    unique_coords, inverse = np.unique(rounded, axis=0, return_inverse=True)
    pv_coords = (unique_coords.astype(np.float32) * PV_COORD_TOLERANCE).astype(np.float32)

    pv_counts = np.bincount(inverse, minlength=pv_coords.shape[0]).astype(np.float32).reshape(-1, 1)
    pv_features = np.concatenate((pv_coords, pv_counts), axis=1)
    return PVDeduplicationResult(
        pv_features=pv_features,
        pv_index_by_track=inverse.astype(np.int64),
    )


def _build_track_features(track_nodes: np.ndarray) -> np.ndarray:
    feature_blocks = (
        track_nodes[:, TRACK_COORDINATE_SLICE],
        track_nodes[:, TRACK_MOMENTUM_SLICE],
        track_nodes[:, TRACK_MINIP_PV_SLICE],
        track_nodes[:, TRACK_CHARGE_INDEX:TRACK_CHARGE_INDEX + 1],
    )
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def _build_track_to_pv_edges(track_nodes: np.ndarray, pv_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_tracks = track_nodes.shape[0]
    num_pvs = pv_features.shape[0]
    if num_tracks == 0 or num_pvs == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    track_index = np.repeat(np.arange(num_tracks, dtype=np.int64), num_pvs)
    pv_index = np.tile(np.arange(num_pvs, dtype=np.int64), num_tracks)

    track_origin = track_nodes[track_index, TRACK_COORDINATE_SLICE].astype(np.float32)
    pv_coords = pv_features[pv_index, :3].astype(np.float32)
    delta = track_origin - pv_coords
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    features = np.concatenate((delta, distance), axis=1).astype(np.float32)
    return np.stack((track_index, pv_index), axis=0), features


def _build_pv_assignment_labels(track_to_pv_edges: np.ndarray, pv_index_by_track: np.ndarray) -> np.ndarray:
    if track_to_pv_edges.shape[1] == 0:
        return np.zeros((0,), dtype=np.int64)
    track_ids = track_to_pv_edges[0]
    pv_ids = track_to_pv_edges[1]
    return (pv_index_by_track[track_ids] == pv_ids).astype(np.int64)


def build_hetero_graph_record(pair: SerializedGraphPair) -> HeteroGraphRecord:
    input_graph = pair.input_graph
    target_graph = pair.target_graph

    track_nodes = _require_key(input_graph, "nodes").astype(np.float32)
    track_edge_features = _require_key(input_graph, "edges").astype(np.float32)
    senders = _require_key(input_graph, "senders").astype(np.int64)
    receivers = _require_key(input_graph, "receivers").astype(np.int64)
    lcag_edge_labels = _require_key(target_graph, "edges").astype(np.float32)

    track_to_track_edges = _as_edge_index(senders, receivers)
    pv_info = _deduplicate_pvs(track_nodes)
    track_to_pv_edges, track_to_pv_features = _build_track_to_pv_edges(track_nodes, pv_info.pv_features)

    track_origin_labels = _infer_track_origin_labels(lcag_edge_labels, track_to_track_edges, track_nodes.shape[0])
    same_b_hadron_labels = (np.argmax(lcag_edge_labels, axis=1) > 0).astype(np.int64)
    pv_assignment_labels = _build_pv_assignment_labels(track_to_pv_edges, pv_info.pv_index_by_track)

    metadata = {
        "event_id": pair.event_id,
        "input_path": str(pair.input_path),
        "target_path": str(pair.target_path),
        "num_tracks": int(track_nodes.shape[0]),
        "num_track_edges": int(track_to_track_edges.shape[1]),
        "num_pvs": int(pv_info.pv_features.shape[0]),
        "num_track_to_pv_edges": int(track_to_pv_edges.shape[1]),
    }

    return HeteroGraphRecord(
        event_id=pair.event_id,
        track_features=_build_track_features(track_nodes),
        pv_features=pv_info.pv_features,
        track_to_track_edges=track_to_track_edges,
        track_to_track_features=track_edge_features,
        track_to_pv_edges=track_to_pv_edges,
        track_to_pv_features=track_to_pv_features,
        track_origin_labels=track_origin_labels,
        same_b_hadron_labels=same_b_hadron_labels,
        pv_assignment_labels=pv_assignment_labels,
        lcag_edge_labels=lcag_edge_labels,
        metadata=metadata,
    )


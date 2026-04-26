from __future__ import annotations

from legacy.src.pcer_types import GraphSchema, TaskTarget


DEFAULT_HETERO_GRAPH_SCHEMA = GraphSchema(
    track_features=(
        "track_px",
        "track_py",
        "track_pz",
        "track_pt",
        "track_ip",
        "track_eta",
        "track_charge",
        "track_origin_x",
        "track_origin_y",
        "track_origin_z",
    ),
    pv_features=(
        "pv_x",
        "pv_y",
        "pv_z",
        "pv_n_tracks",
    ),
    track_to_track_features=(
        "delta_theta",
        "delta_z",
        "distance_perp",
        "same_pv_reco",
    ),
    track_to_pv_features=(
        "delta_x",
        "delta_y",
        "delta_z",
        "distance_3d",
    ),
)


DEFAULT_TASKS = (
    TaskTarget(name="track_origin", target_key="from_b_hadron", level="node", classes=2, weight=1.0),
    TaskTarget(name="same_b_hadron", target_key="same_b_hadron", level="edge", classes=2, weight=1.0),
    TaskTarget(name="pv_assignment", target_key="assigned_pv", level="edge", classes=2, weight=1.0),
)


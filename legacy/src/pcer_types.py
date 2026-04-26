from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


NodeType = Literal["track", "pv"]
EdgeType = Literal["track_to_track", "track_to_pv"]
TaskName = Literal["track_origin", "same_b_hadron", "pv_assignment"]
DatasetType = Literal["heterogeneous", "homogeneous"]
BackendName = Literal["tensorflow_graph_nets", "pytorch_geometric"]


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class GraphSchema:
    track_features: tuple[str, ...]
    pv_features: tuple[str, ...]
    track_to_track_features: tuple[str, ...]
    track_to_pv_features: tuple[str, ...]


@dataclass(frozen=True)
class TaskTarget:
    name: TaskName
    target_key: str
    level: Literal["node", "edge"]
    classes: int
    weight: float = 1.0


@dataclass(frozen=True)
class DatasetConfig:
    train_path: Path
    val_path: Path
    test_path: Path | None
    data_type: DatasetType
    cache_dir: Path
    batch_size: int
    max_events: int | None = None
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: BackendName
    latent_size: int
    num_layers: int
    num_message_passing_steps: int
    use_weighted_message_passing: bool = True
    use_node_pruning: bool = True
    use_edge_pruning: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float = 0.0
    dropped_lr_epochs: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    checkpoint_dir: Path
    model_file: Path
    seed: int = 7
    log_every: int = 25
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    tasks: tuple[TaskTarget, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SerializedGraphPair:
    event_id: int
    input_graph: dict[str, np.ndarray]
    target_graph: dict[str, np.ndarray]
    input_path: Path
    target_path: Path


@dataclass(frozen=True)
class HeteroGraphRecord:
    event_id: int
    track_features: np.ndarray
    pv_features: np.ndarray
    track_to_track_edges: np.ndarray
    track_to_track_features: np.ndarray
    track_to_pv_edges: np.ndarray
    track_to_pv_features: np.ndarray
    track_origin_labels: np.ndarray
    same_b_hadron_labels: np.ndarray
    pv_assignment_labels: np.ndarray
    lcag_edge_labels: np.ndarray
    metadata: dict[str, int | float | str]


@dataclass(frozen=True)
class ModelBuildResult:
    model: Any
    backend: BackendName
    parameter_count: int | None
    notes: tuple[str, ...] = field(default_factory=tuple)

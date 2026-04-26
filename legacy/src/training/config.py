from __future__ import annotations

from pathlib import Path

from legacy.src.data.schema import DEFAULT_TASKS
from legacy.src.pcer_types import DatasetConfig, ExperimentConfig, ModelConfig, OptimizerConfig, TaskTarget, TrainingConfig


def _load_yaml(config_path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load training configs. Install the dependencies from requirements.txt first."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _build_tasks(raw_tasks: list[dict] | None) -> tuple[TaskTarget, ...]:
    task_items = raw_tasks or [task.__dict__ for task in DEFAULT_TASKS]
    return tuple(TaskTarget(**task_item) for task_item in task_items)


def load_config(config_path: str) -> ExperimentConfig:
    path = Path(config_path).resolve()
    raw_config = _load_yaml(path)
    base_dir = path.parent

    dataset = DatasetConfig(
        train_path=_resolve_path(base_dir, raw_config["dataset"]["train_path"]),
        val_path=_resolve_path(base_dir, raw_config["dataset"]["val_path"]),
        test_path=_resolve_path(base_dir, raw_config["dataset"].get("test_path")),
        data_type=raw_config["dataset"].get("data_type", "heterogeneous"),
        cache_dir=_resolve_path(base_dir, raw_config["dataset"].get("cache_dir", "../../data/cache")),
        batch_size=int(raw_config["dataset"].get("batch_size", 8)),
        max_events=raw_config["dataset"].get("max_events"),
        num_workers=int(raw_config["dataset"].get("num_workers", 0)),
    )

    model = ModelConfig(
        name=raw_config["model"].get("name", "paper_hgnn"),
        backend=raw_config["model"].get("backend", "tensorflow_graph_nets"),
        latent_size=int(raw_config["model"]["latent_size"]),
        num_layers=int(raw_config["model"]["num_layers"]),
        num_message_passing_steps=int(raw_config["model"]["num_message_passing_steps"]),
        use_weighted_message_passing=bool(raw_config["model"].get("use_weighted_message_passing", True)),
        use_node_pruning=bool(raw_config["model"].get("use_node_pruning", True)),
        use_edge_pruning=bool(raw_config["model"].get("use_edge_pruning", True)),
    )

    optimizer = OptimizerConfig(
        learning_rate=float(raw_config["optimizer"]["learning_rate"]),
        weight_decay=float(raw_config["optimizer"].get("weight_decay", 0.0)),
        dropped_lr_epochs=int(raw_config["optimizer"].get("dropped_lr_epochs", 0)),
    )

    training = TrainingConfig(
        epochs=int(raw_config["training"]["epochs"]),
        checkpoint_dir=_resolve_path(base_dir, raw_config["training"].get("checkpoint_dir", "../../outputs/checkpoints")),
        model_file=_resolve_path(base_dir, raw_config["training"].get("model_file", "../../outputs/models/paper_hgnn")),
        seed=int(raw_config["training"].get("seed", 7)),
        log_every=int(raw_config["training"].get("log_every", 25)),
        early_stopping_patience=raw_config["training"].get("early_stopping_patience"),
    )

    return ExperimentConfig(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        training=training,
        tasks=_build_tasks(raw_config.get("tasks")),
    )


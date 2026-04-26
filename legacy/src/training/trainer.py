from __future__ import annotations

from legacy.src.data.dataset import summarize_split
from legacy.src.models import build_model, build_model_spec
from legacy.src.pcer_types import ExperimentConfig


def summarize_experiment(config: ExperimentConfig) -> str:
    model_spec = build_model_spec(config)
    task_summary = ", ".join(f"{task.name}:{task.level}:{task.classes}" for task in config.tasks)
    split_summaries: list[str] = []
    for split_name, split_path in (("train", config.dataset.train_path), ("val", config.dataset.val_path)):
        if split_path.exists():
            split_info = summarize_split(split_path, max_events=min(config.dataset.max_events, 32) if config.dataset.max_events else 32)
            split_summaries.append(
                f"{split_name}_events={split_info['loaded_events']} mean_tracks={split_info['mean_tracks']} "
                f"mean_track_edges={split_info['mean_track_edges']} mean_pvs={split_info['mean_pvs']}"
            )
        else:
            split_summaries.append(f"{split_name}_events=unavailable")

    build_summary = "model_build=not_attempted"
    try:
        build_result = build_model(config)
        parameter_text = build_result.parameter_count if build_result.parameter_count is not None else "unknown"
        build_summary = f"model_build=ok backend={build_result.backend} parameters={parameter_text}"
    except Exception as exc:
        build_summary = f"model_build=failed reason={type(exc).__name__}:{exc}"

    return "\n".join(
        (
            f"model={model_spec.name}",
            f"backend={model_spec.backend}",
            f"dataset_type={config.dataset.data_type}",
            f"train_path={config.dataset.train_path}",
            f"val_path={config.dataset.val_path}",
            f"batch_size={config.dataset.batch_size}",
            f"latent_size={model_spec.latent_size}",
            f"num_layers={model_spec.num_layers}",
            f"message_passing_steps={model_spec.num_message_passing_steps}",
            f"weighted_message_passing={model_spec.use_weighted_message_passing}",
            f"node_pruning={model_spec.use_node_pruning}",
            f"edge_pruning={model_spec.use_edge_pruning}",
            f"tasks={task_summary}",
            f"epochs={config.training.epochs}",
            f"learning_rate={config.optimizer.learning_rate}",
            f"checkpoint_dir={config.training.checkpoint_dir}",
            f"model_file={config.training.model_file}",
            build_summary,
            *split_summaries,
        )
    )


def train_experiment(config: ExperimentConfig, dry_run: bool = False) -> dict[str, str | int | float]:
    if dry_run:
        return {
            "status": "dry_run",
            "summary": summarize_experiment(config),
        }

    raise NotImplementedError(
        "The end-to-end training loop is not ported yet. The serialized dataset adapter and PyG hetero model surface are now in place."
    )
